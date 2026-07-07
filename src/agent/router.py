from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Database and Indexing
from sqlalchemy import create_engine
from pymongo import MongoClient
from llama_index.core import SQLDatabase, VectorStoreIndex
from llama_index.core.query_engine import NLSQLTableQueryEngine, RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import ReActAgent

# Models
from src.common.vertex_llm import VertexGeminiLLM
from src.common.chroma_compat import apply_chroma_empty_filter_fix
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# Load env variables
load_dotenv()

# chromadb 0.5.x rejects empty where={}; make unfiltered vector search work
apply_chroma_empty_filter_fix()

app = FastAPI(title="ERP RAG Agent Router", description="Conversational RAG using Vertex AI (Gemini) and LlamaIndex")

# Serve frontend directly from backend
public_dir = os.path.join(os.path.dirname(__file__), "public")
os.makedirs(public_dir, exist_ok=True)

# Database constants
POSTGRES_URI = os.getenv("POSTGRES_URI", "postgresql://admin:adminpassword@localhost:5432/erp_database")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://admin:adminpassword@localhost:27017/")

# Global variables for the engine
router_query_engine = None

def initialize_engines():
    global router_query_engine
    
    print("Initializing Models (Vertex AI Gemini via ADC, with model/region fallback)...")
    # Primary: gemini-2.0-flash-001 (europe-west6), falling back across
    # gemini-2.5-flash / gemini-1.5-flash and the us-central1 region.
    llm = VertexGeminiLLM.from_env()
    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    
    # Global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # 1. Postgres Setup (Structured Data)
    print("Connecting to Postgres...")
    pg_engine = create_engine(POSTGRES_URI)
    sql_database = SQLDatabase(pg_engine, include_tables=["users", "tickets", "ticket_blockers"])
    
    sql_query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=["users", "tickets", "ticket_blockers"],
        llm=llm,
        context_str_prefix=(
            "When filtering text columns, always use case-insensitive matching with short single "
            "keywords, e.g. title ILIKE '%FDA%'. Never use case-sensitive LIKE or long multi-word "
            "phrases: a search for 'FDA compliance' must still match a title like "
            "'FDA 21 CFR Part 11 Compliance Audit'.\n"
        )
    )
    
    # 2. Chroma Setup (Unstructured Data)
    CHROMA_PATH = os.getenv("CHROMA_PATH", "../../chroma_data") # Project Root
    print(f"Connecting to Chroma (Vector Store at {CHROMA_PATH})...")
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import chromadb
    
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = chroma_client.get_or_create_collection("messages_index")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Check item count
    count = chroma_collection.count()
    print(f"Vector Store initialized with {count} documents.")
    
    # Recreate index over Chroma
    chroma_index = VectorStoreIndex.from_vector_store(vector_store)
    mongo_query_engine = chroma_index.as_query_engine(similarity_top_k=10)

    # 3. Build the Agentic Router (Upgraded to Multi-Step Synthesis)
    print("Building Multi-Source Synthesis Agent...")
    
    sql_tool = QueryEngineTool.from_defaults(
        query_engine=sql_query_engine,
        name="sql_analytics_tool",
        description=(
            "Useful for getting strict quantitative data from the Postgres database. "
            "Contains tables: 'users' (user_id, name, department, role, access_level, salary, manager_id) "
            "and 'tickets' (ticket_id, title, status, priority, assignee_id). "
            "Use this for employee lookups, salary data, and ticket status counts. "
            "Always use this when asked for a person's department, role, or salary."
        )
    )
    
    mongo_tool = QueryEngineTool.from_defaults(
        query_engine=mongo_query_engine,
        name="unstructured_chat_tool",
        description=(
            "Useful for searching through Slack channel messages and conversational logs. "
            "Use this for questions about 'what happened', 'why something is blocked', 'project status updates', "
            "and general engineer conversations. If you need context about a specific project or training status "
            "mentioned in chat, use this tool."
        )
    )
    
    # 3. Build Neo4j Custom Tool
    from llama_index.core.query_engine import CustomQueryEngine
    from neo4j import GraphDatabase
    neo4j_driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "adminpassword"))
    )
    neo4j_schema = """
Nodes:
- Person (user_id, name, role, salary)
- Ticket (ticket_id, title, type, priority)
- Department (name)
- Status (name)
- Sprint (name)

Relationships:
- (Person)-[:WORKS_IN]->(Department)
- (Person)-[:REPORTS_TO]->(Person)
- (Person)-[:ASSIGNED_TO]->(Ticket)
- (Person)-[:REPORTED]->(Ticket)
- (Ticket)-[:BLOCKS]->(Ticket)
- (Ticket)-[:CURRENT_STATUS]->(Status)
- (Ticket)-[:ASSIGNED_TO_SPRINT]->(Sprint)

For blocker questions always use BLOCKS, which reads in natural direction:
(a)-[:BLOCKS]->(b) means a blocks b, so a is the blocker of b.
"Who is blocking X": MATCH (blocker:Ticket)-[:BLOCKS]->(x:Ticket {ticket_id:'X'}) RETURN blocker
"Which tickets does X block": MATCH (x:Ticket {ticket_id:'X'})-[:BLOCKS]->(t:Ticket) RETURN t
"""

    class Neo4jCustomQueryEngine(CustomQueryEngine):
        def custom_query(self, query_str: str):
            prompt = f"Given the following Neo4j schema:\n{neo4j_schema}\n\nWrite a Cypher query to answer: {query_str}\nOnly return the pure cypher query without any markdown tags or prefix."
            cypher_query = llm.complete(prompt).text.strip().replace("```cypher", "").replace("```", "")
            with neo4j_driver.session() as session:
                result = session.run(cypher_query)
                records = [dict(record) for record in result]
            return f"Cypher: {cypher_query}\nRecords: {records}"

    neo4j_tool = QueryEngineTool.from_defaults(
        query_engine=Neo4jCustomQueryEngine(),
        name="neo4j_graph_tool",
        description="Useful for querying complex relationships like chains of command, manager dependencies, or complex ticket blockers (e.g. 'what tickets block X' or 'who is the manager of the person blocking Y')."
    )

    # 4. General Knowledge Fallback Tool
    # Handles questions that are NOT about company data — lets the LLM answer from its own knowledge
    class GeneralKnowledgeQueryEngine(CustomQueryEngine):
        def custom_query(self, query_str: str):
            prompt = (
                "You are a helpful enterprise AI assistant with broad general knowledge. "
                "The user is asking a question that does not require looking up any company data — "
                "answer it using your own training knowledge. Be concise and accurate.\n\n"
                f"Question: {query_str}"
            )
            return llm.complete(prompt).text.strip()

    general_tool = QueryEngineTool.from_defaults(
        query_engine=GeneralKnowledgeQueryEngine(),
        name="general_knowledge_tool",
        description=(
            "Use this tool ONLY for general knowledge questions that are NOT about this company's specific "
            "employees, tickets, projects, Slack messages, or data. Examples of questions for this tool: "
            "'what is C-suite?', 'what does Agile mean?', 'explain HL7 FHIR', 'what is FDA 21 CFR Part 11', "
            "'how does a RAG system work?', 'what is a Product Manager?'. "
            "Do NOT use this for anything involving specific people, salaries, tickets, blockers, or Slack logs."
        )
    )

    # LLMMultiSelector allows the router to read the question, determine it needs BOTH SQL and Vector context,
    # pull them both simultaneously, and synthesize an expert final answer.
    from llama_index.core.selectors import LLMMultiSelector
    router_query_engine = RouterQueryEngine(
        selector=LLMMultiSelector.from_defaults(llm=llm),
        query_engine_tools=[sql_tool, mongo_tool, neo4j_tool, general_tool],
    )
    print("Initialization Complete!")

@app.on_event("startup")
def on_startup():
    initialize_engines()

class ChatRequest(BaseModel):
    query: str
    history: list = []
    user_id: str = None # For future RBAC integration

import json
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler, CBEventType

# Log store for the current request
class PayloadLogger:
    def __init__(self):
        self.logs = []
    
    def log(self, step, payload):
        self.logs.append({"step": step, "payload": payload})

payload_logger = PayloadLogger()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    global router_query_engine
    if not router_query_engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    
    logs = []
    try:
        # Multi-Source Synthesis
        # We simulate the payload logging since LlamaIndex makes multi-calls
        # Settings.llm.model will give us the current active model.
        model_used = Settings.llm.model
        
        # 1.5 Context Condensation
        query_to_submit = request.query
        if request.history:
            history_str = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in request.history])
            condense_prompt = (
                "Given the following conversation history and the latest user query, "
                "rewrite the user query to be a standalone query that captures all relevant context from the history. "
                "Only return the standalone query, without any prefixes, quotes or explanations. If the query is already standalone, return it as is.\n\n"
                f"Conversation History:\n{history_str}\n\n"
                f"Latest Query: {request.query}\n\n"
                "Standalone Query:"
            )
            query_to_submit = Settings.llm.complete(condense_prompt).text.strip()
            
            logs.append({
                "step": f"[{model_used}] Context Condensation",
                "payload": {
                    "original": request.query,
                    "condensed": query_to_submit
                }
            })

        logs.append({
            "step": f"[{model_used}] Input",
            "payload": {
                "model": model_used,
                "messages": [{"role": "user", "content": query_to_submit}],
                "temperature": 0.1
            }
        })
        
        result = router_query_engine.query(query_to_submit)
        
        # Get which tools it used safely
        meta = getattr(result, "metadata", None) or {}
        selections = meta.get("selector_result", [])
        
        if hasattr(selections, "selections"):
            source = str([s.reason for s in selections.selections])
        else:
            source = str(selections)
            
        model_used = Settings.llm.model
        logs.append({
            "step": f"[{model_used}] Output",
            "payload": {
                "response": str(result),
                "source_nodes": [str(n.node.get_content()[:200]) + "..." for n in result.source_nodes] if hasattr(result, "source_nodes") else []
            }
        })
            
    except Exception as e:
        print(f"Exception triggered: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    return {
        "query": request.query,
        "response": str(result),
        "source": source,
        "detailed_logs": logs
    }

from fastapi import Form

@app.post("/chat/audio")
async def chat_audio_endpoint(audio: UploadFile = File(...), history: str = Form("[]")):
    global router_query_engine
    if not router_query_engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    
    import tempfile
    from groq import Groq as NativeGroqClient
    
    logs = []
    
    # Save the audio blob temporarily to disk
    audio_data = await audio.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name

    try:
        # 1. Transcribe the audio using Groq's insanely fast Whisper Large v3
        logs.append({
            "step": "Whisper v3 Input",
            "payload": {
                "model": "whisper-large-v3",
                "file_size": len(audio_data),
                "format": "webm"
            }
        })
        
        client = NativeGroqClient(api_key=os.environ.get("GROQ_API_KEY"))
        with open(tmp_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
              file=(os.path.basename(tmp_path), file.read()),
              model="whisper-large-v3",
              response_format="text"
            )
            
        transcribed_text = transcription
        logs.append({
            "step": "Whisper v3 Output",
            "payload": {
                "text": transcribed_text
            }
        })
        
        # 1.5 Context Condensation
        model_used = Settings.llm.model
        history_list = json.loads(history)
        query_to_submit = transcribed_text
        
        if history_list:
            history_str = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history_list])
            condense_prompt = (
                "Given the following conversation history and the latest user query, "
                "rewrite the user query to be a standalone query that captures all relevant context from the history. "
                "Only return the standalone query, without any prefixes, quotes or explanations. If the query is already standalone, return it as is.\n\n"
                f"Conversation History:\n{history_str}\n\n"
                f"Latest Query: {transcribed_text}\n\n"
                "Standalone Query:"
            )
            query_to_submit = Settings.llm.complete(condense_prompt).text.strip()
            
            logs.append({
                "step": f"[{model_used}] Context Condensation",
                "payload": {
                    "original": transcribed_text,
                    "condensed": query_to_submit
                }
            })
            
        # 2. Pass the transcribed text into the Synthesis RAG Router
        logs.append({
            "step": f"[{model_used}] Input",
            "payload": {
                "model": model_used,
                "messages": [{"role": "user", "content": query_to_submit}]
            }
        })
        
        result = router_query_engine.query(query_to_submit)
        
        # 3. Get Source Reasonings safely
        meta = getattr(result, "metadata", None) or {}
        selections = meta.get("selector_result", [])
        
        if hasattr(selections, "selections"):
            source = str([s.reason for s in selections.selections])
        else:
            source = str(selections)
            
        model_used = Settings.llm.model
        logs.append({
            "step": f"[{model_used}] Output",
            "payload": {
                "response": str(result),
                "source": source
            }
        })
            
        return {
            "query": transcribed_text,
            "response": str(result),
            "source": source,
            "detailed_logs": logs
        }
    except Exception as e:
        print(f"Exception triggered in audio endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)

# Serve frontend directly from backend via Root (MUST be at bottom to avoid intercepting APIs)
app.mount("/", StaticFiles(directory=public_dir, html=True), name="public")

if __name__ == "__main__":
    import uvicorn
    # Important: Ensure python-multipart and groq are installed in your environment!
    uvicorn.run(app, host="0.0.0.0", port=8000)
