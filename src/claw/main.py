import os
import json
import traceback
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

STATIC_DIR = Path(__file__).parent / "public"

app = FastAPI(title="OpenClaw Agent", version="1.0.0")

# ─── Global State ─────────────────────────────────────────────────────────────
react_agent = None
llm = None
groq_client = None


# ─── Request Models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    history: List[dict] = []


# ─── Agent Initialization ─────────────────────────────────────────────────────
def initialize_claw_agent():
    from llama_index.core import Settings, SQLDatabase, VectorStoreIndex
    from llama_index.core.query_engine import NLSQLTableQueryEngine, CustomQueryEngine
    from llama_index.core.tools import QueryEngineTool
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from src.common.vertex_llm import VertexGeminiLLM
    from src.common.chroma_compat import apply_chroma_empty_filter_fix
    from sqlalchemy import create_engine
    import chromadb

    # chromadb 0.5.x rejects empty where={}; make unfiltered vector search work
    apply_chroma_empty_filter_fix()

    # ── Agent import ──────────────────────────────────────────────────────────
    # llama-index-core 0.11.x: build via ReActAgent.from_tools (handles memory).
    from llama_index.core.agent import ReActAgent

    # LLM — Vertex AI Gemini via ADC, with model/region fallback
    active_llm = VertexGeminiLLM.from_env()

    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    Settings.embed_model = embed_model
    Settings.llm = active_llm

    # ── 1. SQL Tool (PostgreSQL) ──────────────────────────────────────────────
    pg_uri = os.getenv("POSTGRES_URI", "postgresql://admin:adminpassword@postgres:5432/erp_database")
    pg_engine = create_engine(pg_uri)
    sql_database = SQLDatabase(
        pg_engine,
        include_tables=["users", "tickets", "ticket_blockers"],
        sample_rows_in_table_info=2
    )
    sql_query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        llm=active_llm,
        tables=["users", "tickets", "ticket_blockers"],
        context_str_prefix=(
            "Schema context:\n"
            "- users(user_id, name, role, department, salary, manager_id): "
            "C-suite roles are CEO, CFO, CTO, CPO. Departments include Engineering, QA, DevOps, "
            "Sales, HR, Finance, Product, BA.\n"
            "- tickets(ticket_id, title, type, status, priority, assignee_id, reporter_id, sprint): "
            "status values: Open, In Progress, In Review, Blocked, Done. "
            "priority values: Critical, High, Medium, Low.\n"
            "- ticket_blockers(blocker_id, ticket_id, blocked_by_ticket_id): links blocking relationships.\n"
            "When filtering text columns, always use case-insensitive matching with short single "
            "keywords, e.g. title ILIKE '%FDA%'. Never use case-sensitive LIKE or long multi-word "
            "phrases: a search for 'FDA compliance' must still match a title like "
            "'FDA 21 CFR Part 11 Compliance Audit'.\n"
        )
    )
    sql_tool = QueryEngineTool.from_defaults(
        query_engine=sql_query_engine,
        name="sql_database_tool",
        description=(
            "Query structured company data: employees (name, role, department, salary, manager), "
            "tickets (status, priority, type, assignee, reporter, sprint), and blocker relationships. "
            "Use for questions about people, org hierarchy, ticket status, or workload distribution. "
            "Pass a plain-English question as input, never raw SQL."
        )
    )

    # ── 2. ChromaDB Vector Tool (Slack messages) ──────────────────────────────
    chroma_path = os.getenv("CHROMA_PATH", "/app/chroma_data")
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    try:
        collection = chroma_client.get_collection("messages_index")
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.core import StorageContext
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        chroma_index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context, embed_model=embed_model
        )
        chroma_engine = chroma_index.as_query_engine(llm=active_llm, similarity_top_k=5)
    except Exception:
        from llama_index.core import VectorStoreIndex
        chroma_index = VectorStoreIndex([])
        chroma_engine = chroma_index.as_query_engine(llm=active_llm)

    chroma_tool = QueryEngineTool.from_defaults(
        query_engine=chroma_engine,
        name="slack_vector_tool",
        description=(
            "Search Slack messages semantically. Use for questions about team discussions, "
            "blockers mentioned in conversations, what teams are saying, channel-specific updates, "
            "or any unstructured communication logs."
        )
    )

    # ── 3. Neo4j Graph Tool ───────────────────────────────────────────────────
    neo4j_schema = """
Nodes: Person(user_id, name, role, salary), Ticket(ticket_id, title, type, priority),
Department(name), Status(name), Sprint(name)
Relationships:
- (Person)-[:WORKS_IN]->(Department)
- (Person)-[:REPORTS_TO]->(Person)
- (Person)-[:ASSIGNED_TO]->(Ticket)
- (Ticket)-[:BLOCKS]->(Ticket)
- (Ticket)-[:CURRENT_STATUS]->(Status)
- (Ticket)-[:ASSIGNED_TO_SPRINT]->(Sprint)

For blocker questions always use BLOCKS, which reads in natural direction:
(a)-[:BLOCKS]->(b) means a blocks b, so a is the blocker of b.
"Who is blocking X": MATCH (blocker:Ticket)-[:BLOCKS]->(x:Ticket {ticket_id:'X'}) RETURN blocker
"Which tickets does X block": MATCH (x:Ticket {ticket_id:'X'})-[:BLOCKS]->(t:Ticket) RETURN t
"""

    class Neo4jQueryEngine(CustomQueryEngine):
        def custom_query(self, query_str: str):
            from neo4j import GraphDatabase
            neo4j_driver = GraphDatabase.driver(
                os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
                auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "adminpassword"))
            )
            prompt = (
                f"Given this Neo4j schema:\n{neo4j_schema}\n\n"
                f"Write a Cypher query to answer: {query_str}\n"
                "Return only the pure Cypher query, no markdown or explanation."
            )
            cypher = active_llm.complete(prompt).text.strip().replace("```cypher", "").replace("```", "")
            try:
                with neo4j_driver.session() as session:
                    result = session.run(cypher)
                    records = [dict(r) for r in result]
                return f"Cypher: {cypher}\nResults: {records}"
            except Exception as e:
                return f"Neo4j error: {e}\nCypher attempted: {cypher}"
            finally:
                neo4j_driver.close()

    neo4j_tool = QueryEngineTool.from_defaults(
        query_engine=Neo4jQueryEngine(),
        name="neo4j_graph_tool",
        description=(
            "Query the organizational graph for relationship-based questions: "
            "reporting chains, who manages whom, ticket blocker chains, "
            "department composition, or multi-hop relationship traversal."
        )
    )

    # ── 4. General Knowledge Fallback ─────────────────────────────────────────
    class GeneralKnowledgeEngine(CustomQueryEngine):
        def custom_query(self, query_str: str):
            prompt = (
                "You are OpenClaw, a powerful enterprise AI agent with deep general knowledge. "
                "Answer the following question from your own knowledge — no company data needed.\n\n"
                f"Question: {query_str}"
            )
            return active_llm.complete(prompt).text.strip()

    general_tool = QueryEngineTool.from_defaults(
        query_engine=GeneralKnowledgeEngine(),
        name="general_knowledge_tool",
        description=(
            "Answer general knowledge questions NOT about this company's data. "
            "Examples: 'what is C-suite', 'explain Agile', 'what is FHIR', "
            "'what does a Product Manager do'. Do NOT use for company-specific queries."
        )
    )

    # ── Build agent (llama-index-core 0.11.x) ─────────────────────────────────
    _tools = [sql_tool, chroma_tool, neo4j_tool, general_tool]

    agent = ReActAgent.from_tools(
        tools=_tools,
        llm=active_llm,
        verbose=True,
        max_iterations=10,
    )

    print("STRATA ReACT Agent initialized successfully.")
    return agent, active_llm



# ─── History Condensation ─────────────────────────────────────────────────────
def condense_query(query: str, history: List[dict], llm) -> str:
    if not history:
        return query
    history_str = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in history[-6:]
    )
    prompt = (
        f"Given this conversation:\n{history_str}\n\n"
        f"Rephrase the follow-up question as a standalone query: {query}\n"
        "Return only the rephrased question."
    )
    try:
        return llm.complete(prompt).text.strip()
    except Exception:
        return query


# ─── Step Info Extraction ─────────────────────────────────────────────────────
def extract_step_info(step_output) -> dict:
    content = ""
    try:
        if hasattr(step_output.output, 'message') and step_output.output.message:
            content = step_output.output.message.content
        elif hasattr(step_output.output, 'response'):
            content = str(step_output.output.response)
        else:
            content = str(step_output.output)
    except Exception:
        content = str(step_output)
    return {"content": content, "is_done": step_output.is_done}


# ─── Verbose trace parser (LlamaIndex 0.12.x) ────────────────────────────────
def parse_verbose_trace(raw: str) -> list:
    """Extract Thought/Action/Observation lines from agent verbose stdout."""
    steps = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("thought:") or "> thought" in low:
            steps.append({"content": line, "is_done": False})
        elif low.startswith("action:") or low.startswith("action input:") or "> action" in low:
            steps.append({"content": line, "is_done": False})
        elif low.startswith("observation:") or "> observation" in low:
            steps.append({"content": line, "is_done": False})
    return steps


# ─── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global react_agent, llm
    try:
        react_agent, llm = initialize_claw_agent()
    except Exception as e:
        print(f"[STRATA] Initialization FAILED: {e}")
        print(f"[STRATA] Full traceback:")
        traceback.print_exc()


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "operational", "agent": "openclaw-react", "ready": react_agent is not None}


@app.post("/chat")
async def chat(request: ChatRequest):
    if not react_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized yet")

    condensed = condense_query(request.query, request.history, llm)

    import io, sys, asyncio
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()

    try:
        # llama-index-core 0.11.x: ReActAgent.chat() is sync — run it off the
        # event loop so the verbose ReAct trace is captured from stdout.
        response = await asyncio.get_event_loop().run_in_executor(
            None, react_agent.chat, condensed
        )
        sys.stdout = old_stdout

        verbose_output = captured.getvalue()
        reasoning_steps = parse_verbose_trace(verbose_output)

        # AgentChatResponse.response is the final answer string
        if hasattr(response, 'response') and response.response is not None:
            response_text = str(response.response)
        else:
            response_text = str(response)

        return JSONResponse({
            "query": request.query,
            "condensed_query": condensed,
            "response": response_text,
            "reasoning_chain": reasoning_steps,
            "total_steps": len(reasoning_steps)
        })

    except Exception as e:
        sys.stdout = old_stdout
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/audio")
async def chat_audio(
    audio: UploadFile = File(...),
    history: str = Form(default="[]")
):
    if not react_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized yet")

    import groq as groq_sdk
    audio_bytes = await audio.read()
    client = groq_sdk.Groq(api_key=os.getenv("GROQ_API_KEY"))

    transcription = client.audio.transcriptions.create(
        file=("audio.webm", audio_bytes, "audio/webm"),
        model="whisper-large-v3",
        language="en"
    )
    query = transcription.text

    try:
        history_list = json.loads(history)
    except Exception:
        history_list = []

    chat_request = ChatRequest(query=query, history=history_list)
    return await chat(chat_request)


# ─── Static Files (served last so API routes take priority) ───────────────────
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
