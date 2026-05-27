# Weave — Enterprise Intelligence Platform

A production-grade, multi-source AI intelligence system for enterprise ERP data. The platform ships two agents that share the same data infrastructure:

- **Weave** (port 8000) — multi-source RAG router using LlamaIndex selector
- **STRATA** (port 8001) — autonomous ReACT agent with step-by-step reasoning

Both agents query across structured databases, unstructured conversational logs, and an organizational graph through a conversational interface with voice support.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-Core-purple)](https://llamaindex.ai)
[![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20Kimi%20K2-orange)](https://groq.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)](https://docker.com)


---

## Overview

Built around **NexaMedTech Solutions** — a simulated mid-size MedTech/Pharma ERP startup with 65 employees, 50+ interconnected tickets, and 120+ Slack messages across 10 channels.

Example queries both agents can handle:

- *"Who is blocking the FDA compliance ticket and what did DevOps say about it in Slack?"*
- *"What are all the tickets assigned to engineers who report to the Backend Manager?"*
- *"Full salary breakdown under the CTO, ranked highest to lowest"*

---

## Two Agents, One Data Layer

### Weave (v1) — RAG Router

A fast, single-pass routing engine. Given a query, it selects the best data source(s) and returns a synthesized answer. Best for direct lookups.

```
Query → LLMMultiSelector → [Postgres | ChromaDB | Neo4j | LLM] → Answer
```

### STRATA (v2) — ReACT Agent

An autonomous multi-step agent. It reasons, selects a tool, observes the result, and keeps going until it has a complete answer. Best for complex multi-hop questions.

```
Query → Think → Act (tool call) → Observe → Think → Act → ... → Final Answer
```

Both agents run simultaneously and share the same Postgres, Neo4j, and ChromaDB data volumes.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          GCP Compute Engine VM                        │
│                                                                        │
│  ┌─────────────────────────┐    ┌─────────────────────────┐           │
│  │  Weave  :8000          │    │  STRATA  :8001           │           │
│  │  (RAG Router)            │    │  (ReACT Agent)           │           │
│  │  src/agent/router.py     │    │  src/claw/main.py        │           │
│  └────────────┬────────────┘    └────────────┬────────────┘           │
│               │                              │                         │
│               └──────────────┬───────────────┘                         │
│                              │ shared data layer                        │
│          ┌───────────────────┼───────────────────┐                     │
│          ▼                   ▼                   ▼                     │
│   ┌─────────────┐   ┌─────────────┐   ┌──────────────┐               │
│   │ PostgreSQL  │   │  ChromaDB   │   │    Neo4j     │               │
│   │ :5432       │   │  (volume)   │   │  :7474/7687  │               │
│   │             │   │             │   │              │               │
│   │ Users       │   │ Slack msg   │   │ Org graph    │               │
│   │ Tickets     │   │ embeddings  │   │ Blocker      │               │
│   │ Blockers    │   │             │   │ chains       │               │
│   └─────────────┘   └─────────────┘   └──────────────┘               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Features

**Shared across both agents:**
- Voice interface — in-browser audio transcribed by Groq Whisper-Large-v3
- General knowledge fallback — non-company questions answered from LLM knowledge directly
- Rate-limit resilience — auto-switches between `kimi-k2-instruct` and `llama-3.3-70b-versatile`
- Multi-turn conversation with history condensation
- Fully Dockerized — `docker compose up` starts everything

**Weave specific:**
- Multi-source parallel synthesis via `LLMMultiSelector`
- Real-time routing logs in the side panel

**STRATA specific:**
- Full ReACT loop — Thought, Action, Observation trace rendered live
- Up to 10 reasoning iterations per query
- STRATA UI with live reasoning chain panel and source status indicators

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, FastAPI, Uvicorn |
| AI Orchestration | LlamaIndex Core |
| LLM | Groq API (Kimi K2 Instruct → Llama 3.3 70B fallback) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` via `sentence-transformers` |
| Voice | Groq Whisper-Large-v3 |
| Structured DB | PostgreSQL 15 |
| Vector Store | ChromaDB (persistent volume) |
| Graph DB | Neo4j 5.26 |
| Frontend | Vanilla HTML/CSS/JS |
| Containerization | Docker Compose |

---

## Project Structure

```
Weave/
├── src/
│   ├── agent/                         # Weave v1 (port 8000)
│   │   ├── router.py                  # FastAPI app + LlamaIndex routing engine
│   │   └── public/                    # Weave frontend
│   ├── claw/                          # STRATA v2 (port 8001)
│   │   ├── main.py                    # FastAPI app + ReACT agent
│   │   └── public/                    # STRATA frontend (HTML/CSS/JS)
│   ├── ingestion/
│   │   ├── setup_dbs.py               # Load users/tickets → Postgres + MongoDB
│   │   ├── embed_data.py              # Embed Slack messages → ChromaDB
│   │   └── setup_neo4j.py             # Build org graph → Neo4j
│   └── data_generation/
│       └── generate_mock_data.py      # LLM-generated mock company dataset
├── data/
│   └── raw/
│       ├── users.json                 # 65 employees
│       ├── tickets.json               # 50+ Agile tickets
│       └── messages.json              # 120+ Slack messages
├── Dockerfile                         # Weave app image
├── Dockerfile.claw                    # STRATA agent image
├── docker-compose.yml                 # All services
├── requirements.txt
└── .env.example
```

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- A [Groq API Key](https://console.groq.com) (free tier is sufficient)

---

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/salmanranjhaa/Weave.git
cd Weave

cp .env.example .env
# Open .env and set your GROQ_API_KEY
```

### 2. Start the databases

```bash
docker compose up -d postgres mongodb neo4j pgadmin mongo-express
# Wait ~30 seconds for health checks
docker compose ps
```

### 3. Run the data ingestion pipeline

```bash
docker compose run --rm app python src/ingestion/setup_dbs.py
docker compose run --rm app python src/ingestion/embed_data.py
docker compose run --rm app python src/ingestion/setup_neo4j.py
```

### 4. Start Weave (v1)

```bash
docker compose build app
docker compose up -d app
```

Navigate to **http://localhost:8000**

### 5. Start STRATA (v2)

```bash
docker compose build openclaw
docker compose up -d openclaw
```

Navigate to **http://localhost:8001**

---

## Database Admin UIs

| Service | URL | Purpose |
|---|---|---|
| pgAdmin | `http://localhost:5050` | PostgreSQL management |
| Mongo Express | `http://localhost:8081` | MongoDB management |
| Neo4j Browser | `http://localhost:7474` | Graph visualization and Cypher queries |

---

## Configuration

Copy `.env.example` to `.env`:

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Vector store path (use /app/chroma_data inside Docker)
CHROMA_PATH=/app/chroma_data

# Database connection strings
# Use Docker service names inside Docker, localhost outside
POSTGRES_URI=postgresql://admin:adminpassword@postgres:5432/weave_db
MONGO_URI=mongodb://admin:adminpassword@mongodb:27017/
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=adminpassword
```

---

## GCP Deployment

Recommended VM: `e2-standard-2` (2 vCPU, 8GB RAM), Debian 12.

```bash
# SSH into your VM
gcloud compute ssh --zone "YOUR_ZONE" "YOUR_INSTANCE" --project "YOUR_PROJECT"

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER && newgrp docker
sudo apt-get install -y docker-compose-plugin git

# Clone and configure
git clone https://github.com/salmanranjhaa/Weave.git
cd Weave
# Create .env with your GROQ_API_KEY

# Start databases and ingest data
docker compose up -d postgres mongodb neo4j pgadmin mongo-express
docker compose run --rm app python src/ingestion/setup_dbs.py
docker compose run --rm app python src/ingestion/embed_data.py
docker compose run --rm app python src/ingestion/setup_neo4j.py

# Start Weave
docker compose build app && docker compose up -d app

# Start STRATA
docker compose build openclaw && docker compose up -d openclaw
```

Open firewall ports:
```bash
gcloud compute firewall-rules create Weave-ports \
  --project="YOUR_PROJECT" \
  --allow="tcp:8000,tcp:8001,tcp:5050,tcp:8081,tcp:7474" \
  --source-ranges=0.0.0.0/0
```

---

## API Reference

Both agents expose the same endpoint contract.

### `POST /chat`

```json
{
  "query": "Who is blocking the compliance ticket?",
  "history": [
    {"role": "user", "content": "previous question"},
    {"role": "assistant", "content": "previous answer"}
  ]
}
```

**Weave response:**
```json
{
  "query": "...",
  "response": "...",
  "source": "['sql_tool', 'slack_vector_tool']",
  "detailed_logs": [...]
}
```

**STRATA response:**
```json
{
  "query": "...",
  "condensed_query": "...",
  "response": "...",
  "reasoning_chain": [
    {"step": 1, "content": "Thought: I need to...", "is_done": false},
    {"step": 2, "content": "Action: sql_database_tool", "is_done": false}
  ],
  "total_steps": 4
}
```

### `POST /chat/audio`

Accepts `.webm` audio via multipart form. Returns the same response structure plus the transcribed `query` field.

```bash
curl -X POST http://localhost:8000/chat/audio \
  -F "audio=@recording.webm" \
  -F "history=[]"
```

---

## The Company Dataset

**NexaMedTech Solutions** — A fictional MedTech/Pharma ERP startup.

| Entity | Count |
|---|---|
| Employees | 65 |
| Departments | 9 (Engineering, QA, DevOps, Sales, HR, Finance, Product, BA, C-Suite) |
| Agile Tickets | 50+ |
| Slack Channels | 10 |
| Slack Messages | 120+ |

---

## Useful Commands

```bash
# Status of all services
docker compose ps

# Live logs
docker compose logs -f app
docker compose logs -f openclaw

# Update and restart after code change
git pull origin main
docker compose build app openclaw
docker compose up -d app openclaw

# Re-run ingestion (if data changes)
docker compose run --rm app python src/ingestion/setup_dbs.py
docker compose run --rm app python src/ingestion/embed_data.py
docker compose run --rm app python src/ingestion/setup_neo4j.py

# Resource usage
docker stats
```
