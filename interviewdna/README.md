# InterviewDNA

**A personalized, Agentic RAG AI Interview Coach.**

InterviewDNA is not a chatbot, a static resume matcher, or a fixed
question-bank generator. It runs a genuine **agentic loop**:

```
Observe → Reason → Decide → Select Tool → Execute → Observe Result → Adapt
```

A candidate uploads a resume and a target job description. The system
understands both, finds gaps, builds an interview strategy, conducts an
adaptive interview (technical, behavioral STAR, and technical-STAR deep
dives), evaluates every answer, retrieves grounding knowledge when it detects
a real gap, coaches, retests, and produces a personalized improvement plan.

## Architecture

```
                    STREAMLIT
                        │
                        ▼
                     FastAPI
                        │
                        ▼
                    LangGraph
               Interview Orchestrator
                        │
      ┌─────────────────┼─────────────────┐
      │                 │                 │
      ▼                 ▼                 ▼
 LLM Service          Mem0            LlamaIndex
      │              Memory                │
      ▼                                    ▼
   Ollama                               Pinecone
      │                                    ▲
      ▼                                    │
  Local LLM                       Sentence Transformers
```

| Layer | Responsibility |
|---|---|
| **LangGraph** | Decides — state, routing, which node/tool runs next |
| **LLM (Ollama)** | Understands, reasons, evaluates, generates |
| **LlamaIndex** | Retrieves — ingestion + chunking pipeline |
| **Pinecone** | Stores searchable resume/JD/reference knowledge |
| **Mem0** | Remembers long-term candidate coaching signals |
| **FastAPI** | Exposes application capabilities as an API |
| **Streamlit** | Candidate-facing experience |

The LLM provider lives entirely behind `llm/base.py::LLMService`. Swapping
Ollama for a different Ollama-compatible model, or a different provider
altogether, requires changes **only** in `llm/` — never in
`agents/interview_graph.py` or any other orchestration code.

## The 12 explicit LLM calls

| # | Call | Where |
|---|---|---|
| 1 | Resume DNA Extraction | `services/resume_service.py` |
| 2 | Job DNA Extraction | `services/job_service.py` |
| 3 | Resume↔Job Alignment Reasoning | `services/job_service.py` |
| 4 | Interview Strategy Generation | `agents/strategy_agent.py` |
| 5 | Interview Question Generation | `agents/technical_agent.py` / `star_agent.py` / `technical_star_agent.py` |
| 6 | Candidate Answer Evaluation | `agents/answer_quality_agent.py` |
| 7 | Adaptive Follow-Up Generation | `agents/technical_agent.py` |
| 8 | STAR Evaluation | `agents/star_agent.py` |
| 9 | Technical STAR Evaluation | `agents/technical_star_agent.py` |
| 10 | RAG-Grounded Coaching | `agents/coaching_agent.py` |
| 11 | Retry Evaluation | `agents/coaching_agent.py` |
| 12 | Improvement Plan Generation | `services/evaluation_service.py` |

Every call site is documented in-code with a docstring naming the LangGraph
node that triggers it and *why*. Not every request needs all 12 — LangGraph's
`interview_graph.py` decides which calls are necessary based on state
(coverage, interview mode, evaluation results).

**Deterministic-first**: the Coverage Agent (`agents/coverage_agent.py`) and
all routing decisions in `interview_graph.py` are pure Python — no LLM call —
per the requirement to avoid unnecessary LLM calls wherever rules suffice.

## Repository structure

```
interviewdna/
├── frontend/streamlit_app.py       Candidate-facing UI
├── api/                            FastAPI service boundary
│   ├── main.py
│   ├── session_store.py
│   └── routes/{resume,job,match,interview}.py
├── agents/                         LangGraph orchestrator + agent nodes
│   ├── interview_graph.py          The agentic loop (state, routing, decisions)
│   ├── strategy_agent.py           LLM #4
│   ├── technical_agent.py          LLM #5, #7
│   ├── star_agent.py               LLM #8
│   ├── technical_star_agent.py     LLM #9
│   ├── coaching_agent.py           LLM #10, #11
│   ├── answer_quality_agent.py     LLM #6
│   └── coverage_agent.py           Deterministic coverage/routing
├── llm/                            Provider-agnostic LLM abstraction
│   ├── base.py
│   ├── factory.py
│   └── ollama_service.py
├── rag/                            Ingestion + retrieval
│   ├── ingestion.py                Parse -> chunk -> embed -> store
│   ├── retriever.py                Query-time retrieval
│   ├── embeddings.py               Sentence Transformers
│   └── pinecone_store.py           ALL Pinecone access lives here
├── memory/mem0_service.py          Long-term coaching memory + write rules
├── services/                       Pre-interview pipeline + plan synthesis
│   ├── resume_service.py           LLM #1
│   ├── job_service.py              LLM #2, #3
│   └── evaluation_service.py       LLM #12
├── models/{interview_state.py,schemas.py}
├── prompts/                        All prompt templates, one module per LLM-call family
├── tests/                          pytest suite (dependency-light, uses a FakeLLMService)
├── requirements.txt
├── .env.example
└── README.md
```

## Running locally

### 1. Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally with a model pulled, e.g.:
  ```bash
  ollama pull llama3.1
  ollama serve
  ```
- A free [Pinecone Starter](https://www.pinecone.io/) account + API key
- (Optional) A Mem0 API key — otherwise Mem0 falls back to a local, self-hosted
  `Memory()` instance so the app runs with zero paid dependencies.

### 2. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in PINECONE_API_KEY at minimum
```

### 3. Run

```bash
# Terminal 1
uvicorn api.main:app --reload --port 8000

# Terminal 2
streamlit run frontend/streamlit_app.py
```

| Service | URL |
|---|---|
| Streamlit | http://localhost:8501 |
| FastAPI | http://localhost:8000 (docs at `/docs`) |
| Ollama | http://localhost:11434 |
| Pinecone | cloud |

Streamlit cannot reach a locally-run Ollama/FastAPI if it is itself deployed
to a cloud host — run all local pieces on the same machine, or point
`INTERVIEWDNA_API_BASE` / `OLLAMA_BASE_URL` at reachable hosts.

### 4. Test

```bash
pytest
```

The test suite uses a `FakeLLMService` (see `tests/conftest.py`) so it runs
without a live Ollama/Pinecone/Mem0 connection — it exercises the
deterministic logic (Coverage Agent, routing rules, memory-write rules,
schema validation) that the hackathon judges can trust regardless of model
variance.

## API surface

```
POST /resume/analyze                 multipart file upload -> Resume DNA
POST /job/analyze                    {session_id, job_description_text} -> Job DNA
POST /match                          {session_id} -> Alignment + Strategy

POST /interview/start                {session_id, mode, question_budget} -> first question
POST /interview/answer               {session_id, answer_text} -> evaluation + next question
POST /interview/retry                {session_id, retry_answer_text} -> before/retry comparison

GET  /interview/{session_id}/coverage
GET  /interview/{session_id}/results
```

The interview endpoints are backed by a single LangGraph graph per session,
paused/resumed via LangGraph's `interrupt()` / `Command(resume=...)`
mechanism and a `MemorySaver` checkpointer keyed by `session_id` — so the
candidate's answer, submitted via a separate HTTP request, resumes the graph
exactly where it left off.
