# Agentic Systems Research

A research repository exploring advanced agentic AI system architectures, online learning mechanisms, and multi-agent coordination patterns. Imported from [code-doc-assistant](https://github.com/Chrisys93/code-doc-assistant) and generalised into a domain-agnostic research environment with the code documentation assistant retained as a working PoC.

This repository is not a production system. It is a structured research environment where ideas from the `code-doc-assistant` project are extended, stress-tested, and explored at a depth that would be inappropriate in a product-oriented codebase.

---

## Relationship to `code-doc-assistant`

The `code-doc-assistant` repo is a concrete instantiation of a RAG pipeline applied to a specific domain (code documentation). This repo is the generalisation: the same composable tier system, abstraction layers, and MLflow instrumentation, with the research directions made explicit. The code documentation assistant is retained in `core/` as the working PoC — a real pipeline to instrument, test against, and incrementally adapt (e.g. one approach for Ollama, one for vLLM).

The relationship runs in both directions:

- **This repo as a subsystem of a larger platform**: the pipeline skeleton and research findings here are intended to be composable into a broader AI platform, where multiple domain-specific assistants share infrastructure and coordination mechanisms.
- **The `code-doc-assistant` `dev` branch as a data source**: the MLflow execution traces produced by the `dev` branch feed directly into the research experiments here, particularly the `orchestrated` branch's long-loop optimisation work.

---

## Repository Structure

```
agentic-systems-research/
├── core/                              # Pipeline code (PoC retained from code-doc-assistant)
│   ├── agent_graph.py                 # LangGraph StateGraph — conditional agent pipeline
│   ├── agent_state.py                 # AgentState TypedDict + dataclasses (ToolCall, Chunk, etc.)
│   ├── tools.py                       # Tool wrappers: grep, vector search, AST parse, GitHub fetch
│   └── src/                           # RAG pipeline modules
│       ├── __init__.py
│       ├── app.py                     # Streamlit UI — chat + pipeline visualisation + session tab
│       ├── config.py                  # Tier-aware configuration (mirrors Helm _helpers.tpl)
│       ├── ingest.py                  # Ingestion: clone → discover → AST chunk → embed → store
│       ├── query_engine.py            # Retrieval + prompt assembly + LLM response
│       └── vector_store.py            # VectorStoreBase ABC — ChromaDB impl, swappable
├── instrumentation/                   # MLflow logging contract + per-query metrics
│   ├── __init__.py
│   ├── mlflow_schema.py               # Canonical schema shared with code-doc-assistant/dev
│   └── metrics.py                     # Computed metrics independent of MLflow
├── helm/code-doc-assistant/           # Helm chart (mirrors code-doc-assistant, composable tiers)
│   ├── Chart.yaml
│   ├── values.yaml
│   └── templates/
│       ├── _helpers.tpl               # Tier → model, resources, embedding, chunking strategy
│       ├── app-deployment.yaml
│       ├── chromadb-statefulset.yaml
│       ├── ollama-statefulset.yaml
│       ├── vllm-deployment.yaml
│       ├── mlflow-statefulset.yaml
│       ├── services.yaml
│       └── ingress.yaml
├── argo/
│   └── ingest-workflow.yaml           # Argo Workflows DAG for parallel codebase ingestion
├── notebooks/
│   └── pipeline_walkthrough.ipynb     # LangGraph graph visualisation + HITL demo
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py              # Pipeline validation with in-process ChromaDB
├── Dockerfile                         # App container (copies core/ + instrumentation/)
├── docker-compose.yml                 # Local dev: Ollama + ChromaDB + App
├── docker-compose.dev.yml             # Dev: + MLflow (always-on), vLLM option, HITL config
├── docker-compose.gpu.yml             # GPU override
├── docker-compose.cpu.yml             # CPU-only override
├── requirements.txt
├── run.sh                             # Auto-detect GPU, default to lightweight tier
├── run-cpu.sh                         # Force CPU-only
├── LICENSE                            # Apache 2.0
├── README.md
└── ARCHITECTURE.md                    # Full research programme + design decisions
```

Branch-specific code lives entirely within each branch — `master` contains the shared skeleton, instrumentation layer, and the PoC pipeline that all branches build on.

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU recommended for `full` and `balanced` tiers; CPU-only works for `lightweight`

### Run locally

```bash
git clone https://github.com/Chrisys93/agentic-systems-research.git
cd agentic-systems-research
./run.sh                          # auto-detects GPU, defaults to lightweight tier
```

Or manually:

```bash
MODEL_TIER=lightweight docker compose up --build   # CPU-only
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build  # GPU
```

Open `http://localhost:8501`.

### Dev mode (MLflow + agent pipeline)

The dev compose file mirrors the [`code-doc-assistant` `dev` branch](https://github.com/Chrisys93/code-doc-assistant/tree/dev) environment — LangGraph agent graph with tool selection, HITL checkpoints, supervisor quality gates, and MLflow tracking always on. This is the environment where the `orchestrated` branch's research builds on top of.

```bash
docker compose -f docker-compose.dev.yml up --build

# Automated mode (no human checkpoints):
HITL_ENABLED=false OUTPUT_REVIEW_MODE=off \
  docker compose -f docker-compose.dev.yml up

# Supervisor quality gate instead of human review:
OUTPUT_REVIEW_MODE=supervisor QUALITY_GATE_THRESHOLD=7.0 \
  docker compose -f docker-compose.dev.yml up

# vLLM backend (GPU required):
INFERENCE_BACKEND=vllm \
  docker compose -f docker-compose.dev.yml --profile vllm up
```

Access: `http://localhost:8501` (UI) · `http://localhost:5000` (MLflow)

### Model tiers

Identical to `code-doc-assistant`:

| Tier | Model | RAM | GPU |
|------|-------|-----|-----|
| `full` | Mistral Nemo 12B | ~12Gi+ | Required |
| `balanced` | Qwen2.5-Coder 7B | ~8Gi | Recommended |
| `lightweight` | Phi-3.5 Mini 3.8B | ~4Gi | Optional |

---

## Branches

Each branch is a self-contained research direction built on top of this skeleton. They share the `core/` and `instrumentation/` modules but diverge in architecture.

| Branch | Research direction | Status |
|--------|--------------------|--------|
| `master` | Pipeline skeleton + PoC + instrumentation layer | Active |
| `orchestrated` | LangGraph supervisor, online preference learning, graph structure search | In design |
| `emergent` | Multi-agent coordination through shared state only, no central supervisor | Planned |
| `protocol-driven` | Federated preference aggregation, B2B cold-start collaboration | Planned |
| `fine-tuning` | LoRA/QLoRA on accumulated cross-session data | Deferred |

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full research programme, branch-by-branch design rationale, and the connections between branches.

---

## The MLflow Contract

The `instrumentation/mlflow_schema.py` module defines the canonical logging schema shared between this repo and `code-doc-assistant`'s `dev` branch. Any experiment that consumes `dev` traces must conform to this schema. Any changes to the schema require coordinated updates in both repos.

The core logged fields per query:

| Field | Type | Description |
|-------|------|-------------|
| `query_id` | str | UUID per query |
| `session_id` | str | UUID per session |
| `model_tier` | str | `full` / `balanced` / `lightweight` |
| `retrieval_scores` | list[float] | Similarity scores for retrieved chunks |
| `retrieval_hit` | bool | Whether top-k chunks contained expected content |
| `tool_selections` | list[str] | Tools invoked, in order |
| `supervisor_adjustments` | list[dict] | Any runtime adjustments made by the supervisor node |
| `latency_ms` | int | End-to-end query latency |
| `satisfaction_score` | int \| None | Explicit HITL feedback (1–5), if provided |
| `session_preferences` | dict | Serialised `SessionPreferences` state at query time |

The `instrumentation/metrics.py` module computes `QueryMetrics` from a final `AgentState` without touching MLflow — retrieval confidence stats, tool usage patterns, latency breakdowns, and feedback scores as a clean dataclass that can be logged, displayed, or fed into research pipelines independently.

---

## Engineering Standards

- Python 3.11+, type hints throughout
- Modular design: `core/` is the stable layer; branch-specific code never modifies `core/`
- All experiments are reproducible: seeds, MLflow run IDs, and environment specs logged
- Abstraction layers: `VectorStoreBase` for DB-agnostic design, swappable to Qdrant or FAISS
- Configuration: environment-variable driven, parity between Docker Compose and Helm

---

*This is a research repository. Stability and backwards compatibility are not guaranteed on non-`master` branches.*
