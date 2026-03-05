# llm-agent-research

A research repository exploring advanced agentic AI system architectures, online learning mechanisms, and multi-agent coordination patterns. Forked from [code-doc-assistant](https://github.com/Chrisys93/code-doc-assistant) and generalised into a domain-agnostic pipeline skeleton.

This repository is not a production system. It is a structured research environment where ideas from the `code-doc-assistant` project are extended, stress-tested, and explored at a depth that would be inappropriate in a product-oriented codebase.

---

## Relationship to `code-doc-assistant`

The `code-doc-assistant` repo is a concrete instantiation of a RAG pipeline applied to a specific domain (code documentation). This repo is the generalisation: the same composable tier system, abstraction layers, and MLflow instrumentation, with the domain-specific parts removed. The code documentation assistant can be thought of as one possible instantiation of this skeleton; others could be a test generation assistant, a security audit assistant, a migration planning assistant, and so on.

The relationship runs in both directions:

- **This repo as a subsystem of a larger platform**: the pipeline skeleton and research findings here are intended to be composable into a broader AI platform, where multiple domain-specific assistants share infrastructure and coordination mechanisms.
- **The `code-doc-assistant` `dev` branch as a data source**: the MLflow execution traces produced by the `dev` branch feed directly into the research experiments here, particularly the `orchestrated` branch's long-loop optimisation work.

---

## Repository Structure

```
llm-agent-research/
├── core/                        # Generic pipeline skeleton (this branch)
│   ├── config.py                # Tier-aware configuration, mirrors code-doc-assistant logic
│   ├── vector_store.py          # VectorStoreBase ABC — ChromaDB impl, swappable
│   ├── ingest.py                # Domain-agnostic ingestion: discover, chunk, embed, store
│   ├── query_engine.py          # Retrieval + prompt assembly, no domain assumptions
│   └── app.py                   # Minimal Streamlit interface for pipeline testing
├── instrumentation/
│   ├── mlflow_schema.py         # Canonical MLflow logging schema (shared contract with dev)
│   └── metrics.py               # Per-query metrics: confidence, latency, retrieval hit rate
├── helm/                        # Helm chart (mirrors code-doc-assistant, domain-agnostic)
├── docker-compose.yml           # Local dev: Ollama + ChromaDB + App
├── docker-compose.gpu.yml       # GPU override
├── requirements.txt
├── run.sh
└── ARCHITECTURE.md
```

Branch-specific code lives entirely within each branch — `master` contains only the shared skeleton and instrumentation layer that all branches build on.

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU recommended for `full` and `balanced` tiers; CPU-only works for `lightweight`

### Run locally

```bash
git clone <repo-url>
cd llm-agent-research
./run.sh                          # auto-detects GPU, defaults to lightweight tier
```

Or manually:

```bash
MODEL_TIER=lightweight docker compose up --build   # CPU-only
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build  # GPU
```

Open `http://localhost:8501`.

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
| `master` | Generic pipeline skeleton | Active |
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

---

## Engineering Standards

- Python 3.11+, type hints throughout
- Modular design: `core/` is the stable layer; branch-specific code never modifies `core/`
- All experiments are reproducible: seeds, MLflow run IDs, and environment specs logged
- Abstraction layers: `VectorStoreBase` for DB-agnostic design, swappable to Qdrant or FAISS
- Configuration: environment-variable driven, parity between Docker Compose and Helm

---

*This is a research repository. Stability and backwards compatibility are not guaranteed on non-`master` branches.*
