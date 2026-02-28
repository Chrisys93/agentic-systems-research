# Code Documentation Assistant

A conversational AI assistant that ingests a codebase — from a GitHub repo or local files — and answers questions about it: how it works, where functionality lives, what the API surface looks like, what the dependencies are, and so on.

Fully self-hosted. No API keys required. Code never leaves your machine.

---

## Branch Overview

| Branch | Character | Stack |
|--------|-----------|-------|
| `master` | Linear RAG pipeline — ingest, embed, retrieve, respond | Ollama · ChromaDB · LlamaIndex · Streamlit · Helm |
| `dev` | Conditional agent graph — tool use, HITL, supervisor loops | + LangGraph · MLflow · vLLM option · Argo Workflows |
| `fine-tuning` *(planned)* | LoRA/QLoRA adapter training on accumulated HITL feedback | + PEFT · model registry · evaluation harness |
| `research` *(separate repo)* | Online preference learning · federated aggregation · multi-agent orchestration | TBD per branch |

`master` is the stable, reviewer-friendly branch — `docker compose up` and it works. `dev` is where the pipeline becomes an agent: tool selection, human checkpoints, supervisor quality gates. The two branches share deployment infrastructure (Helm chart, Docker Compose base, ChromaDB, Ollama) and diverge only in the application layer.

---

## Architecture Overview

```
                                    ┌─────────────────────────────┐
                                    │    Inference Backend        │
                                    │  ┌───────────────────────┐  │
                                    │  │ Ollama (default)       │  │
                                    │  │  Tier 1: Mistral Nemo  │  │
                                    │  │  Tier 2: Qwen2.5-Coder │  │
                                    │  │  Tier 3: Phi-3.5 Mini  │  │
                                    │  └───────────────────────┘  │
                                    │  ┌───────────────────────┐  │
                                    │  │ vLLM  (dev, GPU)       │  │
                                    │  │  OpenAI-compatible API │  │
                                    │  │  same model tiers      │  │
                                    │  └───────────────────────┘  │
                                    │  ┌───────────────────────┐  │
                                    │  │ Embeddings (Ollama)    │  │
                                    │  │  nomic-embed-text      │  │
                                    │  │  all-minilm            │  │
                                    │  │  mxbai-embed-large     │  │
                                    │  └───────────────────────┘  │
                                    └──────────┬──────────────────┘
                                               │ ▲
                                    Embeddings │ │ Generated
                                    + Queries  │ │ Responses
                                               ▼ │
┌─────────────┐    Questions    ┌──────────────────────────────┐
│   Web UI    │ ──────────────▶ │       App Server             │
│ (Streamlit) │ ◀────────────── │                              │
│             │    Answers +    │  master: LlamaIndex RAG      │
│  Chat tab   │    Sources      │  dev:    LangGraph agent     │
│  Pipeline   │                 │    tool use · HITL           │
│  Session    │                 │    supervisor loops          │
└─────────────┘                 └──────────────┬───────────────┘
                                               │ ▲
                                  Store chunks │ │ Retrieve top-k
                                               ▼ │
                                    ┌─────────────────────────┐
                                    │  Vector DB (ChromaDB)   │
                                    │  - HNSW index           │
                                    │  - Metadata filtering   │
                                    │  - Persistent storage   │
                                    └─────────────────────────┘
                                               │
                                    ┌──────────▼──────────────┐
                                    │  MLflow Tracking        │
                                    │  master: --profile      │
                                    │    observability        │
                                    │  dev: always-on         │
                                    └─────────────────────────┘
```

**Deployment options:**
- **Docker Compose** (`master`): `docker compose up` — Ollama, ChromaDB, app. MLflow optional via `--profile observability`.
- **Docker Compose** (`dev`): `docker compose -f docker-compose.dev.yml up` — adds MLflow (always-on), vLLM backend option, Argo Workflows client.
- **Helm/K8s**: Ollama as StatefulSet, ChromaDB as StatefulSet, App as Deployment. `modelTier`, `embeddingModel`, and (in `dev`) `outputReviewMode` cascade through `_helpers.tpl`.
- **Access**: `kubectl port-forward` (single developer) · NodePort (team, private network) · Ingress (production)

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU + drivers recommended for full/balanced tiers — Ollama falls back to CPU automatically
- (Optional) A Kubernetes cluster + Helm for production deployment

### Local Development — `master` branch

```bash
git clone https://github.com/Chrisys93/code-doc-assistant
cd code-doc-assistant

# Quickest start (auto-detects GPU, defaults to lightweight tier):
./run.sh

# CPU-only explicit:
./run-cpu.sh

# With GPU acceleration:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

# Specify tier:
MODEL_TIER=balanced docker compose up --build
MODEL_TIER=full docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

# With MLflow tracking (optional):
docker compose --profile observability up --build

# Custom embedding model:
EMBEDDING_MODEL=all-minilm MODEL_TIER=lightweight docker compose up --build

# Point at a local repo:
REPO_PATH=/path/to/your/repo docker compose up
```

Open `http://localhost:8501` in your browser.

On first startup, the `ollama-bootstrap` service pulls the LLM and embedding models — this may take a few minutes. Models are cached in a Docker volume; subsequent starts are fast.

### Local Development — `dev` branch

```bash
git checkout dev

# Default: Ollama backend, HITL enabled, MLflow on
docker compose -f docker-compose.dev.yml up --build

# Automated / CI mode (no human checkpoints):
HITL_ENABLED=false OUTPUT_REVIEW_MODE=off \
  docker compose -f docker-compose.dev.yml up

# Supervisor quality gate instead of human output review:
OUTPUT_REVIEW_MODE=supervisor QUALITY_GATE_THRESHOLD=7.0 \
  docker compose -f docker-compose.dev.yml up

# vLLM inference backend (GPU required):
INFERENCE_BACKEND=vllm \
  docker compose -f docker-compose.dev.yml --profile vllm up

# Lightweight tier, CPU-only:
MODEL_TIER=lightweight docker compose -f docker-compose.dev.yml up
```

Access: `http://localhost:8501` (UI) · `http://localhost:5000` (MLflow)

### Production Deployment (Helm)

```bash
# Default (full tier):
helm install code-doc-assistant ./helm/code-doc-assistant

# Lightweight tier:
helm install code-doc-assistant ./helm/code-doc-assistant \
  --set modelTier=lightweight

# Custom combination:
helm install code-doc-assistant ./helm/code-doc-assistant \
  --set modelTier=balanced \
  --set embeddingModel=lightweight

# dev branch: supervisor quality gate:
helm install code-doc-assistant ./helm/code-doc-assistant \
  --set modelTier=balanced \
  --set outputReviewMode=supervisor \
  --set qualityGateThreshold=7.0

# Access via port-forward (single developer):
kubectl port-forward svc/code-doc-assistant-app 8501:8501

# Access via NodePort (team on private network):
helm install code-doc-assistant ./helm/code-doc-assistant \
  --set app.service.type=NodePort \
  --set app.service.nodePort=30501
```

A single `modelTier` value cascades through model selection, resource requests, GPU requirements, chunking strategy, and context window configuration — no manual YAML editing required.

---

## Model Tiers

| Tier | Model | RAM | GPU | Best for |
|------|-------|-----|-----|----------|
| `full` (default) | Mistral Nemo 12B | ~12Gi+ | Required | Best explanation quality |
| `balanced` | Qwen2.5-Coder 7B | ~8Gi | Recommended | Good quality, moderate hardware |
| `lightweight` | Phi-3.5 Mini 3.8B | ~4Gi | Optional | Edge / CPU-only / low-resource |

For polyglot codebases (multiple languages each >10%), DeepSeek-Coder V2 Lite is the recommended swap for the full tier — its MoE architecture handles multi-language contexts particularly well.

---

## Configuration Reference

### `master` branch

| Variable | Default | Options |
|----------|---------|---------|
| `MODEL_TIER` | `full` | `full`, `balanced`, `lightweight` |
| `EMBEDDING_MODEL` | `nomic-embed-text` | `nomic-embed-text`, `all-minilm`, `mxbai-embed-large` |
| `REPO_PATH` | `./repos` | Any local path |
| `LOG_LEVEL` | `info` | `debug`, `info`, `warning` |

> **Note**: changing `EMBEDDING_MODEL` after ingestion requires full re-ingestion of the codebase. The vector dimension is derived automatically — you don't need to configure it manually.

### `dev` branch additions

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_BACKEND` | `ollama` | `ollama` · `vllm` |
| `VLLM_HOST` | `http://localhost:8080` | vLLM server base URL |
| `VLLM_MODEL` | *(inherits `OLLAMA_MODEL`)* | Model name as served by vLLM |
| `HITL_ENABLED` | `true` | Tool plan approval checkpoint (HITL-1) |
| `OUTPUT_REVIEW_MODE` | `human` | `human` · `supervisor` · `self` · `off` |
| `QUALITY_GATE_THRESHOLD` | `6.0` | Supervisor rubric pass score (0–10) |
| `CONFIDENCE_THRESHOLD` | `0.45` | Retrieval mean score below which supervisor retries |
| `MAX_RETRIEVAL_ATTEMPTS` | `3` | Pre-generation supervisor retry limit |
| `MAX_GENERATION_ATTEMPTS` | `3` | Post-generation regeneration limit |
| `MAX_CONTEXT_TOKENS` | `8000` | Context window cap for assembled chunks |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | Always-on in dev; opt-in in master |

---

## Tech Stack

### `master`

| Component | Choice | Notes |
|-----------|--------|-------|
| LLM inference | Ollama | Self-hosted; model-agnostic |
| LLM (full tier) | Mistral Nemo 12B | Best explanation quality |
| LLM (balanced) | Qwen2.5-Coder 7B | Best code comprehension at 7B |
| LLM (lightweight) | Phi-3.5 Mini 3.8B | Edge / resource-constrained |
| Embedding | nomic-embed-text | Code + text all-rounder (768-dim) |
| Vector DB | ChromaDB | Python-native; metadata filtering |
| RAG framework | LlamaIndex | Native CodeSplitter; AST-aware chunking |
| Code parsing | tree-sitter | AST chunking across 40+ languages |
| UI | Streamlit | Chat interface with source attribution |
| Packaging | Docker Compose + Helm | Local dev + K8s production |

### `dev` additions

| Component | Choice | Notes |
|-----------|--------|-------|
| Agent framework | LangGraph | Conditional `StateGraph`; interrupt-based HITL |
| LLM inference (alt) | vLLM | OpenAI-compatible API; continuous batching |
| LLM client (vLLM) | langchain-openai | `ChatOpenAI` pointed at vLLM's `/v1` endpoint |
| Experiment tracking | MLflow | Per-query run logging; artifact storage |
| Workflow orchestration | Argo Workflows | Parallel ingest DAG; nightly re-ingestion cron |

---

## Productionisation

### Cloud Resources by Model Tier

The estimates below account for the full pipeline — LLM, embedding model, ChromaDB, and the Streamlit app running concurrently. A single GPU serves both the LLM and embedding model via Ollama, with ChromaDB and the app consuming additional CPU and RAM. CPU-only deployment is technically possible for the lightweight tier but would not deliver a responsive conversational experience — even for demonstration purposes, GPU inference should be provisioned.

| Tier | AWS | GCP | GPU | System RAM | Est. cost/hr |
|------|-----|-----|-----|------------|--------------|
| Full (Mistral Nemo 12B) | `g5.2xlarge` | `a2-highgpu-1g` | A10G 24GB / A100 40GB | 32Gi+ | ~$1.50–$5.00 |
| Balanced (Qwen2.5-Coder 7B) | `g5.xlarge` / `g4dn.xlarge` | `n1-standard-8` + T4 | T4 16GB / A10G | 16Gi+ | ~$0.75–$2.00 |
| Lightweight (Phi-3.5 3.8B) | `g4dn.xlarge` | `n1-standard-8` + T4 | T4 (recommended) | 8Gi+ | ~$0.50–$1.00 |

For the full tier: Mistral Nemo 12B uses ~8–10GB VRAM. Add the embedding model and you exceed a T4's 16GB, hence the A10G (24GB) recommendation. Provisioning one size above the theoretical minimum is standard practice.

### Scaling

- **HPA** on the app Deployment — the stateless Streamlit app scales horizontally
- **Ollama scaling** — model replication or request queuing; Ray Serve wraps Ollama for load balancing at higher throughput
- **vLLM** (`dev`) — continuous batching and PagedAttention make it significantly more efficient under concurrent team load than Ollama; the right choice when multiple developers are using the tool simultaneously
- **Vector DB** — for large codebases, migrate to managed options (Pinecone, Weaviate Cloud) or self-hosted Qdrant with persistent volumes

### Infrastructure & Operations

- **Observability**: MLflow for experiment tracking (both branches); Prometheus + Grafana for infrastructure metrics; OpenTelemetry tracing
- **CI/CD**: GitHub Actions → build container images → push to ECR/GCR → Helm upgrade
- **Security**: Network policies between pods, secrets management (Vault / AWS Secrets Manager), RBAC
- **Agent sandboxing**: Docker Sandboxes (GA January 2026) provide microVM-based isolation for AI agents — directly relevant for a code ingestion pipeline running in proximity to proprietary codebases

### Self-Hosting vs. API Trade-off

| Approach | Pros | Cons |
|----------|------|------|
| **Self-hosted (Ollama / vLLM)** | Full control; no API costs; code stays local | Lower quality at smaller parameter counts; requires GPU infra |
| **Hosted API (Claude, GPT-4)** | Highest quality reasoning; no infra to manage | API costs; code leaves your network |
| **Hybrid** | Local for routine queries, API for complex reasoning | Two systems to maintain; routing logic |

For a code documentation tool, keeping code local is often a hard requirement — proprietary codebases can't be sent to external APIs. The self-hosted approach handles this by default.

---

*For design decisions, component trade-offs, agent pipeline architecture, and research directions, see [ARCHITECTURE.md](ARCHITECTURE.md).*
