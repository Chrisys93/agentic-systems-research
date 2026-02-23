# Code Documentation Assistant

A conversational AI assistant that ingests a codebase — from a GitHub repo or local files — and answers questions about it: how it works, where functionality lives, what the API surface looks like, what the dependencies are, and so on.

Fully self-hosted. No API keys required. Code never leaves your machine.

---

## Architecture Overview

```
                                    ┌─────────────────────────────┐
                                    │         Ollama              │
                                    │  ┌───────────────────────┐  │
                                    │  │ Tier 1: Mistral Nemo  │  │
                                    │  │ Tier 2: Qwen2.5-Coder │  │
                                    │  │ Tier 3: Phi-3.5 Mini  │  │
                                    │  └───────────────────────┘  │
                                    │  ┌───────────────────────┐  │
                                    │  │ Embeddings:           │  │
                                    │  │  nomic-embed-text     │  │
                                    │  │  all-minilm           │  │
                                    │  │  mxbai-embed-large    │  │
                                    │  └───────────────────────┘  │
                                    └──────────┬──────────────────┘
                                               │ ▲
                                    Embeddings │ │ Generated
                                    + Queries  │ │ Responses
                                               ▼ │
┌─────────────┐    Questions    ┌──────────────────────────────┐
│   Web UI    │ ──────────────▶ │       App Server             │
│ (Streamlit) │ ◀────────────── │  ┌────────────────────────┐  │
│             │    Answers +    │  │ RAG Pipeline            │  │
│  Chat UI    │    Sources      │  │  - Ingestion (clone/    │  │
│  Sidebar    │                 │  │    discover/chunk)      │  │
│  ingestion  │                 │  │  - AST Chunking         │  │
│  controls   │                 │  │    (tree-sitter)        │  │
└─────────────┘                 │  │  - Query + Retrieval    │  │
                                │  │  - Prompt Assembly      │  │
                                │  │  - Guardrails           │  │
                                │  └────────────────────────┘  │
                                └──────────────┬───────────────┘
                                               │ ▲
                                  Store chunks │ │ Retrieve
                                  (embed time) │ │ top-k
                                               ▼ │
                                    ┌─────────────────────────┐
                                    │  Vector DB (ChromaDB)   │
                                    │  - HNSW index           │
                                    │  - Metadata filtering   │
                                    │  - Persistent storage   │
                                    └─────────────────────────┘
```

**Deployment options:**
- **Docker Compose**: All components in local containers (`docker compose up`)
- **Helm/K8s**: Ollama as StatefulSet, ChromaDB as StatefulSet, App as Deployment
- **Access**: localhost (port-forward) | NodePort (private network) | Ingress (production)

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU + drivers recommended for full/balanced tiers — Ollama falls back to CPU automatically
- (Optional) A Kubernetes cluster + Helm for production deployment

### Local Development (Docker Compose)

```bash
git clone <repo-url>
cd code-doc-assistant

# Quickest start (auto-detects GPU, defaults to lightweight tier):
./run.sh

# CPU-only:
MODEL_TIER=lightweight docker compose up --build

# With GPU acceleration:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

# Specify tier:
MODEL_TIER=balanced docker compose up --build
MODEL_TIER=full docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

# Custom embedding model:
EMBEDDING_MODEL=all-minilm MODEL_TIER=lightweight docker compose up --build
```

Open `http://localhost:8501` in your browser.

On first startup, the `ollama-bootstrap` service pulls the LLM and embedding models — this may take a few minutes. Models are cached in a Docker volume; subsequent starts are fast.

To point the assistant at a local repo, set `REPO_PATH`:

```bash
REPO_PATH=/path/to/your/repo docker compose up
```

### Production Deployment (Helm)

```bash
# Default (full tier):
helm install code-doc-assistant ./helm/code-doc-assistant

# Lightweight tier:
helm install code-doc-assistant ./helm/code-doc-assistant --set modelTier=lightweight

# Custom combination:
helm install code-doc-assistant ./helm/code-doc-assistant \
  --set modelTier=balanced \
  --set embeddingModel=lightweight

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

| Variable | Default | Options |
|----------|---------|---------|
| `MODEL_TIER` | `full` | `full`, `balanced`, `lightweight` |
| `EMBEDDING_MODEL` | `nomic-embed-text` | `nomic-embed-text`, `all-minilm`, `mxbai-embed-large` |
| `REPO_PATH` | `./repos` | Any local path |
| `LOG_LEVEL` | `info` | `debug`, `info`, `warning` |

> **Note**: changing `EMBEDDING_MODEL` after ingestion requires full re-ingestion of the codebase. The vector dimension is derived automatically — you don't need to configure it manually.

---

## Tech Stack

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
- **Vector DB** — for large codebases, migrate to managed options (Pinecone, Weaviate Cloud) or self-hosted Qdrant with persistent volumes

### Infrastructure & Operations

- **Observability**: Structured JSON logging, Prometheus metrics, Grafana dashboards, OpenTelemetry tracing
- **CI/CD**: GitHub Actions → build container images → push to ECR/GCR → Helm upgrade
- **Security**: Network policies between pods, secrets management (Vault / AWS Secrets Manager), RBAC
- **Agent sandboxing**: Docker Sandboxes (GA January 2026) provide microVM-based isolation for AI agents — directly relevant for a code ingestion pipeline running in proximity to proprietary codebases

### Self-Hosting vs. API Trade-off

| Approach | Pros | Cons |
|----------|------|------|
| **Self-hosted (Ollama)** | Full control; no API costs; code stays local | Lower quality at smaller parameter counts; requires GPU infra |
| **Hosted API (Claude, GPT-4)** | Highest quality reasoning; no infra to manage | API costs; code leaves your network |
| **Hybrid** | Local for routine queries, API for complex reasoning | Two systems to maintain; routing logic |

For a code documentation tool, keeping code local is often a hard requirement — proprietary codebases can't be sent to external APIs. The self-hosted approach handles this by default.

---

*For design decisions, component trade-offs, and research directions, see [ARCHITECTURE.md](ARCHITECTURE.md).*
