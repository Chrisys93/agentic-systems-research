# Architecture & Research Programme

This document describes the design of the `llm-agent-research` repository: the `master` branch skeleton, the rationale for each research branch, the connections between them, and the open research questions each branch is intended to answer.

It is written to be updated as the research progresses. Decisions are recorded with their reasoning. Open questions are stated explicitly rather than glossed over.

---

## Origin and Motivation

This repository is forked from [`code-doc-assistant`](https://github.com/Chrisys93/code-doc-assistant), a RAG pipeline for code documentation built with Ollama, LlamaIndex, ChromaDB, and a composable Helm chart tier system. The code documentation assistant demonstrated that a domain-specific AI assistant could be built with clean abstraction boundaries — the LLM tier, embedding model, vector DB, and deployment configuration are all independently swappable.

The research question this repo pursues is: what happens when you take that composable skeleton and extend it toward genuinely agentic behaviour — online learning, multi-agent coordination, federated knowledge sharing?

The `code-doc-assistant` repo continues as a product-oriented codebase (with its own `dev`, `fine-tuning`, and `production` branches). This repo is the research counterpart: the same infrastructure foundations, but with stability guarantees relaxed in favour of experimental depth.

---

## `master` — The Generic Skeleton

The `master` branch strips the `code-doc-assistant` down to its reusable core:

**Removed** (domain-specific to code documentation):
- AST-aware chunking via tree-sitter
- Code-specific prompt templates
- File-type classification (`.py`, `.js`, etc.)
- `CodeSplitter` and its `tree-sitter-language-pack` dependency

**Retained** (generic, reusable):
- Composable model tier system (`full` / `balanced` / `lightweight`)
- `VectorStoreBase` ABC with `ChromaVectorStoreImpl` — swappable to Qdrant or FAISS
- Tier-aware configuration via `config.py` (mirrors `_helpers.tpl` logic)
- MLflow instrumentation hooks throughout the pipeline
- Docker Compose + Helm chart with the same composability patterns
- `SentenceSplitter` as the default chunking strategy (language-agnostic)

The result is a pipeline that can be pointed at any document corpus, ingested, embedded, and queried — with no assumptions about the domain. Any research branch that needs domain-specific behaviour adds it in branch-specific code, never modifying `core/`.

### The MLflow Instrumentation Layer

The most important addition in `master` (relative to `code-doc-assistant`) is the `instrumentation/` module. This defines the canonical MLflow logging schema shared between this repo and the `code-doc-assistant` `dev` branch.

The motivation: the `orchestrated` branch's long-loop optimisation consumes execution traces produced by `dev`. Without a shared schema, the two repos can't interoperate. The schema is defined here (as the research repo is the consumer) and `dev` must conform to it.

See `instrumentation/mlflow_schema.py` for the full field definitions and `README.md` for the field reference table.

---

## Research Branches

### `orchestrated`

**Central question**: can a supervising agent improve system behaviour — within a session, across sessions, and eventually at the level of graph topology — purely from observing execution traces and preference signals?

This branch adds a LangGraph-based supervisor to the pipeline. The supervisor is the frontier agent: it holds the broadest context, manages sub-agents, and is the only node responsible for preference learning. Centralising learning here keeps the failure modes contained and the update surface small.

#### Three optimisation loops

**Short-loop (within a session)**

The supervisor observes confidence scores and latency across queries within a session and adjusts in real time: raises `top_k` if retrieval confidence is low, switches embedding model weighting if a particular file type is consistently missed, lowers the similarity score threshold if the corpus is sparse. This is close to what Self-RAG does internally, but externalised as a named, inspectable supervisor node — auditable via the `supervisor_adjustments` field in `AgentState`.

Research question: what adjustment rules are stable? Which cause oscillation? The `SessionPreferences` dataclass (from `dev`) is the session-scoped preference state the supervisor reads and updates.

**Long-loop (across sessions, via MLflow)**

The supervisor reads the MLflow experiment log, detects patterns across many sessions — e.g. grep-based tools consistently outperform vector retrieval for certain query types — and proposes adjustments to graph parameters. This is not a large model doing the detection; it is closer to a classical anomaly detection or pattern mining problem over structured time-series data. The `supervisor_adjustments` audit trail from short-loop operation is the primary input.

Research question: which failure modes are systematic (appearing across sessions and users) versus idiosyncratic (specific to one session or user)? Systematic failures warrant graph-level changes; idiosyncratic ones are noise.

**Graph structure search (the deep research direction)**

Rather than tuning parameters within a fixed graph, the supervisor treats the graph topology itself as the search space. Which nodes should exist? Which edges should be conditional? This is not Neural Architecture Search in the gradient-based sense — LangGraph topologies are discrete and non-differentiable. The closer analogy is combinatorial search over graph topologies, with evolutionary or Bayesian optimisation over structure. The evaluation function is expensive (requires running the graph), which constrains the search strategy.

This is the most novel direction and the least tractable in the near term. It belongs in `orchestrated` as a long-horizon research target rather than a near-term implementation goal.

#### Online preference learning (sub-module of `orchestrated`)

The `online_preference/` sub-module within this branch implements three levels of preference adaptation, each progressively more aggressive:

**Level 1 — Cross-session prior injection** (technically simple, empirically interesting)

`SessionPreferences` from `dev` is session-scoped and in-memory. Level 1 serialises it to MLflow as a versioned artifact after each session; the next session loads the last artifact as a prior and injects it into the context. No model update — just context priming.

Research question: does prior preference injection actually improve first-response quality? Measurable with the existing satisfaction scoring in the MLflow schema.

**Level 2 — Online retrieval-layer adaptation** (moderate risk, contained failure modes)

Preference feedback adjusts retrieval weights, not LLM weights. Prioritised files get their embeddings boosted in the vector index; format preferences steer the context assembly ranking function. This is online learning operating on the retrieval layer, not the model layer — lower risk of destabilisation, more contained failure modes.

Research question: does retrieval-layer adaptation converge or oscillate? What is the right learning rate? What triggers instability?

**Level 3 — Asynchronous / co-working agent feedback** (decision theory problem)

Agents that operate while users are offline are a qualitatively different deployment context from interactive assistants. The feedback loop is asynchronous — the user reviews completed work post-hoc rather than correcting in real time. The preference update must be applied to the *next* agent invocation, not the current one. Standard RLHF assumes near-synchronous feedback; off-policy correction is needed here.

Research question: how much prior preference data is sufficient to trust an agent operating unsupervised? This is a decision theory question with safety implications, not just a machine learning question. The conservative operating mode (low autonomy, high caution) vs. the extrapolating mode (operate on accumulated priors) represents a fundamental design choice with different risk profiles.

The concrete grounding experiment for Level 3 is the autonomous multi-context documentation task: the orchestrating agent traverses a full codebase, identifies all AST leaves touching a specific subsystem (e.g. the GUI surface), and documents each in turn — without human intervention, supervised only by the orchestrating agent's own context management. The task is bounded (the AST scope is well-defined), verifiable (the output is inspectable), and representative of the broader unsupervised operation problem.

#### Resource elasticity note

Online learning at the retrieval layer has an infrastructure cascade that is easy to miss. If the frontier agent's preference adaptation changes the ChromaDB index size, the embedding model weighting, or the effective memory footprint mid-deployment, that has downstream effects on Kubernetes resource requests, HPA targets, and potentially the Ollama/vLLM serving configuration. This is a day-1 vs day-2 operations distinction:

- **Day-1**: static resource provisioning based on model tier
- **Day-2**: dynamic resource adjustment in response to learned behaviour changes

The `orchestrated` branch needs to document what triggers a day-2 resource event and what the response mechanism is. This is an infrastructure lifecycle question as much as an ML question.

#### The honest constraint

Online learning on LLM weights within a session is not the right target, for two reasons. First, feedback volume is too low — a few HITL signals per session is insufficient gradient signal to update a 7B+ parameter model meaningfully. Second, catastrophic forgetting: fine-tuning on a tiny session-specific dataset without regularisation damages general capabilities. The research opportunity is in the retrieval and context layers, where updates require less data and failure modes are more contained. LoRA/QLoRA across many sessions (the `fine-tuning` branch job) is the right mechanism for model-layer learning — not within-session updates.

---

### `emergent`

**Central question**: can useful multi-agent coordination emerge from shared state alone, without a central supervisor?

This branch removes the orchestrating supervisor entirely. Multiple agents operate in parallel, each with its own pipeline, coordinating only through a shared vector DB and a shared MLflow experiment log. No agent knows about the others' internal state — they only see what has been committed to shared state.

The vector DB topology question is central here. Options include:

- **Agent-local indexes** — each agent has its own embedding model and its own index. Fully private, no cross-contamination, no shared knowledge.
- **Shared DB, shared embedding model** — multiple agents index into the same vector space. Simple, but requires all agents to agree on one embedding model.
- **Shared DB, heterogeneous embedding models** — different agents embed with different models, contributing to the same index via separate namespaces. Requires either namespace isolation within the DB or a cross-embedding-space alignment layer. The latter is an active research problem.
- **Hierarchical DB** — agents have local indexes for fast private retrieval, with periodic propagation of selected knowledge upward to a shared index.

Research question: at what task complexity does emergent coordination through shared state break down, and what coordination mechanism is the minimum viable addition to recover useful behaviour?

---

### `protocol-driven`

**Central question**: can initially informationally isolated agents — potentially belonging to different organisations — build productive collaboration through a negotiated protocol, without violating either party's information boundaries?

This branch implements the B2B cold-start collaboration model: two agents with no shared context, no prior relationship, no knowledge of each other's internal state. The collaboration is initiated through a handshake protocol, scoped to a specific shared goal, and terminated cleanly.

The protocol requires:

- **Capability and consent negotiation** — agents negotiate what they are willing to share, in what format, and for what purpose, before any substantive exchange.
- **Joint context construction** — rather than sharing raw embeddings or documents, agents collaboratively construct a *joint context object*: a deliberately limited, mutually agreed representation of the shared problem space. Neither agent's private knowledge is exposed; only the intersection relevant to the collaborative goal.
- **Directional information flow controls** — strict constraints on what propagates where, with audit trails. Agent A's proprietary knowledge never appears in Agent B's index.
- **Goal anchoring** — the joint context is always scoped to a specific collaborative objective. Without this, shared context tends to drift and expand until privacy boundaries are violated.

#### Federated preference aggregation

The more speculative direction within this branch: if multiple organisations deploy instances of the same agent, each accumulating localised `SessionPreferences` histories, federated averaging over those preference profiles could produce a shared prior that outperforms any individual organisation's data.

The key privacy question: are preference profiles — format preferences, verbosity, file prioritisation patterns — sufficiently abstract to aggregate safely without leaking task-specific or codebase-specific information? This is a non-trivial privacy analysis. The connection to federated learning (KubeFlower, FedAvg over model weights) is direct, but preference profiles are much lower-dimensional than model weights, which changes the re-identification risk calculus.

Research question: does a federated prior outperform a local prior? At what data scale does federation become worthwhile? What is the differential privacy budget required to make preference sharing safe?

---

### `fine-tuning` (deferred)

LoRA/QLoRA on accumulated cross-session preference data. Model-weight learning only after sufficient data accumulation across many users and sessions — not within-session. This is the `fine-tuning` branch of `code-doc-assistant` applied to the research context: accumulated `SessionPreferences` histories from Level 1–3 of the `orchestrated` branch become the training signal.

Not started until the `orchestrated` branch has produced sufficient instrumented data to make fine-tuning meaningful.

---

## Branch Dependency Map

```
master  ──────────────────────────────────────────────────────────
         │                    │                    │
         ▼                    ▼                    ▼
    orchestrated           emergent          protocol-driven
         │
         │  (consumes MLflow traces)
         │◀─────────────────────────────── code-doc-assistant/dev
         │
         ▼
    fine-tuning  (deferred, consumes orchestrated session data)
```

`code-doc-assistant/dev` is an external dependency of `orchestrated`, not a branch of this repo. The MLflow schema defined in `instrumentation/mlflow_schema.py` is the contract between them.

---

## The Research Contribution

The novel framing, stated precisely: **retrieval-layer online learning as a proxy for preference adaptation, with federated aggregation of the resulting retrieval profiles as a privacy-preserving coordination mechanism across deployments.**

This is specific enough to be a research contribution, grounded enough in the existing architecture to be tractable, and connected enough to the infrastructure lifecycle questions (day-1 vs day-2 ops, resource elasticity) to be practically relevant beyond the ML domain.

---

## What This Is Not

This repository does not pursue:

- Online learning on LLM weights within a session (feedback volume too low, catastrophic forgetting risk)
- Full graph-search optimisation in the near term (the search space is discrete, evaluation is expensive, tooling is immature — it is documented as a long-horizon direction)
- General-purpose agent frameworks (LangChain is documented as a future growth path; LlamaIndex is used where its RAG-native tooling is the right fit)

These exclusions are deliberate. The research is constrained to what is tractable with the existing infrastructure and data volumes, while pointing clearly toward longer-horizon directions.
