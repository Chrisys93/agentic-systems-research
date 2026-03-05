"""
mlflow_schema.py — Canonical MLflow logging schema for the agentic pipeline.

This module defines the shared contract between this research repo and
code-doc-assistant's `dev` branch. Any experiment that consumes execution
traces from `dev` must conform to this schema. Any changes here require
coordinated updates in both repos.

The schema is split into three layers:
  1. Run parameters  — static per query (logged once via mlflow.log_param)
  2. Run metrics     — numeric per query (logged via mlflow.log_metric)
  3. Run artifacts   — structured data per query (logged via mlflow.log_text/log_dict)

Usage:
    from instrumentation.mlflow_schema import log_run_params, log_run_metrics, log_run_artifacts

    with mlflow.start_run() as run:
        log_run_params(query=query, repo_path=repo_path, ...)
        # ... run the pipeline ...
        log_run_metrics(final_state)
        log_run_artifacts(final_state)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import mlflow


# ---------------------------------------------------------------------------
# Schema version — bump this when fields are added/removed/renamed.
# Consumers should check this before parsing traces.
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Canonical field definitions
# ---------------------------------------------------------------------------

@dataclass
class RunParams:
    """Static parameters logged once per query run."""
    query_id: str                          # UUID per query
    session_id: str                        # UUID per session (thread_id)
    schema_version: str = SCHEMA_VERSION
    query: str = ""
    repo_path: str = ""
    model_tier: str = "full"               # full / balanced / lightweight
    inference_backend: str = "ollama"       # ollama / vllm
    model_name: str = ""                   # resolved model name
    embedding_model: str = "nomic-embed-text"
    hitl_enabled: bool = True
    output_review_mode: str = "human"      # human / supervisor / self / off


@dataclass
class RunMetrics:
    """Numeric metrics logged per query run. Step-indexed metrics (e.g. per
    retrieval attempt) are logged inline during execution; this captures
    the summary metrics logged at run completion."""
    total_tool_calls: int = 0
    retrieval_attempts: int = 0
    chunks_retrieved: int = 0
    supervisor_adjustments_count: int = 0
    generation_attempts: int = 0
    total_latency_ms: float = 0.0
    # Per-step metrics (logged with step= during execution):
    #   supervisor_mean_score       — step=retrieval_attempt
    #   supervisor_retry            — step=retrieval_attempt
    #   generation_latency_ms       — step=generation_attempt
    #   response_length_chars       — step=generation_attempt
    #   quality_gate_score          — step=generation_attempt (supervisor mode)
    #   quality_gate_pass           — step=generation_attempt (supervisor mode)
    #   user_satisfaction           — step=generation_attempt (human mode)


@dataclass
class RunArtifacts:
    """Structured data logged as artifacts per query run."""
    tool_selections: list[str] = field(default_factory=list)
    confidence_scores: list[float] = field(default_factory=list)
    supervisor_adjustments: list[dict[str, Any]] = field(default_factory=list)
    source_attribution: list[str] = field(default_factory=list)
    execution_trace: list[dict[str, Any]] = field(default_factory=list)
    session_preferences: Optional[dict[str, Any]] = None
    satisfaction_score: Optional[int] = None        # 1-5, from HITL feedback


# ---------------------------------------------------------------------------
# Logging helpers — called from the pipeline entry point
# ---------------------------------------------------------------------------

def generate_query_id() -> str:
    """Generate a unique query ID."""
    return str(uuid.uuid4())


def log_run_params(
    query: str,
    repo_path: str,
    session_id: str,
    inference_backend: str = "ollama",
    model_name: str = "",
    model_tier: str = "full",
    embedding_model: str = "nomic-embed-text",
    hitl_enabled: bool = True,
    output_review_mode: str = "human",
) -> RunParams:
    """Log canonical run parameters to the active MLflow run.

    Returns the RunParams dataclass for downstream reference.
    """
    params = RunParams(
        query_id=generate_query_id(),
        session_id=session_id,
        query=query,
        repo_path=repo_path,
        model_tier=model_tier,
        inference_backend=inference_backend,
        model_name=model_name,
        embedding_model=embedding_model,
        hitl_enabled=hitl_enabled,
        output_review_mode=output_review_mode,
    )
    for k, v in asdict(params).items():
        mlflow.log_param(k, v)
    return params


def log_run_metrics(state: dict[str, Any]) -> RunMetrics:
    """Extract and log summary metrics from final AgentState.

    Returns the RunMetrics dataclass for downstream reference.
    """
    metrics = RunMetrics(
        total_tool_calls=len(state.get("executed_tool_calls", [])),
        retrieval_attempts=state.get("retrieval_attempts", 0),
        chunks_retrieved=len(state.get("retrieved_chunks", [])),
        supervisor_adjustments_count=len(state.get("supervisor_adjustments", [])),
        generation_attempts=state.get("generation_attempts", 0),
        total_latency_ms=state.get("total_latency_ms", 0.0) or 0.0,
    )
    for k, v in asdict(metrics).items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k, v)
    return metrics


def log_run_artifacts(state: dict[str, Any]) -> RunArtifacts:
    """Extract and log structured artifacts from final AgentState.

    Returns the RunArtifacts dataclass for downstream reference.
    """
    # Extract tool names in execution order
    tool_selections = [
        tc.tool_name if hasattr(tc, "tool_name") else tc.get("tool_name", "unknown")
        for tc in state.get("executed_tool_calls", [])
    ]

    # Extract supervisor adjustments as dicts
    raw_adjustments = state.get("supervisor_adjustments", [])
    adjustments = [
        asdict(adj) if hasattr(adj, "__dataclass_fields__") else adj
        for adj in raw_adjustments
    ]

    # Extract session preferences
    prefs = state.get("session_preferences")
    prefs_dict = None
    if prefs is not None:
        prefs_dict = asdict(prefs) if hasattr(prefs, "__dataclass_fields__") else prefs

    # Extract satisfaction from post-generation feedback
    feedback = state.get("post_generation_feedback")
    satisfaction = None
    if feedback is not None:
        satisfaction = (
            feedback.satisfaction_score
            if hasattr(feedback, "satisfaction_score")
            else feedback.get("satisfaction_score")
        )

    artifacts = RunArtifacts(
        tool_selections=tool_selections,
        confidence_scores=state.get("confidence_scores", []),
        supervisor_adjustments=adjustments,
        source_attribution=state.get("source_attribution", []),
        execution_trace=state.get("execution_trace", []),
        session_preferences=prefs_dict,
        satisfaction_score=satisfaction,
    )

    mlflow.log_text(
        json.dumps(asdict(artifacts), indent=2, default=str),
        "query_artifacts.json",
    )
    return artifacts


# ---------------------------------------------------------------------------
# Step-level logging helpers — called from individual graph nodes
# ---------------------------------------------------------------------------

def log_supervisor_step(mean_score: float, retry: bool, step: int) -> None:
    """Log supervisor metrics at a specific retrieval attempt step."""
    mlflow.log_metric("supervisor_mean_score", mean_score, step=step)
    if retry:
        mlflow.log_metric("supervisor_retry", 1, step=step)


def log_generation_step(
    latency_ms: float,
    response_length: int,
    step: int,
    response_text: Optional[str] = None,
) -> None:
    """Log generation metrics at a specific generation attempt step."""
    mlflow.log_metric("generation_latency_ms", latency_ms, step=step)
    mlflow.log_metric("response_length_chars", response_length, step=step)
    mlflow.log_metric("generation_attempts", step, step=step)
    if response_text is not None:
        mlflow.log_text(response_text, f"response_attempt_{step}.txt")


def log_quality_gate(score: float, passed: bool, step: int) -> None:
    """Log supervisor quality gate result."""
    mlflow.log_metric("quality_gate_score", score, step=step)
    mlflow.log_metric("quality_gate_pass", int(passed), step=step)


def log_user_satisfaction(
    satisfaction_score: int,
    decision: str,
    step: int,
    format_preference: Optional[str] = None,
) -> None:
    """Log post-generation human feedback."""
    mlflow.log_metric("user_satisfaction", satisfaction_score, step=step)
    mlflow.log_param("output_decision", decision)
    if format_preference:
        mlflow.log_param("format_preference", format_preference)


def log_session_preferences(prefs: Any) -> None:
    """Log accumulated session preference metrics."""
    if prefs is None:
        return
    avg_sat = prefs.avg_satisfaction if hasattr(prefs, "avg_satisfaction") else prefs.get("avg_satisfaction", 0)
    count = prefs.feedback_count if hasattr(prefs, "feedback_count") else prefs.get("feedback_count", 0)
    mlflow.log_metric("session_avg_satisfaction", avg_sat)
    mlflow.log_metric("session_feedback_count", count)
