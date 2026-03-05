"""
metrics.py — Per-query metrics computed from AgentState.

These metrics are computed from the pipeline's final state and are independent
of the MLflow logging layer. They can be used for:
  - Real-time UI display (confidence indicators, latency badges)
  - Supervisor decision-making (confidence thresholds, retry logic)
  - Downstream research analysis (aggregation, trend detection)

The metrics module does NOT call MLflow directly — it computes values that
the caller can then log, display, or analyse as needed. This separation
keeps the metrics logic testable without MLflow dependencies.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, asdict
from typing import Any, Optional


@dataclass
class QueryMetrics:
    """Computed metrics for a single query execution."""

    # --- Retrieval quality ---
    retrieval_confidence_mean: float        # mean similarity score across retrieved chunks
    retrieval_confidence_min: float         # worst chunk score (flags weak retrievals)
    retrieval_confidence_max: float         # best chunk score
    retrieval_confidence_std: float         # spread — high std means mixed quality
    chunks_above_threshold: int             # chunks scoring above CONFIDENCE_THRESHOLD
    chunks_total: int                       # total chunks retrieved

    # --- Pipeline behaviour ---
    retrieval_attempts: int                 # how many supervisor retry loops
    generation_attempts: int                # how many regeneration loops
    tool_call_count: int                    # total tools executed
    unique_tools_used: int                  # distinct tool names
    supervisor_adjustment_count: int        # how many times supervisor intervened

    # --- Latency ---
    total_latency_ms: float                 # end-to-end query time
    avg_tool_latency_ms: float              # mean per-tool execution time
    max_tool_latency_ms: float              # slowest tool (bottleneck indicator)

    # --- User feedback (if available) ---
    satisfaction_score: Optional[int] = None  # 1–5 from HITL, None if no feedback
    output_decision: Optional[str] = None    # accept / regenerate / add_context

    def to_dict(self) -> dict[str, Any]:
        """Serialise for logging or API response."""
        return asdict(self)


def compute_query_metrics(
    state: dict[str, Any],
    confidence_threshold: float = 0.45,
) -> QueryMetrics:
    """Compute QueryMetrics from a final AgentState dict.

    Args:
        state:                Final AgentState after pipeline execution.
        confidence_threshold: Score threshold for counting high-confidence chunks.
                              Matches the CONFIDENCE_THRESHOLD env var default.
    """
    # --- Retrieval scores ---
    scores = state.get("confidence_scores", [])
    if scores:
        mean_score = statistics.mean(scores)
        min_score = min(scores)
        max_score = max(scores)
        std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
        above_threshold = sum(1 for s in scores if s >= confidence_threshold)
    else:
        mean_score = min_score = max_score = std_score = 0.0
        above_threshold = 0

    # --- Tool latencies ---
    executed = state.get("executed_tool_calls", [])
    tool_latencies = []
    tool_names = set()
    for tc in executed:
        latency = tc.latency_ms if hasattr(tc, "latency_ms") else tc.get("latency_ms")
        name = tc.tool_name if hasattr(tc, "tool_name") else tc.get("tool_name", "unknown")
        tool_names.add(name)
        if latency is not None:
            tool_latencies.append(latency)

    avg_tool_lat = statistics.mean(tool_latencies) if tool_latencies else 0.0
    max_tool_lat = max(tool_latencies) if tool_latencies else 0.0

    # --- Feedback ---
    feedback = state.get("post_generation_feedback")
    satisfaction = None
    decision = None
    if feedback is not None:
        satisfaction = (
            feedback.satisfaction_score
            if hasattr(feedback, "satisfaction_score")
            else feedback.get("satisfaction_score")
        )
        decision = (
            feedback.decision
            if hasattr(feedback, "decision")
            else feedback.get("decision")
        )

    return QueryMetrics(
        retrieval_confidence_mean=round(mean_score, 4),
        retrieval_confidence_min=round(min_score, 4),
        retrieval_confidence_max=round(max_score, 4),
        retrieval_confidence_std=round(std_score, 4),
        chunks_above_threshold=above_threshold,
        chunks_total=len(state.get("retrieved_chunks", [])),
        retrieval_attempts=state.get("retrieval_attempts", 0),
        generation_attempts=state.get("generation_attempts", 0),
        tool_call_count=len(executed),
        unique_tools_used=len(tool_names),
        supervisor_adjustment_count=len(state.get("supervisor_adjustments", [])),
        total_latency_ms=state.get("total_latency_ms", 0.0) or 0.0,
        avg_tool_latency_ms=round(avg_tool_lat, 2),
        max_tool_latency_ms=round(max_tool_lat, 2),
        satisfaction_score=satisfaction,
        output_decision=decision,
    )


def format_metrics_summary(metrics: QueryMetrics) -> str:
    """Human-readable one-line summary for UI display or logging."""
    parts = [
        f"confidence={metrics.retrieval_confidence_mean:.2f}",
        f"chunks={metrics.chunks_total} ({metrics.chunks_above_threshold} above threshold)",
        f"tools={metrics.tool_call_count} ({metrics.unique_tools_used} unique)",
        f"retrieval_loops={metrics.retrieval_attempts}",
        f"gen_loops={metrics.generation_attempts}",
        f"latency={metrics.total_latency_ms:.0f}ms",
    ]
    if metrics.satisfaction_score is not None:
        parts.append(f"satisfaction={metrics.satisfaction_score}/5")
    return " | ".join(parts)
