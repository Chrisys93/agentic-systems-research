"""
instrumentation — Canonical MLflow logging schema and per-query metrics.

This package defines the shared contract between the research repo and
code-doc-assistant's dev branch. The schema is the authoritative source;
dev must conform to it.

Modules:
    mlflow_schema  — Run parameters, metrics, artifacts, and logging helpers
    metrics        — Computed per-query metrics independent of MLflow
"""

from instrumentation.mlflow_schema import (
    SCHEMA_VERSION,
    RunParams,
    RunMetrics,
    RunArtifacts,
    log_run_params,
    log_run_metrics,
    log_run_artifacts,
    log_supervisor_step,
    log_generation_step,
    log_quality_gate,
    log_user_satisfaction,
    log_session_preferences,
    generate_query_id,
)

from instrumentation.metrics import (
    QueryMetrics,
    compute_query_metrics,
    format_metrics_summary,
)
