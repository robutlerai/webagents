"""OpenTelemetry for the agent loop, when the API package is here (plan item 2.4)."""

from .otel import (
    NOOP_AGENT_RUN,
    OBSERVABILITY_KEY,
    OTEL_ENV_VAR,
    OTEL_RUN_CONTEXT_KEY,
    AgentRun,
    ObservabilityConfigError,
    error_type,
    load_otel_api,
    otel_enabled,
    parse_observability,
    set_otel_api_for_tests,
    start_agent_run,
)

__all__ = [
    "NOOP_AGENT_RUN",
    "OBSERVABILITY_KEY",
    "OTEL_ENV_VAR",
    "OTEL_RUN_CONTEXT_KEY",
    "AgentRun",
    "ObservabilityConfigError",
    "error_type",
    "load_otel_api",
    "otel_enabled",
    "parse_observability",
    "set_otel_api_for_tests",
    "start_agent_run",
]
