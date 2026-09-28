"""
OpenTelemetry for the agent loop (2026-09-26, gap-closure plan item 2.4).

Spans and metrics that follow the OpenTelemetry GenAI semantic conventions
(`gen_ai.*` attributes): the agent run (`invoke_agent <agent>`), every model
call (`chat <model>`, with the provider and the tokens), every tool call
(`execute_tool <tool>`) and every payment settle (`settle_payment`, the amount
in credits). The same names and attributes as the TypeScript SDK
(`typescript/src/observability/otel.ts`), pinned by the shared fixture
`tests/fixtures/w2ops/otel.json`.

NO NEW DEPENDENCY. The `opentelemetry` API package is used only when it is
already importable; without it, or with the switch off, every call here is a
no-op that costs one attribute read. Where the spans go is the host's
business: whoever sets a TracerProvider with the API (an SDK, an exporter, a
collector agent) gets them; nothing is exported by default.

SWITCHED ON by `observability: {otel: true}` in the agent file, or by
WEBAGENTS_OTEL=1 in the environment (the file wins when it says something).

NEVER MESSAGE TEXT, TOOL ARGUMENTS OR TOOL RESULTS in an attribute (S-227 in
the portal's security log, the rule the trace lines already follow): a span
is a record that leaves the process. Names, models, token counts, durations
and outcomes only.
"""

from __future__ import annotations

import importlib
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

OBSERVABILITY_KEY = "observability"
OTEL_ENV_VAR = "WEBAGENTS_OTEL"
_ON_VALUES = frozenset({"1", "true", "on", "yes"})
INSTRUMENTATION_NAME = "webagents"

#: The context key the run handle travels under, for skills that settle payments.
OTEL_RUN_CONTEXT_KEY = "_otel_run"


class ObservabilityConfigError(Exception):
    """A block that cannot be used as written, with the fixture's sentence."""


def parse_observability(value: Any) -> Dict[str, Any]:
    """The `observability:` block as an agent file writes it: `{otel: true}`,
    or a bare boolean as the short form. An unknown key or a value that is
    not a boolean is refused with the fixture's sentence, as the TypeScript
    loader refuses it, so a file means one thing to both SDKs."""
    if value is None:
        return {}
    if isinstance(value, bool):
        return {"otel": value}
    if not isinstance(value, Mapping):
        raise ObservabilityConfigError("observability: otel must be true or false")
    for key in value:
        if key != "otel":
            raise ObservabilityConfigError(f"observability: unknown key '{key}'. It takes otel.")
    otel = value.get("otel")
    if otel is None:
        return {}
    if not isinstance(otel, bool):
        raise ObservabilityConfigError("observability: otel must be true or false")
    return {"otel": otel}


def otel_enabled(config: Optional[Mapping[str, Any]], env: Optional[Mapping[str, str]] = None) -> bool:
    """Whether spans are wanted: the file's word, else the environment's."""
    if config is not None and config.get("otel") is not None:
        return bool(config.get("otel"))
    raw = (os.environ if env is None else env).get(OTEL_ENV_VAR)
    return bool(raw) and raw.strip().lower() in _ON_VALUES


# ---------------------------------------------------------------------------
# The API, loaded when it is there
# ---------------------------------------------------------------------------


@dataclass
class _Instruments:
    trace: Any
    tracer: Any
    token_usage: Any
    operation_duration: Any
    payment_credits: Any


_loaded: Optional[_Instruments] = None
_tried = False
_override: Optional[Dict[str, Any]] = None


def _build(trace: Any, metrics: Any, tracer_provider: Any = None, meter_provider: Any = None) -> _Instruments:
    tracer = (tracer_provider.get_tracer(INSTRUMENTATION_NAME) if tracer_provider is not None else trace.get_tracer(INSTRUMENTATION_NAME))
    meter = (meter_provider.get_meter(INSTRUMENTATION_NAME) if meter_provider is not None else metrics.get_meter(INSTRUMENTATION_NAME))
    return _Instruments(
        trace=trace,
        tracer=tracer,
        token_usage=meter.create_histogram("gen_ai.client.token.usage", unit="{token}", description="Tokens used per model call"),
        operation_duration=meter.create_histogram("gen_ai.client.operation.duration", unit="s", description="Duration of a model or tool call"),
        payment_credits=meter.create_counter("webagents.payment.credits", unit="{credit}", description="Credits settled by this agent"),
    )


def _instruments() -> Optional[_Instruments]:
    """The API's instruments, once; None when the package cannot load."""
    global _loaded, _tried
    if _loaded is not None or _tried:
        return _loaded
    _tried = True
    try:
        if _override is not None:
            _loaded = _build(_override["trace"], _override["metrics"], _override.get("tracer_provider"), _override.get("meter_provider"))
            return _loaded
        trace = importlib.import_module("opentelemetry.trace")
        metrics = importlib.import_module("opentelemetry.metrics")
        _loaded = _build(trace, metrics)
    except Exception:  # noqa: BLE001 - not importable, or an API this module does not know: the no-op case
        _loaded = None
    return _loaded


def load_otel_api() -> Optional[Any]:
    """The `opentelemetry.trace` module, or None when the package is not importable here."""
    inst = _instruments()
    return inst.trace if inst is not None else None


def set_otel_api_for_tests(
    trace: Any = None,
    metrics: Any = None,
    tracer_provider: Any = None,
    meter_provider: Any = None,
) -> None:
    """For tests: use these modules and providers (an in-memory recorder)
    instead of the global ones, or, with nothing, load the real API again."""
    global _loaded, _tried, _override
    _override = None if trace is None else {
        "trace": trace,
        "metrics": metrics,
        "tracer_provider": tracer_provider,
        "meter_provider": meter_provider,
    }
    _loaded = None
    _tried = False


# ---------------------------------------------------------------------------
# One agent run
# ---------------------------------------------------------------------------


def _now_ns() -> int:
    return time.time_ns()


class AgentRun:
    """The run's span, and the child spans recorded under it. A no-op when off."""

    active = False

    def model_call_started(self, provider: str, model: str) -> None:
        """A model call is under way: its provider and model, for the span that ends it."""

    def model_call_ended(
        self,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        response_model: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """The model call that started last, with what it reported."""

    def tool_call(self, name: str, call_id: Optional[str], started_at_ns: int, error: Optional[str] = None) -> None:
        """One tool call: its name, id, duration and outcome. Never its arguments."""

    def payment_settle(self, credits: float, lock_id: Optional[str] = None, started_at_ns: Optional[int] = None, error: Optional[str] = None) -> None:
        """One payment settle: the amount in credits, the lock, the outcome."""

    def end(self, error: Optional[str] = None) -> None:
        """The run is over; a model call still open is ended with the same error."""


NOOP_AGENT_RUN = AgentRun()


class _RecordedRun(AgentRun):
    active = True

    def __init__(self, inst: _Instruments, agent_name: str, conversation_id: Optional[str]) -> None:
        self._inst = inst
        self._agent_name = agent_name
        attributes: Dict[str, Any] = {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": agent_name}
        if conversation_id:
            attributes["gen_ai.conversation.id"] = conversation_id
        trace = inst.trace
        self._span = inst.tracer.start_span(f"invoke_agent {agent_name}", kind=trace.SpanKind.INTERNAL, attributes=attributes, start_time=_now_ns())
        self._parent = trace.set_span_in_context(self._span)
        self._pending: Optional[Dict[str, Any]] = None
        self._ended = False

    def _child(self, name: str, kind: Any, started_at_ns: int, attributes: Dict[str, Any], ended_at_ns: Optional[int], error: Optional[str]) -> None:
        trace = self._inst.trace
        span = self._inst.tracer.start_span(name, context=self._parent, kind=kind, attributes=attributes, start_time=started_at_ns)
        if error:
            span.set_attribute("error.type", error)
            span.set_status(trace.Status(trace.StatusCode.ERROR))
        span.end(end_time=ended_at_ns if ended_at_ns is not None else _now_ns())

    def model_call_started(self, provider: str, model: str) -> None:
        if self._pending is not None:
            # A call that was never ended: closed as unfinished, so its span is not lost.
            self.model_call_ended(error="unfinished")
        self._pending = {"provider": provider, "model": model, "started": _now_ns()}

    def model_call_ended(
        self,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        response_model: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        pending, self._pending = self._pending, None
        if pending is None:
            return
        provider, model, started = pending["provider"], pending["model"], pending["started"]
        attributes: Dict[str, Any] = {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": provider,
            "gen_ai.request.model": model,
        }
        if response_model and response_model != model:
            attributes["gen_ai.response.model"] = response_model
        if input_tokens is not None:
            attributes["gen_ai.usage.input_tokens"] = int(input_tokens)
        if output_tokens is not None:
            attributes["gen_ai.usage.output_tokens"] = int(output_tokens)
        ended = _now_ns()
        self._child(f"chat {model}", self._inst.trace.SpanKind.CLIENT, started, attributes, ended, error)
        metric_attributes = {"gen_ai.operation.name": "chat", "gen_ai.provider.name": provider, "gen_ai.request.model": model}
        if input_tokens is not None:
            self._inst.token_usage.record(int(input_tokens), {**metric_attributes, "gen_ai.token.type": "input"})
        if output_tokens is not None:
            self._inst.token_usage.record(int(output_tokens), {**metric_attributes, "gen_ai.token.type": "output"})
        self._inst.operation_duration.record(max(0, ended - started) / 1e9, metric_attributes)

    def tool_call(self, name: str, call_id: Optional[str], started_at_ns: int, error: Optional[str] = None) -> None:
        attributes: Dict[str, Any] = {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": name, "gen_ai.tool.type": "function"}
        if call_id:
            attributes["gen_ai.tool.call.id"] = call_id
        ended = _now_ns()
        self._child(f"execute_tool {name}", self._inst.trace.SpanKind.INTERNAL, started_at_ns, attributes, ended, error)
        self._inst.operation_duration.record(max(0, ended - started_at_ns) / 1e9, {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": name})

    def payment_settle(self, credits: float, lock_id: Optional[str] = None, started_at_ns: Optional[int] = None, error: Optional[str] = None) -> None:
        attributes: Dict[str, Any] = {"gen_ai.agent.name": self._agent_name, "webagents.payment.credits": float(credits)}
        if lock_id:
            attributes["webagents.payment.lock_id"] = lock_id
        self._child("settle_payment", self._inst.trace.SpanKind.CLIENT, started_at_ns if started_at_ns is not None else _now_ns(), attributes, None, error)
        if not error and credits > 0:
            self._inst.payment_credits.add(float(credits), {"gen_ai.agent.name": self._agent_name})

    def end(self, error: Optional[str] = None) -> None:
        if self._ended:
            return
        self._ended = True
        if self._pending is not None:
            self.model_call_ended(error=error or "unfinished")
        trace = self._inst.trace
        if error:
            self._span.set_attribute("error.type", error)
            self._span.set_status(trace.Status(trace.StatusCode.ERROR))
        else:
            self._span.set_status(trace.Status(trace.StatusCode.OK))
        self._span.end(end_time=_now_ns())


def start_agent_run(config: Optional[Mapping[str, Any]], agent_name: str, conversation_id: Optional[str] = None) -> AgentRun:
    """Start the run's span, when the switch is on and the API is here; the
    no-op handle otherwise. Never raises: a tracing failure must not fail a turn."""
    if not otel_enabled(config):
        return NOOP_AGENT_RUN
    try:
        inst = _instruments()
        if inst is None:
            return NOOP_AGENT_RUN
        return _RecordedRun(inst, agent_name, conversation_id)
    except Exception:  # noqa: BLE001 - see the docstring
        return NOOP_AGENT_RUN


def error_type(error: BaseException) -> str:
    """A span's `error.type` for an exception: its code, else its class name."""
    code = getattr(error, "code", None) or getattr(error, "error_code", None)
    if isinstance(code, str) and code:
        return code
    return type(error).__name__
