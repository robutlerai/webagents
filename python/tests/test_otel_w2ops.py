"""
OpenTelemetry for the agent loop (2026-09-26, gap-closure plan item 2.4, lane
w2-ops), against the shared fixture `tests/fixtures/w2ops/otel.json`, which
the TypeScript suite reads too (`otel-w2ops.test.ts`): the switch, the span
names and attributes, the metrics, and the rule that no message text or tool
argument reaches an attribute (S-227).

The API package is what the module needs; where the SDK is installed too, the
spans go through its in-memory exporter, and otherwise through an in-memory
recorder built on the API's own interfaces (the SDK is not a dependency of
this package). Without the API package the tests are skipped, with the reason.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.base import Handoff, Skill
from webagents.agents.tools.decorators import tool
from webagents.observability.otel import (
    OTEL_ENV_VAR,
    ObservabilityConfigError,
    load_otel_api,
    otel_enabled,
    parse_observability,
    set_otel_api_for_tests,
    start_agent_run,
)

FIXTURE = json.loads((Path(__file__).resolve().parent / "fixtures" / "w2ops" / "otel.json").read_text())
S = FIXTURE["scenario"]

otel_trace = pytest.importorskip("opentelemetry.trace", reason="the opentelemetry API package is not installed here")
otel_metrics = pytest.importorskip("opentelemetry.metrics", reason="the opentelemetry API package is not installed here")


# ---------------------------------------------------------------------------
# An in-memory recorder: the SDK's exporter when the SDK is here, else the API
# ---------------------------------------------------------------------------


class _Recorder:
    """Spans and metric points as plain dicts, whichever way they were recorded."""

    def __init__(self) -> None:
        self.spans: List[Dict[str, Any]] = []
        self.points: List[Dict[str, Any]] = []
        self.instruments: List[Dict[str, Any]] = []
        self.tracer_provider: Any = None
        self.meter_provider: Any = None

    def ordered_spans(self) -> List[Dict[str, Any]]:
        return sorted(self.spans, key=lambda s: s["start"])


class _ApiSpan(otel_trace.Span):
    """A recording span on the API's own interface, so `set_span_in_context`
    and `get_current_span` carry it as a parent (they only carry a `Span`)."""

    def __init__(self, recorder: _Recorder, name: str, kind: Any, attributes: Dict[str, Any], parent: Optional["_ApiSpan"], start: int) -> None:
        self.record = {"name": name, "kind": kind, "attributes": dict(attributes or {}), "parent": parent.record if parent else None, "status": None, "start": start, "end": None}
        recorder.spans.append(self.record)

    def set_attribute(self, key: str, value: Any) -> None:
        self.record["attributes"][key] = value

    def set_attributes(self, attributes: Any) -> None:
        self.record["attributes"].update(dict(attributes or {}))

    def set_status(self, status: Any, description: Optional[str] = None) -> None:
        self.record["status"] = getattr(status, "status_code", status)

    def end(self, end_time: Optional[int] = None) -> None:
        self.record["end"] = end_time

    def get_span_context(self) -> Any:
        return otel_trace.INVALID_SPAN_CONTEXT

    def add_event(self, name: str, attributes: Any = None, timestamp: Optional[int] = None) -> None:
        pass

    def add_link(self, context: Any, attributes: Any = None) -> None:
        pass

    def update_name(self, name: str) -> None:
        self.record["name"] = name

    def is_recording(self) -> bool:
        return True

    def record_exception(self, exception: BaseException, attributes: Any = None, timestamp: Optional[int] = None, escaped: bool = False) -> None:
        pass


class _ApiTracer:
    def __init__(self, recorder: _Recorder) -> None:
        self.recorder = recorder

    def start_span(self, name: str, context: Any = None, kind: Any = None, attributes: Any = None, links: Any = None, start_time: Optional[int] = None, **_: Any) -> _ApiSpan:
        parent = otel_trace.get_current_span(context) if context is not None else None
        parent_span = parent if isinstance(parent, _ApiSpan) else None
        return _ApiSpan(self.recorder, name, kind, attributes or {}, parent_span, start_time or 0)


class _ApiInstrument:
    def __init__(self, recorder: _Recorder, name: str) -> None:
        self.recorder, self.name = recorder, name

    def record(self, value: Any, attributes: Any = None, context: Any = None) -> None:
        self.recorder.points.append({"instrument": self.name, "value": value, "attributes": dict(attributes or {})})

    def add(self, value: Any, attributes: Any = None, context: Any = None) -> None:
        self.recorder.points.append({"instrument": self.name, "value": value, "attributes": dict(attributes or {})})


class _ApiMeter:
    def __init__(self, recorder: _Recorder) -> None:
        self.recorder = recorder

    def create_histogram(self, name: str, unit: str = "", description: str = "", **_: Any) -> _ApiInstrument:
        self.recorder.instruments.append({"name": name, "kind": "histogram", "unit": unit})
        return _ApiInstrument(self.recorder, name)

    def create_counter(self, name: str, unit: str = "", description: str = "", **_: Any) -> _ApiInstrument:
        self.recorder.instruments.append({"name": name, "kind": "counter", "unit": unit})
        return _ApiInstrument(self.recorder, name)


class _ApiProviders:
    def __init__(self, recorder: _Recorder) -> None:
        self.recorder = recorder

    def get_tracer(self, name: str, *args: Any, **kwargs: Any) -> _ApiTracer:
        return _ApiTracer(self.recorder)

    def get_meter(self, name: str, *args: Any, **kwargs: Any) -> _ApiMeter:
        return _ApiMeter(self.recorder)


def _sdk_recorder() -> Optional[_Recorder]:
    """The SDK's in-memory exporter and reader, when the SDK is installed."""
    try:
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    except Exception:  # noqa: BLE001 - no SDK here
        return None

    recorder = _Recorder()
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])
    recorder.tracer_provider, recorder.meter_provider = tracer_provider, meter_provider

    def ordered_spans() -> List[Dict[str, Any]]:
        by_id: Dict[Any, Dict[str, Any]] = {}
        out = []
        for span in sorted(exporter.get_finished_spans(), key=lambda s: s.start_time):
            record = {
                "name": span.name,
                "kind": span.kind,
                "attributes": dict(span.attributes or {}),
                "parent": by_id.get(span.parent.span_id) if span.parent else None,
                "status": span.status.status_code,
                "start": span.start_time,
                "end": span.end_time,
            }
            by_id[span.context.span_id] = record
            out.append(record)
        return out

    def collect_points() -> None:
        recorder.points.clear()
        data = reader.get_metrics_data()
        for resource in getattr(data, "resource_metrics", []) or []:
            for scope in resource.scope_metrics:
                for metric in scope.metrics:
                    recorder.instruments.append({"name": metric.name, "kind": type(metric.data).__name__.lower(), "unit": metric.unit})
                    for point in metric.data.data_points:
                        value = getattr(point, "sum", None)
                        if value is None:
                            value = getattr(point, "value", None)
                        recorder.points.append({"instrument": metric.name, "value": value, "attributes": dict(point.attributes or {})})

    recorder.ordered_spans = ordered_spans  # type: ignore[method-assign]
    recorder.collect_points = collect_points  # type: ignore[attr-defined]
    return recorder


def install_recorder() -> _Recorder:
    recorder = _sdk_recorder()
    if recorder is None:
        recorder = _Recorder()
        providers = _ApiProviders(recorder)
        recorder.tracer_provider, recorder.meter_provider = providers, providers
    set_otel_api_for_tests(otel_trace, otel_metrics, recorder.tracer_provider, recorder.meter_provider)
    return recorder


def points_of(recorder: _Recorder, instrument: str) -> List[Dict[str, Any]]:
    collect = getattr(recorder, "collect_points", None)
    if collect is not None:
        collect()
        # The SDK sums a histogram's points per attribute set; the API
        # recorder keeps each record. Both answer the questions below.
    return [p for p in recorder.points if p["instrument"] == instrument]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(OTEL_ENV_VAR, raising=False)
    yield
    set_otel_api_for_tests()


# ---------------------------------------------------------------------------
# The scenario's stub model and tool
# ---------------------------------------------------------------------------


class StubModel(Skill):
    """A model skill: two calls, the first asking for the tool, with the fixture's tokens."""

    provider_id = S["provider"]

    def __init__(self, streaming: bool) -> None:
        super().__init__({})
        self.model = S["model"]
        self.calls = 0
        self.streaming = streaming

    async def initialize(self, agent):
        self.agent = agent
        function = self.chat_completion_stream if self.streaming else self.chat_completion
        agent.register_handoff(
            Handoff(target="stub_model", description="stub", scope="all", metadata={"function": function, "priority": 10}),
            source="stub",
        )

    def _usage(self) -> Dict[str, int]:
        call = S["calls"][min(self.calls, len(S["calls"])) - 1]
        return {"prompt_tokens": call["input_tokens"], "completion_tokens": call["output_tokens"], "total_tokens": call["input_tokens"] + call["output_tokens"]}

    async def chat_completion(self, messages, tools=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            message = {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": S["tool"], "arguments": json.dumps(S["tool_args"])}}]}
        else:
            message = {"role": "assistant", "content": "done"}
        return {"id": "c", "created": 1, "model": S["model"], "object": "chat.completion", "choices": [{"index": 0, "message": message, "finish_reason": "stop"}], "usage": self._usage()}

    async def chat_completion_stream(self, messages, tools=None, **kwargs):
        self.calls += 1
        base = {"id": "c", "created": 1, "model": S["model"], "object": "chat.completion.chunk"}
        if self.calls == 1:
            yield {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [{"index": 0, "id": "call-1", "type": "function", "function": {"name": S["tool"], "arguments": json.dumps(S["tool_args"])}}]}, "finish_reason": None}]}
            yield {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
        else:
            yield {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "done"}, "finish_reason": None}]}
            yield {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield {**base, "choices": [], "usage": self._usage()}


class StubTools(Skill):
    def __init__(self) -> None:
        super().__init__({})

    async def initialize(self, agent):
        self.agent = agent
        self.register_tool(self.echo_tool)

    @tool(name=S["tool"], description="Echoes")
    async def echo_tool(self, text: str) -> str:
        return S["tool_result"]


def run_scenario(observability: Optional[Dict[str, Any]], streaming: bool = False) -> _Recorder:
    recorder = install_recorder()
    agent = BaseAgent(name=S["agent"], instructions="", skills={"model": StubModel(streaming), "tools": StubTools()})
    agent.observability = observability
    messages = [{"role": "user", "content": S["message"]}]

    async def go():
        if streaming:
            async for _chunk in agent.run_streaming(messages):
                pass
        else:
            await agent.run(messages)

    asyncio.run(go())
    return recorder


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------


class TestTheSwitch:
    def test_the_block_its_short_form_and_the_refusals(self):
        assert parse_observability({"otel": True}) == {"otel": True}
        assert parse_observability({"otel": False}) == {"otel": False}
        assert parse_observability(True) == {"otel": True}
        assert parse_observability(None) == {}
        assert parse_observability({}) == {}
        with pytest.raises(ObservabilityConfigError, match=None) as refused:
            parse_observability({"otle": True})
        assert str(refused.value) == FIXTURE["config"]["invalid"].replace("{key}", "otle")
        with pytest.raises(ObservabilityConfigError) as not_bool:
            parse_observability({"otel": "yes"})
        assert str(not_bool.value) == FIXTURE["config"]["not_bool"]
        assert FIXTURE["config"]["keys"] == ["otel"]

    def test_the_environment_switches_it_on_and_the_file_wins(self):
        assert FIXTURE["config"]["env_var"] == OTEL_ENV_VAR
        for value in FIXTURE["config"]["on_values"]:
            assert otel_enabled(None, {OTEL_ENV_VAR: value.upper()}) is True
        assert otel_enabled(None, {OTEL_ENV_VAR: "0"}) is False
        assert otel_enabled(None, {}) is False
        assert otel_enabled({"otel": False}, {OTEL_ENV_VAR: "1"}) is False
        assert otel_enabled({"otel": True}, {}) is True


# ---------------------------------------------------------------------------
# The spans of one run
# ---------------------------------------------------------------------------


def _assert_scenario_spans(recorder: _Recorder) -> None:
    ordered = recorder.ordered_spans()
    assert [s["name"] for s in ordered] == [s["name"] for s in S["expected_spans"]]
    for got, expected in zip(ordered, S["expected_spans"]):
        for key, value in expected["attributes"].items():
            assert got["attributes"].get(key) == value, f"{expected['name']} {key}"
        assert got["end"] is not None, f"{expected['name']} ended"
    run = ordered[0]
    assert run["parent"] is None
    for child in ordered[1:]:
        assert child["parent"] is run
    sentinels = [S["message"], S["tool_args"]["text"], S["tool_result"]]
    for span in ordered:
        for value in span["attributes"].values():
            for sentinel in sentinels:
                assert sentinel not in str(value)


class TestTheSpans:
    def test_the_run_its_model_calls_and_its_tool_call_with_nothing_from_the_messages(self):
        _assert_scenario_spans(run_scenario({"otel": True}))

    def test_the_streaming_run_records_the_same_spans(self):
        _assert_scenario_spans(run_scenario({"otel": True}, streaming=True))

    def test_the_token_usage_and_duration_metrics(self):
        recorder = run_scenario({"otel": True})
        token_points = points_of(recorder, "gen_ai.client.token.usage")
        got = sorted(((p["attributes"]["gen_ai.token.type"], p["value"]) for p in token_points))
        expected = sorted(((p["gen_ai.token.type"], p["value"]) for p in S["expected_token_points"]))
        # The SDK aggregates per attribute set (one point per token type, summed); the API recorder keeps each record.
        if len(got) == len(expected):
            assert got == expected
        else:
            sums: Dict[str, int] = {}
            for kind, value in expected:
                sums[kind] = sums.get(kind, 0) + value
            assert got == sorted(sums.items())
        for p in token_points:
            assert p["attributes"]["gen_ai.operation.name"] == "chat"
            assert p["attributes"]["gen_ai.provider.name"] == S["provider"]
            assert p["attributes"]["gen_ai.request.model"] == S["model"]
        durations = points_of(recorder, "gen_ai.client.operation.duration")
        assert {p["attributes"]["gen_ai.operation.name"] for p in durations} == {"chat", "execute_tool"}
        made = {i["name"]: i for i in recorder.instruments}
        for name, spec in FIXTURE["metrics"].items():
            assert name in made, name
            assert made[name]["unit"] == spec["unit"]

    def test_a_payment_settle_under_the_run_in_credits(self):
        recorder = install_recorder()
        run = start_agent_run({"otel": True}, S["agent"])
        assert run.active is True
        run.payment_settle(S["payment"]["credits"], S["payment"]["lock_id"])
        run.end()
        settle = next(s for s in recorder.ordered_spans() if s["name"] == FIXTURE["spans"]["payment_settle"]["name"])
        assert settle["attributes"]["gen_ai.agent.name"] == S["agent"]
        assert settle["attributes"]["webagents.payment.credits"] == S["payment"]["credits"]
        assert settle["attributes"]["webagents.payment.lock_id"] == S["payment"]["lock_id"]
        assert settle["parent"]["name"] == f"invoke_agent {S['agent']}"
        counted = points_of(recorder, "webagents.payment.credits")
        assert [(p["value"], p["attributes"]) for p in counted] == [(S["payment"]["expected_counter"], {"gen_ai.agent.name": S["agent"]})]

    def test_nothing_when_off_and_the_file_wins_over_the_environment(self, monkeypatch):
        monkeypatch.setenv(OTEL_ENV_VAR, "1")
        assert run_scenario({"otel": False}).ordered_spans() == []
        assert len(run_scenario(None).ordered_spans()) > 0
        monkeypatch.delenv(OTEL_ENV_VAR)
        assert run_scenario(None).ordered_spans() == []

    def test_a_failed_run_ends_its_span_with_the_error(self):
        recorder = install_recorder()

        class Broken(Skill):
            def __init__(self) -> None:
                super().__init__({})
                self.model = S["model"]

            async def initialize(self, agent):
                self.agent = agent
                agent.register_handoff(Handoff(target="broken", metadata={"function": self.chat_completion, "priority": 10}), source="stub")

            async def chat_completion(self, messages, tools=None, **kwargs):
                raise RuntimeError("boom")

        agent = BaseAgent(name=S["agent"], instructions="", skills={"model": Broken()})
        agent.observability = {"otel": True}
        with pytest.raises(RuntimeError):
            asyncio.run(agent.run([{"role": "user", "content": S["message"]}]))
        ordered = recorder.ordered_spans()
        assert [s["name"] for s in ordered] == [f"invoke_agent {S['agent']}", f"chat {S['model']}"]
        assert ordered[0]["attributes"]["error.type"] == "RuntimeError"
        assert ordered[1]["attributes"]["error.type"] == "RuntimeError"

    def test_a_no_op_when_the_package_cannot_be_imported(self, monkeypatch):
        import sys

        # A None entry in sys.modules is what an absent package looks like to importlib.
        set_otel_api_for_tests()
        monkeypatch.setitem(sys.modules, "opentelemetry.trace", None)
        monkeypatch.setitem(sys.modules, "opentelemetry.metrics", None)
        assert load_otel_api() is None
        run = start_agent_run({"otel": True}, S["agent"])
        assert run.active is False
        run.model_call_started("x", "y")
        run.model_call_ended()
        run.end()

    def test_the_real_api_loads(self):
        # The API package is here (the module skipped otherwise): the real load path answers.
        set_otel_api_for_tests()
        assert load_otel_api() is otel_trace
