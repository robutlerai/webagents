"""
What `serve` answers when a run fails (B2 and B1's parity, 2026-09-28).

B2: a provider's refusal in a non-streaming call (the agent's own key out of
credit, a 402 from OpenAI) escaped the route as a text/plain 500 and the
traceback was printed over the terminal. It is JSON now, as the TypeScript
server answers: `{"error": {"code": "completions_error", "message": <the fixed
sentence and a reference>}}`, logged with the whole error (S-228).

B1's parity: a stream that fails after it began ends with OpenAI's in-stream
error `{"error": {"message", "type", "code"}}` in both SDKs (fixture
`cli/final_sdk_serve_model.json`, `stream_error`), and a refusal written for
the caller (the proxy skill's "send a payment token", S-327) is its status
before any stream.
"""

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from webagents.agents.core.base_agent import BaseAgent
from webagents.server.core.app import WebAgentsServer

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "final_sdk_serve_model.json").read_text()
)
CREDENTIAL = {"Authorization": "Bearer any-presented-credential"}
SECRET = "Error code: 402 - {'error': {'message': 'provider body with sk-proj-****abcd'}}"


class ProviderRefusal(Exception):
    """Stands in for `openai.APIStatusError`: another library's error, carrying a status."""

    status_code = 402


@pytest.fixture
def log_records():
    """Records reaching the `webagents` logger (which does not propagate to
    pytest's), configured first as the first server would (the fixture of
    `test_error_replies.py`)."""
    from webagents.utils.logging import logging_explicitly_configured, setup_logging

    if not logging_explicitly_configured():
        setup_logging(level="INFO")
    records = []

    class Keep(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Keep(level=logging.ERROR)
    logger = logging.getLogger("webagents")
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


def _served(run=None, run_streaming=None) -> TestClient:
    agent = BaseAgent(name="served", instructions="x", scopes=["all"])
    if run is not None:
        agent.run = run
    if run_streaming is not None:
        agent.run_streaming = run_streaming
    return TestClient(WebAgentsServer(agents=[agent]).app)


def _post(client, stream):
    body = {"messages": [{"role": "user", "content": "hi"}], "stream": stream}
    return client.post("/served/chat/completions", json=body, headers=CREDENTIAL)


def test_a_provider_refusal_is_json_not_a_text_500(log_records):
    async def run(messages, tools=None, **kwargs):
        raise ProviderRefusal(SECRET)

    response = _post(_served(run=run), stream=False)
    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    error = response.json()["error"]
    assert error["code"] == "completions_error"
    assert error["message"].startswith(FIXTURE["stream_error"]["message_starts"])
    assert "provider body" not in response.text
    reference = error["message"][len(FIXTURE["stream_error"]["message_starts"]):]
    assert any(reference in record.getMessage() for record in log_records)


def test_a_stream_that_fails_after_it_began_ends_with_openais_error_shape(log_records):
    async def run_streaming(messages, tools=None, **kwargs):
        yield {"choices": [{"index": 0, "delta": {"content": "par"}}]}
        raise ProviderRefusal(SECRET)

    response = _post(_served(run_streaming=run_streaming), stream=True)
    assert response.status_code == 200
    events = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    assert events[-1] != "[DONE]"
    error = json.loads(events[-1])["error"]
    assert (error["type"], error["code"]) == (FIXTURE["stream_error"]["type"], FIXTURE["stream_error"]["code"])
    assert error["message"].startswith(FIXTURE["stream_error"]["message_starts"])
    assert "provider body" not in response.text


@pytest.mark.parametrize("stream", [False, True])
def test_a_callers_missing_payment_token_is_its_status_before_any_stream(stream):
    from webagents.agents.skills.core.llm.proxy.skill import CALLERS_PAY_REFUSAL, LLMProxyError

    def refused():
        return LLMProxyError("payment_required", CALLERS_PAY_REFUSAL)

    async def run(messages, tools=None, **kwargs):
        raise refused()

    async def run_streaming(messages, tools=None, **kwargs):
        raise refused()
        yield  # pragma: no cover - a generator that fails on its first step

    response = _post(_served(run=run, run_streaming=run_streaming), stream=stream)
    expected = FIXTURE["no_payment_token"]
    assert response.status_code == expected["status"]
    assert response.json() == {"error": {"code": expected["code"], "message": expected["message"]}}
