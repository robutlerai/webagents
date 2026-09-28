"""
`serve` answers a refused non-streaming call with its status (2026-09-27, the
chat-fixes lane). The platform's LLM proxy refused a call (not enough credits,
a sign-in it did not accept) and the served agent answered HTTP 500 with a
traceback in plain text; `LLMProxyError` now carries the status and a JSON
body, and the completions route answers with them, as it already did for an
auth refusal (S-236).
"""

from fastapi.testclient import TestClient

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.core.llm.proxy.skill import LLMProxyError, PaymentRequiredError
from webagents.server.core.app import WebAgentsServer


def _server(error: Exception) -> WebAgentsServer:
    agent = BaseAgent(name="refused", instructions="Test agent", scopes=["all"])

    async def fake_run(messages, stream=False, tools=None):
        raise error

    async def fake_run_streaming(messages, tools=None, **kwargs):
        raise error
        yield  # pragma: no cover - makes this a generator

    agent.run = fake_run
    agent.run_streaming = fake_run_streaming
    # The route under test is the server's own `/{agent}/chat/completions`.
    return WebAgentsServer(agents=[agent])


def _post(server: WebAgentsServer, stream: bool):
    client = TestClient(server.app, raise_server_exceptions=False)
    return client.post(
        "/refused/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": stream},
        headers={"Authorization": "Bearer local-test-not-a-credential"},
    )


def test_the_error_knows_its_status_and_body():
    refused = LLMProxyError("payment_required", "Not enough credits to start a model call.")
    assert refused.status_code == 402
    assert refused.to_dict() == {"error": {"code": "payment_required", "message": "Not enough credits to start a model call."}}
    assert LLMProxyError("unauthorized", "Invalid or expired payment token").status_code == 401
    assert LLMProxyError("rate_limit", "Slow down").status_code == 429
    assert LLMProxyError("timeout", "Response timeout from LLM proxy").status_code == 504
    assert LLMProxyError("provider_error", "The model is overloaded.").status_code == 502
    assert PaymentRequiredError({"amount": "0.5", "currency": "credits"}).status_code == 402


def test_a_refused_non_streaming_call_is_a_402_with_a_json_body():
    response = _post(_server(LLMProxyError("payment_required", "Not enough credits to start a model call.")), stream=False)
    assert response.status_code == 402, response.text
    assert response.json() == {"error": {"code": "payment_required", "message": "Not enough credits to start a model call."}}
    assert "Traceback" not in response.text


def test_a_refused_sign_in_is_a_401_and_a_streamed_refusal_is_the_same_status():
    unauthorized = LLMProxyError("unauthorized", "Invalid or expired payment token")
    assert _post(_server(unauthorized), stream=False).status_code == 401
    assert _post(_server(unauthorized), stream=True).status_code == 401
