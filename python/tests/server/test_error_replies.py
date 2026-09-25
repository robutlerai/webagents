"""A failed run does not hand its exception text to the caller (S-228, 2026-09-24).

Every door a request can take was answering a failure with the exception's own
text: the skill `@http` dispatcher (`Response(500, str(e))`, the route the
daemon's and a deployed agent's `/chat/completions` go through), the
dedicated route's stream (`{"error": str(e)}`), a static agent's mounted
handlers (`HTTPException(500, str(e))`). Measured on a daemon whose model was
unreachable: `500 Connection error.`. The text is whatever the failing code put
in it, a provider's body quoting a masked key among them.

What is pinned, per door: the reply carries a fixed message and a reference,
never the text; the same reference and the text are in the server's log;
status codes and body shapes are unchanged. And what deliberately keeps its
text: this SDK's own payment and auth errors, and the local daemon on
loopback, whose answers go to the developer's own terminal.
"""

import json
import logging
import re

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.base import Skill
from webagents.agents.skills.core.transport import CompletionsTransportSkill
from webagents.agents.tools.decorators import http
from webagents.server.core.app import WebAgentsServer
from webagents.server.core.error_reply import INTERNAL_ERROR_MESSAGE, is_meant_to_be_shown

#: What a failing provider or tool might say: a masked key and a path on the host.
SECRET = "Incorrect API key provided: sk-proj-****abcd; read /home/owner/agent/secrets.txt"
REFERENCE = re.compile(re.escape(INTERNAL_ERROR_MESSAGE) + r" Reference: ([0-9a-f]{8})")
CREDENTIAL = {"Authorization": "Bearer any-presented-credential"}


class ProviderAuthError(Exception):
    """Shaped like `openai.AuthenticationError`: an HTTP status, and a body in its text."""

    status_code = 401


class AuthenticationError(Exception):
    """Our name, another library's class: its text must not travel."""


class Leaky(Skill):
    """`@http` handlers that fail the ways a run fails."""

    def __init__(self):
        super().__init__({}, scope="all")
        self.raise_this: BaseException = RuntimeError(SECRET)

    @http("/boom", method="post")
    async def boom(self):
        raise self.raise_this

    @http("/boom-stream", method="post")
    async def boom_stream(self):
        raise self.raise_this
        yield  # an async generator: the preflight path


@pytest.fixture
def log_records():
    """Records reaching the `webagents` logger (which does not propagate to pytest's)."""
    from webagents.utils.logging import logging_explicitly_configured, setup_logging

    # The first server built in a process configures logging, which resets the
    # `webagents` logger's handlers; do it first, as that server would.
    if not logging_explicitly_configured():
        setup_logging(level="INFO")
    records = []

    class _Keep(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Keep(level=logging.ERROR)
    logger = logging.getLogger("webagents")
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


def _dynamic_server(skill, **server_kwargs):
    agent = BaseAgent(name="leaky", instructions="x", skills={"leaky": skill})
    return WebAgentsServer(dynamic_agents=lambda name: agent if name == "leaky" else None, **server_kwargs), agent


def _reference_in(text: str) -> str:
    match = REFERENCE.search(text)
    assert match, f"no fixed message with a reference in {text!r}"
    assert SECRET not in text and "sk-proj" not in text and "/home/owner" not in text
    return match.group(1)


def _logged(records, reference: str) -> str:
    logged = "\n".join(record.getMessage() for record in records)
    assert reference in logged and SECRET in logged, logged
    return logged


class TestTheSkillRouteDaemonAndDeployedAgentsTake:
    def test_a_failed_handler_answers_a_reference_and_logs_the_text(self, log_records):
        server, _ = _dynamic_server(Leaky())
        response = TestClient(server.app).post("/leaky/boom", json={})
        assert response.status_code == 500
        reference = _reference_in(response.text)
        _logged(log_records, reference)

    def test_the_completions_route_with_a_model_that_fails(self, log_records, monkeypatch):
        # The measured case: `webagents connect`'s route, a model call that fails
        # before the first chunk.
        transport = CompletionsTransportSkill()

        async def failing_handoff(*args, **kwargs):
            raise RuntimeError(SECRET)
            yield

        monkeypatch.setattr(transport, "execute_handoff", failing_handoff)
        agent = BaseAgent(name="leaky", instructions="x", skills={"completions": transport})
        server = WebAgentsServer(dynamic_agents=lambda name: agent)
        body = {"messages": [{"role": "user", "content": "hi"}], "stream": True}
        response = TestClient(server.app).post("/leaky/chat/completions", json=body, headers=CREDENTIAL)
        assert response.status_code == 500
        _logged(log_records, _reference_in(response.text))

    def test_a_providers_status_error_keeps_its_status_not_its_text(self, log_records):
        skill = Leaky()
        skill.raise_this = ProviderAuthError(SECRET)
        server, _ = _dynamic_server(skill)
        response = TestClient(server.app).post("/leaky/boom-stream", json={})
        assert response.status_code == 401
        _logged(log_records, _reference_in(response.json()["error"]))

    def test_this_sdks_payment_error_is_answered_as_before(self):
        from webagents.agents.skills.robutler.payments.exceptions import PaymentTokenRequiredError

        skill = Leaky()
        skill.raise_this = PaymentTokenRequiredError(agent_name="leaky")
        server, _ = _dynamic_server(skill)
        response = TestClient(server.app).post("/leaky/boom-stream", json={})
        assert response.status_code == 402
        assert response.json()["user_message"] == skill.raise_this.user_message

    def test_this_sdks_auth_refusal_keeps_its_message(self):
        from webagents.agents.skills.robutler.auth.skill import AuthenticationError as OurAuthError

        skill = Leaky()
        skill.raise_this = OurAuthError("Invalid API key")
        server, _ = _dynamic_server(skill)
        response = TestClient(server.app).post("/leaky/boom", json={})
        assert (response.status_code, response.text) == (500, "Invalid API key")

    def test_a_look_alike_from_another_library_is_not_trusted(self, log_records):
        skill = Leaky()
        skill.raise_this = AuthenticationError(SECRET)
        server, _ = _dynamic_server(skill)
        response = TestClient(server.app).post("/leaky/boom", json={})
        _logged(log_records, _reference_in(response.text))

    def test_the_local_daemon_keeps_the_text_for_its_own_terminal(self, log_records):
        # `error_detail=True` is what `webagents daemon start` sets on loopback:
        # the chat shows "Connection error." and a hint, not a reference.
        server, _ = _dynamic_server(Leaky(), error_detail=True)
        response = TestClient(server.app).post("/leaky/boom", json={})
        assert (response.status_code, response.text) == (500, SECRET)
        assert any(SECRET in record.getMessage() for record in log_records)


class TestTheStaticAgentDoors:
    def test_the_dedicated_routes_stream_carries_a_reference(self, log_records):
        agent = BaseAgent(name="static", instructions="x", scopes=["all"])

        async def run_streaming(messages, tools=None, **kwargs):
            yield {"choices": [{"index": 0, "delta": {"content": "par"}}]}
            raise RuntimeError(SECRET)

        agent.run_streaming = run_streaming
        server = WebAgentsServer(agents=[agent])
        body = {"messages": [{"role": "user", "content": "hi"}], "stream": True}
        response = TestClient(server.app).post("/static/chat/completions", json=body, headers=CREDENTIAL)
        assert response.status_code == 200  # the stream had started; the error is an event in it
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
        error = next(event["error"] for event in events if "error" in event)
        _logged(log_records, _reference_in(error))

    def test_a_mounted_handler_answers_a_reference(self, log_records):
        agent = BaseAgent(name="mounted", instructions="x", scopes=["all"])

        async def boom():
            raise RuntimeError(SECRET)

        agent._registered_http_handlers = [
            {"subpath": "/boom", "method": "post", "function": boom, "scope": "all", "description": ""}
        ]
        server = WebAgentsServer(agents=[agent])
        response = TestClient(server.app).post("/mounted/boom", json={})
        assert response.status_code == 500
        _logged(log_records, _reference_in(response.json()["detail"]))


def test_what_counts_as_meant_to_be_shown():
    from fastapi import HTTPException

    from webagents.agents.skills.local.auth.skill import AuthError
    from webagents.agents.skills.robutler.payments.exceptions import InsufficientBalanceError

    assert is_meant_to_be_shown(HTTPException(403, "Access denied"))
    assert is_meant_to_be_shown(AuthError("expired"))
    assert is_meant_to_be_shown(InsufficientBalanceError(current_balance=1.0, required_balance=2.0))
    assert not is_meant_to_be_shown(RuntimeError(SECRET))
    assert not is_meant_to_be_shown(ProviderAuthError(SECRET))
    assert not is_meant_to_be_shown(AuthenticationError(SECRET))


class TestTheDaemonsDecision:
    """Detail on loopback only: a daemon bound elsewhere has remote callers."""

    @pytest.fixture
    def started(self, tmp_path, monkeypatch):
        import uvicorn

        import webagents.server.core.app as app_module
        from webagents.cli.main import app as cli

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        # A fake HOME with the OS keystore backend makes macOS offer to reset
        # the login keychain; `daemon` loads the stored provider keys.
        monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
        monkeypatch.delenv("WEBAGENTS_PROFILE", raising=False)
        monkeypatch.chdir(tmp_path)
        seen = {}

        class _App:
            def add_event_handler(self, event, handler):
                pass

        class _Server:
            app = _App()

        def fake_create_server(**kwargs):
            seen.update(kwargs)
            return _Server()

        monkeypatch.setattr(app_module, "create_server", fake_create_server)
        monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)

        def run(*args):
            result = CliRunner().invoke(cli, ["daemon", "--no-cron", *args])
            assert result.exit_code == 0, result.output
            return seen

        return run

    def test_on_loopback_the_daemon_answers_with_the_text(self, started):
        assert started()["error_detail"] is True

    def test_bound_elsewhere_it_answers_like_any_server(self, started):
        assert started("--host", "0.0.0.0")["error_detail"] is False

    def test_the_older_daemon_class_follows_the_same_rule(self):
        from webagents.cli.daemon.server import WebAgentsDaemon

        local = WebAgentsDaemon(port=0, host="127.0.0.1")
        exposed = WebAgentsDaemon(port=0, host="0.0.0.0")
        assert local._error_text(RuntimeError(SECRET), "a") == SECRET
        _reference_in(exposed._error_text(RuntimeError(SECRET), "a"))


def test_no_door_is_left_answering_with_str_e():
    """The shapes the leak had, gone from the serving code: a new door written
    the old way fails here rather than in a log review."""
    import inspect

    import webagents.server.core.app as app_module

    source = inspect.getsource(app_module)
    for shape in ("content=str(last_error)", "{'error': str(e)}", "detail=str(e))", "reason=str(e)"):
        assert shape not in source, shape


class TestAnAgentThatCannotBeBuilt:
    """Building the agent raised (a model client not installed), before any
    route's own handling: the chat said "Internal Server Error" and nothing else."""

    @staticmethod
    def _server(**kwargs):
        def resolver(name):
            raise ImportError("Google GenAI SDK not available. Install with: pip install google-genai")

        # A resolver mounts the dynamic routes; `resolve_agent` itself is what fails.
        return WebAgentsServer(dynamic_agents=lambda name: None, **kwargs), resolver

    def test_the_local_daemon_says_why(self, monkeypatch):
        server, resolver = self._server(error_detail=True)

        async def failing(name, working_dir=None):
            resolver(name)

        monkeypatch.setattr(server, "resolve_agent", failing)
        # A credential: the command routes need one since S-235.
        response = TestClient(server.app).get("/robutler/command", headers={"Authorization": "Bearer local"})
        assert response.status_code == 500
        assert response.json()["detail"] == (
            "Could not load agent 'robutler': Google GenAI SDK not available. Install with: pip install google-genai"
        )

    def test_anyone_else_gets_a_reference(self, monkeypatch, log_records):
        server, resolver = self._server()

        async def failing(name, working_dir=None):
            resolver(name)

        monkeypatch.setattr(server, "resolve_agent", failing)
        # A credential: the command routes need one since S-235.
        response = TestClient(server.app).get("/robutler/command", headers={"Authorization": "Bearer local"})
        assert response.status_code == 500
        reference = REFERENCE.search(response.json()["detail"]).group(1)
        assert any(reference in record.getMessage() and "google-genai" in record.getMessage() for record in log_records)
