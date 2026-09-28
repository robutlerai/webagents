"""
Local models and model failover (2026-09-26, gap-closure plan items 2.8 and
2.4, lane w2-ops), against the shared fixture `tests/fixtures/w2ops/models.json`,
which the TypeScript suite reads too (`ollama-failover-w2ops.test.ts`):

* `base_url` in a model skill's entry reaches the skill;
* Ollama as a named provider: the registry row, the placeholder key, the
  model-access decision, doctor's words and the `models` row;
* `fallback_models:`: the chain moves on a provider error, says so in the
  transcript, and stops on anything else.

Everything runs against a stub OpenAI-compatible server on loopback: never a
real Ollama, never a real key.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.core.llm.failover import FailoverLLMSkill, failover_note, provider_failure
from webagents.agents.skills.core.llm.ollama import OllamaSkill, ollama_model_check, probe_ollama, serves_model
from webagents.agents.skills.core.llm.ollama.skill import OLLAMA_PLACEHOLDER_KEY
from webagents.agents.skills.core.llm.openai.skill import OpenAISkill
from webagents.agents.skills.core.llm.providers import MODELS_READY_FOOTNOTE, find_provider, provider_base_url, provider_needs
from webagents.cli.agent_builder import apply_fallback_models
from webagents.cli.loader import AgentFormatError
from webagents.cli.loader.schema import AgentMetadata
from webagents.cli.model_access import resolve_model_access

FIXTURE = json.loads((Path(__file__).resolve().parent / "fixtures" / "w2ops" / "models.json").read_text())
SERVED = ["llama3.2:latest", "gemma3:latest"]
NO = lambda: False  # noqa: E731
YES = lambda: True  # noqa: E731


# ---------------------------------------------------------------------------
# The stub: an OpenAI-compatible server on loopback
# ---------------------------------------------------------------------------


class _Stub:
    def __init__(self) -> None:
        self.seen: List[Dict[str, Any]] = []
        #: Per model: the status the stub answers; 200 streams a reply.
        self.statuses: Dict[str, int] = {}
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # noqa: D401 - quiet
                pass

            def do_GET(self) -> None:  # noqa: N802
                if self.path.endswith("/models"):
                    body = json.dumps({"data": [{"id": m} for m in SERVED]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(404)
                self.end_headers()

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                parsed = json.loads(self.rfile.read(length) or b"{}")
                model = parsed.get("model", "")
                stub.seen.append({"path": self.path, "model": model, "authorization": self.headers.get("Authorization")})
                status = stub.statuses.get(model, 200)
                if status != 200:
                    body = json.dumps({"error": {"message": f"stub says {status}"}}).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunks = [
                    {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": model, "choices": [{"index": 0, "delta": {"role": "assistant", "content": f"hello from {model}"}, "finish_reason": None}]},
                    {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                    {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": model, "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
                ]
                for chunk in chunks:
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture(scope="module")
def stub():
    s = _Stub()
    yield s
    s.close()


@pytest.fixture(autouse=True)
def _clean(stub, monkeypatch):
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OLLAMA_BASE_URL", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    stub.seen.clear()
    stub.statuses = {}
    yield


def _no_retries(skill: OpenAISkill) -> OpenAISkill:
    """The client without its own retries, so a 503 moves the chain on at once."""
    from openai import AsyncOpenAI

    skill._client = AsyncOpenAI(api_key=skill.api_key, base_url=skill.base_url, max_retries=0)
    return skill


def turn(skills: Dict[str, Any], owner: bool = False) -> Dict[str, Any]:
    """One streamed turn through an agent built on `skills`: its text, its
    notes and its error. `owner` runs it as the chat does, as the local owner."""
    agent = BaseAgent(name="w2ops", instructions="", skills=skills)
    text: List[str] = []
    notes: List[str] = []
    error: Optional[str] = None

    async def go() -> None:
        nonlocal error
        if owner:
            from webagents.access import run_as_local_owner

            run_as_local_owner(agent)
        try:
            async for chunk in agent.run_streaming([{"role": "user", "content": "hi"}]):
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("webagents_note"):
                    notes.append(chunk["webagents_note"])
                for choice in chunk.get("choices") or []:
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        text.append(content)
        except Exception as exc:  # noqa: BLE001 - the case says whether one is expected
            error = str(exc)

    asyncio.run(go())
    return {"text": "".join(text), "notes": notes, "error": error}


# ---------------------------------------------------------------------------
# base_url (item 5)
# ---------------------------------------------------------------------------


class TestBaseUrl:
    def test_reaches_the_openai_skill_and_the_request_goes_there(self, stub):
        skill = _no_retries(OpenAISkill({FIXTURE["base_url"]["file_key"]: stub.base, "api_key": "k-file", "model": "primary-model"}))
        for key in FIXTURE["base_url"]["skill_config_keys"]["python"]:
            assert getattr(skill, key) == stub.base
        result = turn({"llm": skill})
        assert result == {"text": "hello from primary-model", "notes": [], "error": None}
        assert [(s["path"], s["model"], s["authorization"]) for s in stub.seen] == [("/v1/chat/completions", "primary-model", "Bearer k-file")]

    def test_the_environment_variable_stands_in_when_the_entry_names_none(self, stub, monkeypatch):
        monkeypatch.setenv(FIXTURE["base_url"]["env_var"], stub.base)
        monkeypatch.setenv("OPENAI_API_KEY", "k-env")
        skill = _no_retries(OpenAISkill({"model": "primary-model"}))
        assert turn({"llm": skill})["text"] == "hello from primary-model"
        assert stub.seen[0]["authorization"] == "Bearer k-env"


# ---------------------------------------------------------------------------
# Ollama (item 3)
# ---------------------------------------------------------------------------


class TestOllama:
    def test_is_the_fixture_row_of_the_registry(self, stub):
        p = find_provider("ollama")
        row = FIXTURE["ollama"]["provider"]
        assert {
            "id": p.id, "aliases": list(p.aliases), "description": p.description, "credential": p.credential,
            "model_format": p.model_format, "default_model": p.default_model, "base_url_var": p.base_url_var, "default_base_url": p.default_base_url,
        } == row
        assert provider_base_url(p, {}) == row["default_base_url"]
        assert provider_base_url(p, {"OLLAMA_BASE_URL": stub.base}) == stub.base
        assert OLLAMA_PLACEHOLDER_KEY == FIXTURE["ollama"]["placeholder_key"]

    def test_sends_the_openai_wire_shape_to_ollama_base_url_with_the_placeholder_key(self, stub, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", stub.base)
        skill = _no_retries(OllamaSkill({}))
        assert (skill.model, skill.base_url, skill.provider_id) == ("llama3.2", stub.base, "ollama")
        assert turn({"llm": skill})["text"] == "hello from llama3.2"
        assert [(s["model"], s["authorization"]) for s in stub.seen] == [("llama3.2", f"Bearer {FIXTURE['ollama']['placeholder_key']}")]
        # `ollama/<model>` as the agent's model builds the same skill.
        agent = BaseAgent(name="w2ops", instructions="", model="ollama/gemma3")
        assert isinstance(agent.skills["primary_llm"], OllamaSkill)
        assert agent.skills["primary_llm"].model == "gemma3"

    def test_is_reached_directly_when_named_never_chosen_when_the_file_names_no_model(self):
        named = resolve_model_access("ollama/llama3.2", signed_in=NO, env={})
        assert (named.kind, named.model) == ("direct", "ollama/llama3.2")
        assert named.local_route(env={}) == FIXTURE["ollama"]["status_route"].replace("{model}", "ollama/llama3.2").replace("{base_url}", "http://localhost:11434/v1")
        assert resolve_model_access("openai/x", signed_in=NO, env={"OPENAI_API_KEY": "k"}).local_route() is None
        assert resolve_model_access(None, signed_in=NO, env={}).kind == "none"
        signed = resolve_model_access(None, signed_in=YES, env={})
        assert (signed.kind, signed.model) == ("proxy", "auto/balanced")

    def test_is_listed_by_models_with_the_fixture_words(self):
        p = find_provider("ollama")
        assert p.id == FIXTURE["ollama"]["models_row"]["id"]
        assert p.model_format == FIXTURE["ollama"]["models_row"]["format"]
        assert provider_needs(p, "webagents login") == FIXTURE["ollama"]["models_row"]["needs"]
        assert MODELS_READY_FOOTNOTE.replace("{command}", "webagents secrets set") == FIXTURE["ollama"]["models_footnote"]

    @pytest.mark.parametrize("case", FIXTURE["ollama"]["model_match"], ids=lambda c: f"{c['wanted']}~{c['served']}")
    def test_model_match(self, case):
        assert serves_model([case["served"]], case["wanted"]) is case["match"]

    def test_doctors_model_check(self, stub):
        reachable = probe_ollama(stub.base)
        assert (reachable.ok, reachable.models) == (True, SERVED)
        words = FIXTURE["ollama"]["doctor"]

        def fill(s: str, model: str) -> str:
            return s.replace("{model}", model).replace("{base_url}", stub.base).replace("{bare}", model.split("/", 1)[1])

        assert ollama_model_check("ollama/llama3.2", stub.base, reachable) == {"status": "ok", "detail": fill(words["reachable"], "ollama/llama3.2"), "fix": None}
        assert ollama_model_check("ollama/mistral", stub.base, reachable) == {
            "status": "warn", "detail": fill(words["not_pulled"]["detail"], "ollama/mistral"), "fix": fill(words["not_pulled"]["fix"], "ollama/mistral"),
        }
        dead = "http://127.0.0.1:1/v1"
        unreachable = probe_ollama(dead, timeout=0.5)
        assert unreachable.ok is False
        assert ollama_model_check("ollama/llama3.2", dead, unreachable) == {
            "status": "fail",
            "detail": words["unreachable"]["detail"].replace("{model}", "ollama/llama3.2").replace("{base_url}", dead),
            "fix": words["unreachable"]["fix"].replace("{bare}", "llama3.2"),
        }


# ---------------------------------------------------------------------------
# Failover (item 4)
# ---------------------------------------------------------------------------


class TestFailover:
    def test_the_agent_file_key_a_list_of_provider_model_strings_and_nothing_else(self):
        key = FIXTURE["failover"]["file_key"]
        assert AgentMetadata(**{"name": "t", key: ["openai/b", "anthropic/c"]}).fallback_models == ["openai/b", "anthropic/c"]
        with pytest.raises(AgentFormatError) as refused:
            AgentMetadata(**{"name": "t", key: "openai/b"})
        assert str(refused.value) == FIXTURE["failover"]["invalid"]
        with pytest.raises(AgentFormatError):
            AgentMetadata(**{"name": "t", key: [1]})
        # The sibling key of item 1, checked on the same pass.
        assert AgentMetadata(name="t", observability={"otel": True}).observability == {"otel": True}
        with pytest.raises(AgentFormatError) as bad:
            AgentMetadata(name="t", observability={"otle": True})
        assert str(bad.value) == "observability: unknown key 'otle'. It takes otel."

    def test_names_a_failure_the_way_the_fixture_does_and_moves_on_only_for_a_provider_error(self):
        import httpx
        import openai

        def status_error(status: int) -> openai.APIStatusError:
            request = httpx.Request("POST", "http://127.0.0.1:1/v1/chat/completions")
            return openai.APIStatusError("x", response=httpx.Response(status, request=request), body=None)

        for status in FIXTURE["failover"]["retryable"]["statuses"]:
            assert provider_failure(status_error(status)) == FIXTURE["failover"]["reason"]["status"].replace("{status}", str(status))
        for status in FIXTURE["failover"]["not_retryable"]:
            assert provider_failure(status_error(status)) is None
        connection = openai.APIConnectionError(request=httpx.Request("POST", "http://127.0.0.1:1/v1/chat/completions"))
        assert provider_failure(connection, detail=True) == FIXTURE["failover"]["reason"]["network"].replace("{origin}", "http://127.0.0.1:1")
        # A served agent's caller never reads the upstream origin (S-228).
        assert provider_failure(connection) == FIXTURE["failover"]["reason"]["network_served"]
        assert provider_failure(ValueError("OpenAI API key not configured")) is None
        assert failover_note("a", "HTTP 503", "b") == FIXTURE["failover"]["note"].replace("{failed}", "a").replace("{reason}", "HTTP 503").replace("{next}", "b")

    @pytest.mark.parametrize("case", FIXTURE["failover"]["cases"], ids=lambda c: c["case"])
    def test_cases(self, stub, case):
        stub.statuses = dict(case["server"])
        chain = [(label, _no_retries(OpenAISkill({"base_url": stub.base, "api_key": "k", "model": label.split("/", 1)[1]}))) for label in case["chain"]]
        failover = FailoverLLMSkill(chain)
        assert failover.models == case["chain"]
        result = turn({"llm": failover})
        assert [s["model"] for s in stub.seen] == case["requests"]
        assert result["notes"] == case["notes"]
        if case["answered_by"] is None:
            assert result["text"] == ""
            assert result["error"] is not None
            assert failover.answered_model is None
        else:
            assert result["text"] == f"hello from {case['answered_by'].split('/', 1)[1]}"
            assert result["error"] is None
            assert failover.answered_model == case["answered_by"]

    def test_a_primary_that_cannot_be_reached_moves_on_with_the_origin_in_the_note(self, stub):
        def chain() -> FailoverLLMSkill:
            primary = _no_retries(OpenAISkill({"base_url": "http://127.0.0.1:1/v1", "api_key": "k", "model": "primary-model"}))
            fallback = _no_retries(OpenAISkill({"base_url": stub.base, "api_key": "k", "model": "fallback-model"}))
            return FailoverLLMSkill([("openai/primary-model", primary), ("openai/fallback-model", fallback)])

        # The owner at the terminal reads where the primary was: the chat's run.
        result = turn({"llm": chain()}, owner=True)
        assert result["text"] == "hello from fallback-model"
        assert result["notes"] == [failover_note("openai/primary-model", "could not reach http://127.0.0.1:1", "openai/fallback-model")]
        # A served agent's caller does not (S-228).
        served = turn({"llm": chain()})
        assert served["text"] == "hello from fallback-model"
        assert served["notes"] == [failover_note("openai/primary-model", FIXTURE["failover"]["reason"]["network_served"], "openai/fallback-model")]

    def test_a_members_own_tools_are_the_agents_too(self, stub):
        from webagents.agents.skills.base import Skill
        from webagents.agents.tools.decorators import tool

        class Tooled(OpenAISkill):
            @tool(scope="all")
            async def member_tool(self, text: str) -> str:
                """A member's decorated tool"""
                return text

        member = _no_retries(Tooled({"base_url": stub.base, "api_key": "k", "model": "primary-model"}))
        agent = BaseAgent(name="w2ops", instructions="", skills={"llm": FailoverLLMSkill([("openai/primary-model", member)])})
        asyncio.run(agent._ensure_skills_initialized())
        assert "member_tool" in {t["name"] for t in agent.get_tools_for_scopes(frozenset({"all"}))}
        assert isinstance(member, Skill)

    def test_the_builder_wraps_the_primary_and_reports_what_it_cannot_build(self, stub, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", stub.base)
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.setattr("webagents.cli.model_access.is_signed_in", lambda: False)
        skills: Dict[str, Any] = {}
        model, failed = apply_fallback_models(
            skills, ["anthropic/claude-x", "nope/model", "auto/balanced", "openai/fallback-model"], "openai/primary-model", "t"
        )
        assert model is None
        assert failed == [
            ("fallback anthropic/claude-x", "ANTHROPIC_API_KEY is not set"),
            ("fallback nope/model", "this SDK has no client for nope"),
            ("fallback auto/balanced", "Robutler's models need a sign-in"),
        ]
        assert isinstance(skills["llm"], FailoverLLMSkill)
        assert skills["llm"].models == ["openai/primary-model", "openai/fallback-model"]
        # With nothing buildable, the skills are left as they were.
        untouched: Dict[str, Any] = {}
        assert apply_fallback_models(untouched, ["nope/model"], "openai/primary-model", "t") == ("openai/primary-model", [("fallback nope/model", "this SDK has no client for nope")])
        assert untouched == {}
        # The file's own LLM skill is the primary when the decision left it standing.
        own: Dict[str, Any] = {"openai": OpenAISkill({"model": "primary-model", "api_key": "k"})}
        model, failed = apply_fallback_models(own, ["openai/fallback-model"], None, "t")
        assert (model, failed) == (None, [])
        assert own["openai"].models == ["openai/primary-model", "openai/fallback-model"]
