"""
Who pays for a served agent's model, and where it listens (S-327, 2026-09-28).

`serve` and the daemon ran every caller's turn on the signed-in owner's
Robutler credits when the agent had no provider key, for any bearer string
(exercised: a random bearer, 200, and the platform's trace funding the
owner), and `WEBAGENTS_PUBLIC_URL` alone made `serve` listen on every
interface with no AuthSkill. Pinned here, against the shared fixture
`cli/final_sdk_serve_model.json` the TypeScript suite reads too: the decision
for other callers never uses the sign-in, the proxy skill built for them
carries none and refuses a caller with no payment token before dialling,
`serve` refuses to start with the sentence naming both ways out, and the bind
widens only for an AuthSkill.
"""

import asyncio
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from webagents.cli import model_access
from webagents.cli.model_access import ModelUnavailable, choose_model_access

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "final_sdk_serve_model.json").read_text()
)
runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    from webagents.agents.skills.core.llm.providers import LLM_PROVIDERS

    keys = [v for p in LLM_PROVIDERS for v in p.env_vars] + ["GOOGLE_API_KEY"]
    for name in [
        "WEBAGENTS_PROFILE", "WEBAGENTS_TOKEN", "WEBAGENTS_AGENT_TOKEN", "WEBAGENTS_API_KEY",
        "ROBUTLER_PAYMENT_TOKEN", "WEBAGENTS_PUBLIC_URL", "ROBUTLER_LLM_PROXY_URL", *keys,
    ]:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setenv("ROBUTLER_API_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(model_access, "_client_installed", lambda provider: True)
    monkeypatch.setattr("webagents.cli.commands.secrets.load_into_environment", lambda: 0)
    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path / "work")


def _case(monkeypatch, case):
    for name, value in (case.get("env") or {}).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(model_access, "is_signed_in", lambda: bool(case.get("signed_in")))
    # The owner's login token is there whenever the case says signed in: the
    # proxy skill built for callers must not carry it all the same.
    monkeypatch.setattr("webagents.cli.credentials.get_token", lambda *a, **k: "jwt.owner-login" if case.get("signed_in") else None)
    monkeypatch.setattr(
        model_access, "agent_has_own_platform_credential", lambda name, env=None: bool(case.get("own_credential"))
    )


@pytest.mark.parametrize("case", FIXTURE["decision"], ids=[c["name"] for c in FIXTURE["decision"]])
def test_the_decision_for_other_callers(case, monkeypatch):
    from webagents.agents.skills.core.llm.proxy.skill import LLMProxySkill

    _case(monkeypatch, case)
    skills = {}
    if case["kind"] == "none":
        with pytest.raises(ModelUnavailable) as refused:
            choose_model_access(case["model"], skills, "helper", for_callers=True)
        assert str(refused.value).startswith("No model for this agent's callers: ")
        assert "never runs on your sign-in" in str(refused.value)
        assert skills == {}
        return
    access = choose_model_access(case["model"], skills, "helper", for_callers=True)
    assert (access.kind, access.model) == (case["kind"], case["sent"])
    if case["kind"] == "proxy":
        proxy = skills["llm"]
        assert isinstance(proxy, LLMProxySkill)
        assert proxy.callers_pay is True and proxy.platform_token is None
        assert "Authorization" not in proxy._build_extensions(case["sent"])


def test_the_owners_own_chat_still_runs_on_the_sign_in(monkeypatch):
    # The owner's chat and `-p` are unchanged: their turns are the owner's.
    _case(monkeypatch, {"signed_in": True, "own_credential": False})
    skills = {}
    access = choose_model_access("openai/gpt-4o-mini", skills, "helper")
    assert access.kind == "proxy"
    assert skills["llm"]._build_extensions("openai/gpt-4o-mini")["Authorization"] == "Bearer jwt.owner-login"


@pytest.mark.parametrize("case", FIXTURE["refusals"], ids=[c["name"] for c in FIXTURE["refusals"]])
def test_the_refusal_names_both_ways_out(case, monkeypatch):
    _case(monkeypatch, {"signed_in": True, "own_credential": False})
    with pytest.raises(ModelUnavailable) as refused:
        choose_model_access(case["model"], {}, "helper", for_callers=True)
    assert str(refused.value) == case["message"]


def test_a_listed_proxy_skill_carries_no_sign_in_for_callers(monkeypatch):
    from webagents.cli.agent_builder import load_skills

    _case(monkeypatch, {"signed_in": True, "own_credential": False})
    served = load_skills([{"proxy": {"platform_token": "written-in-the-file"}}], agent_name="helper", for_callers=True)["proxy"]
    assert served.callers_pay is True and served.platform_token is None
    own = load_skills(["proxy"], agent_name="helper")["proxy"]
    assert own.callers_pay is False and callable(own.platform_token)
    # And without its own credential the listed skill is not a way in either.
    with pytest.raises(ModelUnavailable):
        choose_model_access(None, {"proxy": served}, "helper", for_callers=True)


def test_the_daemon_decides_for_callers(monkeypatch):
    from webagents.server.extensions import local_file_source

    _case(monkeypatch, {"signed_in": True, "own_credential": False})
    with pytest.raises(ModelUnavailable) as refused:
        local_file_source._choose_model("openai/gpt-4o-mini", {}, "helper")
    assert str(refused.value) == FIXTURE["refusals"][0]["message"]


def test_a_robutler_fallback_needs_the_agents_own_credential(monkeypatch):
    from webagents.cli.agent_builder import apply_fallback_models

    _case(monkeypatch, {"signed_in": True, "own_credential": False, "env": {"OPENAI_API_KEY": "sk-test"}})
    model, failed = apply_fallback_models({}, ["auto/fast"], "openai/gpt-4o-mini", "helper", for_callers=True)
    assert model == "openai/gpt-4o-mini"
    assert failed == [("fallback auto/fast", "Robutler's models need the agent's own platform credential here")]


class TestTheProxySkillForCallers:
    def _skill(self, **config):
        from webagents.agents.skills.core.llm.proxy.skill import LLMProxySkill

        return LLMProxySkill({"model": "auto/balanced", "callers_pay": True, **config})

    def test_no_payment_token_is_refused_before_anything_is_dialled(self, monkeypatch):
        from webagents.agents.skills.core.llm.proxy import skill as proxy_module

        def dial(*args, **kwargs):
            raise AssertionError("dialled the platform with nothing to pay with")

        monkeypatch.setattr(proxy_module.websockets.client, "connect", dial)
        skill = self._skill(platform_token=lambda: "jwt.owner-login")
        assert skill.platform_token is None

        async def run():
            async for _ in skill.chat_completion_stream([{"role": "user", "content": "hi"}]):
                pass

        with pytest.raises(proxy_module.LLMProxyError) as refused:
            asyncio.run(run())
        expected = FIXTURE["no_payment_token"]
        assert refused.value.status_code == expected["status"]
        assert refused.value.to_dict() == {"error": {"code": expected["code"], "message": expected["message"]}}

    def test_the_callers_own_token_on_its_request_pays(self):
        from webagents.server.context.context_vars import Context, set_context

        class Request:
            headers = {"x-payment-token": "caller.jwt"}

        skill = self._skill()
        set_context(Context(request=Request()))
        try:
            assert skill._build_extensions("auto/balanced")["X-Payment-Token"] == "caller.jwt"
        finally:
            set_context(None)

    def test_a_served_caller_never_reads_the_platforms_protocol_words(self):
        # B11: `session.create requires X-Payment-Token, ...` in a 401 body.
        from webagents.agents.skills.core.llm.proxy.skill import REFUSED_CREDENTIAL_MESSAGE, LLMProxyError

        error = LLMProxyError(
            "unauthorized",
            "session.create requires X-Payment-Token, or the Bearer token `webagents login` stores as "
            "Authorization, in session.extensions",
        )
        assert error.to_dict()["error"]["message"] == REFUSED_CREDENTIAL_MESSAGE
        # The chat's failure presentation still reads the platform's words.
        assert "session.create" in str(error)


class _Built:
    def __init__(self, agent, name="helper"):
        self.agent = agent
        self.name = name


@pytest.mark.parametrize("case", FIXTURE["bind"]["cases"], ids=lambda c: f"{c['public_url']}-{c['auth_skill']}")
def test_the_bind_widens_only_for_an_authskill(case, monkeypatch, capsys):
    from webagents.cli.serve import bind_host

    if case["public_url"]:
        monkeypatch.setenv("WEBAGENTS_PUBLIC_URL", case["public_url"])
    monkeypatch.setattr(
        "webagents.server.core.origin_policy.agent_verifies_credentials", lambda agent: case["auth_skill"]
    )
    assert bind_host(_Built(object()), None) == case["host"]
    printed = capsys.readouterr().out.strip()
    if case["line"] is None:
        assert printed == ""
    else:
        assert printed == FIXTURE["bind"][case["line"]].replace("{name}", "helper")


def test_serve_refuses_to_start_without_a_model_for_its_callers(monkeypatch):
    import uvicorn

    from webagents.cli.main import app

    started = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: started.append(a))
    # Signed in, no provider key, no credential of the agent's own: exactly
    # the S-327 setup, which answered every caller on the owner's credits.
    monkeypatch.setenv("WEBAGENTS_TOKEN", "jwt.owner-login")
    Path("AGENT.md").write_text("---\nname: served\nmodel: openai/gpt-4o-mini\n---\nHelp.\n")
    result = runner.invoke(app, ["serve", "--port", "3998"])
    assert result.exit_code == 1, result.output
    assert started == []
    line = FIXTURE["serve_refusal_line"].replace("{name}", "served").replace("{message}", FIXTURE["refusals"][0]["message"])
    assert line in result.output


def test_serve_starts_on_the_agents_own_credential(monkeypatch):
    import uvicorn

    from webagents.cli.main import app

    started = []
    monkeypatch.setattr(uvicorn, "run", lambda app_, host, port, **k: started.append((host, port)))
    monkeypatch.setenv("WEBAGENTS_AGENT_TOKEN", "agent-own-key")
    Path("AGENT.md").write_text("---\nname: served\nmodel: openai/gpt-4o-mini\n---\nHelp.\n")
    result = runner.invoke(app, ["serve", "--port", "3997"])
    assert result.exit_code == 0, result.output
    assert started == [("127.0.0.1", 3997)]


def test_acp_and_stdio_mcp_are_the_owners_own_turns(monkeypatch):
    # The editor or MCP client the owner started here: the sign-in may pay,
    # as the ACP `login` auth method promises; `mcp serve --http` and `serve`
    # are other callers' and never do.
    from webagents.agents.skills.core.llm.proxy.skill import LLMProxySkill
    from webagents.cli.serve import load_served_agent

    monkeypatch.setenv("WEBAGENTS_TOKEN", "jwt.owner-login")
    Path("AGENT.md").write_text("---\nname: served\nmodel: openai/gpt-4o-mini\n---\nHelp.\n")
    own = load_served_agent("AGENT.md", for_callers=False)
    proxy = next(skill for skill in own.agent.skills.values() if isinstance(skill, LLMProxySkill))
    assert proxy.callers_pay is False
    assert proxy._build_extensions("openai/gpt-4o-mini")["Authorization"] == "Bearer jwt.owner-login"
    callers = load_served_agent("AGENT.md")
    assert callers.model_problem == FIXTURE["refusals"][0]["message"]
    assert not any(isinstance(skill, LLMProxySkill) for skill in callers.agent.skills.values())
