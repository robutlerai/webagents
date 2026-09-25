"""
Which model a CLI agent runs on, and how (2026-09-24, `cli/model_access.py`).

The daemon gave every agent without an LLM skill a Google model, whatever keys
the developer had; without `google-genai` installed the chat said "Internal
Server Error". A person signed in to Robutler, which serves models, was told to
go and find a provider key. Pinned here: the decision, the daemon building the
agent from it (the proxy paid by the signed-in person), and the chat's offer
when nothing works.
"""

import pytest

from webagents.cli import model_access
from webagents.cli.model_access import (
    ModelUnavailable,
    resolve_model_access,
    welcome_model,
)

YES = lambda: True  # noqa: E731
NO = lambda: False  # noqa: E731


@pytest.fixture
def clients(monkeypatch):
    """Which client libraries count as installed; all of them by default."""
    installed = {"openai": True, "anthropic": True, "google": True, "xai": True, "fireworks": True}
    monkeypatch.setattr(model_access, "_client_installed", lambda provider: installed.get(provider.id, True))
    return installed


class TestTheDecision:
    def test_the_declared_model_runs_directly_with_its_key(self, clients):
        access = resolve_model_access("openai/gpt-4o", signed_in=NO, env={"OPENAI_API_KEY": "sk"})
        assert (access.kind, access.model) == ("direct", "openai/gpt-4o")

    def test_without_its_key_it_runs_through_robutler_when_signed_in(self, clients):
        access = resolve_model_access("openai/gpt-4o", signed_in=YES, env={})
        assert (access.kind, access.model) == ("proxy", "openai/gpt-4o")
        assert access.reason == "OPENAI_API_KEY is not set"
        assert access.describe() == "openai/gpt-4o via Robutler"

    def test_a_key_without_its_client_library_is_not_enough(self, clients):
        clients["google"] = False
        access = resolve_model_access("google/gemini-2.5-flash", signed_in=YES, env={"GEMINI_API_KEY": "k"})
        assert access.kind == "proxy" and "client library is not installed" in access.reason

    def test_neither_is_none_with_both_ways_out(self, clients):
        access = resolve_model_access("anthropic/claude-x", signed_in=NO, env={})
        assert access.kind == "none"
        message = str(ModelUnavailable(access))
        assert "webagents login" in message and "webagents secrets set ANTHROPIC_API_KEY" in message

    def test_no_model_takes_any_provider_that_has_a_key(self, clients):
        # "why does it want openai key only?": whichever provider has a key.
        access = resolve_model_access(None, signed_in=YES, env={"XAI_API_KEY": "k"})
        assert access.kind == "direct" and access.model.startswith("xai/")

    def test_no_model_and_no_key_is_robutlers_choice_when_signed_in(self, clients):
        access = resolve_model_access(None, signed_in=YES, env={})
        assert (access.kind, access.model) == ("proxy", "auto/balanced")

    def test_no_model_no_key_not_signed_in_is_none(self, clients):
        access = resolve_model_access(None, signed_in=NO, env={})
        message = str(ModelUnavailable(access))
        assert access.kind == "none" and "webagents secrets set <NAME>` (OPENAI_API_KEY, ANTHROPIC_API_KEY" in message

    @pytest.mark.parametrize("declared,sent", [
        ("auto/fast", "auto/fast"),
        ("proxy/openai/gpt-4o", "openai/gpt-4o"),
        ("robutler/anthropic/claude-x", "anthropic/claude-x"),
        ("proxy/", "auto/balanced"),
    ])
    def test_models_only_robutler_serves(self, clients, declared, sent):
        # The platform does not strip `proxy/`, so it is not sent.
        access = resolve_model_access(declared, signed_in=YES, env={"OPENAI_API_KEY": "sk"})
        assert (access.kind, access.model) == ("proxy", sent)
        assert resolve_model_access(declared, signed_in=NO, env={}).kind == "none"

    def test_signed_in_is_asked_only_when_it_matters(self, clients):
        def never():
            raise AssertionError("asked for a login a key already answers")

        assert resolve_model_access("openai/gpt-4o", signed_in=never, env={"OPENAI_API_KEY": "sk"}).kind == "direct"


class TestTheDaemonBuildsWhatWasDecided:
    def test_the_proxy_skill_is_paid_by_the_signed_in_person(self, clients, monkeypatch):
        from webagents.agents.skills.core.llm.proxy.skill import LLMProxySkill
        from webagents.server.extensions import local_file_source

        monkeypatch.setattr(model_access, "is_signed_in", YES)
        monkeypatch.setattr("webagents.cli.credentials.get_token", lambda *a, **k: "jwt.login")
        monkeypatch.setattr("webagents.cli.commands.secrets.load_into_environment", lambda: 0)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ROBUTLER_API_URL", "https://portal.example")
        monkeypatch.delenv("ROBUTLER_LLM_PROXY_URL", raising=False)
        for var in ("ANTHROPIC_API_KEY", "GOOGLE_GEMINI_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY", "FIREWORKS_API_KEY"):
            monkeypatch.delenv(var, raising=False)

        skills = {}
        model = local_file_source._choose_model("openai/gpt-4o", skills, "helper")
        assert model is None and isinstance(skills["llm"], LLMProxySkill)
        proxy = skills["llm"]
        assert proxy.model == "openai/gpt-4o"
        assert proxy.proxy_url == "wss://portal.example/llm"
        assert proxy._build_extensions("openai/gpt-4o")["Authorization"] == "Bearer jwt.login"

    def test_a_usable_key_builds_the_provider_directly(self, clients, monkeypatch):
        from webagents.server.extensions import local_file_source

        monkeypatch.setattr("webagents.cli.commands.secrets.load_into_environment", lambda: 0)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        skills = {}
        assert local_file_source._choose_model("openai/gpt-4o", skills, "helper") == "openai/gpt-4o"
        assert skills == {}

    def test_nothing_usable_is_an_error_with_the_way_out(self, clients, monkeypatch):
        from webagents.server.extensions import local_file_source

        monkeypatch.setattr(model_access, "is_signed_in", NO)
        monkeypatch.setattr("webagents.cli.commands.secrets.load_into_environment", lambda: 0)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ModelUnavailable) as error:
            local_file_source._choose_model("openai/gpt-4o", {}, "helper")
        assert "webagents login" in str(error.value)

    def test_an_agent_that_names_its_llm_skill_keeps_its_choice(self, clients, monkeypatch):
        from webagents.server.extensions import local_file_source

        skills = {"anthropic": object()}
        assert local_file_source._choose_model(None, skills, "helper") is None
        assert list(skills) == ["anthropic"]


def test_the_welcome_card_says_what_will_run(clients, monkeypatch):
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_GEMINI_API_KEY", "GEMINI_API_KEY",
                "XAI_API_KEY", "FIREWORKS_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(model_access, "is_signed_in", YES)
    assert welcome_model(None, ["filesystem", "shell"]) == ("auto/balanced via Robutler", None)
    monkeypatch.setattr(model_access, "is_signed_in", NO)
    shown, warning = welcome_model("openai/gpt-4o", [])
    assert shown == "openai/gpt-4o" and "webagents login" in warning


class TestAnAgentThatListsItsProvider:
    """`skills: [openai]` names the provider; without its key it is not a dead end (2026-09-24)."""

    @pytest.fixture(autouse=True)
    def no_keys(self, monkeypatch, clients):
        for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_GEMINI_API_KEY", "GEMINI_API_KEY",
                    "XAI_API_KEY", "FIREWORKS_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr("webagents.cli.commands.secrets.load_into_environment", lambda: 0)

    @pytest.mark.parametrize("declared,expected", [
        ("openai/gpt-4.1-mini", "openai/gpt-4.1-mini"),
        ("gpt-4.1-mini", "openai/gpt-4.1-mini"),
        ("anthropic/claude-x", "openai/gpt-4o-mini"),
        (None, "openai/gpt-4o-mini"),
    ])
    def test_the_model_it_runs_is_its_providers(self, declared, expected):
        from webagents.agents.skills.core.llm.providers import find_provider

        assert model_access.model_for_named_provider(find_provider("openai"), declared) == expected

    def test_the_daemon_runs_the_same_model_through_robutler_when_signed_in(self, monkeypatch):
        from webagents.agents.skills.core.llm.proxy.skill import LLMProxySkill
        from webagents.server.extensions import local_file_source

        monkeypatch.setattr(model_access, "is_signed_in", YES)
        monkeypatch.setattr("webagents.cli.credentials.get_token", lambda *a, **k: "jwt.login")

        class KeylessOpenAI:
            api_key = ""

        skills = {"openai": KeylessOpenAI(), "shell": object()}
        assert local_file_source._choose_model("openai/gpt-4o-mini", skills, "helper") is None
        assert "openai" not in skills and "shell" in skills
        assert isinstance(skills["llm"], LLMProxySkill) and skills["llm"].model == "openai/gpt-4o-mini"

    def test_a_listed_provider_with_its_key_keeps_its_skill(self, monkeypatch):
        from webagents.server.extensions import local_file_source

        class OpenAIWithKey:
            api_key = "sk-from-config"

        skill = OpenAIWithKey()
        skills = {"openai": skill}
        # None: the listed skill carries the model itself (`agent_builder.load_skills`),
        # and a model handed to BaseAgent would build a second LLM skill beside it.
        assert local_file_source._choose_model("openai/gpt-4o-mini", skills, "helper") is None
        assert skills == {"openai": skill}

    def test_the_listed_skill_is_built_with_the_files_model(self):
        # It ran its own hard-coded default (gpt-4o) whatever the file said.
        from webagents.cli.agent_builder import load_skills

        skills = load_skills(["openai"], agent_name="helper", model="openai/gpt-4o-mini")
        assert skills["openai"].model == "gpt-4o-mini"
        # No model at all: the provider's own default, as the TypeScript SDK does.
        assert load_skills(["openai"], agent_name="helper")["openai"].model == "gpt-4o-mini"
        # Another provider's model is not this skill's: its default instead.
        assert load_skills(["openai"], agent_name="helper", model="anthropic/claude-x")["openai"].model == "gpt-4o-mini"

    def test_nothing_usable_is_the_same_error_with_both_ways_out(self, monkeypatch):
        from webagents.server.extensions import local_file_source

        monkeypatch.setattr(model_access, "is_signed_in", NO)

        class KeylessOpenAI:
            api_key = ""

        with pytest.raises(ModelUnavailable) as error:
            local_file_source._choose_model("openai/gpt-4o-mini", {"openai": KeylessOpenAI()}, "helper")
        assert "webagents login" in str(error.value) and "OPENAI_API_KEY" in str(error.value)

    def test_the_card_says_what_will_run(self, monkeypatch):
        monkeypatch.setattr(model_access, "is_signed_in", YES)
        assert welcome_model("openai/gpt-4o-mini", ["openai"]) == ("openai/gpt-4o-mini via Robutler", None)
        monkeypatch.setattr(model_access, "is_signed_in", NO)
        shown, warning = welcome_model("openai/gpt-4o-mini", ["openai"])
        assert "webagents login" in warning and "OPENAI_API_KEY" in warning
