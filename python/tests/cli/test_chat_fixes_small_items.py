"""
The smaller CLI items from the 2026-09-27 real-model pass (the chat-fixes
lane; the TypeScript twin is `tests/unit/cli/chat-fixes-small-items.test.ts`):

  * enter on a fully typed argument sends the line (`/agent edit` offered
    `edit` and put a space after it instead);
  * a key hint names the model's provider (`provider_key_for`);
  * `repl.log` follows `--profile`;
  * `models` shows Robutler's models ready when signed in;
  * `-m bedrock/...`, a provider this SDK has no client for, runs through
    Robutler when signed in and says how to get there when not, instead of
    "No handoff registered"; a file's own LLM skill keeps charge;
  * a line break in an answer stays a line break in the chat's markdown.
"""

import json
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from rich.console import Console
from typer.testing import CliRunner

from webagents.cli.model_access import ModelUnavailable, choose_model_access, resolve_model_access
from webagents.cli.repl.failures import provider_key_for
from webagents.cli.repl.session import repl_log_path
from webagents.cli.ui.markdown import ChatMarkdown, _hard_breaks
from webagents.cli.ui.prompt_box import PromptBox, Slot, argument_words
from webagents.cli.ui.theme import theme_for


def _agent_completer(args: str):
    before, partial = argument_words(args)
    if before == ["edit"]:
        return Slot([("helper", "A helper")], partial)
    if before:
        return None
    return Slot([("helper", "A helper"), ("new", "make one here"), ("edit", "open its file")], partial)


def _box() -> PromptBox:
    return PromptBox(
        theme_for(Console(file=StringIO(), width=100, force_terminal=False, color_system=None)),
        commands=[("/help", "Show the commands and keys"), ("/agent", "List this folder")],
        footer=lambda: [],
        completers={"agent": _agent_completer},
    )


class TestEnterWhileTheMenuOffersTheWordAlreadyTyped:
    def test_a_fully_typed_agent_edit_is_sent(self):
        box = _box()
        assert [c[0] for c in box.menu_items("/agent edit")] == ["edit"]
        assert box.enter_choice("/agent edit") == ("send", "/agent edit")

    def test_a_partly_typed_value_is_still_inserted(self):
        assert _box().enter_choice("/agent ed") == ("insert", "edit")

    def test_the_word_matches_without_regard_to_case(self):
        assert _box().enter_choice("/agent EDIT") == ("send", "/agent EDIT")

    def test_a_command_row_runs_the_command(self):
        assert _box().enter_choice("/he") == ("send", "/help")


def test_the_key_a_hint_names_is_the_models_providers():
    assert provider_key_for("google/gemini-2.5-flash") == "GOOGLE_API_KEY"
    assert provider_key_for("anthropic/claude-sonnet-4-5") == "ANTHROPIC_API_KEY"
    assert provider_key_for("xai/grok-4") == "XAI_API_KEY"
    # A model that names no provider this SDK can call keeps the historical default.
    assert provider_key_for("auto/balanced") == "OPENAI_API_KEY"
    assert provider_key_for("proxy/gpt-4o") == "OPENAI_API_KEY"
    assert provider_key_for("bedrock/anthropic.claude") == "OPENAI_API_KEY"
    assert provider_key_for(None) == "OPENAI_API_KEY"


def test_the_repl_log_follows_the_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("WEBAGENTS_PROFILE", raising=False)
    assert repl_log_path() == tmp_path / ".webagents" / "logs" / "repl.log"
    monkeypatch.setenv("WEBAGENTS_PROFILE", "local")
    assert repl_log_path() == tmp_path / ".webagents-local" / "logs" / "repl.log"


def _models_rows(signed_in: bool):
    from webagents.cli.main import app

    with patch("webagents.cli.model_access.is_signed_in", return_value=signed_in), patch(
        "webagents.agents.skills.core.llm.ollama.probe.probe_ollama"
    ) as probe:
        probe.return_value.ok = False
        result = CliRunner().invoke(app, ["--json", "models"])
    assert result.exit_code == 0, result.output
    # The envelope `--json` writes (`cli/output.py`): `{"ok": true, "data": {...}}`.
    return {row["id"]: row for row in json.loads(result.output)["data"]["providers"]}


def test_models_shows_robutlers_models_ready_when_signed_in(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY", "ROBUTLER_LLM_PROXY_URL"):
        monkeypatch.delenv(var, raising=False)
    assert _models_rows(signed_in=True)["proxy"]["ready"] is True
    assert _models_rows(signed_in=False)["proxy"]["ready"] is False


class TestAProviderThisSdkHasNoClientFor:
    def test_runs_through_robutler_when_signed_in(self):
        access = resolve_model_access("bedrock/anthropic.claude-sonnet", signed_in=lambda: True, env={})
        assert (access.kind, access.model, access.provider) == ("proxy", "bedrock/anthropic.claude-sonnet", None)
        assert access.reason == "this SDK has no client for bedrock"

    def test_says_how_to_get_there_when_signed_out(self):
        access = resolve_model_access("bedrock/anthropic.claude-sonnet", signed_in=lambda: False, env={})
        assert access.kind == "none"
        message = str(ModelUnavailable(access))
        assert message.startswith("No model for this agent: this SDK has no client for bedrock.")
        assert "Sign in with `webagents login`" in message
        assert "No handoff registered" not in message

    def test_a_files_own_llm_skill_keeps_charge(self, monkeypatch):
        class CustomLLM:
            async def chat_completion_stream(self, messages, **kwargs):
                yield {}

        monkeypatch.setattr("webagents.cli.model_access.is_signed_in", lambda: False)
        skills = {"litellm": CustomLLM()}
        access = choose_model_access("bedrock/anthropic.claude-sonnet", skills, "custom")
        assert access is not None and (access.kind, access.model) == ("direct", "bedrock/anthropic.claude-sonnet")
        assert list(skills) == ["litellm"]


class TestLineBreaksStayLineBreaks:
    @staticmethod
    def _render(markup: str) -> list:
        console = Console(file=StringIO(), width=80, force_terminal=False, color_system=None)
        theme = theme_for(console, env={})
        with console.capture() as captured:
            console.print(ChatMarkdown(markup, theme))
        return [line.rstrip() for line in captured.get().split("\n") if line.strip()]

    def test_one_per_line_draws_one_per_line(self):
        assert self._render("One\nTwo\nThree") == ["One", "Two", "Three"]

    def test_paragraphs_lists_and_fences_are_left_to_the_parser(self):
        assert _hard_breaks("One\n\nTwo") == "One\n\nTwo"
        assert _hard_breaks("- a\n- b") == "- a\n- b"
        assert _hard_breaks("# Title\ntext") == "# Title\ntext"
        assert _hard_breaks("```\nx = 1\ny = 2\n```") == "```\nx = 1\ny = 2\n```"
        assert _hard_breaks("| a |\n| b |") == "| a |\n| b |"
        assert _hard_breaks("already  \nbroken") == "already  \nbroken"
        assert self._render("```\nx = 1\ny = 2\n```")[0:2] != ["x = 1  ", "y = 2  "]
