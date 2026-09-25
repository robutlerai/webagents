"""
The first-time-developer path, as a test (2026-09-24).

Every case here was hit by walking the quickstart on a clean machine with the
package built from this tree: `init`, then the model key, then `webagents -p`
(it was `run`) and the chat, then `doctor`. None of them was visible to the unit suites, because each was a
gap between two pieces that were individually right.
"""

import json

import pytest
from typer.testing import CliRunner

from webagents.cli.main import app

runner = CliRunner()

KEY_VARS = (
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_GEMINI_API_KEY", "GEMINI_API_KEY",
    "XAI_API_KEY", "FIREWORKS_API_KEY", "OPENAI_KEY", "GOOGLE_API_KEY",
)


@pytest.fixture(autouse=True)
def newcomer(tmp_path, monkeypatch):
    """A clean HOME, no provider keys, and a file-backed store (no real Keychain)."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("COLUMNS", "200")
    for var in KEY_VARS + ("WEBAGENTS_PROFILE", "WEBAGENTS_TOKEN", "ROBUTLER_API_URL", "WEBAGENTS_LOG_LEVEL"):
        # setenv THEN delenv, so monkeypatch records the original state even
        # when the variable was absent. The CLI now loads stored keys INTO
        # os.environ, and a bare delenv of an absent name records nothing to
        # restore, so a key loaded here would leak into every later test.
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    project = tmp_path / "my-first-agent"
    project.mkdir()
    monkeypatch.chdir(project)
    from webagents.cli import credentials

    credentials.set_flag_token(None)
    yield project


def test_version_flag_exists():
    """The first thing most people type. It answered "No such option"."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    from webagents import __version__

    assert result.stdout.strip() == __version__


def _init_and_enter(monkeypatch, project):
    """`webagents init`, then `cd my-agent`, as the next steps say."""
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.output
    monkeypatch.chdir(project / "my-agent")
    return result


def test_init_names_both_ways_to_a_model(monkeypatch, newcomer):
    result = _init_and_enter(monkeypatch, newcomer)
    # `init` writes `model: openai/gpt-4o-mini`; the next step fails without one of these.
    assert "webagents secrets set OPENAI_API_KEY" in result.stdout
    assert "webagents login" in result.stdout


def test_a_prompt_without_the_key_says_which_key_and_exits_non_zero(monkeypatch, newcomer):
    _init_and_enter(monkeypatch, newcomer)
    result = runner.invoke(app, ["-p", "hello"])
    # THE BUG: "No handoff registered for agent", a traceback, and exit 0.
    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.stderr
    assert "webagents login" in result.stderr
    assert "No handoff registered" not in result.output
    assert "Traceback" not in result.output
    # stdout is for the answer, and there is none.
    assert result.stdout == ""


def test_a_prompt_puts_the_answer_and_only_the_answer_on_stdout(monkeypatch, newcomer):
    """`webagents run -p "..." > out.txt` captured the banner, the skill list and
    the spinner along with the answer (2026-09-24)."""
    _init_and_enter(monkeypatch, newcomer)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-not-a-key")
    from webagents.agents.core.base_agent import BaseAgent

    async def canned(self, messages, **kwargs):
        yield {"choices": [{"delta": {"content": "The answer."}}]}

    monkeypatch.setattr(BaseAgent, "run_streaming", canned)
    result = runner.invoke(app, ["-p", "hello"])
    assert result.exit_code == 0, result.output
    assert result.stdout == "The answer.\n"
    # THE BUG: every first run printed this, and then worked anyway, because
    # the agent's LLM comes from its `model:`, not from the Google fallback.
    assert "Failed to load LLM skill" not in result.output

    as_json = runner.invoke(app, ["-p", "hello", "--output-format", "json"])
    assert json.loads(as_json.stdout)["content"] == "The answer."
    lines = runner.invoke(app, ["-p", "hello", "--output-format", "stream-json"]).stdout.splitlines()
    assert [json.loads(line)["type"] for line in lines] == ["delta", "done"]


def test_a_key_stored_with_secrets_set_is_used(monkeypatch, newcomer):
    """`secrets set` then a prompt is the order the docs give."""
    _init_and_enter(monkeypatch, newcomer)
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "dummy-not-a-key")
    runner.invoke(app, ["secrets", "set", "OPENAI_API_KEY"])

    # doctor sees what the chat would see: the stored key.
    result = runner.invoke(app, ["doctor"])
    assert "✓ model" in result.stdout
    assert "with your OPENAI_API_KEY" in result.stdout


def test_doctor_names_the_agents_own_key_not_googles(monkeypatch, newcomer):
    _init_and_enter(monkeypatch, newcomer)
    result = runner.invoke(app, ["doctor"])
    # THE BUG: it advised GOOGLE_GEMINI_API_KEY for an openai/... agent.
    assert "none (OPENAI_API_KEY is not set)" in result.stdout
    assert "GOOGLE" not in result.stdout


def test_the_daemon_client_presents_a_credential():
    """`connect` answered 401 from the credential floor on every message."""
    from webagents.cli.client.daemon_client import DaemonClient

    client = DaemonClient()
    assert client.client.headers.get("authorization", "").startswith("Bearer ")


def test_the_chat_tells_the_truth_about_the_sandbox(newcomer):
    """The footer once said "sandbox: on" for an agent with no sandbox at all.
    The chat now reports what the agent it built can do (`/sandbox`, `/status`)."""
    import asyncio

    from webagents.cli.repl.session import WebAgentsSession

    (newcomer / "AGENT.md").write_text("---\nname: plain\nskills:\n  - shell\n---\nx\n")
    plain = WebAgentsSession(agent_path=newcomer / "AGENT.md")
    asyncio.run(plain.initialize())
    kind, headline, _ = plain.sandbox_summary()
    assert kind == "warn" and headline.startswith("Off")

    boxed_dir = newcomer / "boxed"
    boxed_dir.mkdir()
    (boxed_dir / "AGENT.md").write_text(
        "---\nname: boxed\nskills:\n  - shell\nsandbox:\n  preset: strict\n---\nx\n"
    )
    boxed = WebAgentsSession(agent_path=boxed_dir / "AGENT.md")
    asyncio.run(boxed.initialize())
    kind, headline, _ = boxed.sandbox_summary()
    assert kind == "ok" and headline.startswith("On (strict)")
