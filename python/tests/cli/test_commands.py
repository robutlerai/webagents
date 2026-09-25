"""
The `webagents` commands, as a person runs them (2026-09-24).

The surface is the TypeScript CLI's (`tests/cli/test_cli_parity.py` holds the
two together); these check what each command does and says. Every test runs
under a throwaway HOME with the file secrets backend and no token, so nothing
here reads the machine's keychain, its config or its sign-in.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from webagents.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    from webagents.agents.skills.core.llm.providers import LLM_PROVIDERS

    keys = [v for p in LLM_PROVIDERS for v in p.env_vars] + ["GOOGLE_API_KEY"]
    for name in ["WEBAGENTS_PROFILE", "WEBAGENTS_TOKEN", "ROBUTLER_API_URL", *keys]:
        # setenv THEN delenv, so the original state is restored even for a
        # variable that was absent: the CLI loads stored keys INTO os.environ.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    # Nothing answers here, so a platform call fails fast rather than reaching a real portal.
    monkeypatch.setenv("ROBUTLER_API_URL", "http://127.0.0.1:9")
    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path / "work")


class TestTheBasics:
    def test_help_lists_the_commands(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for command in ("chat", "serve", "daemon", "login", "publish", "init", "doctor", "secrets"):
            assert command in result.output

    def test_version_is_bare(self):
        from webagents import __version__

        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert result.output.strip() == __version__

    def test_the_python_only_commands_are_gone(self):
        # `-p` replaced `run`, `publish` replaced `deploy`, and the chat has
        # /resume and /new; the rest the TypeScript CLI never had.
        for removed in ("run", "dev", "list", "reset", "completion", "version", "deploy", "agent", "auth", "session", "checkpoint", "skill", "ui"):
            result = runner.invoke(app, [removed])
            assert result.exit_code != 0, f"`webagents {removed}` still exists"


class TestInit:
    def test_it_makes_a_project_folder_with_the_agent_file(self):
        result = runner.invoke(app, ["init", "helper"])
        assert result.exit_code == 0, result.output
        text = Path("helper/AGENT.md").read_text()
        assert text.startswith("---\nname: helper\ndescription: A chatbot agent\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n---\n")
        assert "Created agent project: helper/" in result.output
        assert "cd helper" in result.output
        # Both ways to a model, and never `export KEY=...` into shell history.
        assert "add your key with `webagents secrets set OPENAI_API_KEY`, or sign in with `webagents login`" in result.output
        assert "export" not in result.output

    def test_the_default_name(self):
        assert runner.invoke(app, ["init"]).exit_code == 0
        assert Path("my-agent/AGENT.md").is_file()

    def test_the_tool_agent_template(self):
        assert runner.invoke(app, ["init", "tools", "-t", "tool-agent"]).exit_code == 0
        assert "  - filesystem\n  - shell\n" in Path("tools/AGENT.md").read_text()

    def test_an_unknown_template_leaves_nothing_behind(self):
        result = runner.invoke(app, ["init", "x", "--template", "rag-agent"])
        assert result.exit_code == 1
        assert "Unknown template 'rag-agent'. Available: chatbot, tool-agent." in result.output
        assert not Path("x").exists()

    def test_an_existing_folder_is_refused(self):
        Path("taken").mkdir()
        result = runner.invoke(app, ["init", "taken"])
        assert result.exit_code == 1
        assert "Directory taken already exists." in result.output

    def test_the_file_is_one_both_sdks_read(self):
        from webagents.cli.loader.hierarchy import load_agent

        runner.invoke(app, ["init", "helper"])
        merged = load_agent(Path("helper/AGENT.md"))
        assert merged.metadata.name == "helper"
        assert merged.metadata.model == "openai/gpt-4o-mini"


class TestListings:
    def test_templates(self):
        result = runner.invoke(app, ["templates", "list"])
        assert result.exit_code == 0
        assert "chatbot" in result.output and "tool-agent" in result.output
        assert "Use: webagents init <name> --template <template>" in result.output

    def test_skills_names_what_a_file_can_load(self):
        from webagents.cli.agent_builder import SKILL_CLASSES

        result = runner.invoke(app, ["skills", "list"])
        assert result.exit_code == 0
        for name in SKILL_CLASSES:
            assert f"  {name}\n" in result.output

    def test_models_counts_a_stored_key(self, monkeypatch):
        from webagents.cli.commands.secrets import _store

        before = runner.invoke(app, ["models"]).output
        assert "-      openai" in before
        _store(quiet=True).set("OPENAI_API_KEY", "sk-stored")
        after = runner.invoke(app, ["models"]).output
        assert "ready  openai" in after
        assert '"ready" means this machine has its key' in after


class TestConfig:
    def test_set_types_the_value_and_get_reads_it(self):
        result = runner.invoke(app, ["config", "set", "daemon.port", "8821"])
        assert result.exit_code == 0
        assert result.output.startswith("daemon.port = 8821 in ")
        assert runner.invoke(app, ["config", "get", "daemon.port"]).output.strip() == "8821"

    def test_an_unknown_key_is_refused(self):
        result = runner.invoke(app, ["config", "set", "daemon.prot", "1"])
        assert result.exit_code == 1
        assert "Unknown config key: daemon.prot" in result.output
        assert "Known keys:" in result.output

    def test_a_number_key_refuses_text(self):
        result = runner.invoke(app, ["config", "set", "daemon.port", "eighty"])
        assert result.exit_code == 1
        assert 'daemon.port expects a number, got "eighty"' in result.output

    def test_get_with_no_key_is_every_key_as_json(self):
        result = runner.invoke(app, ["config", "get"])
        assert result.exit_code == 0
        assert "daemon.port" in json.loads(result.output)

    def test_unset_and_path(self):
        runner.invoke(app, ["config", "set", "daemon.port", "8821"])
        assert runner.invoke(app, ["config", "unset", "daemon.port"]).output.strip() == "Removed daemon.port"
        assert runner.invoke(app, ["config", "unset", "daemon.port"]).output.strip() == "daemon.port was not set there"
        lines = runner.invoke(app, ["config", "path"]).output.splitlines()
        assert lines[0].startswith("global:  ") and lines[1].startswith("project: ")

    def test_validate(self):
        result = runner.invoke(app, ["config", "validate"])
        assert result.exit_code == 0
        assert result.output.strip() == "Config is valid."


class TestSecrets:
    def test_set_list_get_unset(self, monkeypatch):
        import getpass

        monkeypatch.setattr(getpass, "getpass", lambda prompt="": "sk-entered")
        result = runner.invoke(app, ["secrets", "set", "OPENAI_API_KEY"])
        assert result.exit_code == 0
        assert result.output.strip() == "Stored OPENAI_API_KEY (an owner-only file)."

        listing = runner.invoke(app, ["secrets", "list"]).output
        assert "OPENAI_API_KEY" in listing and "stored in an owner-only file" in listing

        assert runner.invoke(app, ["secrets", "get", "OPENAI_API_KEY"]).output.strip() == "OPENAI_API_KEY is stored. Use --show to print it."
        shown = runner.invoke(app, ["secrets", "get", "OPENAI_API_KEY", "--show"])
        assert shown.output == "sk-entered\n"

        assert runner.invoke(app, ["secrets", "unset", "OPENAI_API_KEY"]).output.strip() == "Removed OPENAI_API_KEY."
        gone = runner.invoke(app, ["secrets", "unset", "OPENAI_API_KEY"])
        assert gone.exit_code == 1
        assert "OPENAI_API_KEY was not stored." in gone.output

    def test_a_name_that_is_not_a_variable_is_refused(self):
        result = runner.invoke(app, ["secrets", "set", "openai key"])
        assert result.exit_code == 1
        assert "does not look like an environment variable name" in result.output

    def test_an_empty_listing_says_how_to_add_one(self):
        result = runner.invoke(app, ["secrets", "list"])
        assert "No keys stored, and none set in this shell." in result.output
        assert "webagents secrets set OPENAI_API_KEY" in result.output


class TestAccount:
    def test_whoami_signed_out_exits_nonzero_and_says_how(self):
        result = runner.invoke(app, ["whoami"])
        assert result.exit_code == 1
        assert "Not signed in to 127.0.0.1:9." in result.output
        assert "Run `webagents login`." in result.output

    def test_whoami_json_is_one_envelope(self):
        result = runner.invoke(app, ["--json", "whoami"])
        assert result.exit_code == 1
        assert json.loads(result.output)["error"]["code"] == "not_signed_in"

    def test_logout(self):
        result = runner.invoke(app, ["logout"])
        assert result.exit_code == 0
        assert result.output.strip() == "Signed out of 127.0.0.1:9."

    def test_login_with_a_rejected_key(self, monkeypatch):
        import webagents.cli.account as account

        monkeypatch.setattr(account, "validate_key", lambda portal, key: {"ok": False, "reason": "the portal rejected that key"})
        result = runner.invoke(app, ["login", "--token", "rok_bad"])
        assert result.exit_code == 1
        assert "Could not authenticate: the portal rejected that key" in result.output

    def test_login_with_a_key_stores_the_exchanged_token_and_the_portal(self, monkeypatch):
        import webagents.cli.account as account
        from webagents.cli.credentials import get_token

        monkeypatch.delenv("ROBUTLER_API_URL")
        monkeypatch.setattr(account, "validate_key", lambda portal, key: {"ok": True, "token": "jwt-exchanged", "username": "me"})
        result = runner.invoke(app, ["login", "--token", "rok_good", "--url", "http://portal.test/"])
        assert result.exit_code == 0, result.output
        assert "platform.url set to http://portal.test" in result.output
        assert "Authenticated as @me on http://portal.test." in result.output
        assert get_token() == "jwt-exchanged"

    def test_link_show_on_an_unlinked_folder(self):
        result = runner.invoke(app, ["link", "--show"])
        assert result.exit_code == 0
        assert "This folder is not linked to an agent on Robutler." in result.output

    def test_unlink_an_unlinked_folder(self):
        assert runner.invoke(app, ["unlink"]).output.strip() == "This folder is not linked."


class TestChatFlags:
    def test_an_unknown_output_format(self):
        result = runner.invoke(app, ["-p", "hi", "--output-format", "yaml"])
        assert result.exit_code == 2
        assert "Unknown --output-format 'yaml'. Expected one of: text, json, stream-json." in result.output

    def test_json_output_needs_a_prompt(self):
        result = runner.invoke(app, ["chat", "--output-format", "json"])
        assert result.exit_code == 2
        assert "--output-format json applies to -p/--prompt only." in result.output

    def test_an_unknown_agent_is_refused_naming_the_ones_here(self):
        Path("AGENT.md").write_text("---\nname: helper\n---\nHelp.\n")
        result = runner.invoke(app, ["-a", "helpr", "-p", "hi"])
        assert result.exit_code == 1
        assert "There is no agent called helpr in this folder. Agents here: helper, and the built-in robutler." in result.output

    def test_no_model_is_refused_before_anything_is_sent(self):
        Path("AGENT.md").write_text("---\nname: helper\n---\nHelp.\n")
        result = runner.invoke(app, ["-p", "hi"])
        assert result.exit_code == 1
        assert "webagents login" in result.output


class TestDaemon:
    def test_it_binds_loopback_by_default(self, monkeypatch):
        # S-218: it was 0.0.0.0, which put the unauthenticated management
        # routes on the LAN.
        import uvicorn

        bound = []
        monkeypatch.setattr(uvicorn, "run", lambda app_, host, port, **kwargs: bound.append((host, port)))
        result = runner.invoke(app, ["daemon", "--no-cron"])
        assert result.exit_code == 0, result.output
        assert bound == [("127.0.0.1", 8765)]

    def test_a_configured_host_off_this_machine_is_refused(self, monkeypatch):
        import uvicorn

        monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("bound"))
        runner.invoke(app, ["config", "set", "daemon.host", "0.0.0.0"])
        result = runner.invoke(app, ["daemon"])
        assert result.exit_code == 1


class TestServe:
    def test_it_binds_loopback_and_says_so(self, monkeypatch):
        import uvicorn

        bound = []
        monkeypatch.setattr(uvicorn, "run", lambda app_, host, port, **kwargs: bound.append((host, port)))
        Path("AGENT.md").write_text("---\nname: helper\n---\nHelp.\n")
        result = runner.invoke(app, ["serve", "--port", "3999"])
        assert result.exit_code == 0, result.output
        assert bound == [("127.0.0.1", 3999)]
        assert "helper: listening on 127.0.0.1 only" in result.output

    def test_the_one_agent_answers_at_the_root_too(self):
        # The TypeScript `serve` answers at /chat/completions, and the docs give
        # that URL for both SDKs; the named path keeps working.
        import asyncio

        from webagents.cli.serve import _AtRoot

        seen = []

        async def app(scope, receive, send):
            seen.append(scope["path"])

        wrapped = _AtRoot(app, "helper")
        for path in ("/chat/completions", "/helper/chat/completions", "/health"):
            asyncio.run(wrapped({"type": "http", "path": path}, None, None))
        assert seen == ["/helper/chat/completions", "/helper/chat/completions", "/health"]

    def test_nothing_to_serve_serves_a_bare_agent(self, monkeypatch):
        import uvicorn

        monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0, result.output
        assert "serving default agent" in result.output
        assert "[webagents] agent on http://127.0.0.1:3000" in result.output


def test_publish_dry_run_needs_no_account():
    Path("AGENT.md").write_text("---\nname: helper\ndescription: Helps.\n---\nHelp.\n")
    result = runner.invoke(app, ["publish", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Would POST http://127.0.0.1:9/api/agents:" in result.output
    assert json.loads(result.output.split(":\n", 1)[1])["name"] == "helper"


def test_doctor_names_what_to_fix(monkeypatch):
    Path("AGENT.md").write_text("---\nname: helper\n---\nHelp.\n")
    result = runner.invoke(app, ["doctor"])
    # No model here, so the check fails and says both ways out.
    assert result.exit_code == 1
    assert result.stdout.startswith("Checks\n")
    for name in ("runtime", "agent", "model", "sign-in", "keys", "sandbox", "config"):
        assert f" {name}" in result.output
    assert "helper (AGENT.md)" in result.output
    # No `skills:`, so no shell: the same answer as the TypeScript doctor.
    assert "not needed: the agent cannot run commands" in result.output
    assert "`webagents login`, or `webagents secrets set <NAME>`" in result.output


def test_doctor_json_is_one_envelope():
    result = runner.invoke(app, ["--json", "doctor"])
    document = json.loads(result.stdout)
    assert [c["name"] for c in document["data"]["checks"]] == ["runtime", "agent", "model", "sign-in", "keys", "sandbox", "config"]
