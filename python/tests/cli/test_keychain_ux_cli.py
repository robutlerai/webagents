"""
Keychain dialogs on macOS, the CLI's half (the keychain-ux lane, 2026-09-27):
`credentials.json` at 0600 (S-315), what `logout`, `whoami`, `secrets list`,
`doctor` and the chat's /status say, and that `serve` and the daemon forbid
dialogs before they read anything. The words come from
`fixtures/keychain_ux/keychain_ux.json`, which the TypeScript twin
(`tests/unit/cli/keychain-ux-cli.test.ts`) reads too.

Every test runs in a scratch HOME with WEBAGENTS_SECRETS_BACKEND=file; the
keychain itself is a fake where one is needed.
"""

import json
import os
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

from webagents.agents.skills.local.secrets import keychain_ux as kx
from webagents.agents.skills.local.secrets.store import SecretStore
from webagents.cli.main import app

FIXTURE = json.loads((Path(__file__).parents[1] / "fixtures" / "keychain_ux" / "keychain_ux.json").read_text())
DUMMY = "dummy-value-not-a-real-secret"
runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("ROBUTLER_API_URL", "http://127.0.0.1:9")
    from webagents.cli.commands.secrets import _known_env_vars

    # Every provider variable the registry knows: another test in the suite
    # may have left one in the environment, and `secrets list` shows it.
    for name in ("WEBAGENTS_PROFILE", "WEBAGENTS_TOKEN", "GEMINI_API_KEY", *_known_env_vars()):
        monkeypatch.delenv(name, raising=False)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    kx.reset_for_tests(interactive=False)
    yield home
    kx.reset_for_tests()
    os.chdir(cwd)


class PlainKeyring:
    def __init__(self):
        self.items = {}

    def get_password(self, service, name):
        return self.items.get((service, name))

    def set_password(self, service, name, value):
        self.items[(service, name)] = value

    def delete_password(self, service, name):
        del self.items[(service, name)]


# ---------------------------------------------------------------------------
# credentials.json is 0600 (S-315)
# ---------------------------------------------------------------------------


class TestCredentialsFile:
    def test_it_is_written_0600_and_never_holds_the_token(self, isolated):
        from webagents.cli.state.local import LocalState

        LocalState(project_dir=Path.cwd()).set_credentials(access_token=DUMMY, username="someone", user_id="u1")
        path = isolated / ".webagents" / FIXTURE["credentials_file"]["name"]
        assert stat.S_IMODE(path.stat().st_mode) == int(FIXTURE["credentials_file"]["mode"], 8)
        assert DUMMY not in path.read_text()
        assert json.loads(path.read_text())["username"] == "someone"

    def test_a_looser_file_an_earlier_version_left_is_repaired_when_read(self, isolated):
        from webagents.cli.state.local import LocalState

        path = isolated / ".webagents" / "credentials.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"username": "someone"}))
        os.chmod(path, 0o644)
        assert LocalState(project_dir=Path.cwd()).get_credentials()["username"] == "someone"
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_an_existing_looser_file_is_tightened_by_the_next_write(self, isolated):
        from webagents.cli.state.local import LocalState

        path = isolated / ".webagents" / "credentials.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        os.chmod(path, 0o644)
        LocalState(project_dir=Path.cwd()).set_credentials(username="someone")
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


# ---------------------------------------------------------------------------
# logout, whoami, secrets
# ---------------------------------------------------------------------------


class TestLogout:
    def test_an_old_item_only_a_dialog_could_remove_is_named(self, monkeypatch):
        from webagents.cli import credentials
        from webagents.cli.state import local

        monkeypatch.setattr(local.LocalState, "clear_credentials", lambda self: None)
        monkeypatch.setattr(credentials, "left_behind", lambda: [{"item": "webagents:cli", "account": "platform_token"}])
        result = runner.invoke(app, ["logout"])
        assert result.exit_code == 0
        assert result.stdout.splitlines() == [
            "Signed out of 127.0.0.1:9.",
            FIXTURE["left_behind"].format(item="webagents:cli", account="platform_token"),
        ]

    def test_a_token_only_a_dialog_could_remove_is_not_signed_out(self, monkeypatch):
        from webagents.cli.state import local

        sentence = FIXTURE["blocked"].format(program="Python", item="webagents (Python) cli", command="webagents whoami")

        def blocked(self):
            raise kx.KeychainDialogBlocked(sentence, "webagents (Python) cli", "platform_token")

        monkeypatch.setattr(local.LocalState, "clear_credentials", blocked)
        result = runner.invoke(app, ["logout"])
        assert result.exit_code == 1
        assert "Signed out" not in result.stdout
        assert sentence in result.stderr


class TestWhoami:
    def test_a_sign_in_this_run_could_not_read_is_the_one_sentence(self, monkeypatch):
        from webagents.cli import account

        kx.note_blocked("webagents (Python) cli", "platform_token")
        result = account.who_am_i()
        assert result.ok is False and result.code == "keychain_needs_terminal"
        assert result.message == FIXTURE["blocked"].format(program=kx.current_program()["program"], item="webagents (Python) cli", command="webagents whoami")

    def test_where_the_other_cli_is_signed_in_it_says_each_keeps_its_own(self, isolated):
        from webagents.cli import account

        record = isolated / ".webagents" / "keychain.json"
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps({"runtimes": {"typescript": {"items": {"webagents (TypeScript) cli": {"platform_token": {"program": "node"}}}}}}))
        result = account.who_am_i()
        assert result.code == "not_signed_in"
        assert result.message == "Not signed in to 127.0.0.1:9. " + FIXTURE["other_runtime"]["signed_in"].format(other="TypeScript CLI")

    def test_whoami_settles_what_a_background_run_could_not_read(self, monkeypatch):
        from webagents.cli import account

        calls = []
        monkeypatch.setattr(account, "settle_keychain", lambda: calls.append(True) or 0)
        runner.invoke(app, ["whoami"])
        assert calls == [True]


class TestSecretsList:
    def test_the_other_clis_keys_are_mentioned_where_this_one_has_none(self, isolated, monkeypatch):
        from webagents.cli.commands import secrets

        keychain = SecretStore(namespace="providers", keyring_module=PlainKeyring(), unavailable_reason="test", file_path=isolated / ".webagents" / "secrets" / "providers.json", quiet=True)
        monkeypatch.setattr(secrets, "_store", lambda quiet=False: keychain)
        record = isolated / ".webagents" / "keychain.json"
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps({"runtimes": {"typescript": {"items": {"webagents (TypeScript) providers": {"OPENAI_API_KEY": {"program": "node"}}}}}}))
        lines = secrets.listing_lines()
        assert lines[:2] == ["No keys stored, and none set in this shell.", "Add one with `webagents secrets set OPENAI_API_KEY`."]
        assert lines[2] == FIXTURE["other_runtime"]["keys"].format(other="TypeScript CLI", command="webagents secrets set NAME")

    def test_the_file_is_shared_so_nothing_is_said_there(self, isolated):
        from webagents.cli.commands import secrets

        record = isolated / ".webagents" / "keychain.json"
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps({"runtimes": {"typescript": {"items": {"webagents (TypeScript) providers": {"OPENAI_API_KEY": {"program": "node"}}}}}}))
        assert secrets.listing_lines() == ["No keys stored, and none set in this shell.", "Add one with `webagents secrets set OPENAI_API_KEY`."]

    def test_remove_names_an_old_item_left_behind(self, isolated, monkeypatch):
        from webagents.cli.commands import secrets

        class Store:
            left_behind = [{"item": "webagents:providers", "account": "OPENAI_API_KEY"}]

            def delete(self, name):
                return True

        monkeypatch.setattr(secrets, "_store", lambda quiet=False: Store())
        result = runner.invoke(app, ["secrets", "remove", "OPENAI_API_KEY"])
        assert result.exit_code == 0
        assert result.stdout.splitlines() == ["Removed OPENAI_API_KEY.", FIXTURE["left_behind"].format(item="webagents:providers", account="OPENAI_API_KEY")]


# ---------------------------------------------------------------------------
# doctor and /status
# ---------------------------------------------------------------------------


class TestDoctorAndStatus:
    def test_doctor_has_a_keychain_line_after_keys(self, isolated):
        from webagents.cli.doctor import run_checks

        checks = run_checks()
        names = [check.name for check in checks]
        assert names.index("keychain") == names.index("keys") + 1
        line = next(check for check in checks if check.name == "keychain")
        assert line.status == "ok"
        assert line.detail == FIXTURE["doctor"]["file"].format(dir="~/.webagents/secrets")

    def test_the_line_never_prints_a_value(self, isolated, monkeypatch):
        monkeypatch.setenv("WEBAGENTS_TOKEN", DUMMY)
        from webagents.cli.doctor import keychain_check

        line = keychain_check()
        assert DUMMY not in line.detail and DUMMY not in (line.fix or "")

    def test_the_chat_status_row_is_the_doctor_detail(self, isolated):
        from webagents.cli.doctor import keychain_check
        from webagents.cli.repl.session import WebAgentsSession

        assert FIXTURE["status_row"] == "Keychain"
        assert WebAgentsSession._keychain_row() == keychain_check().detail


# ---------------------------------------------------------------------------
# serve and the daemon are never asked
# ---------------------------------------------------------------------------


class TestServersForbidDialogs:
    def test_create_server_forbids_them(self):
        from webagents.server.core.app import create_server

        kx.reset_for_tests(interactive=True)
        assert kx.dialogs_allowed() is True
        create_server(agents=[], heartbeat=False, agent_card=False)
        assert kx.dialogs_allowed() is False

    def test_serve_forbids_them_before_the_agent_reads_its_keys(self, monkeypatch):
        from webagents.cli import serve

        kx.reset_for_tests(interactive=True)
        seen = []

        def load(path, **kwargs):
            seen.append(kx.dialogs_allowed())
            raise SystemExit(3)

        monkeypatch.setattr(serve, "load_served_agent", load)
        with pytest.raises(SystemExit):
            serve.serve_command("AGENT.md", 1, None)
        assert seen == [False]

    def test_the_daemon_forbids_them_before_it_loads_an_agent(self, monkeypatch):
        from webagents.cli.commands import daemon

        kx.reset_for_tests(interactive=True)
        seen = []

        def address(port, host=None):
            seen.append(kx.dialogs_allowed())
            raise SystemExit(3)

        monkeypatch.setattr(daemon, "_address", address)
        with pytest.raises(SystemExit):
            daemon.run_daemon()
        assert seen == [False]
