"""
`daemon.port` / `daemon.host` mean the same thing to every command (2026-09-24).

Found by driving the chat: `webagents config set daemon.port 38770` stored the
value and echoed it back, and then nothing used it. Precedence is now flag >
config > 127.0.0.1:8765 everywhere (`cli/daemon_address.py`): `webagents
daemon` (one foreground command since 2026-09-24, as in the TypeScript CLI),
`webagentsd`, and the daemon client.

And the part that must not regress (S-218): a host that comes from CONFIG has
to be a loopback address. The project config is the file people commit, so a
config value would otherwise be a way for a repository to put the daemon's
unauthenticated management routes on the network. Only an explicit `--host`
binds anywhere else.
"""

import asyncio
import json

import pytest
from typer.testing import CliRunner

from webagents.cli.daemon_address import (
    DaemonAddressError,
    resolve_daemon_address,
)
from webagents.cli.main import app

runner = CliRunner()


@pytest.fixture
def places(tmp_path, monkeypatch):
    """A scratch HOME and a project directory to stand in; returns (home, project)."""
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    # A fake HOME with the OS keystore backend makes macOS offer to reset the
    # login keychain; the CLI reads stored keys for `daemon`, `dev`, `connect`.
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.delenv("WEBAGENTS_PROFILE", raising=False)
    monkeypatch.chdir(project)
    return home, project


def write_config(directory, **values):
    """`directory/.webagents/config.json` with dotted keys: write_config(d, **{"daemon.port": 1})."""
    path = directory / ".webagents" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values))


# -- the resolver -------------------------------------------------------------------------


class TestResolution:
    def test_nothing_configured_is_loopback_8765(self, places):
        address = resolve_daemon_address()
        assert (address.host, address.port) == ("127.0.0.1", 8765)
        assert address.base_url == "http://127.0.0.1:8765"
        assert address.describe() == "127.0.0.1:8765"

    def test_flag_beats_project_beats_global(self, places):
        home, project = places
        write_config(home, **{"daemon.port": 1111, "daemon.host": "localhost"})
        assert resolve_daemon_address().describe() == (
            "localhost:1111 (port from ~/.webagents/config.json, host from ~/.webagents/config.json)"
        )
        write_config(project, **{"daemon.port": 2222})
        address = resolve_daemon_address()
        assert (address.host, address.port) == ("localhost", 2222)
        assert address.port_source == "./.webagents/config.json"
        assert resolve_daemon_address(port=3333).port_source == "--port"
        assert resolve_daemon_address(port=3333).port == 3333

    def test_ipv6_loopback_gets_its_brackets(self, places):
        home, _ = places
        write_config(home, **{"daemon.host": "::1", "daemon.port": 38770})
        assert resolve_daemon_address().base_url == "http://[::1]:38770"

    def test_a_var_reference_resolves(self, places, monkeypatch):
        home, _ = places
        write_config(home, **{"daemon.port": "${WA_TEST_DAEMON_PORT:-4444}"})
        assert resolve_daemon_address().port == 4444
        monkeypatch.setenv("WA_TEST_DAEMON_PORT", "5555")
        assert resolve_daemon_address().port == 5555

    def test_the_profile_has_its_own_config(self, places, monkeypatch):
        home, _ = places
        # A profile's global file is `~/.webagents-<name>/config.json`.
        (home / ".webagents-ci").mkdir()
        (home / ".webagents-ci" / "config.json").write_text(json.dumps({"daemon.port": 6666}))
        assert resolve_daemon_address().port == 8765
        monkeypatch.setenv("WEBAGENTS_PROFILE", "ci")
        assert resolve_daemon_address().port == 6666

    @pytest.mark.parametrize("value", ["abc", 70000, 0, True, "${UNSET_WA_TEST_VAR}"])
    def test_a_bad_port_names_its_file_and_the_fix(self, places, value):
        home, _ = places
        write_config(home, **{"daemon.port": value})
        with pytest.raises(DaemonAddressError) as error:
            resolve_daemon_address()
        assert "daemon.port in ~/.webagents/config.json" in str(error.value)
        assert "webagents config set daemon.port" in error.value.fix


class TestConfiguredHostStaysOnThisMachine:
    """S-218: a configured host must not become a quieter `--host 0.0.0.0`."""

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.5", "example.com"])
    def test_a_non_loopback_host_in_the_project_config_is_refused(self, places, host):
        _, project = places
        write_config(project, **{"daemon.host": host})
        with pytest.raises(DaemonAddressError) as error:
            resolve_daemon_address()
        assert "./.webagents/config.json" in str(error.value)
        assert f"--host {host}" in error.value.fix
        # A plain `config set` writes the global file, which this one outranks.
        assert "webagents config set daemon.host 127.0.0.1 --project" in error.value.fix

    def test_the_global_config_is_held_to_the_same_rule(self, places):
        home, _ = places
        write_config(home, **{"daemon.host": "0.0.0.0"})
        with pytest.raises(DaemonAddressError):
            resolve_daemon_address()

    @pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "::1", "[::1]", "localhost"])
    def test_loopback_hosts_are_honoured(self, places, host):
        home, _ = places
        write_config(home, **{"daemon.host": host})
        assert resolve_daemon_address().is_loopback

    def test_the_flag_still_binds_anywhere(self, places):
        # The explicit opt-in S-218 kept: typed on the command line, on purpose.
        address = resolve_daemon_address(host="0.0.0.0")
        assert (address.host, address.host_source) == ("0.0.0.0", "--host")

    def test_a_host_that_would_rewrite_the_url_is_refused(self, places):
        # `http://127.0.0.1@elsewhere:8765` is a URL for `elsewhere`.
        with pytest.raises(DaemonAddressError):
            resolve_daemon_address(host="127.0.0.1@elsewhere.example")


# -- the client and the auto-start ----------------------------------------------------------


class TestClient:
    def test_a_bare_client_dials_the_configured_port(self, places):
        from webagents.cli.client.daemon_client import DaemonClient

        home, _ = places
        write_config(home, **{"daemon.port": 38770})
        client = DaemonClient()
        try:
            assert client.base_url == "http://127.0.0.1:38770"
        finally:
            asyncio.run(client.close())

    def test_the_config_of_the_agents_own_folder_counts(self, places, tmp_path):
        from webagents.cli.client.daemon_client import DaemonClient

        elsewhere = tmp_path / "elsewhere"
        write_config(elsewhere, **{"daemon.port": 24680})
        client = DaemonClient(working_dir=str(elsewhere))
        try:
            assert client.base_url == "http://127.0.0.1:24680"
        finally:
            asyncio.run(client.close())

    def test_an_explicit_base_url_wins(self, places):
        from webagents.cli.client.daemon_client import DaemonClient

        home, _ = places
        write_config(home, **{"daemon.port": 38770})
        client = DaemonClient("http://localhost:9999")
        try:
            assert client.base_url == "http://localhost:9999"
        finally:
            asyncio.run(client.close())


class TestTheDaemonCommand:
    @pytest.fixture
    def served(self, monkeypatch):
        import uvicorn

        served = []
        monkeypatch.setattr(uvicorn, "run", lambda app_, host, port, **kwargs: served.append((host, port)))
        return served

    def test_it_serves_on_the_configured_address(self, places, served):
        home, _ = places
        write_config(home, **{"daemon.port": 38773, "daemon.host": "::1"})
        result = runner.invoke(app, ["daemon", "--no-cron"])
        assert result.exit_code == 0, result.output
        assert served == [("::1", 38773)]

    def test_the_flags_beat_the_config(self, places, served):
        home, _ = places
        write_config(home, **{"daemon.port": 38773})
        runner.invoke(app, ["daemon", "--port", "5555", "--no-cron"])
        assert served == [("127.0.0.1", 5555)]

    def test_a_configured_host_off_this_machine_is_refused(self, places, served):
        _, project = places
        write_config(project, **{"daemon.host": "0.0.0.0"})
        result = runner.invoke(app, ["daemon"])
        assert result.exit_code == 1, result.output
        assert "--host 0.0.0.0" in result.output
        assert served == []

    def test_an_explicit_host_still_binds_where_it_says(self, places, served):
        result = runner.invoke(app, ["daemon", "--host", "0.0.0.0", "--no-cron"])
        assert result.exit_code == 0, result.output
        assert served == [("0.0.0.0", 8765)]

    def test_webagentsd_is_the_same_command(self, places, served, monkeypatch):
        import sys

        from webagents.daemon_entry import main

        home, _ = places
        write_config(home, **{"daemon.port": 38779})
        monkeypatch.setattr(sys, "argv", ["webagentsd", "--no-cron"])
        with pytest.raises(SystemExit) as done:
            main()
        assert done.value.code in (0, None)
        assert served == [("127.0.0.1", 38779)]


def test_no_command_names_a_literal_default_port():
    # `--port` defaults to None and is resolved; a literal 8765 in a signature
    # is the bug coming back.
    import inspect

    import webagents.cli.main as main_module
    from webagents.daemon_entry import _daemon

    for command in (main_module.daemon, _daemon):
        default = inspect.signature(command).parameters["port"].default
        assert default.default is None, command.__name__
