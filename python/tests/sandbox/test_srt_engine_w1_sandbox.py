"""
The srt engine's contract (gap-closure plan item 1.2, 2026-09-26), against the
shared fixture `tests/fixtures/sandbox/srt.json` that the TypeScript suite
reads too: the settings a declaration becomes, the environment srt runs with,
the private settings file, timeouts, the `network:` list against a real local
server, failing closed on a missing or wrong-version srt, the strict schema
(S-270) and the owner-only rule for unconfined commands.

Enforcement tests run srt for real (`conftest.py` says where it comes from)
and skip, with the reason, where it cannot run.
"""

import asyncio
import http.server
import json
import os
import socketserver
import stat
import subprocess
import threading
import time
from pathlib import Path

import pytest

from webagents.sandbox import (
    SandboxUnavailable,
    backend_status,
    policy_from_metadata,
    run_sandboxed,
    sandbox_available,
    sandbox_required_reason,
)
from webagents.sandbox import srt as engine

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "sandbox" / "srt.json").read_text())

requires_backend = pytest.mark.skipif(
    not sandbox_available(),
    reason=f"no srt here: {backend_status()['reason']}",
)


@pytest.fixture
def tree(tmp_path):
    work = tmp_path / "work"
    outside = tmp_path / "outside"
    work.mkdir()
    outside.mkdir()
    (outside / "token").write_text("TOP-SECRET\n")
    return work, outside


@pytest.fixture
def site():
    """A local HTTP server on 127.0.0.1, the one host a test can allow-list."""
    root = Path(__file__).resolve().parent

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def do_GET(self):  # noqa: N802 - http.server's name
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"HELLO-FROM-SITE\n")

        def log_message(self, *args):  # silence
            pass

    server = socketserver.TCPServer(("127.0.0.1", 0), Quiet)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


class TestTheFixtureIsTheContract:
    def test_the_engine_is_pinned(self):
        assert engine.SRT_PACKAGE == FIXTURE["engine"]["package"]
        assert engine.SRT_VERSION == FIXTURE["engine"]["version"]
        assert engine.SRT_PATH == FIXTURE["engine"]["srt_path"]
        assert list(engine.LINUX_DEPS) == FIXTURE["engine"]["linux_deps"]

    def test_the_schema_keys_and_presets(self):
        from webagents.cli.loader.schema import INERT_SANDBOX_FIELDS, SandboxConfig
        from webagents.sandbox import ESCALATION_DENY, PRESETS, CREDENTIAL_DIRS

        assert sorted(SandboxConfig.model_fields) == FIXTURE["schema"]["keys"]
        assert sorted(PRESETS) == sorted(FIXTURE["schema"]["presets"])
        assert SandboxConfig().preset == FIXTURE["schema"]["default_preset"]
        assert sorted(INERT_SANDBOX_FIELDS) == FIXTURE["schema"]["inert"]
        assert list(ESCALATION_DENY) == FIXTURE["escalation_deny"]
        assert list(CREDENTIAL_DIRS) == FIXTURE["credential_dirs_unreadable_under_development"]
        for name, preset in FIXTURE["presets"].items():
            assert PRESETS[name][2] is preset["confined"], name

    def test_the_environment_srt_runs_with(self):
        from webagents.sandbox.runner import SECRET_NAME_PARTS

        assert list(SECRET_NAME_PARTS) == FIXTURE["env"]["secret_name_parts"]
        assert list(engine.DROPPED_ENV) == FIXTURE["env"]["dropped_from_srt"]
        given = {name.lower(): "x" for name in FIXTURE["env"]["dropped_from_srt"]}
        given.update({"PATH": "/home/u/bin:/usr/bin", "HOME": "/home/u", "LANG": "C"})
        env = engine.srt_environment(given, "/tmp/scratch")
        assert not any(name.upper() in {n.upper() for n in FIXTURE["env"]["dropped_from_srt"]} for name in env if name != "CLAUDE_CODE_TMPDIR")
        assert env["PATH"] == FIXTURE["engine"]["srt_path"]
        assert env["CLAUDE_CODE_TMPDIR"] == "/tmp/scratch" and env["TMPDIR"] == "/tmp/scratch"
        assert env["HOME"] == "/home/u" and env["LANG"] == "C"
        # The agent's PATH comes back INSIDE, as the first line of the command.
        assert engine.wrapped_command("echo hi", "/home/u/bin:/usr/bin").startswith("export PATH=/home/u/bin:/usr/bin\n")

    def test_the_settings_cases(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        work = tmp_path / "work"
        work.mkdir()
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        for case in FIXTURE["settings_cases"]:
            declared = json.loads(json.dumps(case["declared"]).replace("{work}", str(work)))
            policy = policy_from_metadata(declared, cwd=str(cwd), tmpdir=str(tmp_path / "tmp"))
            settings = engine.build_settings(policy, deps={})
            fill = lambda value: json.loads(  # noqa: E731
                json.dumps(value)
                .replace("{work}", os.path.realpath(work))
                .replace("{scratch}", policy.scratch)
                .replace("{home}", os.path.realpath(home))
                .replace("{cwd}", os.path.realpath(cwd))
            )
            for key, expected in case["expect"].items():
                section, _, field = key.partition(".")
                includes = field.endswith("_includes")
                actual = settings[section][field[: -len("_includes")] if includes else field]
                expected = fill(expected)
                if includes:
                    assert all(item in actual for item in expected), (case["name"], key, actual)
                else:
                    assert actual == expected, (case["name"], key, actual)
            for key in FIXTURE["never_set"]:
                assert key not in settings and key not in settings["network"] and key not in settings["filesystem"]

    def test_network_entries_that_are_not_hosts_are_refused_at_load(self):
        for case in FIXTURE["network_rejected"]:
            with pytest.raises(ValueError, match="sandbox: network entry"):
                policy_from_metadata({"network": [case["entry"]]}, cwd="/tmp")

    def test_a_misspelled_key_is_rejected_with_the_fixture_sentence(self):
        # S-270. The stricter setting, one letter off, ran with the looser default.
        from webagents.cli.loader.schema import AgentFormatError, AgentMetadata

        for case in FIXTURE["unknown_key"]["cases"]:
            with pytest.raises(AgentFormatError) as caught:
                AgentMetadata(name="a", sandbox=case["declared"])
            assert str(caught.value) == case["message"]

    def test_unrestricted_is_an_opt_out_that_is_said(self, tmp_path, capsys):
        from webagents.agents.skills.local.shell.skill import UNRESTRICTED_WARNING, ShellSkill

        assert UNRESTRICTED_WARNING == FIXTURE["unrestricted"]["warning"]
        skill = ShellSkill({"base_dir": str(tmp_path), "sandbox": {"preset": "unrestricted"}})
        assert skill.policy is not None and skill.policy.confined is False
        assert FIXTURE["unrestricted"]["warning"] in capsys.readouterr().err


class TestTheSettingsFileIsPrivate:
    def test_0600_in_a_0700_directory_outside_every_write_root(self, tree):
        work, _ = tree
        policy = policy_from_metadata({"preset": "development", "allowed_folders": [str(work)]}, cwd=str(work))
        path = engine.write_settings(policy)
        try:
            assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
            assert stat.S_IMODE(os.stat(os.path.dirname(path)).st_mode) == 0o700
            real = os.path.realpath(path)
            assert not any(real.startswith(root.rstrip("/") + "/") for root in policy.write_roots), policy.write_roots
            assert json.loads(Path(path).read_text())["filesystem"]["allowWrite"] == policy.write_roots
        finally:
            os.unlink(path)
            os.rmdir(os.path.dirname(path))

    def test_refused_when_every_place_is_writable(self, tmp_path):
        policy = policy_from_metadata({"preset": "development", "allowed_folders": ["/"]}, cwd=str(tmp_path))
        with pytest.raises(SandboxUnavailable, match="no place for the settings file outside the writable folders"):
            engine.settings_base(policy)


class TestItFailsClosed:
    def test_a_missing_srt_is_reported_and_nothing_runs(self, tmp_path, monkeypatch):
        monkeypatch.setenv(engine.ENV_CLI, str(tmp_path / "nowhere" / "cli.js"))
        engine.reset_backend_status()
        try:
            status = backend_status()
            assert status["available"] is False
            assert f"{engine.SRT_PACKAGE}@{engine.SRT_VERSION} was not found" in status["reason"]
            assert "WEBAGENTS_SRT_CLI" in status["reason"]
            with pytest.raises(SandboxUnavailable, match="was not run"):
                run_sandboxed("echo hi", policy_from_metadata({}, cwd=str(tmp_path)))
        finally:
            engine.reset_backend_status()

    def test_a_wrong_version_is_refused(self, tmp_path, monkeypatch):
        package = tmp_path / "node_modules" / "@anthropic-ai" / "sandbox-runtime"
        (package / "dist").mkdir(parents=True)
        (package / "dist" / "cli.js").write_text("process.exit(0)\n")
        (package / "package.json").write_text(json.dumps({"name": engine.SRT_PACKAGE, "version": "0.0.1"}))
        monkeypatch.setenv(engine.ENV_CLI, str(package / "dist" / "cli.js"))
        engine.reset_backend_status()
        try:
            status = backend_status()
            assert status["available"] is False
            assert f"is {engine.SRT_PACKAGE} 0.0.1; this SDK requires exactly {engine.SRT_VERSION}" in status["reason"]
        finally:
            engine.reset_backend_status()

    def test_the_status_carries_where_srt_came_from(self):
        status = backend_status()
        if status["available"]:
            assert status["backend"] == "srt" and status["version"] == engine.SRT_VERSION
            assert "cli.js from" in status["found"] and "node from" in status["found"]
        else:
            assert status["reason"]


class TestOwnerOnlyWithoutASandbox:
    """A caller other than the owner runs commands only confined (defense in
    depth behind the owner-only tool, S-248)."""

    def _as(self, auth):
        from webagents.server.context.context_vars import create_context, set_context

        context = create_context(messages=[])
        context.auth = auth
        set_context(context)

    def test_the_reasons_match_the_fixture(self, tmp_path):
        reasons = FIXTURE["refusals"]["not_owner_reasons"]
        assert sandbox_required_reason(None) == reasons["undeclared"]
        assert sandbox_required_reason(policy_from_metadata({"preset": "unrestricted"}, cwd=str(tmp_path))) == reasons["unrestricted"]

    def test_a_stranger_is_served_confined_by_default_and_refused_when_the_agent_opted_out(self, tmp_path):
        # The sandbox is on by default (2026-09-27): a file with no `sandbox:`
        # confines, so a stranger is served confined where srt runs, and
        # refused with the engine's reason where it does not. The opt-outs
        # refuse them.
        from webagents.access.caller import LOCAL_OWNER, CallerAuth
        from webagents.agents.skills.local.shell.skill import ShellSkill

        prefix = FIXTURE["refusals"]["not_owner"].split("{reason}")[0]
        reasons = FIXTURE["refusals"]["not_owner_reasons"]
        try:
            self._as(CallerAuth(scope="user"))
            by_default = ShellSkill({"base_dir": str(tmp_path), "env": {}})
            assert by_default.sandbox_origin == "default"
            out = asyncio.run(by_default.run_command("echo hi", timeout=20))
            if sandbox_available():
                assert "hi" in out
            else:
                assert out == prefix + f"the sandbox is unavailable: {backend_status()['reason']}"

            off = ShellSkill({"base_dir": str(tmp_path), "sandbox": "off", "env": {}})
            assert off.sandbox_origin == "agent file"
            assert asyncio.run(off.run_command("echo hi")) == prefix + reasons["unrestricted"]
            self._as(None)
            assert asyncio.run(off.run_command("echo hi")).startswith(prefix)
            self._as(LOCAL_OWNER)
            assert "hi" in asyncio.run(off.run_command("echo hi"))

            flag = ShellSkill({"base_dir": str(tmp_path), "env": {"WEBAGENTS_NO_SANDBOX": "1"}})
            assert flag.sandbox_origin == "--no-sandbox"
            self._as(CallerAuth(scope="user"))
            assert asyncio.run(flag.run_command("echo hi")) == prefix + reasons["unrestricted"]
            self._as(LOCAL_OWNER)
            assert "hi" in asyncio.run(flag.run_command("echo hi"))
        finally:
            self._as(LOCAL_OWNER)

    @requires_backend
    def test_a_stranger_is_served_when_the_agent_declares_a_sandbox(self, tree):
        from webagents.access.caller import LOCAL_OWNER, CallerAuth
        from webagents.agents.skills.local.shell.skill import ShellSkill

        work, outside = tree
        skill = ShellSkill({"base_dir": str(work), "sandbox": {"preset": "strict", "allowed_folders": [str(work)]}})
        try:
            self._as(CallerAuth(scope="user"))
            assert "hi" in asyncio.run(skill.run_command("echo hi", timeout=20))
            assert "TOP-SECRET" not in asyncio.run(skill.run_command(f"cat $(echo {outside})/token", timeout=20))
        finally:
            self._as(LOCAL_OWNER)


@requires_backend
class TestTheEngineForReal:
    def test_write_inside_ok_and_outside_denied(self, tree):
        work, outside = tree
        policy = policy_from_metadata({"preset": "strict", "allowed_folders": [str(work)]}, cwd=str(work))
        assert run_sandboxed(f"echo ok > {work}/f && cat {work}/f", policy, timeout=20).stdout.strip() == "ok"
        result = run_sandboxed(f"echo pwned > {outside}/escaped && echo WROTE", policy, timeout=20)
        assert "WROTE" not in result.stdout and not (outside / "escaped").exists()

    def test_deny_read_unreadable_under_development(self, tree, tmp_path, monkeypatch):
        # The credential folders of the home directory, which `development`
        # otherwise reads freely; through `$(...)`, past any string check.
        work, _ = tree
        home = tmp_path / "home"
        (home / ".ssh").mkdir(parents=True)
        (home / ".ssh" / "id_test").write_text("PRIVATE-KEY-CANARY\n")
        monkeypatch.setenv("HOME", str(home))
        policy = policy_from_metadata({"preset": "development", "allowed_folders": [str(work)]}, cwd=str(work))
        assert os.path.realpath(home / ".ssh") in policy.deny_reads
        result = run_sandboxed(f"cat $(echo {home}/.ssh)/id_test", policy, timeout=20)
        assert "PRIVATE-KEY-CANARY" not in result.stdout

    def test_network_denied_without_allowlist_and_reachable_with_it(self, tree, site):
        work, _ = tree
        denied = policy_from_metadata({"preset": "strict", "allowed_folders": [str(work)]}, cwd=str(work))
        direct = run_sandboxed(f"curl -sf -m 4 http://127.0.0.1:{site}/ && echo NET", denied, timeout=30)
        assert "NET" not in direct.stdout
        # Through srt's proxy too (NO_PROXY sends loopback direct otherwise).
        via_proxy = run_sandboxed(f"curl -sf -m 4 --noproxy '' http://127.0.0.1:{site}/ && echo NET", denied, timeout=30)
        assert "NET" not in via_proxy.stdout
        allowed = policy_from_metadata(
            {"preset": "development", "allowed_folders": [str(work)], "network": [f"127.0.0.1:{site}"]}, cwd=str(work)
        )
        served = run_sandboxed(f"curl -sf -m 4 --noproxy '' http://127.0.0.1:{site}/", allowed, timeout=30)
        assert served.stdout.strip() == "HELLO-FROM-SITE", served.stderr

    def test_secret_env_absent_and_passthrough_present(self, tree, monkeypatch):
        work, _ = tree
        monkeypatch.setenv("OPENAI_API_KEY", "sk-FAKE-canary")
        monkeypatch.setenv("GH_TOKEN", "ghp-FAKE-canary")
        policy = policy_from_metadata(
            {"preset": "strict", "allowed_folders": [str(work)], "env_passthrough": ["GH_TOKEN"]}, cwd=str(work)
        )
        out = run_sandboxed('echo "k=[$OPENAI_API_KEY] g=[$GH_TOKEN] s=[$SANDBOX_RUNTIME]"', policy, timeout=20).stdout
        assert "k=[]" in out and "g=[ghp-FAKE-canary]" in out and "s=[1]" in out

    def test_timeout_kills_the_tree_and_reports_a_timeout(self, tree):
        work, _ = tree
        policy = policy_from_metadata({"preset": "development", "allowed_folders": [str(work)]}, cwd=str(work))
        started = time.monotonic()
        with pytest.raises(subprocess.TimeoutExpired):
            run_sandboxed("sleep 3137 & echo DONE; wait", policy, timeout=2)
        assert time.monotonic() - started < 15
        time.sleep(0.5)
        left = subprocess.run(["pgrep", "-f", "sleep 3137"], capture_output=True, text=True).stdout.strip()
        assert left == "", f"the command's children survived the timeout: {left}"

    def test_the_settings_file_is_gone_after_the_command(self, tree):
        work, _ = tree
        policy = policy_from_metadata({"preset": "development", "allowed_folders": [str(work)]}, cwd=str(work))
        base = engine.settings_base(policy)
        before = {name for name in os.listdir(base) if name.startswith("webagents-srt-")}
        run_sandboxed("true", policy, timeout=20)
        after = {name for name in os.listdir(base) if name.startswith("webagents-srt-")}
        assert after <= before


class TestDoctorSaysWhichEngine:
    def _sandbox_check(self, tmp_path, monkeypatch, block):
        (tmp_path / "AGENT.md").write_text(f"---\nname: a\nskills:\n  - shell\n{block}---\n\nBody.\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
        from webagents.cli.doctor import run_checks

        return next(c for c in run_checks(tmp_path) if c.name == "sandbox")

    def test_a_confined_preset_names_srt(self, tmp_path, monkeypatch):
        check = self._sandbox_check(tmp_path, monkeypatch, "sandbox:\n  preset: strict\n")
        if sandbox_available():
            assert check.status == "ok"
            assert f"strict (agent file), enforced by srt {engine.SRT_VERSION}" in check.detail
        else:
            assert check.status == "fail" and "shell commands are refused" in check.detail
            assert engine.SRT_PACKAGE in check.fix

    def test_unrestricted_is_a_warning_with_the_fixture_words(self, tmp_path, monkeypatch):
        check = self._sandbox_check(tmp_path, monkeypatch, "sandbox:\n  preset: unrestricted\n")
        assert check.status == "warn"
        assert check.detail == FIXTURE["unrestricted"]["doctor_detail"]
