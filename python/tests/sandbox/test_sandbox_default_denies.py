"""
The sandbox-default lane's proofs against REAL srt (2026-09-27): the built-in
denies (S-311 keychain, S-315 profile folders, S-309 `.env` and `.webagents/`
in the agent's folder), the defaults an agent with no `sandbox:` gets,
`network.local`, `files.deny`, `env:` from `.env`, the refusal hints, the
refused-host capture, the ask-on-first-use loop, and that everyday commands
and the common tools still work. The TypeScript twin is
`typescript/tests/unit/sandbox/sandbox-default-denies.test.ts`; the fixture
`tests/fixtures/sandbox/srt.json` (`builtin_denies`, `hints`, `scenarios`)
names what both prove.

Every test skips, with the reason, where srt cannot run (`conftest.py` says
where it comes from). The keychain probe runs on macOS only, creates ONE item
under a service name of its own with this interpreter's `keyring`, reads it
inside srt, expects nothing back, and deletes it in a finally; it never
reads, lists or touches any other keychain item.
"""

from __future__ import annotations

import asyncio
import http.server
import os
import platform
import shutil
import socketserver
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import pytest

from webagents.agents.skills.local.shell.skill import ShellSkill
from webagents.sandbox import REFUSAL_HINTS, backend_status, default_policy, policy_from_metadata, run_sandboxed, sandbox_available

requires_backend = pytest.mark.skipif(not sandbox_available(), reason=f"no srt here: {backend_status()['reason']}")


def _out(command: str, policy) -> str:
    result = run_sandboxed(command, policy, timeout=60)
    return f"{result.stdout}\n{result.stderr}"


@pytest.fixture
def site():
    class Quiet(http.server.BaseHTTPRequestHandler):
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


@requires_backend
class TestTheBuiltInDeniesForReal:
    def test_dotenv_and_webagents_folder_unreadable_under_both_presets(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        (work / ".env").write_text("PROBE_SECRET_KEY=from-dotenv\n")
        (work / ".env.local").write_text("LOCAL=1\n")
        (work / ".envelope").write_text("not-env\n")
        (work / ".webagents").mkdir()
        (work / ".webagents" / "k.json").write_text("KEYDATA\n")
        for policy in (default_policy(cwd=str(work)), policy_from_metadata({"preset": "strict"}, cwd=str(work))):
            assert "from-dotenv" not in _out("cat .env", policy)
            assert "from-dotenv" not in _out(f"cat $(echo {os.path.realpath(work)})/.env", policy)
            assert "LOCAL=1" not in _out("cat .env.local", policy)
            assert "KEYDATA" not in _out("cat .webagents/k.json", policy)
            assert "not-env" in _out("cat .envelope", policy)

    def test_profile_folder_and_files_deny_unreadable_under_development(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        work = tmp_path / "work"
        (home / ".webagents-local").mkdir(parents=True)
        (home / ".webagents-local" / "history").write_text("HIST-CANARY\n")
        (home / "notes").mkdir()
        (home / "notes" / "n.txt").write_text("NOTE-CANARY\n")
        (home / "plain.txt").write_text("PLAIN-OK\n")
        work.mkdir()
        monkeypatch.setenv("HOME", str(home))
        policy = policy_from_metadata({"preset": "development", "files": {"deny": ["~/notes"]}}, cwd=str(work))
        real_home = os.path.realpath(home)
        assert "HIST-CANARY" not in _out(f"cat $(echo {real_home})/.webagents-local/history", policy)
        assert "NOTE-CANARY" not in _out(f"cat $(echo {real_home})/notes/n.txt", policy)
        # `development` still reads broadly: the rest of the home folder is readable.
        assert "PLAIN-OK" in _out(f"cat $(echo {real_home})/plain.txt", policy)

    def test_credential_files_outside_the_first_list_and_every_home_env_unreadable_s343(self, tmp_path, monkeypatch):
        """S-343 (2026-09-29): the same probe with the first eleven credential
        folders read every one of these. Fake files in a scratch HOME only.
        The TypeScript twin is in `sandbox-default-denies.test.ts`."""
        home = Path(os.path.realpath(tmp_path)) / "home"
        work = home / "work" / "agent"
        canaries = {
            ".config/gh/hosts.yml": "GH-CANARY",
            ".git-credentials": "GITCRED-CANARY",
            ".zsh_history": "HISTORY-CANARY",
            ".codex/auth.json": "CODEX-CANARY",
            ".config/op/config": "OP-CANARY",
            "work/other-project/.env": "OTHER-ENV-CANARY",
            "work/other-project/.env.production": "OTHER-ENV-PROD-CANARY",
            "work/agent/packages/api/.env": "NESTED-ENV-CANARY",
        }
        for relative, canary in canaries.items():
            (home / relative).parent.mkdir(parents=True, exist_ok=True)
            (home / relative).write_text(canary + "\n")
        (home / "work" / "other-project" / "README.md").write_text("README-OK\n")
        monkeypatch.setenv("HOME", str(home))
        policy = default_policy(cwd=str(work))
        for relative, canary in canaries.items():
            assert canary not in _out(f"cat $(echo {home})/{relative}", policy), relative
        # The rest of another project stays readable: only its secrets are denied.
        assert "README-OK" in _out(f"cat $(echo {home})/work/other-project/README.md", policy)

    @pytest.mark.skipif(platform.system() != "Darwin", reason="the keychain probe is macOS only")
    def test_a_keychain_item_this_interpreter_created_is_unreadable_inside(self, tmp_path):
        keyring = pytest.importorskip("keyring")
        # With HOME pointed at a folder with no keychain (a scratch HOME), macOS
        # has no default keychain, and the add below shows its "keychain cannot
        # be found" prompt and waits: this test hung a suite on exactly that on
        # 2026-09-27 (the keychain-ux lane). Skip rather than ask.
        if subprocess.run(["/usr/bin/security", "default-keychain", "-d", "user"], capture_output=True, timeout=10).returncode != 0:
            pytest.skip("macOS has no default keychain for this HOME")
        service = f"webagents-sandbox-probe-{uuid.uuid4().hex[:8]}"
        try:
            keyring.set_password(service, "probe", "PROBE-VALUE-42")
        except Exception as error:  # noqa: BLE001 - no keystore here
            pytest.skip(f"no keystore here: {error}")
        try:
            assert keyring.get_password(service, "probe") == "PROBE-VALUE-42"
            work = tmp_path / "work"
            work.mkdir()
            policy = default_policy(cwd=str(work))
            script = f"import keyring; print('INSIDE=' + repr(keyring.get_password({service!r}, 'probe')))"
            result = run_sandboxed(f"{sys.executable} -c {subprocess.list2cmdline([script])}", policy, timeout=60)
            assert "INSIDE=" in result.stdout, result.stderr
            assert "PROBE-VALUE-42" not in result.stdout
        finally:
            keyring.delete_password(service, "probe")
            assert keyring.get_password(service, "probe") is None


@requires_backend
class TestTheDefaultsAndTheSwitchesForReal:
    def test_no_sandbox_block_means_no_network_no_local_servers_no_listening_and_writes_in_the_folder_only(self, tmp_path, site):
        work = tmp_path / "work"
        outside = tmp_path / "outside"
        work.mkdir()
        outside.mkdir()
        skill = ShellSkill({"base_dir": str(work), "env": {}})
        assert skill.sandbox_state_line() == "development (default)"
        assert "DONE" in asyncio.run(skill.run_command(f"echo written > $(echo {os.path.realpath(outside)})/leak.txt; echo DONE", timeout=30))
        assert not (outside / "leak.txt").exists()
        assert "inside" in asyncio.run(skill.run_command("echo inside > here.txt && cat here.txt", timeout=30))
        local = asyncio.run(skill.run_command(f"curl -sS -m 4 http://127.0.0.1:{site}/", timeout=30))
        assert "HELLO-FROM-SITE" not in local and REFUSAL_HINTS["local"] in local
        listen = asyncio.run(skill.run_command("python3 -c 'import socket; s=socket.socket(); s.bind((\"127.0.0.1\", 0)); s.listen(1); print(\"LISTEN\" + \"-OK\")'", timeout=30))
        assert "LISTEN-OK" not in listen

    def test_network_local_opens_local_servers_and_listening(self, tmp_path, site):
        work = tmp_path / "work"
        work.mkdir()
        skill = ShellSkill({"base_dir": str(work), "sandbox": {"network": {"local": True}}, "env": {}})
        local = asyncio.run(skill.run_command(f"curl -sS -m 4 http://127.0.0.1:{site}/", timeout=30))
        assert "HELLO-FROM-SITE" in local and REFUSAL_HINTS["local"] not in local
        listen = asyncio.run(skill.run_command("python3 -c 'import socket; s=socket.socket(); s.bind((\"127.0.0.1\", 0)); s.listen(1); print(\"LISTEN\" + \"-OK\")'", timeout=30))
        assert "LISTEN-OK" in listen

    def test_a_listed_host_is_reached_by_the_common_tools(self, tmp_path, site):
        work = tmp_path / "work"
        work.mkdir()
        policy = policy_from_metadata({"network": {"hosts": [f"127.0.0.1:{site}"]}}, cwd=str(work))
        assert "HELLO-FROM-SITE" in _out(f"curl -sS -m 4 http://127.0.0.1:{site}/", policy)
        assert "HELLO-FROM-SITE" in _out(f"{sys.executable} -c \"import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:{site}/', timeout=4).read())\"", policy)
        pip = _out(f"{sys.executable} -m pip download --no-deps --index-url http://127.0.0.1:{site}/simple/ -d \"$TMPDIR/pipdl\" nonexistent-probe-pkg 2>&1 | tail -3", policy)
        if "No module named pip" not in pip:
            assert "No matching distribution" in pip or f"Skipping page http://127.0.0.1:{site}" in pip, pip
        if shutil.which("node"):
            assert "HELLO-FROM-SITE" in _out(f"node -e \"fetch('http://127.0.0.1:{site}/').then(r => r.text()).then(t => console.log(t)).catch(e => console.log('ERR', e.cause || e))\"", policy)
        if shutil.which("npm"):
            npm = _out(f"npm view nonexistent-probe-pkg --registry http://127.0.0.1:{site}/ 2>&1 | tail -3", policy)
            assert "HELLO-FROM-SITE" in npm or "not valid JSON" in npm or "invalid json" in npm.lower(), npm

    def test_a_listed_env_name_comes_from_dotenv_which_the_command_cannot_read(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        (work / ".env").write_text("PROBE_SECRET_KEY=from-dotenv\nPLAIN=plain-value\n")
        policy = policy_from_metadata({"env": ["PROBE_SECRET_KEY"]}, cwd=str(work))
        shown = _out('echo "[$PROBE_SECRET_KEY] [$PLAIN]"; cat .env', policy)
        assert "[from-dotenv] []" in shown and "plain-value" not in shown
        skill = ShellSkill({"base_dir": str(work), "env": {}})
        assert REFUSAL_HINTS["env"] in asyncio.run(skill.run_command("cat .env", timeout=30))

    def test_everyday_commands_run_under_the_defaults(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        subprocess.run(["git", "init", "-q", "."], cwd=work, check=True)
        subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-q", "--allow-empty", "-m", "init"], cwd=work, check=True)
        (work / "a.txt").write_text("assert 1\n")
        (work / "t_probe.py").write_text("def test_ok():\n    assert 1 == 1\n")
        policy = default_policy(cwd=str(work))
        assert "GIT-OK" in _out("git status --short && echo GIT-OK", policy)
        assert "a.txt" in _out("ls -la", policy)
        if shutil.which("rg"):
            assert "assert 1" in _out("rg -n assert a.txt", policy)
        assert "1 passed" in _out(f"{sys.executable} -m pytest -q -p no:cacheprovider t_probe.py 2>&1 | tail -1", policy)

    def test_an_unlisted_host_gets_the_hint_the_refused_host_comes_from_srt_and_the_ask_loop_reruns(self, tmp_path, site):
        work = tmp_path / "work"
        work.mkdir()
        asked = []
        written = []
        answer = {"value": "no"}

        class Asker:
            async def ask_host(self, host, command):
                asked.append((host, command))
                return answer["value"]

            async def allow_host_always(self, host):
                written.append(host)

        skill = ShellSkill({"base_dir": str(work), "env": {}, "asker": Asker()})
        # Through srt's proxy (`--noproxy ''`): a direct loopback socket is refused by the
        # kernel and never reaches the proxy, so it is a `local` refusal with no host to ask about.
        command = f"curl -sS -m 4 --noproxy '' http://127.0.0.1:{site}/"
        refused = asyncio.run(skill.run_command(command, timeout=30))
        assert asked == [("127.0.0.1", command)]
        assert "HELLO-FROM-SITE" not in refused and REFUSAL_HINTS["hosts"] in refused
        answer["value"] = "once"
        assert "HELLO-FROM-SITE" in asyncio.run(skill.run_command(command, timeout=30))
        assert written == []
        answer["value"] = "always"
        assert "HELLO-FROM-SITE" in asyncio.run(skill.run_command(command, timeout=30))
        assert written == ["127.0.0.1"]
        # Nobody but the owner is asked.
        from webagents.access.caller import LOCAL_OWNER, CallerAuth
        from webagents.server.context.context_vars import create_context, set_context

        before = len(asked)
        context = create_context(messages=[])
        context.auth = CallerAuth(scope="user")
        set_context(context)
        try:
            asyncio.run(skill.run_command(command, timeout=30))
        finally:
            context.auth = LOCAL_OWNER
            set_context(context)
        assert len(asked) == before
