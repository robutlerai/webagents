"""
S-317 and the interrupt, pinned (the ptypass-fixes lane, 2026-09-27).

The real-terminal PTY pass found that every command the Python chat ran,
confined or not, held the owner's terminal as its stdin: a confined command
read "dummyword" typed into the chat and wrote a line onto the screen, and
after an interrupted command Enter stopped sending (srt's node restored the
terminal mode it had inherited). It also found that Esc stopped the reply
while the command ran on for 13 to 18 seconds, until its own timeout.

What these tests hold, from the fixture `tests/fixtures/sandbox/srt.json`
(`interrupt`, scenarios `stdin_is_devnull` and `interrupt_kills_tree`):

  * a command reads EOF at once from stdin, even when the process running it
    has a pipe full of text as its own stdin (a stand-in for the terminal
    that a test cannot own), confined and unconfined;
  * the preflight and `node --version` get `/dev/null` too;
  * an interrupt kills the command and what it started, confined and
    unconfined, through `run_sandboxed(cancel=...)`, through the shell tool
    when its task is cancelled (what the chat's Esc and Ctrl+C do), and
    through the SKILL.md script runner.

The confined cases need real srt and skip, with the reason, where it cannot
run. The TypeScript twin is
`typescript/tests/unit/sandbox/ptypass-fixes-stdin-interrupt.test.ts`.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from webagents.sandbox import (
    INTERRUPTED_RESULT,
    CommandInterrupted,
    backend_status,
    default_policy,
    policy_from_metadata,
    run_sandboxed,
    sandbox_available,
)
from webagents.sandbox import srt

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "sandbox" / "srt.json").read_text())
requires_backend = pytest.mark.skipif(not sandbox_available(), reason=f"no srt here: {backend_status()['reason']}")

CANARY = "CANARY-TYPED-BY-THE-OWNER"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_file(path: Path, seconds: float = 30.0) -> str:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().strip():
            return path.read_text().strip()
        time.sleep(0.05)
    raise AssertionError(f"{path} never appeared")


def _gone_within(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


#: A command that writes its own pid, starts a child that writes its pid,
#: then sleeps: the interrupt must take both.
SLEEPER = (
    "python3 -c \"import os, subprocess, time; "
    "open('parent.pid', 'w').write(str(os.getpid())); "
    "child = subprocess.Popen(['python3', '-c', 'import os, time; open(\\\"child.pid\\\", \\\"w\\\").write(str(os.getpid())); time.sleep(60)']); "
    "time.sleep(60)\""
)


def test_the_fixture_names_the_sentence_and_the_rule():
    assert FIXTURE["interrupt"]["result"] == INTERRUPTED_RESULT
    assert FIXTURE["interrupt"]["stdin"].startswith("/dev/null")
    names = {scenario["name"] for scenario in FIXTURE["scenarios"]}
    assert {"stdin_is_devnull", "interrupt_kills_tree"} <= names


def _run_in_a_child_with_text_on_stdin(tmp_path: Path, declared: object) -> str:
    """Run `cat` through `run_sandboxed` in a separate interpreter whose own
    stdin is a pipe holding CANARY, and return what `cat` printed."""
    helper = (
        "import json, sys\n"
        "from webagents.sandbox import policy_from_metadata, run_sandboxed\n"
        f"policy = policy_from_metadata(json.loads({json.dumps(json.dumps(declared))}), cwd={str(tmp_path)!r})\n"
        "result = run_sandboxed('cat; echo END-OF-CAT', policy, timeout=60)\n"
        "print(json.dumps({'stdout': result.stdout, 'stderr': result.stderr, 'code': result.returncode}))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", helper],
        input=f"{CANARY}\n",
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(tmp_path),
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])["stdout"]


class TestStdinIsDevNull:
    @requires_backend
    def test_a_confined_command_reads_eof_not_the_callers_stdin(self, tmp_path):
        out = _run_in_a_child_with_text_on_stdin(tmp_path, {})
        assert "END-OF-CAT" in out
        assert CANARY not in out

    def test_an_unconfined_command_reads_eof_not_the_callers_stdin(self, tmp_path):
        out = _run_in_a_child_with_text_on_stdin(tmp_path, "off")
        assert "END-OF-CAT" in out
        assert CANARY not in out

    def test_the_preflight_and_node_version_get_devnull(self, monkeypatch, tmp_path):
        seen = []

        def fake_run(argv, **kwargs):
            seen.append(kwargs.get("stdin"))
            return subprocess.CompletedProcess(argv, 0, "v24.7.0\n", "")

        monkeypatch.setattr(srt.subprocess, "run", fake_run)
        srt._NODE_VERSIONS.pop("/nonexistent/node-for-the-test", None)
        srt._node_version("/nonexistent/node-for-the-test")
        location = srt.SrtLocation(node="/x/node", cli="/x/cli.js", version=srt.SRT_VERSION, package_dir="/x", node_from="t", cli_from="t")
        assert srt._preflight(location, {}) == ""
        srt._NODE_VERSIONS.pop("/nonexistent/node-for-the-test", None)
        assert seen == [subprocess.DEVNULL, subprocess.DEVNULL]


def _interrupt_run(tmp_path: Path, policy) -> float:
    """Start SLEEPER through `run_sandboxed(cancel=...)` in a thread, set
    `cancel` once both pids are written, and return how long the call took to
    raise after that; both processes must be gone."""
    cancel = threading.Event()
    outcome: dict = {}

    def target():
        try:
            outcome["result"] = run_sandboxed(SLEEPER, policy, timeout=120, cancel=cancel)
        except BaseException as error:  # noqa: BLE001 - asserted below
            outcome["error"] = error

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    parent = int(_wait_for_file(tmp_path / "parent.pid"))
    child = int(_wait_for_file(tmp_path / "child.pid"))
    assert _alive(parent) and _alive(child)
    started = time.monotonic()
    cancel.set()
    thread.join(timeout=10)
    took = time.monotonic() - started
    assert not thread.is_alive(), "run_sandboxed did not return after the interrupt"
    assert isinstance(outcome.get("error"), CommandInterrupted), outcome
    assert str(outcome["error"]) == INTERRUPTED_RESULT
    assert _gone_within(parent, 3) and _gone_within(child, 3), "the interrupt left a process running"
    return took


class TestTheInterruptKillsTheTree:
    @requires_backend
    def test_confined(self, tmp_path):
        assert _interrupt_run(tmp_path, default_policy(cwd=str(tmp_path))) < 5

    def test_unconfined(self, tmp_path):
        assert _interrupt_run(tmp_path, policy_from_metadata("off", cwd=str(tmp_path))) < 5

    def test_an_interrupt_before_the_start_runs_nothing(self, tmp_path):
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(CommandInterrupted):
            run_sandboxed("touch ran", policy_from_metadata("off", cwd=str(tmp_path)), timeout=10, cancel=cancel)
        assert not (tmp_path / "ran").exists()

    def test_a_timeout_still_kills_the_tree_and_says_so(self, tmp_path):
        policy = policy_from_metadata("off", cwd=str(tmp_path))
        with pytest.raises(subprocess.TimeoutExpired):
            run_sandboxed(SLEEPER, policy, timeout=3)
        child = int(_wait_for_file(tmp_path / "child.pid", 5))
        assert _gone_within(child, 4)


def _shell(tmp_path: Path, sandbox: object):
    from webagents.agents.skills.local.shell.skill import ShellSkill

    config = {"base_dir": str(tmp_path)}
    if sandbox is not None:
        config["sandbox"] = sandbox
    return ShellSkill(config)


async def _cancel_the_tool(tmp_path: Path, skill) -> None:
    task = asyncio.ensure_future(skill.run_command(SLEEPER, timeout=120))
    loop = asyncio.get_running_loop()
    parent = int(await loop.run_in_executor(None, _wait_for_file, tmp_path / "parent.pid"))
    child = int(await loop.run_in_executor(None, _wait_for_file, tmp_path / "child.pid"))
    # What the chat's Esc and Ctrl+C do: cancel the turn's task.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    gone = await loop.run_in_executor(None, lambda: _gone_within(parent, 3) and _gone_within(child, 3))
    assert gone, "cancelling the shell tool left the command running"


class TestTheShellToolStopsWithItsTask:
    @requires_backend
    def test_confined_by_default(self, tmp_path):
        asyncio.run(_cancel_the_tool(tmp_path, _shell(tmp_path, None)))

    def test_sandbox_off(self, tmp_path):
        asyncio.run(_cancel_the_tool(tmp_path, _shell(tmp_path, "off")))

    def test_the_tool_answers_the_sentence_when_the_runner_reports_an_interrupt(self, tmp_path, monkeypatch):
        import webagents.sandbox as sandbox

        def interrupted(*_args, **_kwargs):
            raise CommandInterrupted(INTERRUPTED_RESULT)

        monkeypatch.setattr(sandbox, "run_sandboxed", interrupted)
        assert asyncio.run(_shell(tmp_path, "off").run_command("echo hi")) == INTERRUPTED_RESULT
