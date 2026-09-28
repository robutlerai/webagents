"""
`WebAgentsSession._ask` (interactive-mode part 2, 2026-09-26): one visible
line read on the event loop, so Ctrl+C at a question answers None at once.

In the e2e run, Ctrl+C at the first-run "Choose 1-3" prompt echoed ^C and the
chat sat until enter, then died with exit 130: `input()` ran in a worker
thread the signal could not interrupt. Driven here with a pipe as stdin and a
real SIGINT to this process.
"""

import asyncio
import os
import signal
import sys

import pytest

from webagents.cli.repl.session import WebAgentsSession

pytestmark = pytest.mark.skipif(os.name != "posix", reason="the loop reader and SIGINT are POSIX")


@pytest.fixture
def piped_stdin(monkeypatch):
    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, "r", buffering=1)
    monkeypatch.setattr(sys, "stdin", reader)
    yield write_fd
    try:
        os.close(write_fd)
    except OSError:
        pass
    reader.close()


def _session(monkeypatch, tmp_path) -> WebAgentsSession:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(exist_ok=True)
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.chdir(tmp_path)
    return WebAgentsSession(agent_path=None, interactive=True)


def test_a_typed_line_is_answered(monkeypatch, tmp_path, piped_stdin):
    session = _session(monkeypatch, tmp_path)

    async def main():
        task = asyncio.ensure_future(session._ask("q: "))
        await asyncio.sleep(0.05)
        os.write(piped_stdin, b"2\n")
        return await asyncio.wait_for(task, 5)

    assert asyncio.run(main()) == "2"


def test_the_end_of_input_answers_none(monkeypatch, tmp_path, piped_stdin):
    session = _session(monkeypatch, tmp_path)

    async def main():
        task = asyncio.ensure_future(session._ask("q: "))
        await asyncio.sleep(0.05)
        os.close(piped_stdin)
        return await asyncio.wait_for(task, 5)

    assert asyncio.run(main()) is None


def test_ctrl_c_answers_none_at_once(monkeypatch, tmp_path, piped_stdin):
    session = _session(monkeypatch, tmp_path)

    async def main():
        task = asyncio.ensure_future(session._ask("Choose 1-3 [1]: "))
        await asyncio.sleep(0.05)
        os.kill(os.getpid(), signal.SIGINT)
        return await asyncio.wait_for(task, 5)

    assert asyncio.run(main()) is None


def test_the_offer_says_it_continues_without_a_model_on_ctrl_c(monkeypatch, tmp_path, piped_stdin):
    from io import StringIO

    from rich.console import Console

    (tmp_path / "AGENT.md").write_text("---\nname: bot\nskills:\n  - openai\n---\nHi.\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ROBUTLER_API_URL", "http://127.0.0.1:9")
    session = _session(monkeypatch, tmp_path)
    session.console = Console(file=StringIO(), width=100, force_terminal=False, color_system=None, record=True)

    async def main():
        await session.initialize()
        assert session.model_problem
        task = asyncio.ensure_future(session.offer_model_access())
        await asyncio.sleep(0.1)
        os.kill(os.getpid(), signal.SIGINT)
        await asyncio.wait_for(task, 5)

    asyncio.run(main())
    assert "✦ Continuing without a model." in session.console.export_text(clear=False)
