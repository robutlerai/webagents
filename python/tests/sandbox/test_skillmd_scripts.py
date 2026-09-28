"""
A SKILL.md skill's scripts run through the kernel sandbox (gap-closure plan
item 1.4, 2026-09-26), under real srt (`conftest.py` says where it comes
from; every test here skips, with the reason, where it cannot run). The
TypeScript suite runs the same probes in
`tests/unit/sandbox/skillmd-scripts.test.ts`, against the same fixture
repository and `tests/fixtures/skillmd/skillmd.json`.

What is proved: the sample script runs and its output comes back as the
fixture says; a script cannot write outside the agent's allowed folders, not
even into its own skill folder; it cannot reach the network unless the
agent's `network:` lists the host; an agent with no `sandbox:` still confines
scripts (a synthesised `strict` policy); a timeout is reported as one.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import shutil
import socketserver
import threading
from pathlib import Path

import pytest

from webagents.agents.skills.local.skillmd import SkillMdSkill, discover_skills
from webagents.sandbox import backend_status, sandbox_available

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "skillmd"
FIXTURE = json.loads((FIXTURES / "skillmd.json").read_text())
REPO = FIXTURES / FIXTURE["sample"]["repo"]
PROBE = FIXTURE["scripts"]["probe"]

requires_backend = pytest.mark.skipif(not sandbox_available(), reason=f"no srt here: {backend_status()['reason']}")


@pytest.fixture
def agent_dir(tmp_path):
    """An agent folder with the sample repository's skills installed under
    `.agents/skills`, and a folder outside it that no script may write."""
    folder = tmp_path / "agent"
    skills = folder / ".agents" / "skills"
    skills.mkdir(parents=True)
    for name in ("pdf", "xlsx"):
        shutil.copytree(REPO / "skills" / name, skills / name)
    (tmp_path / "outside").mkdir()
    return folder


@pytest.fixture
def site():
    """A local HTTP server on 127.0.0.1, the one host a test can allow-list."""

    class Quiet(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server's name
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"HELLO-FROM-SITE\n")

        def log_message(self, *args):
            pass

    server = socketserver.TCPServer(("127.0.0.1", 0), Quiet)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def _skill(agent_dir: Path, sandbox=None) -> SkillMdSkill:
    found = discover_skills(str(agent_dir))
    assert found.names() == ["pdf", "xlsx"]
    return SkillMdSkill({"skills": found.skills, "agent_dir": str(agent_dir), "sandbox": sandbox})


def _run(instance: SkillMdSkill, **kwargs) -> str:
    return asyncio.run(instance.run_skill_script(**kwargs))


@requires_backend
class TestScriptsRunConfined:
    def test_the_sample_script_runs_and_answers(self, agent_dir):
        skill = _skill(agent_dir, sandbox={"preset": "development"})
        out = _run(skill, skill="pdf", script="scripts/fill_form.py", args=["input.pdf", "name=Ada"])
        assert out.strip() == FIXTURE["sample"]["fill_form_output"]

    def test_no_output_is_said(self, agent_dir):
        skill = _skill(agent_dir, sandbox={"preset": "development"})
        (agent_dir / ".agents" / "skills" / "pdf" / "scripts" / "quiet.sh").write_text("exit 0\n")
        assert _run(skill, skill="pdf", script="scripts/quiet.sh") == FIXTURE["scripts"]["no_output"]

    def test_stderr_and_the_exit_code_come_back(self, agent_dir):
        skill = _skill(agent_dir, sandbox={"preset": "development"})
        (agent_dir / ".agents" / "skills" / "pdf" / "scripts" / "fail.sh").write_text("echo out; echo err >&2; exit 3\n")
        out = _run(skill, skill="pdf", script="scripts/fail.sh")
        assert out.startswith("out\n")
        assert FIXTURE["scripts"]["stderr_line"].replace("{stderr}", "err\n") in out
        assert out.endswith(FIXTURE["scripts"]["exit_line"].replace("{code}", "3"))

    def test_cannot_write_outside_the_allowed_folders_nor_into_its_own_folder(self, agent_dir, tmp_path):
        skill = _skill(agent_dir, sandbox={"preset": "development"})
        outside = tmp_path / "outside" / "x.txt"
        own = agent_dir / ".agents" / "skills" / "pdf" / "planted.md"
        inside = agent_dir / "work.txt"
        out = _run(skill, skill="pdf", script="scripts/probe.py", args=["--write", str(outside), "--write", str(own), "--write", str(inside)])
        lines = out.strip().split("\n")
        assert lines[0] == PROBE["ok"]
        assert lines[1].startswith(PROBE["write_refused_starts_with"]), out
        assert lines[2].startswith(PROBE["write_refused_starts_with"]), out
        assert lines[3].startswith(PROBE["write_ok_starts_with"]), out
        assert not outside.exists() and not own.exists() and inside.exists()

    def test_cannot_reach_the_network_unless_the_agent_allows_the_host(self, agent_dir, site):
        url = f"http://127.0.0.1:{site}/"
        closed = _skill(agent_dir, sandbox={"preset": "development"})
        out = _run(closed, skill="pdf", script="scripts/probe.py", args=["--fetch", url])
        assert out.strip().split("\n")[1].startswith(PROBE["fetch_refused_starts_with"]), out
        opened = _skill(agent_dir, sandbox={"preset": "development", "network": [f"127.0.0.1:{site}"]})
        out = _run(opened, skill="pdf", script="scripts/probe.py", args=["--fetch", url])
        assert out.strip().split("\n")[1] == PROBE["fetch_ok_starts_with"] + "HELLO-FROM-SITE", out

    def test_an_agent_with_no_sandbox_still_confines_scripts(self, agent_dir, tmp_path, site):
        skill = _skill(agent_dir, sandbox=None)
        outside = tmp_path / "outside" / "x.txt"
        inside = agent_dir / "work.txt"
        url = f"http://127.0.0.1:{site}/"
        out = _run(skill, skill="pdf", script="scripts/probe.py", args=["--write", str(outside), "--write", str(inside), "--fetch", url])
        lines = out.strip().split("\n")
        assert lines[0] == PROBE["ok"]
        # A synthesised strict policy: no writes anywhere, not even the agent folder, and no network.
        assert lines[1].startswith(PROBE["write_refused_starts_with"]), out
        assert lines[2].startswith(PROBE["write_refused_starts_with"]), out
        assert lines[3].startswith(PROBE["fetch_refused_starts_with"]), out
        assert not outside.exists() and not inside.exists()

    def test_a_timeout_is_reported_as_one(self, agent_dir):
        skill = _skill(agent_dir, sandbox={"preset": "development"})
        (agent_dir / ".agents" / "skills" / "pdf" / "scripts" / "slow.sh").write_text("sleep 30\n")
        out = _run(skill, skill="pdf", script="scripts/slow.sh", timeout=1)
        assert out == FIXTURE["scripts"]["timed_out"].replace("{timeout}", "1")

    def test_secrets_are_withheld_from_scripts(self, agent_dir, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
        skill = _skill(agent_dir, sandbox={"preset": "development"})
        (agent_dir / ".agents" / "skills" / "pdf" / "scripts" / "env.sh").write_text('echo "key=${OPENAI_API_KEY:-withheld}"\n')
        assert _run(skill, skill="pdf", script="scripts/env.sh").strip() == "key=withheld"


class TestWithoutSrt:
    def test_a_declared_sandbox_that_cannot_run_refuses_the_script(self, agent_dir, monkeypatch):
        from webagents.sandbox import srt

        monkeypatch.setenv("WEBAGENTS_SRT_CLI", str(agent_dir / "nowhere" / "cli.js"))
        srt.reset_backend_status()
        try:
            skill = _skill(agent_dir, sandbox={"preset": "development"})
            out = _run(skill, skill="pdf", script="scripts/fill_form.py")
            assert out.startswith(FIXTURE["scripts"]["refusals"]["unavailable_starts_with"])
            assert "was not run" in out
        finally:
            srt.reset_backend_status()
