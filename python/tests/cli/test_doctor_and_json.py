"""
`doctor` and the machine-readable output contract (2026-09-23; the doctor
became the TypeScript one's seven checks on 2026-09-24).

`doctor` earns its place by catching what stands between a folder and a
running agent, so the tests assert it CATCHES those things rather than that it
runs. The JSON tests assert the contract a script depends on: one document on
stdout, diagnostics elsewhere, and an error envelope on failure.
"""

import json
import os

import pytest
from typer.testing import CliRunner

from webagents.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Never touch the real profile, and never inherit ambient provider keys."""
    monkeypatch.setenv("WEBAGENTS_PROFILE", "pytest-phase4")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("ROBUTLER_API_URL", "http://127.0.0.1:9")
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)
    for var in (
        "WEBAGENTS_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY", "GOOGLE_GEMINI_API_KEY", "GEMINI_API_KEY",
        "XAI_API_KEY", "FIREWORKS_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    yield
    os.chdir(cwd)


def _checks():
    from webagents.cli.doctor import run_checks

    return {c.name: c for c in run_checks()}


class TestDoctorFindsRealProblems:
    def test_the_seven_checks_in_the_typescript_order(self):
        from webagents.cli.doctor import run_checks

        assert [c.name for c in run_checks()] == ["runtime", "agent", "model", "sign-in", "keys", "sandbox", "config"]

    def test_no_model_is_a_failure_with_both_ways_out(self):
        check = _checks()["model"]
        assert check.status == "fail"
        # Under the profile these tests run in, the commands name it: typed as
        # shown, they fix THIS profile (`cli_command`, 2026-09-25).
        assert check.fix == "`webagents --profile pytest-phase4 login`, or `webagents --profile pytest-phase4 secrets set <NAME>`"

    def test_a_key_is_the_model_it_runs_on(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        check = _checks()["model"]
        assert check.status == "ok"
        assert "with your OPENAI_API_KEY" in check.detail

    def test_reports_an_invalid_config(self, tmp_path):
        (tmp_path / ".webagents").mkdir(exist_ok=True)
        (tmp_path / ".webagents" / "config.json").write_text(json.dumps({"nosuchkey": 1}))
        check = _checks()["config"]
        assert check.status == "fail"
        assert check.detail == "1 problem"
        assert check.fix == "`webagents config validate`"

    def test_every_failure_offers_a_next_step(self, tmp_path):
        (tmp_path / ".webagents").mkdir(exist_ok=True)
        (tmp_path / ".webagents" / "config.json").write_text(json.dumps({"nosuchkey": 1}))
        for check in _checks().values():
            if check.status == "fail":
                assert check.fix, f"{check.name} reports a problem with no fix"

    def test_a_broken_agent_file_does_not_take_doctor_down(self, tmp_path):
        # doctor is what you run when things are broken; it must survive them.
        (tmp_path / "AGENT.md").write_text("---\nname: a\nnosuchfield: 1\n---\nBody\n")
        checks = _checks()
        assert checks["agent"].status == "fail"
        assert "webagents init" in checks["agent"].fix
        assert checks["sandbox"].detail == "not needed: the agent cannot run commands"

    def test_a_shell_without_a_sandbox_is_named(self, tmp_path):
        (tmp_path / "AGENT.md").write_text("---\nname: a\nskills:\n  - shell\n---\nBody\n")
        check = _checks()["sandbox"]
        assert check.status == "warn"
        assert check.detail == "off: shell commands run with your permissions"

    def test_signed_out_is_a_warning_with_the_command(self):
        check = _checks()["sign-in"]
        assert check.status == "warn"
        assert check.detail == "Not signed in to 127.0.0.1:9."
        assert check.fix == "`webagents --profile pytest-phase4 login`"


class TestTheReport:
    def test_it_prints_as_the_typescript_doctor_does(self):
        from webagents.cli.doctor import Check, report_lines

        lines = report_lines([
            Check("runtime", "ok", "Python 3.12.1"),
            Check("model", "fail", "none (no provider key is set)", "`webagents login`"),
        ])
        assert lines == [
            "Checks",
            "  ✓ runtime   Python 3.12.1",
            "  ✗ model     none (no provider key is set)",
            "",
            "To fix",
            "  model       `webagents login`",
        ]


class TestJsonContract:
    def test_stdout_is_exactly_one_document(self):
        result = runner.invoke(app, ["--json", "doctor"])
        # No banner, no table, no colour: it must parse whole.
        parsed = json.loads(result.stdout)
        assert parsed["ok"] is True
        assert [c["name"] for c in parsed["data"]["checks"]][0] == "runtime"

    def test_failure_is_an_envelope_with_a_fix_and_a_nonzero_exit(self):
        result = runner.invoke(app, ["--json", "whoami"])
        assert result.exit_code == 1
        parsed = json.loads(result.stdout)
        assert parsed["ok"] is False
        assert parsed["error"]["code"] == "not_signed_in"
        # An agent reading this is exactly who benefits from being told what
        # to do, not only what broke.
        assert parsed["error"]["fix"]
