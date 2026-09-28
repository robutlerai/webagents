"""
`webagents cron list` and `webagents cron run` (plan item 1.7, 2026-09-26):
the lines are `tests/fixtures/cli/cron.json`, the same the TypeScript CLI
prints (`tests/unit/cli/cron-w1daemon.test.ts`); the help words are held to
the TypeScript declarations by `test_cli_parity.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from typer.testing import CliRunner

from webagents.cli.cron_command import (
    EXIT_CODES,
    folder_schedules,
    human_duration,
    list_schedules_command,
    render_schedule_table,
    run_line,
    run_schedule_command,
)
from webagents.cli.daemon.schedule_runner import parse_iso, state_path
from webagents.cli.main import app

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FIXTURE = json.loads((FIXTURES / "cli" / "cron.json").read_text())
DAEMON = json.loads((FIXTURES / "daemon" / "cron.json").read_text())
T0 = parse_iso("2026-09-26T10:00:30Z")


class FakeAgent:
    def __init__(self, reply: Any = "Nothing happened."):
        self.name = "reporter"
        self.reply = reply
        self.calls: List[Any] = []

    async def run(self, messages, **kwargs):
        self.calls.append(messages)
        if isinstance(self.reply, BaseException):
            raise self.reply
        return {"content": self.reply}


def project(tmp_path: Path) -> Path:
    folder = tmp_path / "project"
    folder.mkdir()
    (folder / "AGENT.md").write_text(DAEMON["agent_file"])
    return folder


# -- the words -----------------------------------------------------------------------------------


def test_the_help_words_are_the_fixture_s():
    runner = CliRunner()
    group = runner.invoke(app, ["cron", "-h"]).output
    assert FIXTURE["help"]["group"] in group
    assert FIXTURE["help"]["list"] in group and FIXTURE["help"]["run"] in group
    listing = runner.invoke(app, ["cron", "list", "-h"]).output
    assert FIXTURE["help"]["watch"] in listing
    running = runner.invoke(app, ["cron", "run", "-h"]).output
    assert FIXTURE["help"]["agent"] in running and FIXTURE["help"]["name"] in running and FIXTURE["help"]["watch"] in running


def test_durations():
    for seconds, text in FIXTURE["list"]["durations"].items():
        assert human_duration(int(seconds)) == text, seconds


@pytest.mark.parametrize("case", FIXTURE["list"]["cases"], ids=[c["name"] for c in FIXTURE["list"]["cases"]])
def test_the_table_is_the_fixture_s(case):
    assert render_schedule_table(case["schedules"]) == case["lines"]


@pytest.mark.parametrize("case", FIXTURE["run"]["cases"], ids=[c["line"] for c in FIXTURE["run"]["cases"]])
def test_the_run_line_is_the_fixture_s(case):
    assert run_line(case["record"]) == case["line"]
    assert EXIT_CODES[case["record"]["outcome"]] == case["exit"]
    assert EXIT_CODES == FIXTURE["run"]["exit"]


# -- list ----------------------------------------------------------------------------------------


def test_list_reads_the_folder_as_the_daemon_would_and_writes_nothing(tmp_path):
    folder = project(tmp_path)
    (folder / "AGENT-broken.md").write_text('---\nname: broken\ncron: "0 9 * * *"\n---\nReport.\n')
    (folder / "AGENT-plain.md").write_text("---\nname: plain\n---\nNo schedules here.\n")
    lines: List[str] = []
    errors: List[str] = []
    list_schedules_command(folder, clock=lambda: T0, log=lines.append, error=errors.append)
    assert lines == [
        "AGENT     SCHEDULE      WHEN                       NEXT                  LAST",
        "reporter  daily-report  0 9 * * 1-5 Europe/Berlin  2026-09-28T07:00:00Z  never",
        "reporter  queue         every 30m                  off                   never",
        "reporter  watch         every 1h (heartbeat)       2026-09-26T11:00:30Z  never",
        "reporter  ping          */5 * * * * UTC            2026-09-26T10:05:00Z  never",
    ]
    # `cron`'s own words since 2026-09-28 (`cli/cron.json`, `list.refused_line`).
    assert errors == [f"{folder / 'AGENT-broken.md'}: {DAEMON['string_form_refused']} Its schedules do not run until the file loads."]
    # Nothing written: the daemon's clock is the daemon's.
    assert not (folder / ".webagents").exists()
    # The daemon's walk order: by file name, `AGENT-plain.md` before `AGENT.md`.
    assert [f.name for f in folder_schedules(folder, errors.append)] == ["plain", "reporter"]


def test_list_shows_the_daemon_s_state_and_says_when_there_is_none(tmp_path):
    folder = project(tmp_path)
    state = state_path(folder, "reporter")
    state.parent.mkdir(parents=True)
    state.write_text(
        json.dumps(
            {
                "schedules": {
                    "daily-report": {
                        "spec": "cron 0 9 * * 1-5 Europe/Berlin",
                        "nextRun": "2026-09-28T07:00:00Z",
                        "lastFire": "2026-09-25T07:00:00Z",
                        "lastRun": {"at": "2026-09-25T07:00:00Z", "outcome": "delivered", "detail": "file reports/daily.md"},
                    }
                }
            }
        )
    )
    lines: List[str] = []
    list_schedules_command(folder, clock=lambda: T0, log=lines.append)
    assert lines[1] == "reporter  daily-report  0 9 * * 1-5 Europe/Berlin  2026-09-28T07:00:00Z  delivered 2026-09-25T07:00:00Z"

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "AGENT.md").write_text("---\nname: quiet\n---\nNothing scheduled.\n")
    lines.clear()
    list_schedules_command(empty, log=lines.append)
    assert lines == [FIXTURE["list"]["none"].format(folder=empty.resolve())]

    emitted: List[Dict[str, Any]] = []
    list_schedules_command(folder, json_out=True, clock=lambda: T0, emit=emitted.append)
    assert [s["name"] for s in emitted[0]["schedules"]] == [s["name"] for s in DAEMON["parsed"]]
    assert emitted[0]["schedules"][0]["deliver"] == DAEMON["parsed"][0]["deliver"]


def test_list_through_the_cli(tmp_path, monkeypatch):
    folder = project(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    result = CliRunner().invoke(app, ["cron", "list", "-w", str(folder)])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].split() == FIXTURE["list"]["columns"]
    assert lines[1].startswith("reporter  daily-report  0 9 * * 1-5 Europe/Berlin  ")
    assert [line.split()[1] for line in lines[1:]] == [s["name"] for s in DAEMON["parsed"]]

    as_json = CliRunner().invoke(app, ["--json", "cron", "list", "-w", str(folder)])
    assert as_json.exit_code == 0, as_json.output
    assert [s["name"] for s in json.loads(as_json.output)["data"]["schedules"]] == [s["name"] for s in DAEMON["parsed"]]


# -- run -----------------------------------------------------------------------------------------


def test_run_builds_the_agent_runs_the_schedule_now_and_records_it(tmp_path):
    folder = project(tmp_path)
    agent = FakeAgent("Nothing happened.")
    built: List[Any] = []

    async def build(agent_file, working_dir):
        built.append((Path(agent_file.path), working_dir))
        return agent

    lines: List[str] = []
    code = run_schedule_command("reporter", "daily-report", folder, build=build, clock=lambda: T0, log=lines.append)
    assert code == 0
    assert built == [(folder / "AGENT.md", folder.resolve())]
    assert agent.calls == [[{"role": "user", "content": "Summarize yesterday's activity."}]]
    assert lines == ["reporter/daily-report: delivered (file reports/daily.md)"]
    assert (folder / "reports" / "daily.md").read_text() == DAEMON["file_entry"].format(
        schedule="daily-report", ran_at="2026-09-26T10:00:30Z", content="Nothing happened."
    )
    # Recorded where the daemon's listing reads it; the next fire is the schedule's own.
    state = json.loads(state_path(folder, "reporter").read_text())
    assert state["schedules"]["daily-report"]["lastRun"] == {"at": "2026-09-26T10:00:30Z", "outcome": "delivered", "detail": "file reports/daily.md"}
    assert state["schedules"]["daily-report"]["nextRun"] == "2026-09-28T07:00:00Z"
    listed: List[str] = []
    list_schedules_command(folder, clock=lambda: T0, log=listed.append)
    assert listed[1].endswith("  delivered 2026-09-26T10:00:30Z")

    # A failing turn: the record, and exit 1.
    agent.reply = RuntimeError("no model key")
    lines.clear()
    assert run_schedule_command("reporter", "daily-report", folder, build=build, clock=lambda: T0, log=lines.append) == 1
    assert lines == ["reporter/daily-report: failed (turn failed: no model key)"]

    # As JSON.
    emitted: List[Dict[str, Any]] = []
    agent.reply = "Fine."
    assert run_schedule_command("reporter", "daily-report", folder, json_out=True, build=build, clock=lambda: T0, emit=emitted.append) == 0
    assert emitted == [{"agent": "reporter", "name": "daily-report", "at": "2026-09-26T10:00:30Z", "outcome": "delivered", "detail": "file reports/daily.md"}]


def test_run_names_what_it_cannot_find(tmp_path):
    folder = project(tmp_path)
    errors: List[str] = []
    assert run_schedule_command("nobody", "daily-report", folder, error=errors.append) == 1
    assert run_schedule_command("reporter", "nightly", folder, error=errors.append) == 1
    assert errors == [
        FIXTURE["run"]["no_agent"].format(agent="nobody", folder=folder.resolve()),
        FIXTURE["run"]["no_schedule"].format(name="nightly", agent="reporter", folder=folder.resolve()),
    ]


def test_run_through_the_cli_names_what_it_cannot_find(tmp_path, monkeypatch):
    folder = project(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    result = CliRunner().invoke(app, ["cron", "run", "nobody", "daily-report", "-w", str(folder)])
    assert result.exit_code == 1
    assert FIXTURE["run"]["no_agent"].format(agent="nobody", folder=folder.resolve()) in result.output
