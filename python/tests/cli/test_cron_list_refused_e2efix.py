"""
`webagents cron list` exits non-zero when a file the daemon would refuse sits
in the folder (2026-09-26, the new-developer e2e run): the old string `cron:`
form was said on stderr and the command still exited 0 after "No schedules",
as if the file were fine. Pinned by `list.refused_exit` in
`tests/fixtures/cli/cron.json` and the refusal sentence in
`tests/fixtures/daemon/cron.json`, which the TypeScript suite reads too
(`tests/unit/cli/cron-list-refused-e2efix.test.ts`).
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from webagents.cli.cron_command import LIST_REFUSED_EXIT, list_schedules_command
from webagents.cli.main import app

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
LIST = json.loads((FIXTURES / "cli" / "cron.json").read_text())["list"]
DAEMON = json.loads((FIXTURES / "daemon" / "cron.json").read_text())
runner = CliRunner()


def test_the_old_string_form_is_said_lists_nothing_and_exits_non_zero(tmp_path):
    assert LIST_REFUSED_EXIT == LIST["refused_exit"]
    file = tmp_path / "AGENT.md"
    file.write_text('---\nname: old\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\ncron: "0 9 * * 1-5"\n---\nBody\n')
    out, err = [], []
    code = list_schedules_command(tmp_path, log=out.append, error=err.append)
    assert code == LIST["refused_exit"]
    assert err == [LIST["refused_line"].format(file=file, reason=DAEMON["string_form_refused"])]
    assert out == [LIST["none_refused"].format(folder=tmp_path.resolve())]


def test_every_file_loading_exits_zero(tmp_path):
    (tmp_path / "AGENT.md").write_text("---\nname: fine\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n---\nBody\n")
    err = []
    assert list_schedules_command(tmp_path, log=lambda line: None, error=err.append) == 0
    assert err == []


def test_json_carries_the_count_and_the_command_exits_non_zero(tmp_path, monkeypatch):
    (tmp_path / "AGENT.md").write_text('---\nname: old\ncron: "0 9 * * 1-5"\n---\nBody\n')
    emitted = []
    code = list_schedules_command(tmp_path, json_out=True, emit=emitted.append, error=lambda line: None)
    assert code == LIST["refused_exit"]
    assert emitted == [{"schedules": [], "refused": 1}]
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["cron", "list"])
    assert result.exit_code == LIST["refused_exit"], result.output
    assert DAEMON["string_form_refused"] in result.output
