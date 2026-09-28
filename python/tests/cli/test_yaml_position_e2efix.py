"""
Invalid front matter is refused with the line and column (2026-09-26, the
new-developer e2e run: "is not valid YAML" and nothing else). The sentence is
`loader.not_yaml_at` in `tests/fixtures/cli/chat_edits.json`, which the
TypeScript loader prints too (`tests/unit/cli/yaml-position-e2efix.test.ts`);
the numbers are the FILE's line (the opening fence is line 1) and a 1-based
column, from each SDK's own parser.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from webagents.cli.loader.agent_md import AgentFile
from webagents.cli.loader.schema import AgentFormatError

LOADER = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "chat_edits.json").read_text())["loader"]


def _pattern(template: str, path: Path) -> re.Pattern:
    escaped = re.escape(template).replace(re.escape("{path}"), re.escape(str(path))).replace(re.escape("{line}"), r"(\d+)").replace(re.escape("{column}"), r"(\d+)")
    return re.compile(f"^{escaped}$")


def test_a_file_whose_third_line_is_not_yaml(tmp_path):
    file = tmp_path / "AGENT.md"
    file.write_text("---\nname: a\nskills: [\n  - openai\n---\nBody\n")
    with pytest.raises(AgentFormatError) as raised:
        AgentFile(file)
    match = _pattern(LOADER["not_yaml_at"], file).match(str(raised.value))
    assert match, str(raised.value)
    line, column = int(match.group(1)), int(match.group(2))
    # Inside the front matter of the file: after the opening fence, before the closing one.
    assert 2 <= line <= 5
    assert column >= 1


def test_the_plain_sentence_stays_when_the_parser_gives_no_position(tmp_path, monkeypatch):
    import yaml

    file = tmp_path / "AGENT.md"
    file.write_text("---\nname: a\n---\nBody\n")
    monkeypatch.setattr(yaml, "safe_load", lambda text: (_ for _ in ()).throw(yaml.YAMLError("no mark")))
    with pytest.raises(AgentFormatError) as raised:
        AgentFile(file)
    assert str(raised.value) == LOADER["not_yaml"].format(path=file)
