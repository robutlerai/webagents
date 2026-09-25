"""Which URLs in an answer are drawn as links, the same in both CLIs (2026-09-25).

The cases are `tests/fixtures/cli/bare_urls.json`, which the TypeScript chat
runs through its `inline()` (`typescript/tests/unit/cli/bare-urls.test.ts`).
This chat drew no bare URL as a link; it now finds them by the TypeScript
pattern, which stops at a quote.
"""

import json
import re
from pathlib import Path

import pytest
from rich.console import Console

from webagents.cli.ui.markdown import ChatMarkdown
from webagents.cli.ui.theme import markdown_styles, theme_for

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "bare_urls.json").read_text())


def _links(markup: str) -> list:
    console = Console(force_terminal=True, color_system="truecolor", width=200, record=False)
    theme = theme_for(console, env={})
    from rich.theme import Theme

    console.push_theme(Theme(markdown_styles(theme)))
    with console.capture() as captured:
        console.print(ChatMarkdown(markup, theme))
    targets = re.findall(r"\x1b\]8;[^;]*;([^\x1b\x07]+)", captured.get())
    # Rich opens a link once per styled run; the same target back to back is one link.
    return [t for i, t in enumerate(targets) if i == 0 or targets[i - 1] != t]


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["line"][:40])
def test_the_same_urls_are_links(case):
    markup = f"- {case['line']}" if case.get("block") == "bullet" else case["line"]
    assert _links(markup) == case["links"]


def test_code_quotes_and_other_headings_are_left_alone():
    fenced = "```\nhttps://a.example/x\n```"
    assert _links(fenced) == []
    assert _links("> https://b.example/y") == []
    assert _links("## https://c.example/z") == []
    assert _links("### https://d.example/w") == ["https://d.example/w"]
