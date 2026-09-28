"""No docs page nests a fenced block inside one with the same fence (the
keychain-ux lane, 2026-09-27, item 7).

WHY. Three pages under `docs/testing/` showed a ```` ```markdown ```` example
that itself held a ```` ```yaml ```` block, all with three backticks. Markdown
closes a fence at the first bare fence of the same length, so the inner
block's closing line ended the OUTER block and the rest of the page rendered
as code, or as prose with stray fences. The outer fences are four backticks
now; this keeps it so for every page, the new ones included.

The rule checked: inside an open fence of N backticks (or tildes), a line that
opens with exactly N of the same character AND carries an info string is an
intended nested fence, and it would close nothing. Every fence must also close.
"""

import re
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parents[3] / "docs"
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def _problems(text: str):
    open_marker = None
    open_line = 0
    for number, line in enumerate(text.splitlines(), 1):
        match = FENCE.match(line)
        if not match:
            continue
        marker, info = match.group(1), match.group(2).strip()
        if open_marker is None:
            open_marker, open_line = marker, number
            continue
        same_char = marker[0] == open_marker[0]
        if same_char and len(marker) >= len(open_marker) and not info:
            open_marker = None
            continue
        if same_char and len(marker) == len(open_marker) and info:
            yield f"line {number}: `{line.strip()}` opens a block inside the one opened at line {open_line} with the same fence"
    if open_marker is not None:
        yield f"line {open_line}: a fence that never closes"


def test_no_page_nests_a_fence_inside_the_same_fence():
    pages = sorted(list(DOCS_ROOT.rglob("*.md")) + list(DOCS_ROOT.rglob("*.mdx")))
    assert pages, DOCS_ROOT
    found = {str(page.relative_to(DOCS_ROOT)): list(_problems(page.read_text(encoding="utf-8"))) for page in pages}
    assert {page: problems for page, problems in found.items() if problems} == {}


def test_the_rule_catches_the_shape_it_is_for():
    nested = "```markdown\n**Strict:**\n```yaml\nstatus: 200\n```\n```\n"
    assert list(_problems(nested))
    fixed = "````markdown\n**Strict:**\n```yaml\nstatus: 200\n```\n````\n"
    assert list(_problems(fixed)) == []
