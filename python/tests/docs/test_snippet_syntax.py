"""Validate that every Python code block in the documentation parses correctly.

Extracts fenced ```python blocks from all markdown files under webagents/docs/
and runs ast.parse() on each one. This catches syntax errors, wrong constructors,
and typos before they reach users.
"""

import ast
import re
from pathlib import Path

import pytest

DOCS_ROOT = Path(__file__).resolve().parents[3] / "docs"

FENCE_RE = re.compile(
    r"^```python\s*\n(.*?)^```",
    re.MULTILINE | re.DOTALL,
)

# Blocks that are intentionally pseudo-code (API signatures without bodies).
# Key: (relative_path, block_index)
PSEUDO_CODE_BLOCKS: set = {
    ("agent/handoffs.md", 1),
    ("agent/router.md", 14),
    ("agent/widgets.md", 12),
    ("agent/widgets.md", 13),
    ("agent/widgets.md", 14),
}


def _collect_snippets():
    """Yield (relative_path, block_index, code) for every Python fenced block."""
    snippets = []
    for md_file in sorted(DOCS_ROOT.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        for idx, match in enumerate(FENCE_RE.finditer(text)):
            rel = str(md_file.relative_to(DOCS_ROOT))
            if (rel, idx) in PSEUDO_CODE_BLOCKS:
                continue
            snippets.append((rel, idx, match.group(1)))
    return snippets


_SNIPPETS = _collect_snippets()


@pytest.mark.docs
@pytest.mark.parametrize(
    "file,block_idx,code",
    _SNIPPETS,
    ids=[f"{s[0]}#{s[1]}" for s in _SNIPPETS],
)
def test_python_snippet_parses(file: str, block_idx: int, code: str):
    """Each Python doc snippet must be valid syntax."""
    try:
        ast.parse(code, filename=f"{file}#block{block_idx}")
    except SyntaxError as exc:
        pytest.fail(
            f"Syntax error in {file} block {block_idx}, "
            f"line {exc.lineno}: {exc.msg}\n\n{code}"
        )
