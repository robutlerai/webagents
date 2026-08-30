#!/usr/bin/env python3
"""Inject runnable example files into the docs, verbatim.

The documented quickstart snippets rotted precisely because they were
retyped: the old Python portal-connect snippet was syntactically valid,
passed the syntax test, and connected never. Snippets between the markers
below are GENERATED from the files under `python/examples/` and
`typescript/examples/`, and `python/tests/docs/test_doc_examples.py`
fails when a doc block and its example file drift.

Marker format (in any .md under docs/):

    <!-- BEGIN GENERATED: python/examples/portal_connect_minimal.py -->
    ```python tab="Python"
    ...replaced by this script...
    ```
    <!-- END GENERATED -->

Usage: python3 scripts/sync_doc_examples.py [--check]
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"

MARKER_RE = re.compile(
    r"(<!-- BEGIN GENERATED: (?P<path>[^ ]+) -->\n)(?P<body>.*?)(<!-- END GENERATED -->)",
    re.DOTALL,
)

LANG_BY_SUFFIX = {".py": "python", ".ts": "typescript"}
TAB_BY_SUFFIX = {".py": "Python", ".ts": "TypeScript"}


def example_snippet(rel_path: str) -> str:
    """The doc snippet for one example file: the code after the leading
    module docstring / header comment, verbatim."""
    src_path = REPO / rel_path
    text = src_path.read_text()
    suffix = src_path.suffix
    if suffix == ".py":
        # Strip the module docstring (the doc prose around the snippet does
        # that job); everything after it is the runnable snippet.
        m = re.match(r'\s*""".*?"""\s*\n', text, re.DOTALL)
        code = text[m.end():] if m else text
    elif suffix == ".ts":
        m = re.match(r"\s*/\*\*.*?\*/\s*\n", text, re.DOTALL)
        code = text[m.end():] if m else text
    else:
        code = text
    lang = LANG_BY_SUFFIX.get(suffix, "")
    tab = TAB_BY_SUFFIX.get(suffix, "")
    fence_info = f'{lang} tab="{tab}"' if tab else lang
    return f"```{fence_info}\n{code.strip()}\n```\n"


def sync(check_only: bool = False) -> int:
    drift = 0
    for md in sorted(DOCS.rglob("*.md")):
        text = md.read_text()

        def _replace(m: "re.Match[str]") -> str:
            rel = m.group("path")
            return m.group(1) + example_snippet(rel) + m.group(4)

        new_text = MARKER_RE.sub(_replace, text)
        if new_text != text:
            drift += 1
            if check_only:
                print(f"DRIFT: {md.relative_to(REPO)}")
            else:
                md.write_text(new_text)
                print(f"updated: {md.relative_to(REPO)}")
    if check_only and drift:
        print(f"{drift} doc file(s) out of sync. Run scripts/sync_doc_examples.py")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(sync(check_only="--check" in sys.argv))
