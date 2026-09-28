"""
The frozen notes in the prompt (gap-closure plan item 2.1, principle 6,
2026-09-26): a bounded rendering of the notes a caller may see, computed once
per session and never updated mid-session, so the provider's prompt cache
holds across the turns of a conversation. The TypeScript twin is
``typescript/src/skills/memory/notes.ts``; both are pinned by the ``notes``
cases of ``tests/fixtures/memory_tool/definition.json``.

The budget counts the entry lines only, so the structure is always whole;
entries come newest first and stop at the first that does not fit, and the
rest are counted rather than shown (the model has ``memory_search``).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

NOTES_HEADING = "## Memory"
NOTES_TITLES = {
    "owner": "Your notes (owner)",
    "shared": "Shared notes",
    "caller": "Notes about this caller",
}
DEFAULT_NOTES_BUDGET = 4000

_NEWLINES = re.compile(r"\s*\n+\s*")


def _one_line(content: str) -> str:
    return _NEWLINES.sub(" ", content).strip()


def render_notes(sections: Sequence[Dict[str, Any]], budget: int) -> str:
    """The prompt text, or an empty string when there is nothing to show.

    ``sections`` are ``{"title": str, "entries": [{"key": str, "content": str}, ...]}``."""
    lines: List[str] = [NOTES_HEADING]
    used = 0
    shown = 0
    more = 0
    stopped = False
    for section in sections:
        fitting: List[str] = []
        for entry in section.get("entries", []):
            if stopped:
                more += 1
                continue
            line = f"- {entry['key']}: {_one_line(str(entry.get('content', '')))}"
            if used + len(line) > budget:
                stopped = True
                more += 1
                continue
            used += len(line)
            fitting.append(line)
        if fitting:
            lines.append(f"{section['title']}:")
            lines.extend(fitting)
            shown += len(fitting)
    if not shown:
        return ""
    if more:
        lines.append(f"({more} more notes; use memory_search to find them.)")
    return "\n".join(lines)
