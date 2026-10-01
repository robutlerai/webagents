"""
The memory index in the prompt (gap-closure plan item 2.1, principle 6,
2026-09-26; an index since 2026-09-29): a bounded list of the notes a caller
may see, one line each, computed once per session and never updated
mid-session, so the provider's prompt cache holds across the turns of a
conversation. The TypeScript twin is ``typescript/src/skills/memory/notes.ts``;
both are pinned by the ``notes`` cases of ``tests/fixtures/memory_tool/definition.json``.

AN INDEX, NOT THE NOTES (2026-09-29, the owner: "is it index based like in
claude?"). The block carried the newest notes' full text, one line each, until
4,000 characters ran out, and the rest were only counted, so the model did not
know what else it had. It is now what Claude Code's ``MEMORY.md`` is: every
note's key and a one-line description (the note's own, or its first line),
newest first, so the model sees everything it remembers and reads a note in
full with ``memory_read`` when it needs it. The budget counts the entry lines
only, so the structure is always whole; the entries that do not fit are counted
(``memory_list`` shows them all).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

NOTES_HEADING = "## Memory"
NOTES_GUIDE = "One line per note you keep, newest first. memory_read gives a note in full; memory_write keeps one, with a one-line description."
NOTES_TITLES = {
    "owner": "Your notes (owner)",
    "shared": "Shared notes",
    "caller": "Notes about this caller",
}
DEFAULT_NOTES_BUDGET = 4000
#: The most of a description (or a first line) an index line shows.
DESCRIPTION_CHARS = 120

_NEWLINES = re.compile(r"\s*\n+\s*")
_MARKS = re.compile(r"^(?:#{1,6}\s+|[-*+]\s+|>\s*)")


def _one_line(content: str) -> str:
    return _NEWLINES.sub(" ", content).strip()


def _cut(text: str, limit: int = DESCRIPTION_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def first_line(content: str) -> str:
    """A note's first line with text, without a heading's or a list's mark."""
    for line in str(content or "").splitlines():
        text = _MARKS.sub("", line.strip()).strip()
        if text:
            return text
    return ""


def note_line(entry: Dict[str, Any]) -> str:
    """One note in the index: `- key: description`, the description the
    note's own or its first line, cut to `DESCRIPTION_CHARS`."""
    description = _one_line(str(entry.get("description") or "")) or first_line(str(entry.get("content") or ""))
    return f"- {entry['key']}: {_cut(description)}" if description else f"- {entry['key']}"


def render_notes(sections: Sequence[Dict[str, Any]], budget: int) -> str:
    """The prompt text, or an empty string when there is nothing to show.

    ``sections`` are ``{"title": str, "entries": [{"key", "description", "content"}, ...]}``."""
    lines: List[str] = [NOTES_HEADING, NOTES_GUIDE]
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
            line = note_line(entry)
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
        lines.append(f"({more} more notes; memory_list shows them all.)")
    return "\n".join(lines)
