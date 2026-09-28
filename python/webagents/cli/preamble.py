"""
A few lines the CLI adds after an agent's own instructions (2026-09-28).

WHY. An agent with ``filesystem`` or ``shell`` listed its folder and read its
files on every message, "hi" included: nothing told it not to (the small-talk
rule went into the embedded ROBUTLER.md only), and nothing told it that the
``AGENT.md`` it saw in the listing was its own definition, so it opened that
too. For an agent loaded from a file that has one of those skills, the CLI
adds where its instructions come from, its working folder, and the small-talk
rule. Agent-facing words; the person never sees them.

ONE TEXT IN BOTH SDKS: ``typescript/src/cli/preamble.ts``, pinned by
``tests/fixtures/chat/turn_history.json`` (``preamble``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Optional, Union

#: The skills that let an agent look around its folder.
EXPLORING_SKILLS = ("filesystem", "shell")


def cli_preamble(agent_file: Union[str, Path], folder: Union[str, Path]) -> str:
    return (
        "Notes from the webagents CLI:\n"
        f"- Your instructions above come from {agent_file}; you already have them, so there is no need to open that file to learn who you are.\n"
        f"- Your working folder is {folder}.\n"
        "- Answer greetings, thanks and small talk directly, without tools. Use a tool only when the request needs one, "
        "and do not explore the folder unless you are asked to."
    )


def _skill_names(skills: Optional[Iterable[Any]]) -> Iterable[str]:
    for entry in skills or []:
        if isinstance(entry, str):
            yield entry.strip().lower()
        elif isinstance(entry, dict):
            for key in entry:
                yield str(key).strip().lower()


def with_cli_preamble(instructions: str, agent_file: Optional[Union[str, Path]], skills: Optional[Iterable[Any]]) -> str:
    """``instructions`` with the preamble after them, for an agent loaded from a
    file that can explore its folder; unchanged otherwise (the embedded agent
    has its own rule)."""
    if agent_file is None or not any(n in EXPLORING_SKILLS for n in _skill_names(skills)):
        return instructions
    # abspath, not resolve(): the TypeScript twin does not follow symlinks either.
    path = Path(os.path.abspath(agent_file))
    note = cli_preamble(path, path.parent)
    return f"{instructions.rstrip()}\n\n{note}" if instructions and instructions.strip() else note
