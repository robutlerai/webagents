"""
Which agent a command means (2026-09-24), by the rules and in the words of the
TypeScript CLI (`typescript/src/cli/agent-files.ts`):

  * no name: this folder's AGENT.md, else its only AGENT-<name>.md, else the
    built-in assistant;
  * `-a <name>`: the agent in this folder called that (its `name:`, or the
    `<name>` of AGENT-<name>.md), or the built-in one by its own name.

A name that matches nothing is refused, naming the agents that are here. The
old lookup tried it as a path, then as AGENT-<name>.md, and then gave up
quietly, so a typo opened some other agent under the name that was typed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, NamedTuple, Optional

#: The assistant that runs where there is no agent file, in both CLIs.
BUILT_IN_AGENT = "robutler"


class FolderAgent(NamedTuple):
    name: str
    file: Path
    description: str


class AgentNotFound(Exception):
    """`-a <name>` named no agent in this folder."""


def folder_agents(folder: Path) -> List[FolderAgent]:
    """The agent files in `folder` (AGENT.md and AGENT-<name>.md), with their names."""
    from .loader.hierarchy import load_agent

    try:
        names = sorted(p.name for p in folder.iterdir() if p.name == "AGENT.md" or re.fullmatch(r"AGENT-.+\.md", p.name))
    except OSError:
        return []
    out: List[FolderAgent] = []
    for name in names:
        file = folder / name
        try:
            merged = load_agent(file)
        except Exception:  # noqa: BLE001 - an unreadable file is not an agent to offer
            continue
        out.append(FolderAgent(merged.metadata.name or merged.name, file, merged.metadata.description or ""))
    return out


def default_agent_file(folder: Path) -> Optional[Path]:
    """AGENT.md, else the only AGENT-<name>.md; None means the built-in agent.

    Two or more AGENT-<name>.md and no AGENT.md is a choice for the person to
    make (`-a`, or `/agent` in the chat), not one to guess.
    """
    agent_md = folder / "AGENT.md"
    if agent_md.is_file():
        return agent_md
    named = sorted(p for p in folder.glob("AGENT-*.md") if p.is_file())
    return named[0] if len(named) == 1 else None


def agent_file_for(folder: Path, name: str) -> Optional[Path]:
    """`-a <name>`: that agent's file, or None for the built-in one.

    Raises `AgentNotFound`, whose message names the agents that are here.
    """
    wanted = name.strip()
    agents = folder_agents(folder)
    for agent in agents:
        if agent.name == wanted or agent.file.name == f"AGENT-{wanted}.md":
            return agent.file
    if wanted == BUILT_IN_AGENT:
        return None
    here = ", ".join(a.name for a in agents)
    raise AgentNotFound(
        f"There is no agent called {wanted} in this folder. "
        + (f"Agents here: {here}, and the built-in {BUILT_IN_AGENT}." if here else f"The built-in {BUILT_IN_AGENT} runs without -a.")
    )
