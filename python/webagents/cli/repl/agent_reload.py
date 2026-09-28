"""
The reload engine's fingerprint and diff (2026-09-26, interactive-mode spec
3.1): what "the loaded version" is, and what "changed since the chat loaded
it" reports.

S-283 is why a reload is never automatic: the agent's files are control files,
so the chat reads them again only on `/reload` or `/agent edit`, and asks
before it takes a version it did not write whose policy-bearing parts changed.
This holds the pure parts; the sentences are the shared fixture's
(`chat_edits.json`). The TypeScript chat mirrors it in `cli/agent-reload.ts`.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence


def _digest(path: str) -> str:
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return ""


def _canonical(value: Any) -> str:
    return "" if value is None else json.dumps(value, sort_keys=False, default=str)


@dataclass
class LoadedAgent:
    """The parts of a loaded agent whose change the chat reports and, for some, asks about."""

    sha: str
    name: str
    description: str
    model: str
    instructions: str
    skills: List[str]
    agent_skills: List[str]
    access: str
    sandbox: str
    cron: str
    skillmd: List[str]
    lock: str
    mcp: str


def loaded_agent_of(file: str, folder: str, metadata: Any, instructions: str) -> LoadedAgent:
    """The fingerprint of the agent `file` declares, in `folder` (module docstring).

    `metadata` is the merged `AgentMetadata`; `instructions` the merged body."""
    from webagents.agents.skills.local.skillmd.skillmd_loader import discover_skills

    explicit = list(getattr(metadata, "agent_skills", None) or [])
    skillmd = sorted(s.name for s in discover_skills(folder, explicit).skills)
    return LoadedAgent(
        sha=_digest(file),
        name=metadata.name or "",
        description=getattr(metadata, "description", "") or "",
        model=getattr(metadata, "model", None) or "",
        instructions=instructions or "",
        skills=[_entry_name(s) for s in (metadata.skills or [])],
        agent_skills=explicit,
        access=_canonical(getattr(metadata, "access", None)),
        sandbox=_canonical(_sandbox_dump(metadata)),
        cron=_canonical(getattr(metadata, "cron", None)),
        skillmd=skillmd,
        lock=_digest(os.path.join(folder, ".webagents", "skills.lock")),
        mcp=_digest(os.path.join(folder, "mcp.json")),
    )


def _entry_name(entry: Any) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict) and entry:
        return str(next(iter(entry)))
    return str(entry)


def _sandbox_dump(metadata: Any) -> Any:
    sandbox = getattr(metadata, "sandbox", None)
    if sandbox is None:
        return None
    return sandbox.model_dump() if hasattr(sandbox, "model_dump") else sandbox


#: The parts whose change makes the chat ASK before it takes a version it did not write.
_POLICY_PARTS = {"access", "sandbox", "cron", "skills", "agent_skills", "SKILL.md skills", "mcp.json"}


@dataclass
class ReloadDiff:
    parts: List[str] = field(default_factory=list)
    policy_changed: bool = False


def reload_diff(before: LoadedAgent, after: LoadedAgent) -> ReloadDiff:
    """What changed between `before` and `after`, as the lines `/reload` lists (spec 3.1)."""
    diff = ReloadDiff()

    def note(label: str, changed: bool) -> None:
        if not changed:
            return
        diff.parts.append(label)
        if label in _POLICY_PARTS:
            diff.policy_changed = True

    note("model", before.model != after.model)
    note("skills", before.skills != after.skills)
    note("agent_skills", before.agent_skills != after.agent_skills)
    note("access", before.access != after.access)
    note("sandbox", before.sandbox != after.sandbox)
    note("cron", before.cron != after.cron)
    note("description", before.description != after.description)
    note("name", before.name != after.name)
    if before.instructions != after.instructions:
        diff.parts.append(f"instructions ({len(after.instructions.splitlines())} lines)")
    if before.skillmd != after.skillmd:
        added = len([n for n in after.skillmd if n not in before.skillmd])
        gone = len([n for n in before.skillmd if n not in after.skillmd])
        diff.parts.append(f"SKILL.md skills: +{added} -{gone}")
        diff.policy_changed = True
    note("mcp.json", before.mcp != after.mcp)
    return diff


def same_version(a: LoadedAgent, b: LoadedAgent) -> bool:
    """Whether two fingerprints are the same version (the file bytes and everything the folder adds)."""
    return a.sha == b.sha and a.lock == b.lock and a.mcp == b.mcp and a.skillmd == b.skillmd


def tool_change_lines(before: Sequence[str], after: Sequence[str]) -> List[str]:
    """`Tools added: a, b.` / `Tools gone: c.`, empty parts omitted; the tool sets are sorted."""
    added = sorted(t for t in after if t not in before)
    gone = sorted(t for t in before if t not in after)
    lines: List[str] = []
    if added:
        lines.append(f"Tools added: {', '.join(added)}.")
    if gone:
        lines.append(f"Tools gone: {', '.join(gone)}.")
    return lines
