"""
`webagents conversations list | delete | prune` (2026-09-29, the owner: "how
do we start/load/delete conversations?").

The chat keeps every conversation under the profile
(`sessions/<folder>/<agent>/<id>.json`, `sessions.py`) and, until this, nothing
but a hand-run `rm` removed one: they piled up forever, the owner's `local`
profile holding 30 across 18 folders, 16 of them test folders. These commands
show and remove them from outside the chat; inside it, `/resume` lists and
continues, and `/resume delete <number>` removes one.

  * `list` shows this folder's conversations, every agent's, newest first,
    with the start of each id; `--all` shows every folder's.
  * `delete <id>` removes the one whose id starts so, after asking, from this
    folder or, with `--all`, from anywhere; `--yes` skips the question, and
    without a terminal the question cannot be asked, so it is required.
  * `prune --older-than <age>` removes the conversations last used longer ago
    than `30d`, `12h`, `2w` or `90m`; `--dry-run` shows them and removes
    nothing.

A copy kept on Robutler (the `session: {backend: robutler}` skill) is never
touched, and the sentence says so. The words and the age grammar are the
shared fixture `tests/fixtures/cli/conversations.json`; the TypeScript CLI is
`src/cli/conversations-command.ts`, rule for rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

WORDS: Dict[str, str] = {
    "none": "No conversations in {folder}.",
    "noneAnywhere": "No conversations kept under this profile.",
    "row": "    {id}  {when}{count}{preview}",
    "hint": "`webagents -r <id>` in its folder continues one; `{delete}` deletes one.",
    "noMatch": "No conversation {id} in {folder}; `{list}` shows them.",
    "noMatchAnywhere": "No conversation {id} under this profile.",
    "ambiguous": "{id} starts {count} conversations' ids; give more of it.",
    "askDelete": "Delete the conversation last used {when} ({count} messages)? [y/N] ",
    "deleted": "Deleted the conversation last used {when}.",
    "remoteStays": "Its copy on Robutler stays; delete it there.",
    "notDeleted": "Nothing deleted.",
    "needsYes": "{command} asks before it deletes, and this is not a terminal; add --yes.",
    "badAge": "--older-than takes a number and a unit: 30d, 12h, 2w or 90m.",
    "pruneNone": "No conversations last used more than {age} ago.",
    "wouldPrune": "Would delete {count} conversations last used more than {age} ago:",
    "askPrune": "Delete {count} conversations last used more than {age} ago? [y/N] ",
    "pruned": "Deleted {count} conversations.",
}

_AGE = re.compile(r"^(\d+)([mhdw])$")
_UNIT_SECONDS = {"m": 60, "h": 3600, "d": 86400, "w": 7 * 86400}
_ID_WIDTH = 8


def fill(word: str, **values: object) -> str:
    text = WORDS[word]
    for key, value in values.items():
        text = text.replace("{" + key + "}", str(value))
    return text


def parse_age(text: str) -> Optional[int]:
    """`30d` as seconds; None for anything else (`badAge`)."""
    match = _AGE.match(text.strip())
    if not match or int(match.group(1)) == 0:
        return None
    return int(match.group(1)) * _UNIT_SECONDS[match.group(2)]


@dataclass
class Kept:
    """One conversation as it is kept."""

    directory: Path
    folder: str
    agent: str
    id: str
    updated_at: str
    messages: int
    preview: str
    on_robutler: bool


def _when(iso: str) -> datetime:
    try:
        then = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return datetime.min.replace(tzinfo=timezone.utc)
    return then if then.tzinfo else then.replace(tzinfo=timezone.utc)


def kept_in(folder_dir: Path) -> List[Kept]:
    """The conversations under one folder's directory, every agent's, newest first.
    A file with no message from the person (a conversation never started) is left out,
    as `/resume` leaves it out."""
    from .sessions import load_session, session_preview

    out: List[Kept] = []
    try:
        agents = sorted(p for p in folder_dir.iterdir() if p.is_dir())
    except OSError:
        return []
    for agent_dir in agents:
        try:
            files = sorted(p for p in agent_dir.iterdir() if p.suffix == ".json" and not p.name.startswith("."))
        except OSError:
            continue
        for file in files:
            session = load_session(agent_dir, file.stem)
            if not session or not any(isinstance(m, dict) and m.get("role") == "user" for m in session["messages"]):
                continue
            metadata = session["metadata"]
            folder = metadata.get("folder") if isinstance(metadata.get("folder"), str) else folder_dir.name
            out.append(
                Kept(
                    directory=agent_dir,
                    folder=folder,
                    agent=str(session.get("agent_name") or agent_dir.name),
                    id=str(session["session_id"]),
                    updated_at=str(session["updated_at"]),
                    messages=len(session["messages"]),
                    preview=session_preview(session["messages"]),
                    on_robutler=isinstance(metadata.get("robutler_chat_id"), str) and bool(metadata.get("robutler_chat_id")),
                )
            )
    out.sort(key=lambda k: _when(k.updated_at), reverse=True)
    return out


def sessions_root(profile: Optional[str] = None) -> Path:
    from .config_store import global_dir, profile_name

    return global_dir(profile_name(profile)) / "sessions"


def folder_dir(folder: Path, profile: Optional[str] = None) -> Path:
    from .sessions import slug_for

    return sessions_root(profile) / slug_for(str(Path(folder).resolve()))


def gather(folder: Path, everywhere: bool, profile: Optional[str] = None) -> List[Kept]:
    """This folder's conversations, or every folder's (`--all`), newest first."""
    if not everywhere:
        # This folder's own path, for the files written before a conversation
        # kept its folder (their directory holds only the lossy slug).
        here = str(Path(folder).resolve())
        kept = kept_in(folder_dir(folder, profile))
        for k in kept:
            k.folder = here
        return kept
    root = sessions_root(profile)
    try:
        folders = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        return []
    out: List[Kept] = []
    for each in folders:
        out.extend(kept_in(each))
    out.sort(key=lambda k: _when(k.updated_at), reverse=True)
    return out


def list_lines(kept: List[Kept], folder: str, everywhere: bool, width: int, delete_command: str, now: Optional[datetime] = None) -> List[str]:
    """What `list` prints: a folder line, an agent line, a row per conversation, then the hint."""
    from .sessions import when_label

    if not kept:
        return [WORDS["noneAnywhere"] if everywhere else fill("none", folder=folder)]
    lines: List[str] = []
    order: Dict[str, Dict[str, List[Kept]]] = {}
    for k in kept:
        order.setdefault(k.folder, {}).setdefault(k.agent, []).append(k)
    for shown_folder, agents in order.items():
        lines.append(shown_folder)
        for agent, rows in agents.items():
            lines.append(f"  {agent}")
            for k in rows:
                when = when_label(k.updated_at, now).ljust(12)
                count = f"{k.messages} messages".ljust(14)
                room = max(10, width - 4 - _ID_WIDTH - 2 - 12 - 14)
                preview = k.preview or "(no text)"
                preview = preview if len(preview) <= room else preview[: room - 1] + "…"
                lines.append(fill("row", id=k.id[:_ID_WIDTH], when=when, count=count, preview=preview))
    lines.append("")
    lines.append(fill("hint", delete=delete_command))
    return lines


class NotOne(Exception):
    """`delete <id>` matched none, or more than one; the message says which."""


def pick(kept: List[Kept], prefix: str, folder: str, everywhere: bool, list_command: str) -> Kept:
    matches = [k for k in kept if k.id.startswith(prefix)] if prefix else []
    if not matches:
        raise NotOne(fill("noMatchAnywhere", id=prefix) if everywhere else fill("noMatch", id=prefix, folder=folder, list=list_command))
    if len(matches) > 1:
        raise NotOne(fill("ambiguous", id=prefix, count=len(matches)))
    return matches[0]


def older_than(kept: List[Kept], seconds: int, now: Optional[datetime] = None) -> List[Kept]:
    current = now or datetime.now(timezone.utc)
    return [k for k in kept if (current - _when(k.updated_at)).total_seconds() > seconds]


def delete(kept: Kept) -> bool:
    from .sessions import delete_session

    return delete_session(kept.directory, kept.id)


def confirm_at_terminal(question: str, ask: Optional[Callable[[str], str]] = None) -> Optional[bool]:
    """A yes to `question`; None when there is no terminal to ask at."""
    import sys

    if ask is None:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return None
        ask = input
    try:
        answer = ask(question)
    except EOFError:
        return False
    return (answer or "").strip().lower() in ("y", "yes")


# -- the commands -----------------------------------------------------------------------------


def _shown(kept: Kept) -> Dict[str, object]:
    return {
        "folder": kept.folder,
        "agent": kept.agent,
        "id": kept.id,
        "updated_at": kept.updated_at,
        "messages": kept.messages,
        "preview": kept.preview,
        "on_robutler": kept.on_robutler,
    }


def list_command(folder: Path, everywhere: bool, json_out: bool, width: Optional[int] = None) -> int:
    """`conversations list [--all]`."""
    import shutil

    from .config_store import cli_command
    from .output import emit

    kept = gather(folder, everywhere)
    if json_out:
        emit({"conversations": [_shown(k) for k in kept]})
        return 0
    columns = width or shutil.get_terminal_size((100, 24)).columns
    for line in list_lines(kept, str(Path(folder).resolve()), everywhere, columns, cli_command("conversations delete <id>")):
        print(line)
    return 0


def delete_command(folder: Path, prefix: str, everywhere: bool, yes: bool, json_out: bool, ask: Optional[Callable[[str], str]] = None) -> int:
    """`conversations delete <id> [--all] [--yes]`."""
    import sys

    from .config_store import cli_command
    from .output import emit, fail
    from .sessions import when_label

    kept = gather(folder, everywhere)
    try:
        chosen = pick(kept, prefix, str(Path(folder).resolve()), everywhere, cli_command("conversations list"))
    except NotOne as refused:
        if json_out:
            fail("not_found" if "No conversation" in str(refused) else "ambiguous", str(refused))
        print(str(refused), file=sys.stderr)
        return 1
    when = when_label(chosen.updated_at)
    if not yes:
        answer = confirm_at_terminal(fill("askDelete", when=when, count=chosen.messages), ask)
        if answer is None:
            message = fill("needsYes", command=cli_command("conversations delete"))
            if json_out:
                fail("needs_yes", message)
            print(message, file=sys.stderr)
            return 1
        if not answer:
            print(WORDS["notDeleted"])
            return 0
    delete(chosen)
    if json_out:
        emit({"deleted": chosen.id, "on_robutler": chosen.on_robutler})
        return 0
    print(fill("deleted", when=when))
    if chosen.on_robutler:
        print(WORDS["remoteStays"])
    return 0


def prune_command(
    folder: Path,
    age: Optional[str],
    everywhere: bool,
    yes: bool,
    dry_run: bool,
    json_out: bool,
    ask: Optional[Callable[[str], str]] = None,
    now: Optional[datetime] = None,
) -> int:
    """`conversations prune --older-than <age> [--all] [--yes] [--dry-run]`."""
    import sys

    from .config_store import cli_command
    from .output import emit, fail
    from .sessions import when_label

    seconds = parse_age(age or "")
    if seconds is None:
        if json_out:
            fail("bad_age", WORDS["badAge"], exit_code=2)
        print(WORDS["badAge"], file=sys.stderr)
        return 2
    old = older_than(gather(folder, everywhere), seconds, now)
    if not old:
        if json_out:
            emit({"would_delete" if dry_run else "deleted": []})
            return 0
        print(fill("pruneNone", age=age))
        return 0
    if dry_run:
        if json_out:
            emit({"would_delete": [_shown(k) for k in old]})
            return 0
        print(fill("wouldPrune", count=len(old), age=age))
        for k in old:
            print(f"  {k.id[:_ID_WIDTH]}  {when_label(k.updated_at, now).ljust(12)}{k.agent}  {k.folder}")
        return 0
    if not yes:
        answer = confirm_at_terminal(fill("askPrune", count=len(old), age=age), ask)
        if answer is None:
            message = fill("needsYes", command=cli_command("conversations prune"))
            if json_out:
                fail("needs_yes", message)
            print(message, file=sys.stderr)
            return 1
        if not answer:
            print(WORDS["notDeleted"])
            return 0
    removed = [k for k in old if delete(k)]
    if json_out:
        emit({"deleted": [k.id for k in removed]})
        return 0
    print(fill("pruned", count=len(removed)))
    if any(k.on_robutler for k in removed):
        print(WORDS["remoteStays"])
    return 0
