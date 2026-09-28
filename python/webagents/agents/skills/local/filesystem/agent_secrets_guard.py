"""
The two sets of names the file tools guard (2026-09-27, the agent-secrets
lane of the gap-closure build). The TypeScript twin is
`skills/filesystem/agent-secrets-guard.ts`; the shared fixture
`tests/fixtures/agent_secrets/file_tools.json` pins the names, the sentences
and the cases, and both suites read it.

WHY. The file tools allowed everything under the agent's folder, and the
folder is where the secrets are: the CLI loads provider keys from its `.env`
(`cli/config_store.py`), and `.webagents/` holds the schedule state and, for
a day, a served agent's private signing key. The default agent pairs these
tools with `rest_request`, which posts to any public URL, so a prompt
injection in the owner's own session could read a key and send it out
(S-312). The same tools could also rewrite the agent's control files: the
real-model pass wrote `AGENT.md` to `preset: unrestricted`, planted an
`AGENT-evil.md`, `WEBAGENTS.md`, `mcp.json`, another skill's `SKILL.md`,
`.env` and a `.git/hooks` file through `write_file`, each a persistent change
the daemon reloads on the write itself (S-314). S-283 had write-denied
exactly that set for confined shell commands; the file tools were a second
path to the same files, with nothing in the way but the model's own caution.

TWO SETS, TWO ANSWERS.

  * SECRETS (`.env`, `.env.*`, anything under `.webagents/`): never read,
    written, searched or moved by a file tool, for any caller, with one
    sentence that says why. A listing may still show the names.
  * CONTROL FILES: the sandbox's `ESCALATION_DENY` and `AGENT_FILE_PATTERNS`
    (imported from `sandbox/policy`, never copied, so the two lists cannot
    drift) plus every `.env*` name and the running agent's own file. A write
    to one of these happens only with the owner's yes in the interactive
    chat, which shows the diff; `serve`, the daemon, `-p` and any caller who
    is not the owner get the refusal sentence. Reads are fine: the built-in
    agent's job of writing agent files keeps working, behind the prompt.

HOW A NAME MATCHES. On any component of the path, not only at the folder's
root: the daemon serves `sub/AGENT-helper.md` as readily as `AGENT.md`, a
nested repository's `.git/hooks` runs on the owner's next git command all
the same, and a whitelisted `~` reaches `~/.bashrc`. Symbolic links are
resolved first (on the deepest ancestor that exists, so a file about to be
created is judged by where it would land), and both the path as written and
the real path are checked, so a link named `config.txt` that points at
`.env` is refused and a link named `.env` that points elsewhere still asks.
"""

from __future__ import annotations

import difflib
import os
from pathlib import Path
from typing import Callable, Awaitable, List, Optional, Sequence

from webagents.sandbox.policy import AGENT_FILE_PATTERNS, ESCALATION_DENY, matches_agent_file_pattern

#: The secrets set (the fixture's `secrets`): exact names, name prefixes and folder names.
SECRET_NAMES: Sequence[str] = (".env",)
SECRET_PREFIXES: Sequence[str] = (".env.",)
SECRET_FOLDERS: Sequence[str] = (".webagents",)

#: Every `.env*` name is a control file too (the fixture's `control.prefixes`).
CONTROL_PREFIXES: Sequence[str] = (".env",)

_SECRET_REFUSAL = (
    "Refused: {path} is where the agent's secrets live (.env files and the .webagents folder), "
    "and the file tools never read, write, search or move them."
)
_CONTROL_REFUSAL = (
    "Refused: {path} is one of the agent's control files (its agent files, WEBAGENTS.md, mcp.json, "
    "its skills, git hooks and .env), which the file tools change only when the owner says yes in "
    "the interactive chat."
)
#: What the chat prints above the diff, and asks below it (the fixture's `control.header` and `control.question`).
CONTROL_HEADER = "The agent wants to change {path}, one of its control files:"
CONTROL_QUESTION = "Make this change? [y/N] "
_CONTROL_DECLINED = "The owner declined the change to {path}; nothing was written."

#: The chat's yes/no for a control-file write: the file as the model named it, and the unified diff to show.
ConfirmControlWrite = Callable[[str, str], Awaitable[bool]]


def secret_refusal(shown: str) -> str:
    return _SECRET_REFUSAL.format(path=shown)


def control_refusal(shown: str) -> str:
    return _CONTROL_REFUSAL.format(path=shown)


def control_declined(shown: str) -> str:
    return _CONTROL_DECLINED.format(path=shown)


def control_entries() -> List[str]:
    """The control entries, as the sandbox denies them (`ESCALATION_DENY`), for the fixture to pin."""
    return list(ESCALATION_DENY)


def control_patterns() -> List[str]:
    """The agent-file name patterns (`AGENT_FILE_PATTERNS`), for the fixture to pin."""
    return list(AGENT_FILE_PATTERNS)


def real_path_of(target: os.PathLike | str) -> Path:
    """The real path of `target`: symbolic links resolved on the deepest
    ancestor that exists, the rest appended as written. Never raises."""
    absolute = Path(os.path.abspath(target))
    try:
        if os.path.lexists(absolute):
            return Path(os.path.realpath(absolute))
    except OSError:
        pass
    parent = absolute.parent
    if parent == absolute:
        return absolute
    return real_path_of(parent) / absolute.name


def _segments_of(target: os.PathLike | str) -> List[str]:
    return [part for part in Path(os.path.abspath(target)).parts if part not in ("/", "\\", "")]


def _contains_entry(segments: Sequence[str], entry: str) -> bool:
    """Whether `entry` (one or more segments, `/`-joined) appears as consecutive segments anywhere in `segments`."""
    wanted = [part for part in entry.split("/") if part]
    if not wanted:
        return False
    width = len(wanted)
    return any(list(segments[start : start + width]) == wanted for start in range(0, len(segments) - width + 1))


def _segments_are_secret(segments: Sequence[str]) -> bool:
    return any(
        segment in SECRET_NAMES or segment in SECRET_FOLDERS or any(segment.startswith(prefix) for prefix in SECRET_PREFIXES)
        for segment in segments
    )


def _segments_are_control(segments: Sequence[str]) -> bool:
    if any(_contains_entry(segments, entry) for entry in ESCALATION_DENY):
        return True
    return any(
        matches_agent_file_pattern(segment) or any(segment.startswith(prefix) for prefix in CONTROL_PREFIXES)
        for segment in segments
    )


def is_secret_path(target: os.PathLike | str) -> bool:
    """Whether `target`, as written or where it really points, is in the secrets set (S-312)."""
    return _segments_are_secret(_segments_of(target)) or _segments_are_secret(_segments_of(real_path_of(target)))


def is_control_path(target: os.PathLike | str, agent_file: Optional[os.PathLike | str] = None) -> bool:
    """Whether `target`, as written or where it really points, is a control
    file (S-314): a sandbox deny entry, an agent-file name, a `.env*` name, or
    the running agent's own file when the loader named it."""
    real = real_path_of(target)
    if _segments_are_control(_segments_of(target)) or _segments_are_control(_segments_of(real)):
        return True
    if agent_file:
        own = real_path_of(agent_file)
        if real == own or os.path.abspath(target) == os.path.abspath(agent_file):
            return True
    return False


#: More lines than this on a side, and the diff is a count rather than a table the chat could not show anyway.
DIFF_LINE_LIMIT = 4000


def unified_diff(before: str, after: str, shown: str) -> str:
    """A unified diff of `before` to `after` for the chat to show, three lines
    of context, headers naming `shown`. A new file is all additions. Very
    large files get a summary instead of a diff."""
    a = before.split("\n") if before else []
    b = after.split("\n") if after else []
    header = [f"--- {shown}", f"+++ {shown}"]
    if len(a) > DIFF_LINE_LIMIT or len(b) > DIFF_LINE_LIMIT:
        return "\n".join(header + [f"@@ {len(a)} lines replaced by {len(b)} lines (too large to show) @@"])
    body = list(difflib.unified_diff(a, b, fromfile=shown, tofile=shown, lineterm="", n=3))
    # difflib writes its own two header lines; keep ours so both SDKs agree on the shape.
    return "\n".join(header + body[2:])
