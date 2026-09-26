"""
`webagents skills add` and `webagents skills remove` (2026-09-25).

THE FILE IS THE PERSON'S. These change the `skills:` list of an agent file and
nothing else: every other byte (comments, key order, the body, the config of a
`- rest: {...}` entry, CRLF line ends) comes back as it was written. So the
list is edited line by line and never re-serialised: `AgentFile._save`
(`loader/agent_md.py`) learned on 2026-09-23 what a `yaml.dump` rewrite costs
(every comment in the front matter). A layout this cannot edit safely, such as
a comment at column zero inside the list or a flow list over several lines, is
refused with the file untouched, and every edit is read back as YAML and
compared with what was meant before it is written.

ONE EDITOR IN BOTH CLIS. The TypeScript CLI's `src/cli/skills-edit.ts` is this,
rule for rule, and `tests/fixtures/cli/skills_edit.json` holds the cases and
the words both suites run.

WHAT A NAME IS. One `skills list` shows, or a model provider's other name
(`claude`, `gemini`, `grok`). `claude` and `anthropic` are one skill to the
loaders, so adding one where the other is listed changes nothing. A name this
SDK cannot load is refused with a "did you mean"; removing takes any name the
file lists, known here or not, since the other SDK may load it.

WHAT A SKILL STILL NEEDS is said after an add, and only when it is missing: a
model provider's key, the sign-in for Robutler's models (`proxy`) or for
searching the platform from the chat (`discovery`).
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import yaml

def canonical_skill_name(name: str) -> str:
    """One skill, however a file names it: a provider's other names are the provider (`claude` is `anthropic`)."""
    from webagents.agents.skills.core.llm.providers import find_provider

    lower = name.strip().lower()
    provider = find_provider(lower)
    return provider.id if provider is not None else lower


#: Why a list could not be edited; each has one sentence (`problem_message`).
PROBLEMS = ("unclosed", "not_yaml", "not_list", "layout")


def problem_message(problem: str, file: str) -> str:
    if problem == "unclosed":
        return f"{file} opens its front matter with --- and never closes it."
    if problem == "not_yaml":
        return f"The front matter of {file} is not valid YAML. Fix it, then try again."
    if problem == "not_list":
        return f"skills: in {file} is not a list. Fix it, then try again."
    return f"The skills: list in {file} is laid out in a way this command cannot change safely. Edit it by hand."


class SkillListError(Exception):
    def __init__(self, problem: str, file: str):
        super().__init__(problem_message(problem, file))
        self.problem = problem


@dataclass
class SkillListEdit:
    """What an edit did. `text` is the file as it is to be written; unchanged when `changed` is false."""

    text: str
    changed: bool
    #: Names written into the file, as they were asked for.
    added: List[str] = field(default_factory=list)
    #: Asked-for names the file already lists, as the file writes them.
    already: List[str] = field(default_factory=list)
    #: Names taken out, as the file wrote them.
    removed: List[str] = field(default_factory=list)
    #: Asked-for names the file does not list, as they were asked for.
    absent: List[str] = field(default_factory=list)
    #: The file's skill names after the edit.
    skills: List[str] = field(default_factory=list)


def _entry_name(entry: Any) -> Optional[str]:
    """The name a `skills:` entry uses: `shell`, or the key of `{shell: {...}}`; None for anything else."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict) and len(entry) == 1:
        return str(next(iter(entry)))
    return None


def _name_for_file(file: str) -> str:
    """The name an agent file with no front matter goes by in both loaders:
    AGENT.md is `default`, AGENT-<name>.md is `<name>`. Written into the front
    matter an add creates, because a front matter without `name:` is named
    `assistant` by this SDK's loader."""
    base = Path(file).name
    base = base[:-3] if base.lower().endswith(".md") else base
    if base == "AGENT":
        return "default"
    return base[len("AGENT-"):] if base.startswith("AGENT-") else base


def _is_fence(line: str) -> bool:
    return line.rstrip(" \t") == "---"


def _is_trivia(line: str) -> bool:
    return line.strip() == "" or line.strip().startswith("#")


@dataclass
class _Parsed:
    bom: str
    nl: str
    lines: List[str]
    #: Index of the closing `---`, or -1 when the file has no front matter.
    close: int
    data: Dict[str, Any]


def _parse(text: str, file: str) -> _Parsed:
    bom = "﻿" if text.startswith("﻿") else ""
    body = text[1:] if bom else text
    nl = "\r\n" if "\r\n" in body else "\n"
    lines = body.replace("\r\n", "\n").split("\n")
    if not _is_fence(lines[0]):
        return _Parsed(bom, nl, lines, -1, {})
    close = next((i for i, line in enumerate(lines) if i > 0 and _is_fence(line)), -1)
    if close < 0:
        raise SkillListError("unclosed", file)
    try:
        data = yaml.safe_load("\n".join(lines[1:close]))
    except yaml.YAMLError:
        raise SkillListError("not_yaml", file) from None
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise SkillListError("not_yaml", file)
    return _Parsed(bom, nl, lines, close, data)


def _entries_of(data: Dict[str, Any], file: str) -> List[Any]:
    """The `skills:` entries of parsed front matter; raises when it is not a list."""
    raw = data.get("skills")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise SkillListError("not_list", file)
    return raw


def _flow_list(line: str, open_at: int) -> Optional[Tuple[int, List[int]]]:
    """The position of a one-line flow list's `]` and of its top-level commas,
    or None when the `]` is not on this line."""
    depth = 0
    quote: Optional[str] = None
    commas: List[int] = []
    i = open_at
    while i < len(line):
        c = line[i]
        if quote:
            if quote == '"' and c == "\\":
                i += 1
            elif c == quote:
                if quote == "'" and line[i + 1:i + 2] == "'":
                    i += 1
                else:
                    quote = None
            i += 1
            continue
        if c in "\"'":
            quote = c
        elif c in "[{":
            depth += 1
        elif c in "]}":
            depth -= 1
            if depth == 0:
                return i, commas
        elif c == "," and depth == 1:
            commas.append(i)
        i += 1
    return None


def _empty_list_line(line: str, colon: int) -> str:
    """`skills: []`, keeping a comment the key line carried."""
    rest = line[colon + 1:].strip()
    return f"{line[:colon + 1]} []" + (f" {rest}" if rest.startswith("#") else "")


def edit_skill_list(text: str, action: str, requested: Sequence[str], file: str) -> SkillListEdit:
    """Add or remove skills in the text of the agent file `file` (module
    docstring). Raises `SkillListError` rather than write anything it cannot
    stand behind."""
    shown = Path(file).name

    def same(a: str, b: str) -> bool:
        return canonical_skill_name(a) == canonical_skill_name(b)

    parsed = _parse(text, shown)
    entries = _entries_of(parsed.data, shown)
    names = [_entry_name(e) for e in entries]

    added: List[str] = []
    already: List[str] = []
    removed: List[str] = []
    absent: List[str] = []
    dropped: Set[int] = set()

    for wanted in requested:
        hits = [i for i, name in enumerate(names) if name is not None and same(name, wanted)]
        if action == "add":
            if hits:
                written = names[hits[0]]
                if written not in already:
                    already.append(written)
            elif not any(same(name, wanted) for name in added):
                added.append(wanted)
        elif not hits:
            if not any(same(name, wanted) for name in absent):
                absent.append(wanted)
        else:
            for i in hits:
                if i in dropped:
                    continue
                dropped.add(i)
                if names[i] not in removed:
                    removed.append(names[i])

    after = [name for i, name in enumerate(names) if i not in dropped] + added
    result = dict(added=added, already=already, removed=removed, absent=absent, skills=[n for n in after if n is not None])
    if not added and not dropped:
        return SkillListEdit(text=text, changed=False, **result)

    lines = list(parsed.lines)
    if parsed.close < 0:
        # No front matter: one is written, naming the agent as the loaders did.
        lines[:0] = ["---", f"name: {_name_for_file(file)}", "skills:", *[f"  - {n}" for n in added], "---", ""]
    else:
        _edit_front_matter(lines, parsed.close, parsed.data, len(entries), dropped, added, shown)
    edited = parsed.bom + parsed.nl.join(lines)

    # Read back as YAML: the list must be exactly what was meant, or nothing is written.
    check = _parse(edited, shown)
    got = [_entry_name(e) for e in _entries_of(check.data, shown)]
    if got != after:
        raise SkillListError("layout", shown)
    return SkillListEdit(text=edited, changed=True, **result)


def _edit_front_matter(
    lines: List[str],
    close: int,
    data: Dict[str, Any],
    count: int,
    dropped: Set[int],
    added: List[str],
    file: str,
) -> None:
    """The line edit itself, in place on `lines` (front matter is lines 1 to `close - 1`)."""
    key = next((i for i in range(1, close) if re.match(r"skills[ \t]*:", lines[i])), -1)
    if key < 0:
        # Written some other way (quoted, or a complex key): not this command's to touch.
        if "skills" in data:
            raise SkillListError("layout", file)
        at = close
        while at > 1 and lines[at - 1].strip() == "":
            at -= 1
        lines[at:at] = ["skills:", *[f"  - {n}" for n in added]]
        return

    line = lines[key]
    colon = line.index(":")
    value = line[colon + 1:].strip()

    if value.startswith("["):
        open_at = line.index("[", colon)
        found = _flow_list(line, open_at)
        if found is None:
            raise SkillListError("layout", file)
        close_at, commas = found
        tail = line[close_at + 1:].strip()
        if tail and not tail.startswith("#"):
            raise SkillListError("layout", file)
        bounds = [open_at, *commas, close_at]
        items = [line[bounds[i] + 1:end].strip() for i, end in enumerate(bounds[1:])]
        if items and items[-1] == "":
            items.pop()
        if len(items) != count or any(item == "" for item in items):
            raise SkillListError("layout", file)
        items = [item for i, item in enumerate(items) if i not in dropped] + added
        lines[key] = f"{line[:open_at]}[{', '.join(items)}]{line[close_at + 1:]}"
        return
    if value and not value.startswith("#"):
        raise SkillListError("layout", file)

    # A block list: the lines after the key that are blank, indented, or items
    # at column zero, less the blanks and comments that lead into what follows.
    end = key + 1
    while end < close and (lines[end].strip() == "" or lines[end][:1] in (" ", "\t", "-")):
        end += 1
    while end > key + 1 and _is_trivia(lines[end - 1]):
        end -= 1
    indent: Optional[str] = None
    starts: List[int] = []
    for i in range(key + 1, end):
        match = re.match(r"( *)-(?:[ \t]|$)", lines[i])
        if not match:
            continue
        if indent is None:
            indent = match.group(1)
        if match.group(1) == indent:
            starts.append(i)
    if len(starts) != count:
        raise SkillListError("layout", file)

    def entry_end(j: int) -> int:
        """An entry runs to the next one, less the blanks and comments before it."""
        stop = starts[j + 1] if j + 1 < len(starts) else end
        while stop > starts[j] + 1 and _is_trivia(lines[stop - 1]):
            stop -= 1
        return stop

    if added:
        at = entry_end(len(starts) - 1) if starts else key + 1
        lines[at:at] = [f"{indent if indent is not None else '  '}- {n}" for n in added]
    for j in sorted(dropped, reverse=True):
        del lines[starts[j]:entry_end(j)]
    if len(dropped) == count and not added:
        lines[key] = _empty_list_line(line, colon)


# ============================================================================
# The command
# ============================================================================


@dataclass
class SkillsFacts:
    """What this machine has, for saying what an added skill still needs."""

    #: Whether a key variable has a value here: set in this shell, or stored with `secrets set`.
    has_key: Callable[[str], bool]
    #: Whether `webagents login` has signed this profile in.
    signed_in: bool


def spoken_list(names: Sequence[str]) -> str:
    """`a`, `a and b`, `a, b and c`."""
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _say(line: str) -> None:
    print(line)


def _complain(line: str) -> None:
    print(line, file=sys.stderr)


def skills_command(
    action: str,
    requested: Sequence[str],
    agent: Optional[str] = None,
    folder: Optional[Path] = None,
    facts: Optional[Callable[[], SkillsFacts]] = None,
    out: Callable[[str], None] = _say,
    err: Callable[[str], None] = _complain,
) -> int:
    """Run `skills add` or `skills remove`; returns the exit code. Refusals
    (an unknown name, no agent file, a list it cannot edit) change nothing."""
    from webagents.agents.skills.core.llm.providers import find_provider

    from .agent_builder import SKILL_CLASSES
    from .config_store import cli_command
    from .help_format import suggest_similar

    folder = folder or Path.cwd()
    file = _choose_agent_file(folder, agent, err, cli_command)
    if file is None:
        return 1
    shown = file.name

    known = sorted(SKILL_CLASSES)
    canonical = canonical_skill_name

    wanted = [n for n in (name.strip().lower() for name in requested) if n]

    try:
        text = file.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        err(f"Could not read {shown}: {error}")
        return 1

    # Every name is checked before anything is written.
    listed: List[str] = []
    if action == "remove":
        try:
            listed = edit_skill_list(text, "add", [], str(file)).skills
        except SkillListError as error:
            err(str(error))
            return 1
    unknown = [
        n for n in wanted
        if canonical(n) not in known and not any(canonical(have) == canonical(n) for have in listed)
    ]
    if unknown:
        candidates = list(dict.fromkeys([*listed, *known])) if action == "remove" else known
        for name in unknown:
            err(f"Unknown skill '{name}'.{suggest_similar(name, candidates)}")
        err(f"Run `{cli_command('skills list')}` for the skills an agent file can name.")
        return 1

    try:
        edit = edit_skill_list(text, action, wanted, str(file))
    except SkillListError as error:
        err(str(error))
        return 1
    if edit.changed:
        try:
            # Bytes, so the file's own line ends are the ones written.
            file.write_bytes(edit.text.encode("utf-8"))
        except OSError as error:
            err(f"Could not write {shown}: {error}")
            return 1

    if edit.added:
        out(f"Added {spoken_list(edit.added)} to {shown}.")
    if edit.already:
        out(f"{shown} already names {spoken_list(edit.already)}.")
    if edit.removed:
        out(f"Removed {spoken_list(edit.removed)} from {shown}.")
    if edit.absent:
        out(f"{shown} does not name {spoken_list(edit.absent)}.")
    out(f"Skills: {', '.join(edit.skills) if edit.skills else 'none'}")

    # What an added skill still needs, asked of this machine only when one could need something.
    def needs_something(name: str) -> bool:
        provider = find_provider(canonical(name))
        return (provider is not None and provider.credential != "local") or canonical(name) == "discovery"

    needy = [n for n in edit.added if needs_something(n)]
    if needy:
        have = (facts or machine_facts)()
        for name in needy:
            provider = find_provider(canonical(name))
            if provider is not None and provider.credential == "api_key" and provider.env_vars:
                if not any(have.has_key(variable) for variable in provider.env_vars):
                    first = provider.env_vars[0]
                    out(f"{name} needs {first}: add it with `{cli_command(f'secrets set {first}')}`.")
            elif not have.signed_in:
                out(f"{name} needs you signed in to Robutler: run `{cli_command('login')}`.")
    return 0


def _choose_agent_file(
    folder: Path,
    agent: Optional[str],
    err: Callable[[str], None],
    cli_command: Callable[..., str],
) -> Optional[Path]:
    """The file `-a` names, else this folder's one agent file; None (said) when there is none to change."""
    from .agent_files import BUILT_IN_AGENT, AgentNotFound, agent_file_for, folder_agents

    if agent:
        try:
            file = agent_file_for(folder, agent)
        except AgentNotFound as error:
            err(str(error))
            return None
        if file is None:
            err(f"The built-in {BUILT_IN_AGENT} has no agent file to change. Create one with `{cli_command('init')}`.")
            return None
        return file
    agent_md = folder / "AGENT.md"
    if agent_md.exists():
        return agent_md
    try:
        named = sorted(p.name for p in folder.iterdir() if re.fullmatch(r"AGENT-.+\.md", p.name))
    except OSError:
        # An unreadable folder has no agent file to offer.
        named = []
    if len(named) == 1:
        return folder / named[0]
    if not named:
        err(f"No agent file in this folder. Create one with `{cli_command('init')}`.")
        return None
    names = [a.name for a in folder_agents(folder)]
    err(f"More than one agent in this folder: pick one with -a ({', '.join(names)}).")
    return None


def machine_facts() -> SkillsFacts:
    """This machine's keys (the shell, or `secrets set`) and sign-in."""
    from .commands.secrets import _store
    from .model_access import is_signed_in

    try:
        store = _store(quiet=True)
    except Exception:  # noqa: BLE001 - the shell alone still answers
        store = None

    def has_key(variable: str) -> bool:
        if os.environ.get(variable):
            return True
        try:
            return bool(store is not None and store.get(variable))
        except Exception:  # noqa: BLE001 - unreadable is absent
            return False

    return SkillsFacts(has_key=has_key, signed_in=is_signed_in())
