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

SKILL.md SKILLS FROM OUTSIDE (plan item 1.4, 2026-09-26). A name that is a
SOURCE (`owner/repo`, a git URL, a `.../tree/<ref>/<path>` page, a folder:
`skillmd_install.parse_source`) is not looked for in this SDK's skill table:
it is fetched at one commit, its skills shown file by file, and, once the
person confirms (`--yes` without a terminal), installed into
`.agents/skills/<name>` and recorded in `.webagents/skills.lock`. The agent
file is not edited for those: every agent in the folder finds `.agents/skills`
on its own. `remove <name>` takes an installed skill out the same way when
the name is not a coded skill's. `--skill <name>` picks one skill from a
source. The TypeScript CLI does the same, and `skills list` in both shows
the coded names and the folder's SKILL.md skills apart.
"""

from __future__ import annotations

import json
import os
import re
import stat
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


def unsafe_target_reason(file: Path, folder: Path, who: str = "this command") -> Optional[str]:
    """Why an agent file may not be changed in place (2026-09-26, S-290, spec
    W3): it is a symbolic link, lies outside its folder, or (POSIX) belongs to
    another user; None when it is safe to write. A cloned repository can carry
    `AGENT-x.md -> ../outside/rc`, and the editor wrote through it. The guard
    sits here, so `webagents skills add|remove` gets it, and the chat's
    `/skills`, `/agent edit` and `/agent new` reuse it. The TypeScript editor
    guards the same way (`skills-edit.ts`, `unsafeTargetReason`)."""
    file = Path(file)
    folder = Path(folder)
    shown = file.name
    refusal = f"{shown} is a link or lies outside this folder, so {who} will not change it."
    # A direct child of the folder, by name.
    if os.path.abspath(os.path.join(str(folder), shown)) != os.path.abspath(str(file)):
        return refusal
    try:
        info = os.lstat(str(file))
    except OSError:
        # Not there yet (a file `/agent new` is about to create): no link to follow.
        return None
    if stat.S_ISLNK(info.st_mode):
        return refusal
    try:
        if os.path.dirname(os.path.realpath(str(file))) != os.path.realpath(str(folder)):
            return refusal
    except OSError:
        return refusal
    getuid = getattr(os, "getuid", None)
    if getuid is not None and info.st_uid != getuid():
        return refusal
    return None


#: Why a list could not be edited; each has one sentence (`problem_message`).
PROBLEMS = ("unclosed", "not_yaml", "not_list", "layout")


def problem_message(problem: str, file: str, key: str = "model") -> str:
    if problem == "unclosed":
        return f"{file} opens its front matter with --- and never closes it."
    if problem == "not_yaml":
        return f"The front matter of {file} is not valid YAML. Fix it, then try again."
    if problem == "not_list":
        return f"skills: in {file} is not a list. Fix it, then try again."
    if problem == "scalar_layout":
        # The chat's `/model --save` (spec 3.5): the list sentence's shape,
        # naming the key (`repl/chat_words.py` `modelLayout`).
        return f"The {key}: line in {file} is laid out in a way this command cannot change safely. Edit it by hand."
    return f"The skills: list in {file} is laid out in a way this command cannot change safely. Edit it by hand."


class SkillListError(Exception):
    def __init__(self, problem: str, file: str, key: Optional[str] = None):
        super().__init__(problem_message(problem, file, key or "model"))
        self.problem = problem


@dataclass
class ScalarEdit:
    """What a scalar edit did. `text` is the file as it is to be written; unchanged when `changed` is false."""

    text: str
    changed: bool
    #: The value the file had, as YAML read it; None when the key was absent.
    previous: Optional[str] = None


def _yaml_scalar(value: str) -> str:
    """`value` as a plain scalar when YAML reads it back as itself, else double-quoted."""
    if (
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:+-]*", value)
        and not re.fullmatch(r"(true|false|null|yes|no|on|off)", value, re.IGNORECASE)
        and not re.fullmatch(r"[0-9.]+", value)
    ):
        return value
    return json.dumps(value)


def edit_front_matter_scalar(text: str, key: str, value: str, file: str) -> ScalarEdit:
    """Set one top-level front-matter scalar (`model: openai/gpt-4o`) in the
    text of the agent file `file`, the way `edit_skill_list` edits the list
    (2026-09-26, interactive-mode spec 3.5, `/model --save`): the same parse,
    one line changed or added, then read back as YAML. A key line written some
    other way (quoted, a block scalar, an anchor) is refused with the
    `scalar_layout` sentence rather than guessed at. A file with no front
    matter gets one, naming the agent as the loaders name it."""
    shown = Path(file).name
    parsed = _parse(text, shown)
    had = parsed.data.get(key)
    previous = None if had is None else str(had)
    if previous == value:
        return ScalarEdit(text=text, changed=False, previous=previous)

    lines = list(parsed.lines)
    written = f"{key}: {_yaml_scalar(value)}"
    if parsed.close < 0:
        lines[:0] = ["---", f"name: {_name_for_file(file)}", written, "---", ""]
    else:
        key_re = re.compile(rf"^{re.escape(key)}[ \t]*:")
        at = next((i for i in range(1, parsed.close) if key_re.match(lines[i])), -1)
        if at < 0:
            # Present under another spelling (quoted, or a complex key): not this command's to touch.
            if key in parsed.data:
                raise SkillListError("scalar_layout", shown, key)
            # After `name:` when there is one, else at the end of the front matter.
            name_at = next((i for i in range(1, parsed.close) if re.match(r"^name[ \t]*:", lines[i])), -1)
            insert_at = parsed.close
            if name_at >= 0:
                insert_at = name_at + 1
            else:
                while insert_at > 1 and lines[insert_at - 1].strip() == "":
                    insert_at -= 1
            lines.insert(insert_at, written)
        else:
            line = lines[at]
            colon = line.index(":")
            rest = line[colon + 1:]
            # The value up to a trailing comment, which is kept. A value that
            # continues on the next line (a block scalar, a mapping) is not one line.
            comment = re.search(r"(\s+#.*)$", rest)
            body = rest[: len(rest) - len(comment.group(1))] if comment else rest
            if re.match(r"^\s*[|>&*]", body) or body.strip() == "":
                raise SkillListError("scalar_layout", shown, key)
            nxt = lines[at + 1] if at + 1 < parsed.close else None
            if nxt is not None and re.match(r"^\s+\S", nxt) and not re.match(r"^\s*#", nxt):
                raise SkillListError("scalar_layout", shown, key)
            lines[at] = f"{line[:colon + 1]} {_yaml_scalar(value)}{comment.group(1) if comment else ''}"
    edited = parsed.bom + parsed.nl.join(lines)
    # Read back as YAML: the value must be exactly what was meant, or nothing is written.
    check = _parse(edited, shown)
    if str(check.data.get(key)) != value:
        raise SkillListError("scalar_layout", shown, key)
    return ScalarEdit(text=edited, changed=True, previous=previous)


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


#: What a model provider's skill imports, as a person installs it (B3,
#: 2026-09-28): `model_access._CLIENT_MODULES` names the modules; these are
#: the packages. Both ship in the `llm` extra.
_CLIENT_PACKAGES: Dict[str, str] = {"google": "google-genai", "anthropic": "anthropic"}


def missing_client_line(name: str) -> Optional[str]:
    """Why a model provider's skill cannot be added here, with the install
    command, or None when it can (B3, 2026-09-28). `/skills add google`
    without the Google SDK wrote the name into the file, and every start then
    said "Skill google failed to load" while the file kept naming it. The
    TypeScript providers call their APIs over HTTP and need no install."""
    import sys

    from webagents.agents.skills.core.llm.providers import find_provider

    from .model_access import _CLIENT_MODULES, _client_installed

    provider = find_provider(canonical_skill_name(name))
    if provider is None or provider.id not in _CLIENT_MODULES or _client_installed(provider):
        return None
    package = _CLIENT_PACKAGES.get(provider.id, _CLIENT_MODULES[provider.id])
    return (
        f"{name} needs the {package} library, which this Python does not have. "
        f"Install it with `{sys.executable} -m pip install 'webagents[llm]'`, then add {name} again."
    )


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
    skill: Optional[str] = None,
    yes: bool = False,
    tty: Optional[bool] = None,
    confirm: Optional[Callable[[str], bool]] = None,
    edited: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> int:
    """Run `skills add` or `skills remove`; returns the exit code. Refusals
    (an unknown name, no agent file, a list it cannot edit) change nothing.
    Sources among `requested` install SKILL.md skills (module docstring);
    `skill`, `yes`, `tty` and `confirm` belong to that path. `edited` is told
    what the editor changed, once, after a successful edit (the `--json`
    document's facts, 2026-09-27; the TypeScript `SkillsCommandIO.edited`)."""
    from webagents.agents.skills.local.skillmd.skillmd_install import install_from_source, parse_source, read_lock, remove_installed

    from .agent_builder import SKILL_CLASSES
    from .config_store import cli_command

    folder = folder or Path.cwd()
    known = sorted(SKILL_CLASSES)
    canonical = canonical_skill_name

    # Sources apart from names: a source never reaches the editor, and a name
    # never reaches the installer. An installed SKILL.md skill's name, when it
    # is not a coded skill's, is removed from `.agents/skills` rather than
    # from the file.
    names: List[str] = []
    sources = []
    exit_code = 0
    for entry in requested:
        source = parse_source(entry)
        if source.kind == "name":
            names.append(entry)
        elif action == "add":
            sources.append(source)
        else:
            err(f"{entry} is not a skill name; remove takes the names `skills list` shows.")
            exit_code = 1
    if action == "remove":
        remaining: List[str] = []
        installed = read_lock(str(folder))["skills"]
        for entry in names:
            lower = entry.strip().lower()
            if lower and canonical(lower) not in known and lower in installed:
                code = remove_installed(lower, str(folder), out=out, err=err)
                exit_code = exit_code or (code or 0)
            else:
                remaining.append(entry)
        names = remaining

    if names or agent:
        code = _edit_agent_file(action, names, agent, folder, facts, out, err, edited) if names or action == "add" else 0
        if names:
            exit_code = exit_code or code
        elif code:
            exit_code = code
    for source in sources:
        code = install_from_source(
            source,
            str(folder),
            skill=skill,
            yes=yes,
            tty=sys.stdin.isatty() if tty is None else tty,
            confirm=confirm,
            out=out,
            err=err,
        )
        exit_code = exit_code or code
    return exit_code


def _edit_agent_file(
    action: str,
    requested: Sequence[str],
    agent: Optional[str],
    folder: Path,
    facts: Optional[Callable[[], SkillsFacts]],
    out: Callable[[str], None],
    err: Callable[[str], None],
    edited: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> int:
    """The editor's part of `skills add|remove`: the coded names, in the agent file."""
    from webagents.agents.skills.core.llm.providers import find_provider

    from .agent_builder import SKILL_CLASSES
    from .config_store import cli_command
    from .help_format import suggest_similar

    file = _choose_agent_file(folder, agent, err, cli_command)
    if file is None:
        return 1
    if not requested:
        return 0
    shown = file.name

    # W3 / S-290: never write through a symbolic link, a file outside the
    # folder, or another user's file. The guard is here so both CLIs get it.
    unsafe = unsafe_target_reason(file, folder)
    if unsafe is not None:
        err(unsafe)
        return 1

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
    # A model provider's skill whose library is not installed is refused with
    # the install command, before the file names it (B3).
    missing = [line for line in (missing_client_line(n) for n in wanted) if line] if action == "add" else []
    if missing:
        for line in missing:
            err(line)
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
    if edited is not None:
        edited({"file": shown, "added": list(edit.added), "already": list(edit.already), "removed": list(edit.removed), "absent": list(edit.absent), "skills": list(edit.skills)})

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


def skillmd_list_lines(folder: Path) -> List[str]:
    """The SKILL.md part of `skills list`: what an agent in `folder` would
    load (`.agents/skills`, plus the default agent file's `agent_skills:`),
    the skipped ones with their reasons, and how to add one."""
    from webagents.agents.skills.local.skillmd.skillmd_loader import discover_skills, list_lines

    from .agent_files import default_agent_file
    from .config_store import cli_command

    explicit: List[str] = []
    file = default_agent_file(folder)
    if file is not None:
        try:
            from .loader.hierarchy import load_agent

            explicit = list(load_agent(file).metadata.agent_skills or [])
        except Exception:  # noqa: BLE001 - a broken agent file is doctor's to report
            explicit = []
    return list_lines(discover_skills(str(folder), explicit), cli_command("skills add <owner/repo | git URL | folder>"))


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


# ============================================================================
# plan_skills / apply_skills: the chat's /skills, and the CLI's checks, in two
# halves (2026-09-26, interactive-mode spec 3.4). `plan_skills` does today's
# checks and no write; `apply_skills` writes. The chat shows the plan, asks,
# snapshots and applies; the fixture `chat_edits.json` and the `plans` block of
# `skills_edit.json` pin them, and the TypeScript chat mirrors both.
# ============================================================================


@dataclass
class SkillsPlan:
    """The checks for `/skills add|remove`, no write (module docstring)."""

    file: Optional[Path]
    errors: List[str] = field(default_factory=list)
    add: List[str] = field(default_factory=list)
    remove: List[str] = field(default_factory=list)
    already: List[str] = field(default_factory=list)
    absent: List[str] = field(default_factory=list)
    skills_after: List[str] = field(default_factory=list)
    installed_removals: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    changed: bool = False


@dataclass
class SkillsApplied:
    """What `apply_skills` did, for the caller to word (the chat's ✓ lines)."""

    added: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    skills_after: List[str] = field(default_factory=list)
    installed_removed: List[str] = field(default_factory=list)


def _agent_file_in_folder(folder: Path, file: Optional[Path]) -> Optional[Path]:
    if file is not None:
        return file
    agent_md = folder / "AGENT.md"
    if agent_md.exists() or os.path.islink(str(agent_md)):
        return agent_md
    try:
        named = sorted(p.name for p in folder.iterdir() if re.fullmatch(r"AGENT-.+\.md", p.name))
    except OSError:
        named = []
    return folder / named[0] if len(named) == 1 else None


def plan_skills(action: str, requested: Sequence[str], *, folder: Path, file: Optional[Path] = None, who: str = "this command") -> SkillsPlan:
    """The checks for `/skills add|remove`, no write (module docstring)."""
    from webagents.agents.skills.local.skillmd.skillmd_install import installed_skill_names, parse_source
    from webagents.agents.skills.core.llm.providers import find_provider  # noqa: F401 - warms the registry

    from .agent_builder import SKILL_CLASSES
    from .config_store import cli_command
    from .help_format import suggest_similar

    folder = Path(folder)
    known = sorted(SKILL_CLASSES)
    canonical = canonical_skill_name
    try:
        installed = installed_skill_names(str(folder))
    except Exception:  # noqa: BLE001
        installed = []

    names: List[str] = []
    sources: List[str] = []
    installed_removals: List[str] = []
    errors: List[str] = []
    for entry in requested:
        lower = entry.strip().lower()
        if parse_source(entry).kind != "name":
            if action == "add":
                sources.append(entry)
            else:
                errors.append(f"{entry} is not a skill name; remove takes the names `skills list` shows.")
        elif action == "remove" and lower and canonical(lower) not in known and lower in installed:
            installed_removals.append(lower)
        else:
            names.append(entry)

    if not names:
        return SkillsPlan(file=None, sources=sources, installed_removals=installed_removals, errors=errors, changed=bool(installed_removals))

    chosen = _agent_file_in_folder(folder, file)
    if chosen is None:
        errors.append(f"No agent file in this folder. Create one with `{cli_command('init')}`.")
        return SkillsPlan(file=None, sources=sources, installed_removals=installed_removals, errors=errors, changed=bool(installed_removals))
    unsafe = unsafe_target_reason(chosen, folder, who)
    if unsafe is not None:
        errors.append(unsafe)
        return SkillsPlan(file=chosen, sources=sources, installed_removals=installed_removals, errors=errors, changed=False)

    try:
        text = chosen.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        errors.append(f"Could not read {chosen.name}: {error}")
        return SkillsPlan(file=chosen, sources=sources, installed_removals=installed_removals, errors=errors, changed=False)

    wanted = [n for n in (name.strip().lower() for name in names) if n]
    try:
        listed = edit_skill_list(text, "add", [], str(chosen)).skills
    except SkillListError as error:
        errors.append(str(error))
        return SkillsPlan(file=chosen, sources=sources, installed_removals=installed_removals, errors=errors, changed=False)
    unknown = [n for n in wanted if canonical(n) not in known and not any(canonical(have) == canonical(n) for have in listed)]
    if unknown:
        candidates = list(dict.fromkeys([*listed, *known])) if action == "remove" else known
        for name in unknown:
            errors.append(f"Unknown skill '{name}'.")
            suggestion = suggest_similar(name, candidates).lstrip("\n")
            if suggestion:
                errors.append(suggestion)
        errors.append(f"Run `{cli_command('skills list')}` for the skills an agent file can name.")
        return SkillsPlan(file=chosen, sources=sources, installed_removals=installed_removals, errors=errors, changed=False)
    # A model provider's skill whose library is not installed: the install
    # command, and no change (B3, `missing_client_line`).
    missing = [line for line in (missing_client_line(n) for n in wanted) if line] if action == "add" else []
    if missing:
        errors.extend(missing)
        return SkillsPlan(file=chosen, sources=sources, installed_removals=installed_removals, errors=errors, changed=False)

    try:
        edit = edit_skill_list(text, action, wanted, str(chosen))
    except SkillListError as error:
        errors.append(str(error))
        return SkillsPlan(file=chosen, sources=sources, installed_removals=installed_removals, errors=errors, changed=False)
    return SkillsPlan(
        file=chosen,
        errors=errors,
        add=edit.added,
        remove=edit.removed,
        already=edit.already,
        absent=edit.absent,
        skills_after=edit.skills,
        installed_removals=installed_removals,
        sources=sources,
        changed=edit.changed or bool(installed_removals),
    )


def apply_skills(action: str, plan: SkillsPlan, folder: Path) -> SkillsApplied:
    """Apply `plan`'s file edit and installed removals; nothing is printed, so
    the caller words the result. The file is read again and written by rename,
    so a link is never written through. Sources are the caller's to install."""
    from webagents.agents.skills.local.skillmd.skillmd_install import remove_installed

    folder = Path(folder)
    installed_removed: List[str] = []
    for name in plan.installed_removals:
        if remove_installed(name, str(folder), out=lambda _line: None, err=lambda _line: None) == 0:
            installed_removed.append(name)
    if plan.file is None or (not plan.add and not plan.remove):
        return SkillsApplied(skills_after=plan.skills_after, installed_removed=installed_removed)
    text = Path(plan.file).read_bytes().decode("utf-8")
    names = plan.add if action == "add" else plan.remove
    edit = edit_skill_list(text, action, names, str(plan.file))
    if edit.changed:
        _write_by_rename(Path(plan.file), edit.text)
    return SkillsApplied(added=edit.added, removed=edit.removed, skills_after=edit.skills, installed_removed=installed_removed)


def _write_by_rename(file: Path, text: str) -> None:
    """Write `text` by renaming a temporary file in the same folder, so a link is never written through."""
    temp = file.with_name(f".{file.name}.{os.getpid()}.tmp")
    temp.write_bytes(text.encode("utf-8"))
    os.replace(str(temp), str(file))


#: The chat's `/model --save` writes its one line the same way (spec 3.5, W3).
write_by_rename = _write_by_rename
