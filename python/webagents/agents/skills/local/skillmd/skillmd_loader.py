"""
Loading SKILL.md skills (the Agent Skills format, agentskills.io), gap-closure
plan item 1.4, 2026-09-26. The TypeScript twin is
`typescript/src/skills/skillmd/skillmd-loader.ts`; both run
`tests/fixtures/skillmd/skillmd.json`.

WHAT A SKILL.md SKILL IS. A folder holding `SKILL.md` (frontmatter with a name
and a description, then instructions) plus whatever files the instructions
refer to: `scripts/`, `references/`, `assets/`, or anything else. It is the
format Claude Code, Codex, OpenCode and Hermes all read, so a skill written
once runs under every one of them and under both of these SDKs. It is what a
person who does not program can write.

WHERE THEY COME FROM (`discover_skills`). Two places, the same in both SDKs:

  1. `<agent folder>/.agents/skills/<name>/SKILL.md`, found on its own. This
     is the folder Codex, OpenCode and Hermes scan and the one `webagents
     skills add <source>` installs into.
  2. `agent_skills:` in the agent file: a list of paths, each a skill folder
     or a folder of skill folders, for skills kept somewhere else.

`skills:` was NOT reused for this, because in an agent file it already names
CODED skills (`shell`, `openai`, `mcp`); a path in that list would be refused
as an unknown skill by every existing loader. A separate key keeps both
meanings unambiguous and lets `webagents skills list` show the two kinds
apart. An explicit entry wins over a discovered one of the same name.

LENIENT LOAD, STRICT VALIDATION. The file is read as its author wrote it:
CRLF line ends, a byte-order mark, a closing `---` with no newline after it,
`skill.md` in lower case, a description with an unquoted colon (the YAML is
retried with the value quoted), `metadata` values written as numbers
(`version: 1.0` is the string "1.0" here, in both SDKs), Claude Code's,
Codex's and Hermes's own keys (kept in `extra`, never refused). A skill is
SKIPPED, and said so in `doctor` and in the loader's report, only when the
description is missing or the frontmatter cannot be parsed; everything else
is a warning. The name is the folder's name, as the specification requires;
a frontmatter `name` that differs is a warning, not a rename.

THE PARSER THIS REPLACES (`plugin/components/skill_runner.py`, kept for the
plugin skill) fell back to `path.stem` ("SKILL") for the name, split
`allowed-tools` on commas only, swallowed YAML errors into `{}`, needed a
newline after the closing `---`, and failed on CRLF. None of that is repeated
here, and the fixture pins each case.

`allowed-tools` IS A HINT, NEVER A GRANT: it says which tools the skill's
author expected, and nothing here widens or narrows a tool's scope from it.
`!`cmd`` substitutions in a body are text and are never run.
"""

from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import yaml

#: The file names a skill folder is recognised by, in the order tried.
SKILL_FILE_NAMES = ("SKILL.md", "skill.md")

#: Where an agent folder's own SKILL.md skills live, relative to it.
AUTO_SKILLS_DIR = os.path.join(".agents", "skills")

#: The agent-file key naming other skill folders (a list of paths).
EXPLICIT_KEY = "agent_skills"

#: Folders never scanned for skills or listed as bundled files.
SKIPPED_DIRS = frozenset({".git", "node_modules", "__pycache__"})

#: The frontmatter keys the specification defines; anything else is kept in `extra`.
KNOWN_KEYS = ("name", "description", "license", "compatibility", "metadata", "allowed-tools")

#: The specification's limits, applied as warnings.
NAME_MAX = 64
DESCRIPTION_MAX = 1024
COMPATIBILITY_MAX = 500
BODY_MAX_LINES = 500

#: At most this many bundled files are listed on activation.
LISTED_FILES_MAX = 200

_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", re.UNICODE)


def is_skill_name(name: str) -> bool:
    """Whether `name` is a skill name: 1 to `NAME_MAX` lower-case letters, digits and single hyphens."""
    return 1 <= len(name) <= NAME_MAX and bool(_NAME_RE.match(name)) and name == name.lower()
_KEY_LINE_RE = re.compile(r"^([ \t]*)([^\s#:][^:]*?):[ \t]+(.+?)[ \t]*$")


@dataclass
class SkillMd:
    """A loaded SKILL.md skill."""

    #: The folder's name, which is the skill's name.
    name: str
    description: str
    #: Absolute, symlink-resolved path of the SKILL.md file: the catalog's `location`.
    location: str
    #: Absolute, symlink-resolved path of the skill folder.
    directory: str
    #: The instructions, frontmatter removed, surrounding blank lines trimmed.
    body: str
    #: The frontmatter's `name:`, when it is a valid skill name (1 to 64
    #: lower-case letters, digits and single hyphens); None otherwise. The
    #: installer names the installed folder after it (2026-09-26): it named
    #: the folder after the repository, so a skill declared `greeter` in a
    #: repository `greeter-skill` landed as `greeter-skill` and warned about
    #: its own name on every load.
    declared_name: Optional[str] = None
    allowed_tools: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    license: Optional[str] = None
    compatibility: Optional[str] = None
    #: Frontmatter keys the specification does not define (Claude Code's, Codex's, Hermes's), as written.
    extra: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    #: `auto` (found under .agents/skills) or `explicit` (named by `agent_skills:`).
    source: str = "auto"


@dataclass
class SkippedSkill:
    """A skill folder that could not be loaded, and why."""

    location: str
    name: str
    #: `no_description`, `bad_yaml`, `not_found` or `unreadable`.
    problem: str
    reason: str


@dataclass
class Discovery:
    skills: List[SkillMd] = field(default_factory=list)
    skipped: List[SkippedSkill] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def names(self) -> List[str]:
        return [skill.name for skill in self.skills]


class _Loader(yaml.SafeLoader):
    """PyYAML's safe loader without the timestamp resolver, so `2026-01-01`
    stays the string the JavaScript `yaml` parser (YAML 1.2 core schema)
    makes of it and both SDKs read the same frontmatter."""


_Loader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:timestamp"]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


# ---------------------------------------------------------------------------
# The frontmatter
# ---------------------------------------------------------------------------


def _is_fence(line: str) -> bool:
    return line.rstrip(" \t") == "---"


def split_frontmatter(text: str) -> Tuple[Optional[str], str, Optional[str]]:
    """`(frontmatter, body, problem)`. The BOM is dropped and CRLF becomes LF
    first, and a closing `---` on the last line with no newline after it is a
    closing fence (the old regex required the newline). `problem` is
    `no_frontmatter` or `unclosed` when there is none to give."""
    if text.startswith("﻿"):
        text = text[1:]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    if not lines or not _is_fence(lines[0]):
        return None, text, "no_frontmatter"
    close = next((i for i in range(1, len(lines)) if _is_fence(lines[i])), -1)
    if close < 0:
        return None, text, "unclosed"
    return "\n".join(lines[1:close]), "\n".join(lines[close + 1:]), None


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _quotable(value: str) -> bool:
    """A plain scalar this may wrap in double quotes without changing YAML's
    reading of it: not already quoted, not a flow collection or a block
    scalar indicator, not an anchor, alias or tag."""
    return not value[:1] in ('"', "'", "[", "{", "|", ">", "&", "*", "!")


def quote_metadata_values(front: str) -> str:
    """The values of a block-style `metadata:` mapping, quoted before parsing,
    so `version: 1.0` reaches both SDKs as the string "1.0" rather than as a
    float that Python prints "1.0" and JavaScript prints "1"."""
    lines = front.split("\n")
    out: List[str] = []
    in_block = False
    for line in lines:
        stripped = line.lstrip(" \t")
        indent = len(line) - len(stripped)
        if in_block and stripped and indent == 0:
            in_block = False
        if in_block and stripped and not stripped.startswith("#"):
            match = _KEY_LINE_RE.match(line)
            if match and _quotable(match.group(3)):
                line = f"{match.group(1)}{match.group(2)}: {_quote(match.group(3))}"
        elif indent == 0 and re.match(r"^metadata[ \t]*:[ \t]*$", stripped):
            in_block = True
        out.append(line)
    return "\n".join(out)


def repair_unquoted_colons(front: str) -> str:
    """The retry after a parse error: every top-level `key: value` whose plain
    value holds `: ` (a description such as "Use when: the file is a PDF")
    gets its value quoted. Only the lines a colon breaks are touched."""
    out: List[str] = []
    for line in front.split("\n"):
        match = _KEY_LINE_RE.match(line)
        if match and match.group(1) == "" and _quotable(match.group(3)) and (": " in match.group(3) or match.group(3).endswith(":")):
            line = f"{match.group(2)}: {_quote(match.group(3))}"
        out.append(line)
    return "\n".join(out)


def parse_yaml_frontmatter(front: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """`(mapping, None)`, or `(None, reason)` when the frontmatter is not a
    YAML mapping even after the colon repair."""
    prepared = quote_metadata_values(front)
    try:
        data = yaml.load(prepared, Loader=_Loader)  # noqa: S506 - SafeLoader subclass
    except yaml.YAMLError as first:
        try:
            data = yaml.load(repair_unquoted_colons(prepared), Loader=_Loader)  # noqa: S506
        except yaml.YAMLError:
            detail = str(first).strip().split("\n")[0]
            return None, f"frontmatter is not valid YAML: {detail}"
    if data is None:
        data = {}
    if not isinstance(data, dict):
        return None, "frontmatter is not a mapping of keys to values"
    return data, None


def scalar_text(value: Any) -> Optional[str]:
    """A scalar as the string both SDKs make of it; None for anything else.
    An integral float prints without its `.0`, as JavaScript prints it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return str(int(value)) if value.is_integer() else repr(value)
    return None


def _tool_list(value: Any, warnings: List[str]) -> List[str]:
    """`allowed-tools`: space-separated by the specification, comma-separated
    or a YAML list as Claude Code also writes it."""
    if value is None:
        return []
    if isinstance(value, str):
        return [token for token in re.split(r"[\s,]+", value) if token]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [token for item in value for token in re.split(r"[\s,]+", item) if token]
    warnings.append("allowed-tools is not a space-separated string; ignored")
    return []


def parse_skill_md(text: str, dir_name: str) -> Union[SkillMd, Tuple[str, str]]:
    """The skill a SKILL.md's text describes, named for its folder, or
    `(problem, reason)` when it must be skipped: `no_description` or
    `bad_yaml`. `location` and `directory` are left empty for the caller."""
    front, body, split_problem = split_frontmatter(text)
    if split_problem == "no_frontmatter":
        return "no_description", "SKILL.md has no frontmatter (a --- block with name and description)"
    if split_problem == "unclosed":
        return "bad_yaml", "SKILL.md opens its frontmatter with --- and never closes it"
    data, reason = parse_yaml_frontmatter(front or "")
    if data is None:
        return "bad_yaml", reason or "frontmatter is not valid YAML"

    warnings: List[str] = []
    description = scalar_text(data.get("description"))
    if description is None or not description.strip():
        return "no_description", "SKILL.md has no description in its frontmatter"
    description = description.strip()
    if len(description) > DESCRIPTION_MAX:
        warnings.append(f"description is longer than {DESCRIPTION_MAX} characters")

    declared = scalar_text(data.get("name"))
    if declared is not None and declared.strip() and declared.strip() != dir_name:
        warnings.append(f'name "{declared.strip()}" does not match the folder name "{dir_name}"; the folder name is used')
    if not is_skill_name(dir_name):
        warnings.append(f'name "{dir_name}" is not a skill name (1 to {NAME_MAX} lower-case letters, digits and single hyphens)')
    # The declared name the installer may use for the folder (`SkillMd.declared_name`).
    declared_name = declared.strip() if declared is not None and is_skill_name(declared.strip()) else None

    metadata: Dict[str, str] = {}
    raw_metadata = data.get("metadata")
    if isinstance(raw_metadata, dict):
        for key, value in raw_metadata.items():
            text_value = scalar_text(value)
            if text_value is None:
                warnings.append(f"metadata.{key} is not a string; ignored")
            else:
                metadata[str(key)] = text_value
    elif raw_metadata is not None:
        warnings.append("metadata is not a mapping of strings; ignored")

    license_text = scalar_text(data.get("license")) if data.get("license") is not None else None
    if data.get("license") is not None and license_text is None:
        warnings.append("license is not a string; ignored")
    compatibility = scalar_text(data.get("compatibility")) if data.get("compatibility") is not None else None
    if data.get("compatibility") is not None and compatibility is None:
        warnings.append("compatibility is not a string; ignored")
    elif compatibility is not None and len(compatibility) > COMPATIBILITY_MAX:
        warnings.append(f"compatibility is longer than {COMPATIBILITY_MAX} characters")

    allowed_tools = _tool_list(data.get("allowed-tools"), warnings)
    extra = {str(key): value for key, value in data.items() if key not in KNOWN_KEYS}

    body = body.strip()
    if body.count("\n") + 1 > BODY_MAX_LINES:
        warnings.append(f"body is longer than {BODY_MAX_LINES} lines")

    return SkillMd(
        name=dir_name,
        description=description,
        location="",
        directory="",
        body=body,
        declared_name=declared_name,
        allowed_tools=allowed_tools,
        metadata=metadata,
        license=license_text,
        compatibility=compatibility,
        extra=extra,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------


def skill_file_in(directory: str) -> Optional[str]:
    """The SKILL.md (or skill.md) in `directory`, or None."""
    for name in SKILL_FILE_NAMES:
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def load_skill_dir(directory: str, source: str = "auto") -> Union[SkillMd, SkippedSkill]:
    """The skill in a folder, or why it was skipped."""
    real = os.path.realpath(directory)
    name = os.path.basename(real.rstrip(os.sep)) or real
    file = skill_file_in(real)
    if file is None:
        return SkippedSkill(location=real, name=name, problem="not_found", reason="no SKILL.md in the folder")
    location = os.path.realpath(file)
    try:
        with open(location, "rb") as handle:
            text = handle.read().decode("utf-8", errors="replace")
    except OSError as error:
        return SkippedSkill(location=location, name=name, problem="unreadable", reason=f"SKILL.md cannot be read: {error.strerror or error}")
    parsed = parse_skill_md(text, name)
    if isinstance(parsed, tuple):
        problem, reason = parsed
        return SkippedSkill(location=location, name=name, problem=problem, reason=reason)
    parsed.location = location
    parsed.directory = real
    parsed.source = source
    return parsed


def _skill_dirs_under(directory: str) -> List[str]:
    """The immediate sub-folders of `directory` that hold a SKILL.md, sorted."""
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return []
    found = []
    for name in names:
        if name in SKIPPED_DIRS or name.startswith("."):
            continue
        child = os.path.join(directory, name)
        if os.path.isdir(child) and skill_file_in(child):
            found.append(child)
    return found


def discover_skills(agent_dir: str, explicit: Optional[Sequence[str]] = None) -> Discovery:
    """Every SKILL.md skill an agent in `agent_dir` has: the `agent_skills:`
    entries (each a skill folder, or a folder of skill folders, relative to
    the agent folder), then `.agents/skills/*`. The first of a name wins and
    the rest are warned about; a folder that is missing or holds no SKILL.md
    is reported as skipped, never raised."""
    base = os.path.realpath(agent_dir)
    result = Discovery()
    seen: Dict[str, SkillMd] = {}

    def take(directory: str, source: str) -> None:
        loaded = load_skill_dir(directory, source)
        if isinstance(loaded, SkippedSkill):
            result.skipped.append(loaded)
            return
        earlier = seen.get(loaded.name)
        if earlier is not None:
            result.warnings.append(f'skill "{loaded.name}" at {loaded.location} is shadowed by {earlier.location}')
            return
        seen[loaded.name] = loaded
        result.skills.append(loaded)

    for entry in explicit or []:
        text = str(entry)
        expanded = os.path.expanduser(text)
        path = os.path.realpath(expanded if os.path.isabs(expanded) else os.path.join(base, expanded))
        if not os.path.isdir(path):
            result.skipped.append(SkippedSkill(location=path, name=os.path.basename(path.rstrip(os.sep)), problem="not_found", reason=f"{EXPLICIT_KEY}: {text} is not a folder"))
            continue
        if skill_file_in(path):
            take(path, "explicit")
            continue
        children = _skill_dirs_under(path)
        if not children:
            result.skipped.append(SkippedSkill(location=path, name=os.path.basename(path.rstrip(os.sep)), problem="not_found", reason=f"{EXPLICIT_KEY}: {text} holds no SKILL.md and no folder with one"))
            continue
        for child in children:
            take(child, "explicit")

    for child in _skill_dirs_under(os.path.join(base, AUTO_SKILLS_DIR)):
        take(child, "auto")

    result.skills.sort(key=lambda skill: skill.name)
    return result


def bundled_files(directory: str) -> List[str]:
    """The files under a skill folder other than SKILL.md, as relative paths
    with `/`, sorted; symbolic links and the skipped folders are left out.
    Listed, never read, on activation."""
    real = os.path.realpath(directory)
    found: List[str] = []
    for root, dirs, files in os.walk(real):
        dirs[:] = sorted(d for d in dirs if d not in SKIPPED_DIRS and not os.path.islink(os.path.join(root, d)))
        for name in sorted(files):
            full = os.path.join(root, name)
            if os.path.islink(full):
                continue
            relative = os.path.relpath(full, real).replace(os.sep, "/")
            if root == real and name in SKILL_FILE_NAMES:
                continue
            found.append(relative)
    return sorted(found)


# ---------------------------------------------------------------------------
# The words the model sees (pinned by the fixture)
# ---------------------------------------------------------------------------

CATALOG_PREAMBLE = "Skills available to you. Activate one with activate_skill before doing what it covers."


def escape_xml(text: str) -> str:
    return html.escape(text, quote=False)


def catalog_text(skills: Sequence[SkillMd]) -> str:
    """The tier-1 catalog for the system prompt, in the skills-ref
    `to_prompt` format; empty when there are no skills."""
    if not skills:
        return ""
    lines = [CATALOG_PREAMBLE, "", "<available_skills>"]
    for skill in sorted(skills, key=lambda s: s.name):
        lines += [
            "<skill>",
            f"<name>{escape_xml(skill.name)}</name>",
            f"<description>{escape_xml(skill.description)}</description>",
            f"<location>{escape_xml(skill.location)}</location>",
            "</skill>",
        ]
    lines.append("</available_skills>")
    return "\n".join(lines)


def doctor_report(found: Discovery) -> Dict[str, Optional[str]]:
    """The `skills` line of `webagents doctor`: `status`, `detail` and `fix`,
    the same words in both CLIs (fixture `doctor`)."""
    names = ", ".join(found.names())
    count = len(found.skills)
    if not found.skills and not found.skipped and not found.warnings:
        return {"status": "ok", "detail": "no SKILL.md skills in this folder", "fix": None}
    detail = f"{count} SKILL.md skill{'' if count == 1 else 's'}: {names or 'none'}"
    for skipped in found.skipped:
        detail += f"; skipped: {skipped.name} ({skipped.reason})"
    for warning in found.warnings + [f"{skill.name}: {w}" for skill in found.skills for w in skill.warnings]:
        detail += f"; warning: {warning}"
    problems = bool(found.skipped or found.warnings or any(skill.warnings for skill in found.skills))
    # The fix names the folders (2026-09-26): the sentence used to stop at
    # "named". Skipped folders first, then the skills that warned; a folder
    # warning with no skill behind it is said by the detail alone.
    named = list(dict.fromkeys([s.name for s in found.skipped] + [s.name for s in found.skills if s.warnings]))
    fix = f"Fix or remove the SKILL.md folders named: {', '.join(named)}" if named else "Fix or remove the SKILL.md folders the skills line names"
    return {
        "status": "warn" if problems else "ok",
        "detail": detail,
        "fix": fix if problems else None,
    }


def list_lines(found: Discovery, hint_command: str) -> List[str]:
    """The SKILL.md part of `webagents skills list`, the same lines in both CLIs (fixture `install.messages.list_*`)."""
    lines = ["SKILL.md skills in this folder:", ""]
    for skill in found.skills:
        lines.append(f"  {skill.name}   {skill.location}")
    for skipped in found.skipped:
        lines.append(f"  {skipped.name}   skipped: {skipped.reason}")
    if not found.skills and not found.skipped:
        lines.append("  (none)")
    lines += ["", f"Add one with `{hint_command}`."]
    return lines


def activation_marker(name: str) -> str:
    """The opening tag an activation puts in the conversation; its presence in
    an earlier tool result is what "already active" means."""
    return f'<skill_content name="{escape_xml(name)}">'


def activation_text(skill: SkillMd) -> str:
    """The tier-2 text `activate_skill` returns: the body, then where the
    skill lives and what it bundles (listed, not read)."""
    files = bundled_files(skill.directory)
    if not files:
        listing = "no other files"
    else:
        shown = files[:LISTED_FILES_MAX]
        listing = ", ".join(shown)
        if len(files) > LISTED_FILES_MAX:
            listing += f", and {len(files) - LISTED_FILES_MAX} more"
    return "\n".join([
        activation_marker(skill.name),
        skill.body,
        "",
        f"Skill directory: {skill.directory}",
        f"Files in the skill directory (read one with read_skill_file, run a script with run_skill_script): {listing}",
        "</skill_content>",
    ])
