"""
`webagents mcp list` and `webagents mcp add`, and the chat's `/mcp list` and
`/mcp add` (2026-09-29, the owner: "can we list available local mcps?").

WHAT IS LISTED. The MCP servers other apps on this machine already use, read
from their own settings files and never written to: Claude Desktop
(`claude_desktop_config.json`), Claude Code (`~/.claude.json`, its servers and
this folder's, and the folder's `.mcp.json`), Cursor (`~/.cursor/mcp.json` and
the folder's), VS Code (the user folder's `mcp.json`, `servers:`, and the
folder's `.vscode/mcp.json`) and Windsurf (`~/.codeium/windsurf/mcp_config.json`).
VS Code allows comments and trailing commas; they are dropped before reading.
Each entry is read into this SDK's shape (`command`/`args`/`env`/`cwd`, or
`url`/`transport`/`headers`); an entry with neither a command nor an address is
skipped. A listing names environment variables and headers, never their values.

WHAT ADD DOES. It copies one of those entries into an agent. A value that looks
like a key (`looks_like_secret`, or a variable named like one) is stored in this
profile's secrets, the store `webagents secrets set` writes, and the entry
reads `${secret:NAME}`; an `Authorization: Bearer <token>` header keeps its
`Bearer `. VS Code's `${workspaceFolder}` becomes the folder; its
`${input:NAME}` becomes `${secret:NAME}` for the person to set. A key in the
address's query is stored the same way. A key on the command line (one that
looks like a key, or follows a flag named like one), where a reference is
refused (S-292), is refused here too, and so is any other `${...}` that only
the other app fills in. A listing shows every such key as ***. The entry goes where the
agent reads its servers: the agent file's own `- mcp:` block when it has one
(edited line by line, every other byte as it was, read back as YAML and
compared before it is written: the `skills_edit.py` rule; a layout that cannot
be edited safely is refused with the lines to paste), else `mcp.json` next to
the agent file, with a bare `- mcp` added to its skills when it has none.

The TypeScript CLI is `src/cli/mcp-import.ts`, rule for rule;
`tests/fixtures/mcp_tool/import.json` holds the words and cases both suites run.
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import yaml

from webagents.agents.skills.local.secrets.references import REFERENCE_NAME, looks_like_secret, suggested_secret_name

WORDS: Dict[str, str] = {
    "heading": "MCP servers other apps on this machine use",
    "group": "{app}  {file}",
    "server": "  {name}  {what}",
    "unreadable": "{app}  {file}: could not read it ({reason})",
    "none": "No MCP servers in the settings of Claude Desktop, Claude Code, Cursor, VS Code or Windsurf.",
    "addHint": "Add one to an agent: {command}, or /mcp add <name> in the chat.",
    "added": "Added {name} to {agent} ({where}).",
    "stored": "Stored {names} in this profile's secrets; the entry reads them as ${secret:NAME}.",
    "toSet": "Set {names} before it connects: {hints}.",
    "reload": "/reload connects it.",
    "restart": "The agent connects it the next time it starts.",
    "notFound": "No MCP server named {name} in other apps' settings; {command} shows them.",
    "ambiguous": "{name} is in more than one app's settings ({apps}); say which with --from <app>.",
    "unknownApp": "No app named {app}; use one of {apps}.",
    "exists": "{agent} already has an MCP server named {name}.",
    "keyOnCommandLine": "{name} carries what looks like a key in its {field}; move it to env in {app}, then add it again.",
    "otherAppReference": "{name} uses {reference}, which only {app} fills in; add it by hand with the value in place.",
    "cannotEdit": "{file} lists its MCP servers in a layout this cannot edit safely. Add these lines under its mcp entry:",
    "noAgent": "No agent file in {folder}; {command} makes one.",
    "manyAgents": "More than one agent file in {folder}; name one: {files}.",
    "removed": "Removed {name} from {agent} ({where}).",
    "notThere": "{agent} has no MCP server named {name}.",
    "secretsStay": "It read {names} from this profile's secrets; they stay stored (`{command}` removes one).",
    "cannotRemove": "{file} lists its MCP servers in a layout this cannot edit safely; take {name} out of it by hand.",
    "reloadRemoved": "/reload stops it in this chat.",
    "restartRemoved": "The agent stops using it the next time it starts.",
}

#: The apps read, in the order a listing shows them: id, name.
APPS: Tuple[Tuple[str, str], ...] = (
    ("claude-desktop", "Claude Desktop"),
    ("claude-code", "Claude Code"),
    ("cursor", "Cursor"),
    ("vscode", "VS Code"),
    ("windsurf", "Windsurf"),
)

#: An environment variable named like a credential: its value is stored as a secret whatever it looks like.
_KEY_NAME = re.compile(r"(TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|ACCESS_KEY|PRIVATE_KEY|CREDENTIALS?|_PAT)$", re.IGNORECASE)
#: A header or an address's query parameter named like a credential.
_KEY_FIELD = re.compile(r"(api[-_]?key|token|secret|auth|password|passwd|^key$)", re.IGNORECASE)
#: A command-line flag named like a credential: the argument after it, or after its `=`, is a key.
_KEY_FLAG = re.compile(r"^-{1,2}(?!-)(?!no-)[A-Za-z0-9_-]*?(api[-_]?key|token|secret|password|passwd)$", re.IGNORECASE)
#: A query parameter in an address: the separator, the name, the value.
_QUERY = re.compile(r"([?&])([^=&#]+)=([^&#]*)")


@dataclass
class FoundServer:
    """One server another app's settings name."""

    app_id: str
    app: str
    file: str
    name: str
    entry: Dict[str, Any]


@dataclass
class Discovery:
    found: List[FoundServer] = field(default_factory=list)
    #: `(app, file, reason)` for a settings file that is there and could not be read.
    unreadable: List[Tuple[str, str, str]] = field(default_factory=list)


def settings_files(home: Path, folder: Path, system: str = sys.platform, appdata: Optional[str] = None) -> List[Tuple[str, str, Path, str]]:
    """`(app id, app, file, shape)` for every settings file read, in listing
    order. `shape` is `mcpServers`, `servers` (VS Code) or `claude-json`."""
    if system == "darwin":
        support = home / "Library" / "Application Support"
    elif system.startswith("win"):
        support = Path(appdata) if appdata else home / "AppData" / "Roaming"
    else:
        support = home / ".config"
    return [
        ("claude-desktop", "Claude Desktop", support / "Claude" / "claude_desktop_config.json", "mcpServers"),
        ("claude-code", "Claude Code", home / ".claude.json", "claude-json"),
        ("claude-code", "Claude Code", folder / ".mcp.json", "mcpServers"),
        ("cursor", "Cursor", home / ".cursor" / "mcp.json", "mcpServers"),
        ("cursor", "Cursor", folder / ".cursor" / "mcp.json", "mcpServers"),
        ("vscode", "VS Code", support / "Code" / "User" / "mcp.json", "servers"),
        ("vscode", "VS Code", folder / ".vscode" / "mcp.json", "servers"),
        ("windsurf", "Windsurf", home / ".codeium" / "windsurf" / "mcp_config.json", "mcpServers"),
    ]


def _outside_strings(text: str, step: Callable[[str, int], Tuple[str, int]]) -> str:
    """`text` with `step(text, i)` deciding, at every character outside a JSON
    string, what to write and where to go on; strings are copied as they are."""
    out: List[str] = []
    i, n = 0, len(text)
    in_string = False
    while i < n:
        ch = text[i]
        if in_string:
            out.append(text[i : i + 2] if ch == "\\" else ch)
            in_string = ch != '"'
            i += 2 if ch == "\\" else 1
        elif ch == '"':
            out.append(ch)
            in_string = True
            i += 1
        else:
            written, i = step(text, i)
            out.append(written)
    return "".join(out)


def _comment(text: str, i: int) -> Tuple[str, int]:
    if text.startswith("//", i):
        end = text.find("\n", i)
        return "", len(text) if end < 0 else end
    if text.startswith("/*", i):
        end = text.find("*/", i + 2)
        return "", len(text) if end < 0 else end + 2
    return text[i], i + 1


def _trailing_comma(text: str, i: int) -> Tuple[str, int]:
    if text[i] == ",":
        j = i + 1
        while j < len(text) and text[j] in " \t\r\n":
            j += 1
        if j < len(text) and text[j] in "}]":
            return "", i + 1
    return text[i], i + 1


def strip_jsonc(text: str) -> str:
    """`text` without `//` and `/* */` comments and trailing commas; a string
    is never touched, whatever it holds."""
    return _outside_strings(_outside_strings(text, _comment), _trailing_comma)


def _text(value: Any) -> Optional[str]:
    """A scalar from another app's JSON as text, the way JavaScript's `String`
    writes it (`true`, not `True`), so both CLIs copy the same value; None for
    null, a list or a mapping, which are dropped."""
    if value is None or isinstance(value, (dict, list)):
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _strings(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(k): text for k, text in ((k, _text(v)) for k, v in value.items()) if text is not None}


def normalize_entry(raw: Any) -> Optional[Dict[str, Any]]:
    """Another app's entry in this SDK's shape; None when it names neither a command nor an address."""
    if not isinstance(raw, dict):
        return None
    command = raw.get("command")
    if isinstance(command, str) and command:
        listed = raw.get("args") if isinstance(raw.get("args"), list) else []
        args = [text for text in (_text(a) for a in listed) if text is not None]
        # A whole command line in `command` with no `args` (Cursor writes
        # `"command": "npx -y chrome-devtools-mcp@latest"`) is split the way a
        # shell would, unless it is the path of a file that is there.
        if not args and re.search(r"\s", command.strip()) and not os.path.isfile(command):
            import shlex

            try:
                words = shlex.split(command)
            except ValueError:
                words = []
            if words:
                command, args = words[0], words[1:]
        entry: Dict[str, Any] = {"command": command, "args": args}
        env = _strings(raw.get("env"))
        if env:
            entry["env"] = env
        if isinstance(raw.get("cwd"), str) and raw["cwd"]:
            entry["cwd"] = raw["cwd"]
        return entry
    url = next((raw[k] for k in ("url", "serverUrl", "httpUrl") if isinstance(raw.get(k), str) and raw.get(k)), None)
    if not url:
        return None
    entry = {"url": url}
    kind = str(raw.get("type") or raw.get("transport") or "").lower().replace("_", "-")
    if kind == "sse":
        entry["transport"] = "sse"
    elif kind in ("http", "streamable-http", "streamablehttp"):
        entry["transport"] = "http"
    headers = _strings(raw.get("headers"))
    if headers:
        entry["headers"] = headers
    return entry


def _servers_in(data: Any, shape: str, folder: Path) -> List[Tuple[str, Any]]:
    if not isinstance(data, dict):
        return []
    if shape == "claude-json":
        pairs = list((data.get("mcpServers") or {}).items()) if isinstance(data.get("mcpServers"), dict) else []
        project = (data.get("projects") or {}).get(str(folder)) if isinstance(data.get("projects"), dict) else None
        if isinstance(project, dict) and isinstance(project.get("mcpServers"), dict):
            pairs += list(project["mcpServers"].items())
        return pairs
    key = "servers" if shape == "servers" else "mcpServers"
    return list(data[key].items()) if isinstance(data.get(key), dict) else []


#: Settings files as read, by path, kept while their time stamp and size stay
#: the same: the chat reads them before every prompt (for completion), and
#: Claude Code's `~/.claude.json` can run to megabytes.
_READ: Dict[str, Tuple[Tuple[int, int], Tuple[bool, Any]]] = {}


def _read_settings(file: Path) -> Optional[Tuple[bool, Any]]:
    """`(True, data)` for a settings file, `(False, None)` for one that is not
    valid JSON, None for one that is not there or cannot be opened. Plain JSON
    is read as it is; only a file that fails is read again without comments
    and trailing commas. A byte-order mark is dropped and bytes that are not
    UTF-8 are replaced, as Node's `readFileSync(file, 'utf8')` does."""
    try:
        stat = file.stat()
        if not file.is_file():
            return None
        key = (stat.st_mtime_ns, stat.st_size)
        cached = _READ.get(str(file))
        if cached is not None and cached[0] == key:
            return cached[1]
        text = file.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return None
    text = text[1:] if text.startswith("\ufeff") else text
    try:
        read: Tuple[bool, Any] = (True, json.loads(text))
    except ValueError:
        try:
            read = (True, json.loads(strip_jsonc(text)))
        except ValueError:
            read = (False, None)
    _READ[str(file)] = (key, read)
    return read


def discover(home: Optional[Path] = None, folder: Optional[Path] = None, system: str = sys.platform, appdata: Optional[str] = None) -> Discovery:
    """Every server the apps' settings files name, in listing order."""
    home = home or Path.home()
    folder = (folder or Path.cwd()).resolve()
    result = Discovery()
    for app_id, app, file, shape in settings_files(home, folder, system, appdata if appdata is not None else os.environ.get("APPDATA")):
        read = _read_settings(file)
        if read is None:
            continue
        ok, data = read
        if not ok:
            result.unreadable.append((app, str(file), "not valid JSON"))
            continue
        for name, raw in _servers_in(data, shape, folder):
            entry = normalize_entry(raw)
            if entry is not None:
                result.found.append(FoundServer(app_id, app, str(file), str(name), entry))
    return result


def _shown_path(path: str, home: Path) -> str:
    text = str(path)
    root = str(home)
    return "~" + text[len(root):] if root and (text == root or text.startswith(root + os.sep)) else text


def masked_args(args: Sequence[str]) -> Tuple[List[str], bool]:
    """`args` with every key in them shown as ***, and whether there was one:
    an argument that looks like a key, the value of `--api-key=VALUE`, and the
    argument after a bare `--api-key` (a flag named like a credential)."""
    out: List[str] = []
    carried = after_flag = False
    for arg in args:
        flag, eq, value = arg.partition("=")
        if (after_flag and not arg.startswith("-")) or looks_like_secret(arg):
            out.append("***")
            carried = True
        elif eq and value and _KEY_FLAG.match(flag):
            out.append(flag + "=***")
            carried = True
        else:
            out.append(arg)
        after_flag = not eq and bool(_KEY_FLAG.match(arg))
    return out, carried


def _query_key(name: str, value: str) -> bool:
    """Whether an address's query value is a key: it looks like one, or its parameter is named like one."""
    return bool(value) and "${" not in value and (looks_like_secret(value) or bool(_KEY_FIELD.search(name)))


def describe(entry: Dict[str, Any]) -> str:
    """One line for an entry: the command and its arguments (`masked_args`),
    or the address (a key in its query as ***) and its transport; variable and
    header names, never their values."""
    if "command" in entry:
        command = "***" if looks_like_secret(entry["command"]) else entry["command"]
        text = " ".join([command, *masked_args(entry.get("args", []))[0]])
        if entry.get("env"):
            text += "  env " + ", ".join(entry["env"])
        return text
    url = _QUERY.sub(lambda m: m.group(1) + m.group(2) + "=" + ("***" if _query_key(m.group(2), m.group(3)) else m.group(3)), entry["url"])
    text = url + (f" ({entry['transport']})" if entry.get("transport") else "")
    if entry.get("headers"):
        text += "  headers " + ", ".join(entry["headers"])
    return text


def list_lines(discovery: Discovery, home: Optional[Path] = None, add_command: str = "webagents mcp add <name>") -> List[str]:
    """What `mcp list` prints."""
    home = home or Path.home()
    if not discovery.found and not discovery.unreadable:
        return [WORDS["none"]]
    lines = [WORDS["heading"]]
    groups: Dict[Tuple[str, str], List[FoundServer]] = {}
    for found in discovery.found:
        groups.setdefault((found.app, found.file), []).append(found)
    for (app, file), servers in groups.items():
        lines.append(WORDS["group"].replace("{app}", app).replace("{file}", _shown_path(file, home)))
        for found in servers:
            lines.append(WORDS["server"].replace("{name}", found.name).replace("{what}", describe(found.entry)))
    for app, file, reason in discovery.unreadable:
        lines.append(WORDS["unreadable"].replace("{app}", app).replace("{file}", _shown_path(file, home)).replace("{reason}", reason))
    if discovery.found:
        lines.append(WORDS["addHint"].replace("{command}", add_command))
    return lines


class AddRefused(Exception):
    """`mcp add` would not copy the server; the message says why."""


class SecretStoreLike:
    """What the conversion needs of a secret store."""

    def get(self, name: str) -> Optional[str]:  # pragma: no cover - interface
        raise NotImplementedError

    def set(self, name: str, value: str) -> Any:  # pragma: no cover - interface
        raise NotImplementedError


def _secret_name(store: SecretStoreLike, preferred: str, fallback: str, value: str) -> str:
    """A name for `value`: `preferred` when free or already holding it, else
    `fallback`, else `fallback` with a number."""
    candidates = [preferred] if REFERENCE_NAME.match(preferred) else []
    candidates.append(fallback)
    for name in candidates:
        held = store.get(name)
        if held is None or held == value:
            return name
    number = 2
    while True:
        name = f"{fallback[:120]}_{number}"
        held = store.get(name)
        if held is None or held == value:
            return name
        number += 1


@dataclass
class Converted:
    entry: Dict[str, Any]
    #: Secret names stored with a value from the other app's settings.
    stored: List[str] = field(default_factory=list)
    #: Secret names the entry reads that have no value yet (VS Code's inputs).
    to_set: List[str] = field(default_factory=list)


def convert_entry(found: FoundServer, folder: Path, store: SecretStoreLike) -> Converted:
    """`found.entry` as the agent will hold it (module docstring): keys moved to
    this profile's secrets, VS Code's variables filled in or turned into
    references. Raises `AddRefused` for what cannot be copied safely."""
    entry = copy.deepcopy(found.entry)
    result = Converted(entry)
    folder_text = str(folder)

    def other_app_reference(value: str) -> Optional[str]:
        match = re.search(r"\$\{(?!env:|secret:)[^}]*\}", value)
        return match.group(0) if match else None

    def refuse_reference(reference: str) -> None:
        raise AddRefused(
            WORDS["otherAppReference"].replace("{name}", found.name).replace("{reference}", reference).replace("{app}", found.app)
        )

    def plain(value: str, where: str) -> str:
        """A command-line value: VS Code's folder filled in; any other `${...}`
        refused, since a reference on the command line is (S-292)."""
        value = value.replace("${workspaceFolder}", folder_text)
        reference = re.search(r"\$\{[^}]*\}", value)
        if reference:
            refuse_reference(reference.group(0))
        if looks_like_secret(value):
            raise AddRefused(
                WORDS["keyOnCommandLine"].replace("{name}", found.name).replace("{field}", where).replace("{app}", found.app)
            )
        return value

    def inputs(value: str) -> str:
        """VS Code's `${input:NAME}` as `${secret:NAME}`, recorded for the person to set."""

        def swap(match: "re.Match[str]") -> str:
            # Folded to the reference grammar (`REFERENCE_NAME`), which is also
            # what `webagents secrets set` accepts: `api-key` becomes `API_KEY`.
            name = re.sub(r"[^A-Z0-9_]", "_", match.group(1).upper())
            name = ("_" + name if re.match(r"^[0-9]", name) else name)[:128]
            if name not in result.to_set:
                result.to_set.append(name)
            return "${secret:" + name + "}"

        value = re.sub(r"\$\{input:([^}]+)\}", swap, value.replace("${workspaceFolder}", folder_text))
        reference = other_app_reference(value)
        if reference:
            refuse_reference(reference)
        return value

    def stored(preferred: str, fallback: str, value: str) -> str:
        name = _secret_name(store, preferred, fallback, value)
        if store.get(name) != value:
            store.set(name, value)
        if name not in result.stored:
            result.stored.append(name)
        return name

    if "command" in entry:
        entry["command"] = plain(entry["command"], "command")
        entry["args"] = [plain(a, "arguments") for a in entry.get("args", [])]
        if masked_args(entry["args"])[1]:
            raise AddRefused(
                WORDS["keyOnCommandLine"].replace("{name}", found.name).replace("{field}", "arguments").replace("{app}", found.app)
            )
        if "cwd" in entry:
            entry["cwd"] = plain(entry["cwd"], "cwd")
        # Every value is checked before any key is stored, so a refusal leaves the store as it was.
        checked = {key: inputs(value) for key, value in entry.get("env", {}).items()}
        env: Dict[str, str] = {}
        for key, value in checked.items():
            if "${" not in value and value and (looks_like_secret(value) or _KEY_NAME.search(key)):
                value = "${secret:" + stored(key, suggested_secret_name(found.name, key), value) + "}"
            env[key] = value
        if env:
            entry["env"] = env
        return result
    # A key in the address's query is stored and read back by reference: an
    # address is not on the process list, and references resolve in it.
    url = inputs(entry["url"])
    checked_headers = {key: inputs(value) for key, value in entry.get("headers", {}).items()}
    parts: List[str] = []
    last = 0
    for match in _QUERY.finditer(url):
        separator, param, value = match.groups()
        if _query_key(param, value):
            name = stored(suggested_secret_name(found.name, param), suggested_secret_name(found.name, param), value)
            parts += [url[last : match.start()], f"{separator}{param}=${{secret:{name}}}"]
            last = match.end()
    entry["url"] = "".join(parts) + url[last:]
    headers: Dict[str, str] = {}
    for key, value in checked_headers.items():
        bearer = re.match(r"^(Bearer\s+)(\S.*)$", value)
        if bearer and "${" not in bearer.group(2):
            value = bearer.group(1) + "${secret:" + stored(suggested_secret_name(found.name, "token"), suggested_secret_name(found.name, "token"), bearer.group(2)) + "}"
        elif "${" not in value and value and (looks_like_secret(value) or _KEY_FIELD.search(key)):
            value = "${secret:" + stored(suggested_secret_name(found.name, key), suggested_secret_name(found.name, key), value) + "}"
        headers[key] = value
    if headers:
        entry["headers"] = headers
    return result


def _yaml_scalar(value: str) -> str:
    """`value` as `skills_edit._yaml_scalar` writes it, but with non-ASCII kept
    as it is, the way the TypeScript `yamlScalar` writes it, so both CLIs write
    the same bytes (and an emoji is not split into two escaped surrogates)."""
    from .skills_edit import _yaml_scalar as scalar

    plain = scalar(value)
    return plain if plain == value else json.dumps(value, ensure_ascii=False)


def entry_lines(name: str, entry: Dict[str, Any], indent: int, step: int) -> List[str]:
    """`name: {entry}` as block YAML at `indent`, fields `step` deeper."""
    pad, inner, deeper = " " * indent, " " * (indent + step), " " * (indent + 2 * step)
    lines = [f"{pad}{_yaml_scalar(name)}:"]
    for key in ("command", "args", "cwd", "url", "transport"):
        if key not in entry:
            continue
        value = entry[key]
        lines.append(f"{inner}{key}: {json.dumps(value, ensure_ascii=False) if isinstance(value, list) else _yaml_scalar(str(value))}")
    for key in ("env", "headers"):
        if entry.get(key):
            lines.append(f"{inner}{key}:")
            lines.extend(f"{deeper}{_yaml_scalar(k)}: {_yaml_scalar(v)}" for k, v in entry[key].items())
    return lines


class CannotEdit(Exception):
    """The agent file's `mcp` block is in a layout this does not edit."""


def insert_into_agent_file(text: str, name: str, entry: Dict[str, Any], file: str) -> str:
    """The agent file `text` with `name` added to its `- mcp:` block (module
    docstring). Raises `CannotEdit` for a layout it will not touch."""
    from .skills_edit import SkillListError, _entries_of, _is_trivia, _parse

    try:
        parsed = _parse(text, file)
        entries = _entries_of(parsed.data, file)
    except SkillListError:
        raise CannotEdit(file) from None
    block = next((e["mcp"] for e in entries if isinstance(e, dict) and isinstance(e.get("mcp"), dict)), None)
    if parsed.close < 0 or block is None:
        raise CannotEdit(file)
    lines = parsed.lines
    wrapper = isinstance(block.get("mcpServers"), dict)

    def indent_of(line: str) -> int:
        return len(line) - len(line.lstrip(" "))

    start = next((i for i in range(1, parsed.close) if re.match(r"^\s*-\s+mcp:\s*(#.*)?$", lines[i])), None)
    if start is None:
        raise CannotEdit(file)
    parent = indent_of(lines[start])
    if wrapper:
        start = next(
            (i for i in range(start + 1, parsed.close) if re.match(r"^\s*mcpServers:\s*(#.*)?$", lines[i]) and indent_of(lines[i]) > parent),
            None,
        )
        if start is None:
            raise CannotEdit(file)
        parent = indent_of(lines[start])
    first = next((i for i in range(start + 1, parsed.close) if not _is_trivia(lines[i])), None)
    if first is None or indent_of(lines[first]) <= parent or "\t" in lines[first][: indent_of(lines[first]) + 1]:
        raise CannotEdit(file)
    server_indent = indent_of(lines[first])
    deeper = next((i for i in range(first + 1, parsed.close) if not _is_trivia(lines[i]) and indent_of(lines[i]) > server_indent), None)
    step = indent_of(lines[deeper]) - server_indent if deeper is not None else 2
    end = first
    for i in range(first + 1, parsed.close):
        if _is_trivia(lines[i]):
            continue
        if indent_of(lines[i]) < server_indent:
            break
        end = i
    added = lines[: end + 1] + entry_lines(name, entry, server_indent, step) + lines[end + 1 :]
    new_text = parsed.bom + parsed.nl.join(added)
    expected = copy.deepcopy(parsed.data)
    target = next(e["mcp"] for e in _entries_of(expected, file) if isinstance(e, dict) and isinstance(e.get("mcp"), dict))
    (target["mcpServers"] if wrapper else target)[name] = entry
    try:
        if _parse(new_text, file).data != expected:
            raise CannotEdit(file)
    except SkillListError:
        raise CannotEdit(file) from None
    return new_text


@dataclass
class AddPlan:
    """What `add` will write: the files and their new text, and what to say."""

    writes: List[Tuple[Path, str]]
    where: str
    converted: Converted


def _agent_servers(data: Dict[str, Any], mcp_json: Optional[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """Where the agent reads its servers, and their names: `inline` (its own
    `- mcp:` block), `json` (a bare `- mcp`, or one with nothing under it, reads
    `mcp.json`), or `none` (no mcp skill yet; `mcp.json` once one is added)."""
    raw = mcp_json or {}
    json_servers = raw["mcpServers"] if isinstance(raw.get("mcpServers"), dict) else raw
    entries = data.get("skills") if isinstance(data.get("skills"), list) else []
    for entry in entries:
        if entry == "mcp":
            return "json", json_servers
        if isinstance(entry, dict) and "mcp" in entry:
            block = entry["mcp"]
            if isinstance(block, dict) and block:
                return "inline", block["mcpServers"] if isinstance(block.get("mcpServers"), dict) else block
            return "json", json_servers
    return "none", json_servers


def choose(discovery: Discovery, name: str, app: Optional[str], list_command: str) -> FoundServer:
    """The one server `name` (from `app` when given) names; raises `AddRefused` otherwise."""
    if app is not None and app not in [a for a, _ in APPS]:
        raise AddRefused(WORDS["unknownApp"].replace("{app}", app).replace("{apps}", ", ".join(a for a, _ in APPS)))
    matches = [f for f in discovery.found if f.name == name and (app is None or f.app_id == app)]
    if not matches:
        raise AddRefused(WORDS["notFound"].replace("{name}", name).replace("{command}", list_command))
    distinct = {json.dumps(m.entry, sort_keys=True) for m in matches}
    if len(distinct) > 1:
        apps = ", ".join(dict.fromkeys(m.app_id for m in matches))
        raise AddRefused(WORDS["ambiguous"].replace("{name}", name).replace("{apps}", apps))
    return matches[0]


def plan_add(found: FoundServer, agent_file: Path, store: SecretStoreLike, agent_name: str) -> AddPlan:
    """The files `add` writes for `found`, nothing written yet (secrets aside)."""
    from .skills_edit import _parse, edit_skill_list

    text = agent_file.read_text(encoding="utf-8")
    parsed = _parse(text, agent_file.name)
    mcp_json_file = agent_file.parent / "mcp.json"
    mcp_json: Optional[Dict[str, Any]] = None
    if mcp_json_file.is_file():
        loaded = json.loads(mcp_json_file.read_text(encoding="utf-8"))
        mcp_json = loaded if isinstance(loaded, dict) else {}
    kind, servers = _agent_servers(parsed.data, mcp_json)
    if found.name in servers:
        raise AddRefused(WORDS["exists"].replace("{agent}", agent_name).replace("{name}", found.name))
    converted = convert_entry(found, agent_file.parent.resolve(), store)
    if kind == "inline":
        try:
            new_text = insert_into_agent_file(text, found.name, converted.entry, agent_file.name)
        except CannotEdit:
            # The lines to paste read the keys by reference; say where they went.
            snippet = "\n".join(entry_lines(found.name, converted.entry, 2, 2))
            said = "\n" + WORDS["stored"].replace("{names}", ", ".join(converted.stored)) if converted.stored else ""
            raise AddRefused(WORDS["cannotEdit"].replace("{file}", agent_file.name) + "\n" + snippet + said) from None
        return AddPlan([(agent_file, new_text)], agent_file.name, converted)
    data = mcp_json if mcp_json is not None else {"mcpServers": {}}
    target = data["mcpServers"] if isinstance(data.get("mcpServers"), dict) or mcp_json is None else data
    target[found.name] = converted.entry
    writes: List[Tuple[Path, str]] = [(mcp_json_file, json.dumps(data, indent=2, ensure_ascii=False) + "\n")]
    if kind == "none":
        edit = edit_skill_list(text, "add", ["mcp"], agent_file.name)
        if edit.changed:
            writes.append((agent_file, edit.text))
    return AddPlan(writes, "mcp.json", converted)


def result_lines(plan: AddPlan, name: str, agent_name: str, in_chat: bool) -> List[str]:
    """What `add` says after writing."""
    from .config_store import cli_command

    lines = [WORDS["added"].replace("{name}", name).replace("{agent}", agent_name).replace("{where}", plan.where)]
    if plan.converted.stored:
        lines.append(WORDS["stored"].replace("{names}", ", ".join(plan.converted.stored)))
    if plan.converted.to_set:
        hints = ", ".join(f"`{cli_command(f'secrets set {n}')}`" for n in plan.converted.to_set)
        lines.append(WORDS["toSet"].replace("{names}", ", ".join(plan.converted.to_set)).replace("{hints}", hints))
    lines.append(WORDS["reload"] if in_chat else WORDS["restart"])
    return lines


def write_plan(plan: AddPlan) -> None:
    """Write every file of `plan`, each by renaming a temporary file."""
    from .skills_edit import write_by_rename

    for file, text in plan.writes:
        write_by_rename(file, text)


def secret_store() -> SecretStoreLike:
    """The store `${secret:NAME}` reads (`owner_reference_sources`)."""
    from webagents.cli.commands.secrets import _store

    return _store(quiet=True)


# -- remove -------------------------------------------------------------------------------------
#
# `mcp remove <name>` and `/mcp remove <name>` (2026-09-29, the subcommand rule:
# what `add` puts in, `remove` takes out). The entry goes from where the agent
# reads it: its line block in the agent file's own `- mcp:` block (every other
# byte as it was, read back as YAML and compared before it is written), or its
# key in `mcp.json`. The last server of an agent file's own block takes the
# whole `- mcp` entry with it: a bare `- mcp` left behind would read
# `mcp.json`, and servers the person never chose would start. The secrets the
# entry read stay stored, since another entry may read them too; the sentence
# names them.


class RemoveRefused(Exception):
    """`mcp remove` would not take the server out; the message says why."""


_KEY_LINE = re.compile(r"""^(\s*)(?:"((?:[^"\\]|\\.)*)"|'((?:[^']|'')*)'|([^\s:#'"][^:#]*?))\s*:(?:\s|$)""")


def _line_key(line: str) -> Optional[str]:
    """The mapping key a block line opens, unquoted; None for any other line."""
    match = _KEY_LINE.match(line)
    if not match:
        return None
    if match.group(2) is not None:
        return json.loads(f'"{match.group(2)}"')
    if match.group(3) is not None:
        return match.group(3).replace("''", "'")
    return match.group(4).strip()


def remove_from_agent_file(text: str, name: str, file: str) -> str:
    """The agent file `text` with the server `name` taken out of its `- mcp:`
    block (the note above); raises `CannotEdit` for a layout it will not touch."""
    from .skills_edit import SkillListError, _entries_of, _is_trivia, _parse

    try:
        parsed = _parse(text, file)
        entries = _entries_of(parsed.data, file)
    except SkillListError:
        raise CannotEdit(file) from None
    holder = next((e for e in entries if isinstance(e, dict) and isinstance(e.get("mcp"), dict)), None)
    if parsed.close < 0 or holder is None:
        raise CannotEdit(file)
    block = holder["mcp"]
    wrapper = isinstance(block.get("mcpServers"), dict)
    servers = block["mcpServers"] if wrapper else block
    if name not in servers:
        raise CannotEdit(file)
    lines = parsed.lines

    def indent_of(line: str) -> int:
        return len(line) - len(line.lstrip(" "))

    def end_of(start: int, parent: int) -> int:
        """The last line of the block opened at `start`, whose children are deeper than `parent`."""
        end = start
        for i in range(start + 1, parsed.close):
            if _is_trivia(lines[i]):
                continue
            if indent_of(lines[i]) <= parent:
                break
            end = i
        return end

    entry_line = next((i for i in range(1, parsed.close) if re.match(r"^\s*-\s+mcp:\s*(#.*)?$", lines[i])), None)
    if entry_line is None:
        raise CannotEdit(file)
    parent_indent = indent_of(lines[entry_line])
    top = entry_line
    if wrapper:
        top = next(
            (i for i in range(entry_line + 1, parsed.close) if re.match(r"^\s*mcpServers:\s*(#.*)?$", lines[i]) and indent_of(lines[i]) > parent_indent),
            None,
        )
        if top is None:
            raise CannotEdit(file)
    first = next((i for i in range(top + 1, parsed.close) if not _is_trivia(lines[i])), None)
    if first is None:
        raise CannotEdit(file)
    server_indent = indent_of(lines[first])
    at = next(
        (i for i in range(first, parsed.close) if not _is_trivia(lines[i]) and indent_of(lines[i]) == server_indent and _line_key(lines[i]) == name),
        None,
    )
    if at is None:
        raise CannotEdit(file)
    expected = copy.deepcopy(parsed.data)
    target_entries = _entries_of(expected, file)
    target = next(e for e in target_entries if isinstance(e, dict) and isinstance(e.get("mcp"), dict))
    (target["mcp"]["mcpServers"] if wrapper else target["mcp"]).pop(name)
    if len(servers) == 1 and set(block) <= ({"mcpServers"} if wrapper else set(block)):
        # The last server: the whole `- mcp` entry goes (the note above).
        target_entries.remove(target)
        start, end = entry_line, end_of(entry_line, parent_indent)
    else:
        start, end = at, end_of(at, server_indent)
        # The comment lines right above a server, at its indent, are about it.
        while start > first and lines[start - 1].strip().startswith("#") and indent_of(lines[start - 1]) == server_indent:
            start -= 1
    kept = lines[:start] + lines[end + 1 :]
    new_text = parsed.bom + parsed.nl.join(kept)
    try:
        if _parse(new_text, file).data != expected:
            raise CannotEdit(file)
    except SkillListError:
        raise CannotEdit(file) from None
    return new_text


def _secret_names(entry: Any) -> List[str]:
    """The `${secret:NAME}` references in an entry, in order, once each."""
    return list(dict.fromkeys(re.findall(r"\$\{secret:([A-Za-z_][A-Za-z0-9_]*)\}", json.dumps(entry))))


def plan_remove(name: str, agent_file: Path, agent_name: str) -> Tuple[List[Tuple[Path, str]], str, List[str]]:
    """The files `remove` writes, where the entry was, and the secrets it read."""
    from .skills_edit import _parse

    text = agent_file.read_text(encoding="utf-8")
    parsed = _parse(text, agent_file.name)
    mcp_json_file = agent_file.parent / "mcp.json"
    mcp_json: Optional[Dict[str, Any]] = None
    if mcp_json_file.is_file():
        loaded = json.loads(mcp_json_file.read_text(encoding="utf-8"))
        mcp_json = loaded if isinstance(loaded, dict) else {}
    kind, servers = _agent_servers(parsed.data, mcp_json)
    if kind == "none" or name not in servers:
        raise RemoveRefused(WORDS["notThere"].replace("{agent}", agent_name).replace("{name}", name))
    secrets = _secret_names(servers[name])
    if kind == "inline":
        try:
            return [(agent_file, remove_from_agent_file(text, name, agent_file.name))], agent_file.name, secrets
        except CannotEdit:
            raise RemoveRefused(WORDS["cannotRemove"].replace("{file}", agent_file.name).replace("{name}", name)) from None
    data = mcp_json if mcp_json is not None else {}
    (data["mcpServers"] if isinstance(data.get("mcpServers"), dict) else data).pop(name)
    return [(mcp_json_file, json.dumps(data, indent=2, ensure_ascii=False) + "\n")], "mcp.json", secrets


def remove_from_agent(name: str, path: Path, *, in_chat: bool) -> List[str]:
    """`mcp remove` and `/mcp remove`: take the server `name` out of the agent
    `path` names, and say what was done. Raises `RemoveRefused` with the reason."""
    from .config_store import cli_command
    from .skills_edit import write_by_rename

    try:
        agent_file = agent_file_for(path)
    except AddRefused as refused:
        raise RemoveRefused(str(refused)) from None
    agent_name = agent_name_of(agent_file, agent_file.read_text(encoding="utf-8"))
    writes, where, secrets = plan_remove(name, agent_file, agent_name)
    for file, text in writes:
        write_by_rename(file, text)
    lines = [WORDS["removed"].replace("{name}", name).replace("{agent}", agent_name).replace("{where}", where)]
    if secrets:
        lines.append(WORDS["secretsStay"].replace("{names}", ", ".join(secrets)).replace("{command}", cli_command("secrets remove <NAME>")))
    lines.append(WORDS["reloadRemoved"] if in_chat else WORDS["restartRemoved"])
    return lines


def agent_file_for(path: Path) -> Path:
    """The agent file `path` names: the file itself, or a folder's `AGENT.md`,
    or its one `AGENT-<name>.md`; raises `AddRefused` otherwise."""
    from .config_store import cli_command

    if path.is_file():
        return path
    folder = path if path.is_dir() else path.parent
    if (folder / "AGENT.md").is_file():
        return folder / "AGENT.md"
    named = sorted(p for p in folder.iterdir() if re.fullmatch(r"AGENT-.+\.md", p.name)) if folder.is_dir() else []
    if len(named) == 1:
        return named[0]
    if named:
        raise AddRefused(WORDS["manyAgents"].replace("{folder}", str(folder)).replace("{files}", ", ".join(p.name for p in named)))
    raise AddRefused(WORDS["noAgent"].replace("{folder}", str(folder)).replace("{command}", cli_command("init")))


def agent_name_of(agent_file: Path, text: str) -> str:
    """The agent's name as its file gives it: `name:`, else as the loaders name the file."""
    from .skills_edit import SkillListError, _parse

    try:
        name = _parse(text, agent_file.name).data.get("name")
    except SkillListError:
        name = None
    if isinstance(name, str) and name.strip():
        return name.strip()
    stem = agent_file.name[: -len(".md")] if agent_file.name.lower().endswith(".md") else agent_file.name
    return "default" if stem == "AGENT" else stem[len("AGENT-"):] if stem.startswith("AGENT-") else stem


def add_to_agent(
    name: str,
    path: Path,
    source: Optional[str],
    *,
    in_chat: bool,
    list_command: str,
    store: Optional[SecretStoreLike] = None,
    discovery: Optional[Discovery] = None,
) -> List[str]:
    """`mcp add` and `/mcp add`: copy the server `name` into the agent `path`
    names, and say what was done. Raises `AddRefused` with the reason."""
    agent_file = agent_file_for(path)
    found = choose(discovery if discovery is not None else discover(folder=agent_file.parent), name, source, list_command)
    agent_name = agent_name_of(agent_file, agent_file.read_text(encoding="utf-8"))
    plan = plan_add(found, agent_file, store if store is not None else secret_store(), agent_name)
    write_plan(plan)
    return result_lines(plan, found.name, agent_name, in_chat)
