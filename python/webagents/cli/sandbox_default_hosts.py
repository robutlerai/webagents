"""
Writing a host into an agent file's `sandbox: network: hosts:` (the
sandbox-default lane, 2026-09-27), for the chat's ask-on-first-use: when a
confined command was refused a host and the owner answers "always", the host
goes into the file, where the owner can see and review it, and the command
re-runs.

A LINE EDIT, NOT A REWRITE, as the chat's other edits are (`skills_edit.py`):
a YAML dump would drop the file's comments and reorder its keys. Four shapes
are handled, by indentation: no `sandbox:` key at all (a block is added
before the closing fence); `sandbox:` with `network:` holding a `hosts:`
block list (the host is appended to it); `sandbox:` with the old bare
`network:` list (appended there); an empty `hosts: []` or `network: []`
(replaced by a one-entry block list). Anything else (a flow list with
entries, `sandbox: off`, a `network:` mapping without `hosts:`) is left alone
with a sentence, and the owner adds the host by hand. The result is read back
through the loader by the caller. The TypeScript twin is
`cli/sandbox-default-hosts.ts`; the fixture
`tests/fixtures/cli/sandbox_default_hosts.json` pins the cases.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple, Union

#: What the chat says and asks (fixture `words`).
HOST_WORDS = {
    "hostRefused": "The sandbox refused {host} for `{command}`.",
    "hostQuestion": "Allow it? [o]nce, [a]lways (adds it to the agent file), [N]o: ",
    "hostWritten": "Added {host} to network.hosts in {file}.",
    "hostNotWritten": "Could not add {host} to {file}: {problem}. Add it under `sandbox: network: hosts:` by hand.",
    "hostNoFile": "the built-in agent has no file",
}


def fill_words(template: str, **values: str) -> str:
    """`{name}` placeholders filled by plain replacement (a command may hold braces of its own)."""
    out = template
    for name, value in values.items():
        out = out.replace("{" + name + "}", value)
    return out


def host_answer(typed: Optional[str]) -> str:
    """The owner's typed answer: `o`/`once`, `a`/`always`, anything else is no (fixture `answers`)."""
    word = (typed or "").strip().lower()
    if word in ("o", "once"):
        return "once"
    if word in ("a", "always"):
        return "always"
    return "no"


_FENCE = re.compile(r"^---\s*$")


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _key_line(line: str, key: str, indent: int) -> Optional[str]:
    """What follows `key:` on the line, at exactly `indent` spaces; None when it is not that key."""
    if _indent_of(line) != indent:
        return None
    trimmed = line.strip()
    if trimmed == f"{key}:":
        return ""
    if trimmed.startswith(f"{key}: "):
        return trimmed[len(key) + 2:].strip()
    return None


def _block_end(lines: Sequence[str], start: int, indent: int) -> int:
    """The index after `start` where a block at `indent` ends: the first line with a smaller indent that is not blank or a comment."""
    for index in range(start, len(lines)):
        line = lines[index]
        if not line.strip() or line.strip().startswith("#"):
            continue
        if _indent_of(line) < indent:
            return index
    return len(lines)


def _last_item(lines: Sequence[str], list_line: int, indent: int, end: int) -> Tuple[int, int]:
    """The last item line of the block list under `list_line`, and its indent; the key's line when the list is empty."""
    at, item_indent = list_line, indent + 2
    for index in range(list_line + 1, end):
        line = lines[index]
        if not line.strip() or line.strip().startswith("#"):
            continue
        if line.strip().startswith("- ") or line.strip() == "-":
            at, item_indent = index, _indent_of(line)
    return at, item_indent


def add_network_host(text: str, host: str) -> Union[Tuple[str, None], Tuple[None, str]]:
    """`(new_text, None)` with `host` in `sandbox.network.hosts`, or `(None, problem)`."""
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.split("\r\n") if newline == "\r\n" else text.split("\n")
    if not lines or not _FENCE.match(lines[0]):
        return None, "the file has no front matter"
    close = next((index for index, line in enumerate(lines) if index > 0 and _FENCE.match(line)), -1)
    if close < 0:
        return None, "the front matter never closes"
    front: List[str] = lines[:close]
    tail = lines[close:]

    def joined(parts: Sequence[str]) -> Tuple[str, None]:
        return newline.join([*parts, *tail]), None

    sandbox_at = next((index for index, line in enumerate(front) if _key_line(line, "sandbox", 0) is not None), -1)
    if sandbox_at < 0:
        return joined([*front, "sandbox:", "  network:", "    hosts:", f"      - {host}"])
    sandbox_rest = _key_line(front[sandbox_at], "sandbox", 0) or ""
    if sandbox_rest:
        return None, f"`sandbox: {sandbox_rest}` is not a block that can take a host"
    sandbox_end = _block_end(front, sandbox_at + 1, 1)
    inner = next((line for line in front[sandbox_at + 1:sandbox_end] if line.strip() and not line.strip().startswith("#")), None)
    indent = _indent_of(inner) if inner is not None else 2

    network_at = next((index for index in range(sandbox_at + 1, sandbox_end) if _key_line(front[index], "network", indent) is not None), -1)
    if network_at < 0:
        pad = " " * indent
        return joined([*front[:sandbox_end], f"{pad}network:", f"{pad}  hosts:", f"{pad}    - {host}", *front[sandbox_end:]])
    network_rest = _key_line(front[network_at], "network", indent) or ""
    network_end = _block_end(front, network_at + 1, indent + 1)
    if network_rest == "[]":
        pad = " " * indent
        return joined([*front[:network_at], f"{pad}network:", f"{pad}  - {host}", *front[network_at + 1:]])
    if network_rest:
        return None, f"`network: {network_rest}` is not a block that can take a host"

    body = [line for line in front[network_at + 1:network_end] if line.strip() and not line.strip().startswith("#")]
    is_bare_list = bool(body) and all(line.strip().startswith("- ") or line.strip() == "-" for line in body)
    if is_bare_list:
        at, item_indent = _last_item(front, network_at, indent, network_end)
        return joined([*front[: at + 1], f"{' ' * item_indent}- {host}", *front[at + 1:]])
    inner_indent = _indent_of(body[0]) if body else indent + 2
    hosts_at = next((index for index in range(network_at + 1, network_end) if _key_line(front[index], "hosts", inner_indent) is not None), -1)
    if hosts_at < 0:
        if not body:
            pad = " " * (indent + 2)
            return joined([*front[: network_at + 1], f"{pad}hosts:", f"{pad}  - {host}", *front[network_at + 1:]])
        pad = " " * inner_indent
        return joined([*front[:network_end], f"{pad}hosts:", f"{pad}  - {host}", *front[network_end:]])
    hosts_rest = _key_line(front[hosts_at], "hosts", inner_indent) or ""
    hosts_end = _block_end(front, hosts_at + 1, inner_indent + 1)
    if hosts_rest == "[]":
        pad = " " * inner_indent
        return joined([*front[:hosts_at], f"{pad}hosts:", f"{pad}  - {host}", *front[hosts_at + 1:]])
    if hosts_rest:
        return None, f"`hosts: {hosts_rest}` is not a block list that can take a host"
    at, item_indent = _last_item(front, hosts_at, inner_indent, hosts_end)
    return joined([*front[: at + 1], f"{' ' * item_indent}- {host}", *front[at + 1:]])
