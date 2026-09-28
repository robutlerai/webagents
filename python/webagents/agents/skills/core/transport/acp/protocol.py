"""
The Agent Client Protocol (ACP v1) as this SDK speaks it: the constants,
the error codes, what `initialize` answers, the kind a tool name maps to,
and which kinds ask the client for permission (gap-closure plan item 1.6,
2026-09-26).

WHY A SEPARATE MODULE. The TypeScript SDK carries the same decisions in
`src/skills/transport/acp/protocol.ts`, and both are pinned by one fixture,
`tests/fixtures/acp/acp_protocol.json`, so an editor sees the same agent
whichever SDK runs it. Keeping the decisions in one small file, with no
transport code around them, is what keeps that comparison honest.

ACP IS STDIO. The editor spawns `webagents acp` and talks JSON-RPC 2.0 over
the process's stdin and stdout, one message per line. There is no HTTP or
WebSocket ACP endpoint any more: the one this SDK used to mount served the
CLIENT's `fs/*` and `terminal/*` methods to any caller with any credential
(S-269), duplicated `/chat/completions` and `webagents mcp serve` for the
rest, and matched no editor, since ACP's HTTP binding is still a draft.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

#: The one protocol version this agent speaks (an integer; ACP v2 is a draft).
PROTOCOL_VERSION = 1

#: The `name` of `agentInfo`; the agent's own name goes in `title`.
AGENT_INFO_NAME = "webagents"

#: Session ids start with this, so a client can tell ours apart in a log.
SESSION_ID_PREFIX = "sess_"

#: JSON-RPC 2.0 and ACP error codes (`errors` in the fixture).
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
AUTH_REQUIRED = -32000
RESOURCE_NOT_FOUND = -32002
REQUEST_CANCELLED = -32800

#: What `initialize` advertises: only what the agent answers. `loadSession`
#: because `session/load` replays history; `list` because `session/list`
#: answers; embedded context because a `resource` block's text reaches the
#: model; no image or audio prompts; MCP servers over stdio (always), HTTP
#: and SSE, the transports the `mcp` skill serves.
AGENT_CAPABILITIES: Dict[str, Any] = {
    "loadSession": True,
    "promptCapabilities": {"image": False, "audio": False, "embeddedContext": True},
    "mcpCapabilities": {"http": True, "sse": True},
    "sessionCapabilities": {"list": {}},
}

#: The registry requires at least one auth method of type `agent` or
#: `terminal`. `terminal` args REPLACE the normal args, so the client runs
#: `webagents login` in a terminal of its own.
AUTH_METHODS: List[Dict[str, Any]] = [
    {
        "id": "login",
        "name": "Log in to Robutler",
        "description": "Signs in with `webagents login`. A provider key set with `webagents secrets set` works without signing in.",
        "type": "terminal",
        "args": ["login"],
    }
]

#: Tool kinds that ask the client before the tool runs.
PERMISSION_KINDS = frozenset({"edit", "delete", "move", "execute"})

#: The choices offered with `session/request_permission`.
PERMISSION_OPTIONS: List[Dict[str, str]] = [
    {"optionId": "allow", "name": "Allow", "kind": "allow_once"},
    {"optionId": "reject", "name": "Reject", "kind": "reject_once"},
]

#: What the model is told when the user rejects a tool, or the prompt was
#: cancelled while the question was open.
REJECTED = "Rejected by the user: {tool} was not run."
CANCELLED = "Cancelled: {tool} was not run."

STOP_END_TURN = "end_turn"
STOP_CANCELLED = "cancelled"
#: A turn the agent's tool budget ended (2026-09-28, `core/tool_budget.py`):
#: its rounds ran out, or it repeated one call. The precise reason goes in
#: the response's `_meta.webagents_finish`.
STOP_MAX_TURN_REQUESTS = "max_turn_requests"

#: The kind of a tool from its name (the fixture's `tool_kinds.rules`, in
#: order; the first match wins). `execute` before `read` so `run_command` is
#: not a read; `search` before `read` so `search_file_content` is not a read.
_KIND_RULES: List[Dict[str, Any]] = [
    {"kind": "execute", "starts": ["run_", "exec", "shell", "terminal", "bash", "spawn"], "contains": ["command"]},
    {"kind": "search", "starts": ["search", "grep", "find", "glob", "discover"], "contains": ["search"]},
    {"kind": "delete", "starts": ["delete", "remove", "rm_", "unlink", "drop"]},
    {"kind": "move", "starts": ["move", "rename", "mv_"]},
    {"kind": "edit", "starts": ["write", "edit", "replace", "create", "put_", "append", "patch", "update", "set_"], "contains": ["_write"]},
    {"kind": "fetch", "starts": ["fetch", "http", "web_", "rest_", "curl", "download", "get_url"]},
    {"kind": "think", "starts": ["todo", "think", "plan", "note"]},
    {"kind": "read", "starts": ["read", "list", "get_", "cat", "view", "show", "ls_", "stat"]},
]

#: Between an MCP server's name and its tool's name (`mcp/config.py`).
_MCP_SEPARATOR = "__"


def tool_kind(name: str) -> str:
    """The ACP `kind` of a tool: `read`, `edit`, `delete`, `move`, `search`,
    `execute`, `think`, `fetch` or `other`. An MCP tool (`<server>__<tool>`)
    is judged on the part after the separator."""
    bare = (name or "").lower()
    if _MCP_SEPARATOR in bare:
        bare = bare.split(_MCP_SEPARATOR, 1)[1]
    for rule in _KIND_RULES:
        if any(bare.startswith(prefix) for prefix in rule.get("starts", ())):
            return rule["kind"]
        if any(part in bare for part in rule.get("contains", ())):
            return rule["kind"]
    return "other"


def needs_permission(kind: str) -> bool:
    return kind in PERMISSION_KINDS


class AcpError(Exception):
    """A JSON-RPC error answer: `code`, `message` and optional `data`."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_dict(self) -> Dict[str, Any]:
        error: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            error["data"] = self.data
        return error


def prompt_text(blocks: Any) -> str:
    """The user message a `session/prompt` carries: its text blocks, an
    embedded `resource`'s text under its uri, and a `resource_link` by uri.
    Image and audio blocks are refused, because `initialize` said so."""
    if not isinstance(blocks, list) or not blocks:
        raise AcpError(INVALID_PARAMS, "prompt must be a non-empty array of content blocks")
    parts: List[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            raise AcpError(INVALID_PARAMS, "each prompt block must be an object")
        kind = block.get("type")
        if kind == "text":
            parts.append(str(block.get("text", "")))
        elif kind == "resource":
            resource = block.get("resource") if isinstance(block.get("resource"), dict) else {}
            uri = resource.get("uri", "")
            text = resource.get("text")
            parts.append(f"[Resource: {uri}]\n{text}" if isinstance(text, str) else f"[Resource: {uri}]")
        elif kind == "resource_link":
            parts.append(f"[Resource link: {block.get('uri', '')}]")
        elif kind in ("image", "audio"):
            raise AcpError(INVALID_PARAMS, f"{kind} prompt blocks are not supported by this agent")
        else:
            raise AcpError(INVALID_PARAMS, f"unknown prompt block type: {kind!r}")
    return "\n\n".join(parts)


def parse_arguments(arguments: Any) -> Dict[str, Any]:
    """A tool call's arguments as the client shows them (`rawInput`): the
    JSON object, or an empty one when the text is not a JSON object."""
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        try:
            parsed = json.loads(arguments)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def text_content(text: str) -> Dict[str, Any]:
    """A tool call's `content` entry for a text result."""
    return {"type": "content", "content": {"type": "text", "text": text}}


def mcp_servers_config(entries: Any) -> Dict[str, Dict[str, Any]]:
    """The `mcp` skill's config for the `mcpServers` a session names: a stdio
    entry is `{name, command, args, env: [{name, value}]}`; an `http` or `sse`
    entry is `{type, name, url, headers: [{name, value}]}`. Entries that name
    neither a command nor a url are left out (the skill would refuse them by
    name anyway)."""
    servers: Dict[str, Dict[str, Any]] = {}
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not entry["name"]:
            continue
        name = entry["name"]
        kind = entry.get("type")
        if kind in ("http", "sse") or (kind is None and isinstance(entry.get("url"), str)):
            if not isinstance(entry.get("url"), str):
                continue
            servers[name] = {
                "url": entry["url"],
                "headers": _pairs(entry.get("headers")),
                "transport": kind or "http",
            }
        elif isinstance(entry.get("command"), str) and entry["command"]:
            servers[name] = {
                "command": entry["command"],
                "args": [str(a) for a in entry.get("args") or []],
                "env": _pairs(entry.get("env")),
            }
    return servers


def _pairs(value: Any) -> Dict[str, str]:
    """`[{name, value}]` as a mapping; a mapping is taken as it is."""
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    out: Dict[str, str] = {}
    for item in value if isinstance(value, list) else []:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            out[item["name"]] = str(item.get("value", ""))
    return out


def plan_entries(items: Any) -> Optional[List[Dict[str, str]]]:
    """A todo list as an ACP plan: `content`, `priority` (`critical` reads as
    `high`), `status` (`cancelled` items are left out)."""
    if not isinstance(items, list):
        return None
    entries: List[Dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "pending")
        if status == "cancelled":
            continue
        if status not in ("pending", "in_progress", "completed"):
            status = "pending"
        priority = str(item.get("priority") or "medium")
        if priority == "critical":
            priority = "high"
        if priority not in ("high", "medium", "low"):
            priority = "medium"
        entries.append({"content": str(item.get("content") or ""), "priority": priority, "status": status})
    return entries
