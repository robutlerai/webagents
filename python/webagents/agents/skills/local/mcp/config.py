"""
The `mcp` entry of an agent file, normalized (plan item 0.3, 2026-09-26).

WHAT A FILE MAY WRITE. Two shapes: the servers at the top level
(`- mcp: {sqlite: {command: npx, ...}}`) or under `mcpServers`, the key
`mcp.json` files use, so a file copied from another tool's config loads
unchanged. A server is stdio (`command`, with `args`, `env`, `cwd`) or remote
(`url`, or `httpUrl`; `transport` names `http` for Streamable HTTP, `sse`, or
`auto` when unset: Streamable HTTP first, then SSE; `headers` go on every
request). The shapes and what each resolves to are pinned by
`tests/fixtures/mcp_tool/config_shapes.json`, which both SDKs run.

THE `mcpServers` SHAPE LOADED NOTHING HERE (2026-09-26). The agent builder
handed it to the skill unwrapped, and the skill's config scan looked for
`command` or `url` one level too high, so a file written in the shape the
documentation showed connected to no server and said nothing. Both shapes now
go through this one function.

A server that names neither is rejected BY NAME, and the others still load.

ONE NAME PER TOOL. A server's tool is `<server>__<tool>`, always, as in
TypeScript. This skill used the bare name unless it collided with one already
registered, which made a tool's name depend on which OTHER servers the file
named and on the order they connected in. An `access.tools` rule, and the
model's tool calls, refer to the name, so a name that can change when a server
is added is a rule that can stop matching without anyone touching it.

The extra keys a hosted platform sets on a server (`pricing`, `enabledTools`,
`toolPolicies`, `auth`, `mcpUrlTemplate`, `urlQuery`, `prompt`) are kept on the
server's `configs` entry untouched; only the transport shape is normalized.
`mcpUrlTemplate` counts as a remote address, composed at connect time.

SECRETS (S-292, 2026-09-26). A value in `env`, `headers` or the address may be
`${secret:NAME}` or `${env:NAME}` (`skills/local/secrets/references.py`),
resolved by the skill at connect time and never here, so nothing that reads a
resolution sees a value. Two things ARE decided here, at load, so `doctor` and
the chat can say them before any server starts: a reference in `command` or
`args` rejects that server by name (a command line is readable by every local
account), and a literal that looks like a key (`sk-`, `ghp_`, a long bearer)
in `env` or `headers` draws a warning that names the reference and the
`webagents secrets set` command to use instead. Both sentences are pinned by
the fixture's `secret_refs` section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..secrets.references import command_line_refusal, literal_warning, looks_like_secret, mentions_reference

#: Between a server's name and its tool's name.
TOOL_SEPARATOR = "__"

_TRANSPORTS = ("http", "sse", "auto")

#: The refusal for a server config that asks for the `notify` tool policy
#: (S-286, 2026-09-26). TypeScript honours `toolPolicies: notify` through a
#: host-provided `policyHook` (`skills/mcp/skill.ts`), gating each call on an
#: approval decision. Python has no such hook and no approval channel, so a
#: `notify` policy here would silently run the tool the owner meant to approve.
#: Rather than that, the server is refused by name with this sentence, so the
#: owner sees it at load. `block` (withhold a tool) and `allow` (run it) are
#: fine and are kept on the config untouched. Pinned by the shared fixture
#: `tests/fixtures/mcp_tool/config_shapes.json` (`tool_policies`).
NOTIFY_POLICY_UNSUPPORTED = (
    "the notify tool policy is not supported: this SDK has no approval gate for MCP calls, "
    "so use block to withhold a tool or allow to run it"
)


def _notify_policy_reason(entry: Dict[str, Any]) -> str:
    """The refusal when `toolPolicies` maps any tool to `notify`, else empty."""
    policies = entry.get("toolPolicies")
    if isinstance(policies, dict) and any(value == "notify" for value in policies.values()):
        return NOTIFY_POLICY_UNSUPPORTED
    return ""


#: The refusal for an entry that asks for a sandbox (`sandbox: true`, or any
#: value but false and null) the agent cannot give it (S-313, 2026-09-27).
#: The skill used to print "Running locally" and start the server with the
#: owner's permissions when no Docker `SandboxSkill` was loaded: the S-217
#: shape, a declaration that does something other than it says. The skill now
#: refuses the entry by name when it initializes (it is the one place the
#: agent's other skills are known), and `server_report()` lists it as
#: `rejected`. The TypeScript skill, which has no sandbox for MCP servers at
#: all, refuses the key at load with its own sentence. Pinned by the shared
#: fixture `tests/fixtures/mcp_tool/config_shapes.json` (`sandbox_key`).
SANDBOX_UNAVAILABLE = (
    "asks for a sandbox (sandbox: true) that this agent cannot provide, so it did not start: "
    "add the sandbox skill (Docker) to the agent, or remove the key to run it with your own permissions"
)


def asks_for_sandbox(entry: Any) -> bool:
    """Whether an entry asks for a sandbox: the key present with any value but false and None."""
    if not isinstance(entry, dict) or "sandbox" not in entry:
        return False
    value = entry.get("sandbox")
    return value is not None and value is not False


#: What a CLI says on stderr about a server that did not load or connect, or
#: carries a literal that looks like a key (2026-09-26, the e2e run: `-p` was
#: silent about MCP problems while the TypeScript CLI said them). The
#: TypeScript skill prints them itself at load and connect; `webagents -p`
#: prints them from the skill's `server_report()` after the agent is built
#: (`problem_lines`), the chat logs them to its file, and `doctor` prints
#: neither, its `mcp` check carrying the same findings. Pinned by the shared
#: fixture (`problem_lines`).
MCP_PROBLEM_LINES = {
    "failed": '[MCPSkill] Server "{name}" failed to connect: {error}',
    "rejected": '[MCPSkill] Server "{name}" {reason}; skipping it.',
    "warning": '[MCPSkill] Server "{name}" {warning}',
}


def problem_lines(report: List[Dict[str, Any]]) -> List[str]:
    """The stderr lines for a `server_report()`: a rejected entry, a failed
    connection, and each literal warning, in the report's order."""
    lines: List[str] = []
    for row in report:
        name = str(row.get("name", ""))
        if row.get("rejected"):
            lines.append(MCP_PROBLEM_LINES["rejected"].format(name=name, reason=row["rejected"]))
        elif row.get("error"):
            lines.append(MCP_PROBLEM_LINES["failed"].format(name=name, error=row["error"]))
        for warning in row.get("warnings") or []:
            lines.append(MCP_PROBLEM_LINES["warning"].format(name=name, warning=warning))
    return lines


def qualified_tool_name(server: str, tool: str) -> str:
    """The name a server's tool gets in the agent: `<server>__<tool>`, always."""
    return f"{server}{TOOL_SEPARATOR}{tool}"


def filter_discovered_tools(tools: List[Any], entry: Any) -> List[Any]:
    """The tools of a server that are registered, from what it listed
    (S-286 addendum, 2026-09-26): only those in `enabledTools` when the entry
    sets that list (the allowlist a hosted platform writes), and never one
    whose `toolPolicies` entry is `block`. TypeScript applied both at
    discovery (`skills/mcp/skill.ts`, `_discoverCapabilities`); this SDK
    registered every tool a server listed, so a tool the owner blocked stayed
    callable. Pinned by the shared fixture (`tool_policies.filtering`)."""
    if not isinstance(entry, dict):
        return list(tools)
    kept = list(tools)
    enabled = entry.get("enabledTools")
    if isinstance(enabled, list):
        allowed = {str(name) for name in enabled}
        kept = [tool for tool in kept if getattr(tool, "name", None) in allowed]
    policies = entry.get("toolPolicies")
    if isinstance(policies, dict):
        kept = [tool for tool in kept if policies.get(getattr(tool, "name", None)) != "block"]
    return kept


def sdk_missing(reason: str) -> str:
    """The load-time error when the MCP SDK cannot be loaded (fixture `missing_sdk.python`)."""
    return (
        f"The mcp skill needs the MCP SDK, which could not be loaded: {reason}. "
        "Install it next to webagents (pip install mcp)."
    )


@dataclass
class McpServersResolution:
    #: The servers that load, in the file's order, as the fixture describes them.
    servers: List[Dict[str, Any]] = field(default_factory=list)
    #: The entries that do not, each with the fixture's reason.
    rejected: List[Dict[str, str]] = field(default_factory=list)
    #: Each loading server's config as written, with `transport` filled in, for the skill's extra keys.
    configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    #: Servers that load but carry a secret-looking literal in `env` or
    #: `headers` (S-292): one entry per offending key, said once at load and
    #: again by `doctor`. Never a value.
    warnings: List[Dict[str, str]] = field(default_factory=list)


def _string_map(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(entry) for key, entry in value.items() if entry is not None}


def servers_from_config(raw: Any) -> McpServersResolution:
    """The servers a `mcp` entry names. Accepts the top-level shape and the
    `mcpServers` wrapper; anything that is not a mapping names no server."""
    resolution = McpServersResolution()
    if not isinstance(raw, dict):
        return resolution
    entries = raw["mcpServers"] if isinstance(raw.get("mcpServers"), dict) else raw
    for name, entry in entries.items():
        name = str(name)
        if name == "mcpServers":
            continue
        if not isinstance(entry, dict):
            resolution.rejected.append({"name": name, "reason": "is not a mapping"})
            continue
        notify_reason = _notify_policy_reason(entry)
        if notify_reason:
            resolution.rejected.append({"name": name, "reason": notify_reason})
            continue
        command = entry.get("command") if isinstance(entry.get("command"), str) and entry.get("command") else None
        address = next(
            (entry[key] for key in ("url", "httpUrl", "mcpUrlTemplate") if isinstance(entry.get(key), str) and entry.get(key)),
            None,
        )
        if command:
            args = [str(a) for a in entry["args"]] if isinstance(entry.get("args"), list) else []
            # A reference on the command line is refused, not resolved (S-292):
            # `ps` shows it to every local account, so no store can keep it secret.
            on_command_line = "command" if mentions_reference(command) else "args" if any(mentions_reference(a) for a in args) else None
            if on_command_line:
                resolution.rejected.append({"name": name, "reason": command_line_refusal(on_command_line)})
                continue
            server: Dict[str, Any] = {"name": name, "transport": "stdio", "command": command, "args": args}
            if isinstance(entry.get("env"), dict):
                server["env"] = _string_map(entry["env"])
            if isinstance(entry.get("cwd"), str) and entry["cwd"]:
                server["cwd"] = entry["cwd"]
            resolution.servers.append(server)
            resolution.configs[name] = {**entry, "transport": "stdio"}
            for key, value in server.get("env", {}).items():
                if looks_like_secret(value):
                    resolution.warnings.append({"name": name, "reason": literal_warning(name, "env", key)})
            continue
        if not address:
            resolution.rejected.append({"name": name, "reason": "has neither command nor url"})
            continue
        transport = entry.get("transport")
        transport = "auto" if transport is None else transport
        if transport not in _TRANSPORTS:
            resolution.rejected.append({"name": name, "reason": "transport must be http, sse or auto"})
            continue
        headers = _string_map(entry.get("headers"))
        resolution.servers.append({"name": name, "transport": transport, "url": address, "headers": headers})
        resolution.configs[name] = {**entry, "transport": transport}
        for key, value in headers.items():
            if looks_like_secret(value):
                resolution.warnings.append({"name": name, "reason": literal_warning(name, "headers", key)})
    return resolution
