---
title: MCP Skill
description: "Connect any Model Context Protocol server to your agent: it discovers tools, resources, and prompts, with secrets kept out of the agent file."
---

# MCP Skill

Connect any [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server to your agent. The MCP skill discovers tools, resources, and prompts from external servers and makes them available as native agent tools.

## Overview

MCP is the general-purpose integration path for tool ecosystems. Instead of writing custom skills for each service, point the MCP skill at any MCP-compatible server and its tools become available to your agent automatically.

The skill supports multiple transport types (SSE, HTTP, WebSocket), automatic reconnection, and background capability refresh.

## Configuration

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';
import { MCPSkill } from 'webagents/skills/mcp';

const agent = new BaseAgent({
  name: 'mcp-agent',
  model: 'openai/gpt-4o',
  skills: [
    new MCPSkill({
      servers: [
        {
          name: 'weather',
          url: 'https://weather-mcp.example.com/mcp',
          transport: 'sse',
        },
        {
          name: 'database',
          url: 'https://db-mcp.example.com/mcp',
          transport: 'http',
          auth: { type: 'bearer', token: process.env.DB_MCP_TOKEN! },
        },
      ],
      timeout: 30_000,
      reconnectInterval: 60_000,
    }),
  ],
});
```

```python tab="Python"
import os

from webagents import BaseAgent
from webagents.agents.skills.core.mcp import MCPSkill

agent = BaseAgent(
    name="mcp-agent",
    model="openai/gpt-4o",
    skills={
        "mcp": MCPSkill({
            "servers": [
                {
                    "name": "weather",
                    "url": "https://weather-mcp.example.com/mcp",
                    "transport": "sse",
                },
                {
                    "name": "database",
                    "url": "https://db-mcp.example.com/mcp",
                    "transport": "http",
                    "auth": {"type": "bearer", "token": os.environ["DB_MCP_TOKEN"]},
                },
            ],
            "timeout": 30.0,
            "reconnect_interval": 60.0,
        }),
    },
)
```

### In an agent file

Name the servers under `mcp` in the `skills:` list. Both SDKs read the same
two shapes: the servers at the top level, or under `mcpServers` as `mcp.json`
files write them, so a config copied from another tool loads unchanged. A bare
`- mcp` reads `mcp.json` next to the agent file.

```yaml
skills:
  - openai
  - mcp:
      sqlite:                       # stdio: the server is a command
        command: uvx
        args: [mcp-server-sqlite, --db, app.db]
        env: { SQLITE_READONLY: "1" }
      docs:                         # remote: http (Streamable HTTP) or sse; unset tries http, then sse
        url: https://mcp.example.com/mcp
        transport: http
```

A remote server also takes `headers`, sent on every request. Every tool is
named `<server>__<tool>` (`sqlite__query`, `docs__search`), in both SDKs and
whatever else the file names, so an `access.tools` rule written for it keeps
matching it. A server with neither `command` nor `url` is skipped by name and
the others still load; when the MCP SDK itself cannot load, the agent reports
the `mcp` skill as failed, with the reason. `access.tools` can name the skill
(`friends: [mcp]`) to keep every tool its servers provide to that group.

In the chat, `/mcp` lists the servers with each one's transport and tools, or
why it did not connect, and `webagents doctor` connects each one in its `mcp`
check.

### Secrets in server settings

Keep keys out of the agent file. A server's `env`, `headers` and `url` can name
a secret stored on this machine, or a variable from the environment, and the
SDK puts the value in when it connects:

```yaml
skills:
  - mcp:
      github:
        command: npx
        args: [-y, "@modelcontextprotocol/server-github"]
        env:
          GITHUB_PERSONAL_ACCESS_TOKEN: ${secret:GITHUB_TOKEN}
      docs:
        url: https://mcp.example.com/mcp
        headers:
          Authorization: Bearer ${secret:DOCS_TOKEN}
      reports:
        command: uvx
        args: [mcp-server-sqlite, --db-path, ./reports.db]
        env:
          SQLITE_HOME: ${env:DATA_HOME}
```

- `${secret:NAME}` reads the store `webagents secrets set NAME` writes, for the
  active profile. `/secrets set NAME` does the same from the chat, asking for
  the value with echo off; the value never reaches the model.
- `${env:NAME}` reads the environment the agent was started in.
- `$${` is a literal `${`. Any other `${...}` is refused, with the forms it
  takes.
- References resolve in `env`, `headers` and the server's address (`url`, and
  the `httpUrl` and `mcpUrlTemplate` forms) when the server connects, in
  memory only; the file keeps the reference.
- A reference in `command` or `args` is refused when the file loads, because
  other accounts on the machine can read a command line from the process list.
  Pass the value through `env` instead.
- A reference that cannot be resolved (a secret not stored, a variable not
  set) stops that one server, with a sentence that names the reference and the
  command that stores it; the other servers still load. Values are masked in
  every message, in `/mcp` and in `webagents doctor`.

**Only your own agent files resolve references**: the `mcp:` entry of an agent
file you run and the `mcp.json` beside it. An `MCPSkill` built in code resolves
nothing unless the code passes `references`, so an application that builds MCP
settings from what its users typed never expands them against its own
environment. The MCP servers a code editor names over ACP resolve nothing
either.

**Moving a key out of a file.** When the loader finds a value that looks like
a key written into a server's `env` or `headers` (an `sk-`, `ghp_`,
`github_pat_`, `xox` or `AKIA` prefix, or a long `Bearer` value), it warns and
suggests a name, `<SERVER>_<KEY>`. To move it:

1. `webagents secrets set GITHUB_TOKEN`, and paste the value when asked.
2. Replace the value in the file with `${secret:GITHUB_TOKEN}`.
3. Run `webagents doctor`: its `mcp` check connects each server and names any
   reference that is still missing.

### The environment a stdio server sees

A stdio server (one with `command`) starts with the MCP SDK's default
environment (such as `PATH` and `HOME`) plus its own resolved `env`, and
nothing else: the keys in the agent's own environment are not passed to it. A
server that needs a variable names it, for example `API_KEY: ${env:API_KEY}`.

### Tool policies

`toolPolicies` on a server entry names what happens to one of its tools:
`allow` runs it, and `block` withholds it, so it is never registered. `notify`
asks before each call, which only a host that provides an approval hook can
do: in TypeScript a `notify` tool with no hook is refused rather than run, and
Python, which has no approval hook, refuses a server that names `notify` when
it loads. `enabledTools` on an entry keeps only the tools it lists.

### Config Reference

| Parameter (Python / TS) | Type | Default | Description |
|------------------------|------|---------|-------------|
| `servers` | list | `[]` | MCP server definitions |
| `timeout` / `timeout` | seconds (Py) / ms (TS) | 30 / 30 000 | Request timeout |
| `reconnect_interval` / `reconnectInterval` | seconds (Py) / ms (TS) | 60 / 60 000 | Reconnect delay |
| `max_connection_errors` / `maxConnectionErrors` | int | 5 | Errors before giving up on a server |
| `capability_refresh_interval` / `capabilityRefreshInterval` | seconds (Py) / ms (TS) | 300 / 300 000 | Capability re-discovery cadence |

### Server Config

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Identifier for this server |
| `url` | Yes | Server endpoint URL |
| `transport` | No | `sse`, `http`, or `websocket` (default: `sse`) |
| `auth` | No | Authentication config (`{ type: 'bearer', token: '...' }`) |

## How It Works

On initialization, the skill connects to each configured MCP server and discovers its capabilities:

1. **Tools** are registered as agent tools, so the LLM can call them directly.
2. **Resources** are exposed for data retrieval.
3. **Prompts** are available for prompt injection.

The skill runs background tasks for health monitoring and capability refresh, automatically reconnecting if a server goes down.

## Platform MCP Proxy

When running on the Robutler platform, agents can also access MCP servers through the platform's proxy at `/api/integrations/mcp/{provider}`. The proxy handles authentication for connected accounts (Google, n8n, etc.) and supports tool-level [pricing](../../payments/tool-pricing.md) with `_metering`.

See the [MCP Integration Guide](../../guides/mcp-integration.md) for platform-specific setup.

## Dynamic Tool Registration

Skills can register additional MCP servers at runtime:

```typescript tab="TypeScript"
import { Skill, tool } from 'webagents';

class MySkill extends Skill {
  readonly name = 'my-skill';

  @tool({ description: 'Dynamically add an MCP server' })
  async addServer(params: { name: string; url: string }): Promise<string> {
    const mcp = this.agent!.skills.find((s) => s.name === 'mcp') as MCPSkill;
    await mcp.registerServer({ name: params.name, url: params.url });
    return `Connected to ${params.name}`;
  }
}
```

```python tab="Python"
class MySkill(Skill):
    @tool
    async def add_server(self, name: str, url: str) -> str:
        """Dynamically add an MCP server."""
        mcp = self.agent.skills["mcp"]
        await mcp._register_mcp_server({"name": name, "url": url})
        return f"Connected to {name}"
```

## Serving an agent over MCP

The other direction: `webagents mcp serve [path]` puts an agent's tools behind
an MCP server, over stdio by default (how Claude Code, Codex and OpenCode start
a local server) or over Streamable HTTP at `/mcp` with `--http <port>`. Over
stdio the caller is you, the agent's owner. Over HTTP a caller must send a
credential; the agent's auth skills and `access:` block say who it is, and it
lists and calls only the tools it may use. A tool call runs the way the agent
runs one for the model: the same scope checks, hooks and pricing. See
[CLI commands](../../cli/commands.md#serving).

## See Also

- [MCP Integration Guide](../../guides/mcp-integration.md): platform proxy and connected accounts
- [OpenAPI Skill](../platform/openapi.md): auto-generate tools from API specs
