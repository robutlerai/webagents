---
title: MCP Skill
description: Connect any Model Context Protocol server to your agent — auto-discovers tools, resources, and prompts.
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
                    "auth": {"type": "bearer", "token": "${DB_MCP_TOKEN}"},
                },
            ],
            "timeout": 30.0,
            "reconnect_interval": 60.0,
        }),
    },
)
```

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

1. **Tools** are registered as agent tools — the LLM can call them directly.
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

## See Also

- [MCP Integration Guide](../../guides/mcp-integration.md) — Platform proxy and connected accounts
- [OAuth Client Skill](../platform/oauth-client.md) — Authenticate with OAuth APIs
- [OpenAPI Skill](../platform/openapi.md) — Auto-generate tools from API specs
