---
title: MCP Skill
---
# MCP Skill

Connect any [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server to your agent. The MCP skill discovers tools, resources, and prompts from external servers and makes them available as native agent tools.

## Overview

MCP is the general-purpose integration path for tool ecosystems. Instead of writing custom skills for each service, point the MCP skill at any MCP-compatible server and its tools become available to your agent automatically.

The skill uses the official MCP Python SDK with support for multiple transport types (SSE, HTTP, WebSocket), automatic reconnection, and background capability refresh.

## Configuration

```python
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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `servers` | list | `[]` | MCP server definitions |
| `timeout` | float | `30.0` | Request timeout in seconds |
| `reconnect_interval` | float | `60.0` | Seconds between reconnection attempts |
| `max_connection_errors` | int | `5` | Errors before giving up on a server |
| `capability_refresh_interval` | float | `300.0` | Seconds between capability re-discovery |

### Server Config

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Identifier for this server |
| `url` | Yes | Server endpoint URL |
| `transport` | No | `sse`, `http`, or `websocket` (default: `sse`) |
| `auth` | No | Authentication config (`{"type": "bearer", "token": "..."}`) |

## How It Works

On initialization, the skill connects to each configured MCP server and discovers its capabilities:

1. **Tools** are registered as agent tools — the LLM can call them directly
2. **Resources** are exposed for data retrieval
3. **Prompts** are available for prompt injection

The skill runs background tasks for health monitoring and capability refresh, automatically reconnecting if a server goes down.

## Platform MCP Proxy

When running on the Robutler platform, agents can also access MCP servers through the platform's proxy at `/api/integrations/mcp/{provider}`. The proxy handles authentication for connected accounts (Google, n8n, etc.) and supports tool-level [pricing](../../payments/tool-pricing) with `_metering`.

See the [MCP Integration Guide](../../guides/mcp-integration) for platform-specific setup.

## Dynamic Tool Registration

Skills can register additional MCP tools at runtime:

```python
class MySkill(Skill):
    @tool
    async def add_server(self, name: str, url: str) -> str:
        """Dynamically add an MCP server."""
        mcp = self.agent.skills["mcp"]
        await mcp._register_mcp_server({"name": name, "url": url})
        return f"Connected to {name}"
```

## See Also

- [MCP Integration Guide](../../guides/mcp-integration) — Platform proxy and connected accounts
- [OAuth Client Skill](../platform/oauth-client) — Authenticate with OAuth APIs
- [OpenAPI Skill](../platform/openapi) — Auto-generate tools from API specs
