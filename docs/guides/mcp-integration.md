---
title: MCP Integration
description: Connecting MCP tool servers to your agent.
---

# MCP Integration

WebAgents supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) for connecting external tool servers to your agent.

## Adding MCP Tools

### Via Platform UI

In the agent configuration page, add an integration of type "Custom MCP" and provide the server URL.

### Via API

```bash
curl -X POST https://robutler.ai/api/agents/{id}/integrations \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "type": "custom_mcp",
    "name": "My Tools",
    "mcpServerUrl": "https://my-mcp-server.com/mcp"
  }'
```

### Via SDK

```python
from webagents.agents.skills.core.mcp import MCPSkill

agent = BaseAgent(
    name="my-agent",
    skills={"mcp": MCPSkill(server_url="https://my-mcp-server.com/mcp")},
)
```

```typescript
import { BaseAgent, MCPSkill } from 'webagents';

const agent = new BaseAgent({
  name: 'my-agent',
  skills: [new MCPSkill({ serverUrl: 'https://my-mcp-server.com/mcp' })],
});
```

## Platform MCP Proxy

The platform provides a JSON-RPC proxy at `/api/integrations/mcp/{provider}` that routes MCP calls through connected accounts (Google, Zapier, n8n, etc.), handling authentication automatically.

## Executing Tools

Use the platform's MCP execution endpoint:

```bash
curl -X POST https://robutler.ai/api/mcp/execute \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"tool": "search_web", "args": {"query": "latest news"}}'
```

Or list available tools:

```bash
curl https://robutler.ai/api/mcp/tools \
  -H "Authorization: Bearer $TOKEN"
```

## Tool Pricing

MCP tools can be monetized. See [Tool Pricing](../payments/tool-pricing) for details on the `_metering` convention and commission distribution.
