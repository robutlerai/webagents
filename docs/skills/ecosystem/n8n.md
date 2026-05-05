---
title: n8n Skill
description: Trigger n8n workflows from your agent — bridges agent reasoning with n8n's 400+ service integrations.
---

# n8n Skill

> [!WARNING]
> This skill is in **alpha stage** and under active development. APIs, features, and functionality may change without notice.

> [!NOTE]
> The dedicated `N8nSkill` is currently **Python-only**. TypeScript users can call n8n's REST API today using the [OpenAPI Skill](../platform/openapi.md), the platform's MCP n8n proxy (`/api/integrations/mcp/n8n` via [MCPSkill](../core/mcp.md)), or raw `fetch` from a custom skill. Track parity at [internal/python-typescript-parity.md](../../internal/python-typescript-parity.md).

Minimalistic n8n integration for workflow automation. Execute workflows, monitor status, and manage automation tasks with secure credential storage.

## Features

- **Secure API key storage** via auth and KV skills
- **Execute workflows** with custom input data
- **List workflows** from your n8n instance
- **Monitor execution status** in real-time
- **Multi-instance support** (localhost, self-hosted, n8n Cloud)

## Quick Setup

```typescript tab="TypeScript"
// Coming soon — track at https://github.com/robutlerai/webagents/issues
// Recommended workaround: use MCPSkill with the platform n8n proxy:
//
// import { BaseAgent } from 'webagents';
// import { MCPSkill } from 'webagents/skills/mcp';
// const agent = new BaseAgent({
//   name: 'n8n-agent',
//   model: 'openai/gpt-4o',
//   skills: [
//     new MCPSkill({
//       servers: [
//         {
//           name: 'n8n',
//           url: 'https://robutler.ai/api/integrations/mcp/n8n',
//           transport: 'http',
//           auth: { type: 'bearer', token: process.env.ROBUTLER_API_KEY! },
//         },
//       ],
//     }),
//   ],
// });
```

```python tab="Python"
from webagents.agents import BaseAgent
from webagents.agents.skills.ecosystem.n8n import N8nSkill

agent = BaseAgent(
    name="n8n-agent",
    model="openai/gpt-4o",
    skills={
        "n8n": N8nSkill(),  # Auto-resolves: auth, kv
    },
)
```

## Core Tools

### `n8n_setup(api_key, base_url)`
Set up n8n API credentials with automatic validation.

### `n8n_execute(workflow_id, data)`
Execute an n8n workflow with optional input data.

### `n8n_list_workflows()`
List all available workflows in your n8n instance.

### `n8n_status(execution_id)`
Check the status of a workflow execution.

## Usage Example

```typescript tab="TypeScript"
// Coming soon — track at https://github.com/robutlerai/webagents/issues
```

```python tab="Python"
messages = [
    {"role": "user", "content": "Set up n8n with API key your_api_key, list workflows, then execute workflow 123 with customer data"}
]
response = await agent.run(messages=messages)
```

## Getting Your n8n API Key

**n8n Cloud**: Settings > n8n API > Create an API key
**Self-hosted**: Settings > n8n API > Create an API key  
**Local**: Start n8n (`npx n8n start`) > Settings > n8n API > Create key

## Troubleshooting

**Connection Issues** - Verify n8n instance is running and base URL is correct
**Authentication Problems** - Check API key is active and has required permissions
**Workflow Execution Issues** - Confirm workflow exists and is properly configured