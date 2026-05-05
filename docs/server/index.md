---
title: Server Overview
description: Deploy agents as OpenAI-compatible API servers — single-agent and multi-agent setups.
---

# Server Overview

Deploy agents as OpenAI-compatible API servers.

## Quick Start

### Basic Server

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';
import { serve } from 'webagents/server/node';

const agent = new BaseAgent({
  name: 'assistant',
  instructions: 'You are a helpful assistant',
  model: 'openai/gpt-4o',
});

await serve(agent, { host: '0.0.0.0', port: 8000 });
```

```python tab="Python"
from webagents.server.core.app import create_server
from webagents.agents import BaseAgent

agent = BaseAgent(
    name="assistant",
    instructions="You are a helpful assistant",
    model="openai/gpt-4o",
)

server = create_server(agents=[agent])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(server.app, host="0.0.0.0", port=8000)
```

### Multiple Agents

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';
import { createAgentApp } from 'webagents/server';
import { serve } from 'webagents/server/node';

const agents = [
  new BaseAgent({
    name: 'support',
    instructions: 'You are a customer service agent',
    model: 'openai/gpt-4o',
  }),
  new BaseAgent({
    name: 'analyst',
    instructions: 'You are a data analyst',
    model: 'anthropic/claude-3-sonnet',
  }),
];

const app = createAgentApp({ agents, title: 'Multi-Agent Server' });
await serve(app, { host: '0.0.0.0', port: 8000 });
```

```python tab="Python"
from webagents.agents.skills.core.memory import ShortTermMemorySkill

agents = [
    BaseAgent(
        name="support",
        instructions="You are a customer service agent",
        model="openai/gpt-4o",
        skills={"memory": ShortTermMemorySkill()},
    ),
    BaseAgent(
        name="analyst",
        instructions="You are a data analyst",
        model="anthropic/claude-3-sonnet",
    ),
]

server = create_server(
    title="Multi-Agent Server",
    agents=agents,
)
```

> [!NOTE]
> Agent names can include dots for namespace hierarchy (e.g. `alice.my-bot`, `alice.my-bot.helper`).
> Dots are ordinary characters in URL path segments, so names like `alice.my-bot` are served at
> `/agents/alice.my-bot/chat/completions` with zero routing changes.

## Server Parameters

The `create_server()` function accepts these key parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `title` | str | "WebAgents Server" | Server title for OpenAPI docs |
| `description` | str | "AI Agent Server..." | Server description |
| `version` | str | "1.0.0" | API version |
| `agents` | List[BaseAgent] | [] | Static agents to serve |
| `dynamic_agents` | Callable | None | Dynamic agent resolver function |
| `url_prefix` | str | "" | URL prefix (e.g., "/agents") |

### Advanced Parameters

```typescript tab="TypeScript"
import { createAgentApp } from 'webagents/server';

const app = createAgentApp({
  agents,
  title: 'Production Server',
  urlPrefix: '/api/v1',
  enableCors: true,
  requestTimeoutMs: 300_000,
  // dynamicAgents: resolveAgent,  // see ../server/dynamic-agents.md
});
```

```python tab="Python"
server = create_server(
    title="Production Server",
    agents=agents,
    dynamic_agents=resolve_agent,
    url_prefix="/api/v1",
    enable_monitoring=True,
    enable_cors=True,
    request_timeout=300.0,
)
```

## API Endpoints

The server automatically creates these endpoints for each agent:

```
GET  /                              # Server info
GET  /health                        # Health check
GET  /{agent_name}                  # Agent info
POST /{agent_name}/chat/completions # OpenAI-compatible chat
GET  /{agent_name}/health           # Agent health
```

With `url_prefix="/agents"`:
```
POST /agents/{agent_name}/chat/completions
```

## Client Usage

### OpenAI SDK Compatible

```typescript tab="TypeScript"
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:8000/assistant',
  apiKey: 'your-api-key',
});

const response = await client.chat.completions.create({
  model: 'gpt-4o',
  messages: [{ role: 'user', content: 'Hello!' }],
});
```

```python tab="Python"
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/assistant",
    api_key="your-api-key",
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### Streaming

```typescript tab="TypeScript"
const stream = await client.chat.completions.create({
  model: 'gpt-4o',
  messages: [{ role: 'user', content: 'Tell me a story' }],
  stream: true,
});

for await (const chunk of stream) {
  const delta = chunk.choices[0].delta.content;
  if (delta) process.stdout.write(delta);
}
```

```python tab="Python"
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Environment Variables

```bash
# LLM Provider Keys
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Optional Server Configuration
ROBUTLER_HOST=0.0.0.0
ROBUTLER_PORT=8000
```

## See Also

- **[Dynamic Agents](./dynamic-agents.md)** - Runtime agent loading
- **[Architecture](./architecture.md)** - Production patterns
- **[Agent Endpoints](../agent/endpoints.md)** - Custom HTTP endpoints