---
title: Server Overview
description: Deploy agents as OpenAI-compatible API servers — single-agent and multi-agent setups.
---

# Server Overview

Deploy agents as OpenAI-compatible API servers.

## Quick Start

### Basic Server

```typescript tab="TypeScript"
// npm i webagents @hono/node-server
import { BaseAgent, OpenAISkill, serve } from 'webagents';

const agent = new BaseAgent({
  name: 'assistant',
  instructions: 'You are a helpful assistant',
  skills: [new OpenAISkill({ model: 'gpt-4o' })],
});

await serve(agent, { hostname: '0.0.0.0', port: 8000 });
// POST http://localhost:8000/chat/completions
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

`serve()` takes a single agent. To host several from one process, use `WebAgentsServer`:
each agent is added by name and gets its own mount.

```typescript tab="TypeScript"
// npm i webagents @hono/node-server
import { BaseAgent, OpenAISkill, AnthropicSkill, WebAgentsServer } from 'webagents';

const server = new WebAgentsServer({ hostname: '0.0.0.0', port: 8000 });

await server.addAgent('support', new BaseAgent({
  name: 'support',
  instructions: 'You are a customer service agent',
  skills: [new OpenAISkill({ model: 'gpt-4o' })],
}));

await server.addAgent('analyst', new BaseAgent({
  name: 'analyst',
  instructions: 'You are a data analyst',
  skills: [new AnthropicSkill({ model: 'claude-sonnet-5' })],
}));

await server.start();
// POST http://localhost:8000/agents/support/chat/completions
// POST http://localhost:8000/agents/analyst/chat/completions
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
        model="anthropic/claude-sonnet-5",
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
| `error_detail` | bool | False | Answer a failed run with the error's own text rather than a fixed message and a reference; see [When a Run Fails](#when-a-run-fails) |

### Advanced Parameters

The table above is the Python signature. TypeScript takes a different, smaller set of options,
and passes them to `serve()` or to the `WebAgentsServer` constructor rather than to a factory:
`port`, `hostname`, `cors`, `logging`, `basePath`, `identity` and `publicUrl`, plus
`metricsPath`, `rateLimit`, `defaultScopes` and `extensions` on `WebAgentsServer`.

```typescript tab="TypeScript"
import { BaseAgent, OpenAISkill, WebAgentsServer } from 'webagents';

const server = new WebAgentsServer({
  hostname: '0.0.0.0',
  port: 8000,
  basePath: '/api/v1',
  cors: true,
  logging: true,
});

await server.addAgent('assistant', new BaseAgent({
  name: 'assistant',
  instructions: 'You are a helpful assistant',
  skills: [new OpenAISkill({ model: 'gpt-4o' })],
}));

await server.start();
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

There is no `title`, `urlPrefix`, `enableCors`, `requestTimeoutMs`, `maxRequestSizeBytes` or
`corsOrigins` in TypeScript, and dynamic agent resolution is a Python feature. Passing one of
those to the server config is a compile error (TS2353), so the compiler catches it. Skill
configs behave differently: `SkillConfig` accepts unknown keys, so a misspelled skill option
such as `baseUrl` for `baseURL` type-checks and is then silently ignored.

## API Endpoints

The two SDKs mount differently, so check the shape for the language you are using.

**Python** puts the agent name in the path, and `url_prefix` shifts the whole set:

```
GET  /                              # Server info
GET  /health                        # Health check
GET  /{agent_name}                  # Agent info
POST /{agent_name}/chat/completions # OpenAI-compatible chat
GET  /{agent_name}/health           # Agent health
```

With `url_prefix="/agents"` the chat endpoint becomes
`POST /agents/{agent_name}/chat/completions`.

**TypeScript** has no `url_prefix` and adds no implicit agent segment. `serve(agent)` mounts at
the root, and `basePath` moves the whole mount:

```
POST /chat/completions              # serve(agent)
POST /v1/chat/completions
POST /uamp    POST /uamp/stream
GET  /health    GET  /info
GET  /.well-known/agent.json
GET  /.well-known/jwks.json
```

With `serve(agent, { basePath: '/agents/assistant' })` the chat endpoint becomes
`POST /agents/assistant/chat/completions`. `WebAgentsServer` mounts each agent at
`/agents/{name}` for you, and also serves `GET /health` at the origin.

## When a Run Fails

A request whose run fails gets a fixed message and a short reference, in the shape that
endpoint always uses for errors (a `{"error": ...}` body, or an `error` event in a stream):

```
The agent could not complete this request. Reference: 3f9a1c07
```

The error itself, stack trace included, is in the server's log under the same reference, so
search the log for it. What the failing code said, whether a model provider's response, a
tool's output or a file path, stays on the server. Errors written for the caller keep their
message: an authentication refusal (401), a payment requirement (402), and in Python a
FastAPI `HTTPException`.

In Python, `create_server(error_detail=True)` answers with the error's own text instead.
`webagents daemon` turns it on only while the daemon listens on loopback (this machine
only), where the caller is your own terminal. Leave it off on any server someone else can
reach.

## Client Usage

### OpenAI SDK Compatible

```typescript tab="TypeScript"
import OpenAI from 'openai';

const client = new OpenAI({
  // serve(agent) mounts at the root, so there is no agent segment here.
  // With basePath or WebAgentsServer, use the mount, e.g.
  // 'http://localhost:8000/agents/assistant'.
  baseURL: 'http://localhost:8000',
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