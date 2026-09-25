---
title: Agent Endpoints
description: HTTP and WebSocket endpoints exposed by your agent — `@http`, `@websocket`, scopes, SSE streaming.
---

# Agent Endpoints

Expose custom HTTP API endpoints for your agent using the `@http` decorator. Endpoints are mounted under the agent's base path and are served by the same app used for chat completions.

- Simple, declarative decorator: TypeScript `@http({ path, method, scopes })`, Python `@http("/path", method="get|post", scope="...")`.
- Path parameters and query strings supported.
- Scope-based access control (`all`, `owner`, `admin`).
- Plays nicely with skills, tools, and hooks.

## Basic Usage

```typescript tab="TypeScript"
import { BaseAgent, Skill, http, serve } from 'webagents';

class StatusSkill extends Skill {
  readonly name = 'status';

  @http({ path: '/status', method: 'GET' })
  async getStatus(_req: Request): Promise<Response> {
    return Response.json({ status: 'healthy' });
  }
}

const agent = new BaseAgent({
  name: 'assistant',
  model: 'openai/gpt-4o-mini',
  skills: [new StatusSkill()],
});

await serve(agent, { port: 8000 });
```

```python tab="Python"
from webagents import BaseAgent, http
from webagents.server.core.app import create_server
import uvicorn

@http("/status", method="get")
def get_status() -> dict:
    return {"status": "healthy"}

agent = BaseAgent(
    name="assistant",
    model="openai/gpt-4o-mini",
    capabilities=[get_status],
)

server = create_server(agents=[agent])
uvicorn.run(server.app, host="0.0.0.0", port=8000)
```

Both expose `GET /assistant/status`.

## Methods, Path, and Query

```typescript tab="TypeScript"
class UsersSkill extends Skill {
  readonly name = 'users';

  @http({ path: '/users', method: 'GET' })
  async listUsers(_req: Request): Promise<Response> {
    return Response.json({ users: ['alice', 'bob', 'charlie'] });
  }

  @http({ path: '/users', method: 'POST' })
  async createUser(req: Request): Promise<Response> {
    const data = await req.json();
    return Response.json({ created: data.name, id: 'user_123' });
  }

  @http({ path: '/users/:userId', method: 'GET' })
  async getUser(req: Request): Promise<Response> {
    const url = new URL(req.url);
    const userId = url.pathname.split('/').pop()!;
    const includeDetails = url.searchParams.get('include_details') === 'true';
    const user: Record<string, unknown> = { id: userId, name: `User ${userId}` };
    if (includeDetails) user.details = 'Extended info';
    return Response.json(user);
  }
}
```

```python tab="Python"
from webagents import http

@http("/users", method="get")
def list_users() -> dict:
    return {"users": ["alice", "bob", "charlie"]}

@http("/users", method="post")
def create_user(data: dict) -> dict:
    return {"created": data.get("name"), "id": "user_123"}

@http("/users/{user_id}", method="get")
def get_user(user_id: str, include_details: bool = False) -> dict:
    user = {"id": user_id, "name": f"User {user_id}"}
    if include_details:
        user["details"] = "Extended info"
    return user
```

Example requests:

```bash
# List users
curl http://localhost:8000/assistant/users

# Create user
curl -X POST http://localhost:8000/assistant/users \
  -H "Content-Type: application/json" \
  -d '{"name": "dana"}'

# Get user with query param
curl "http://localhost:8000/assistant/users/42?include_details=true"

# Missing or wrong Content-Type
curl -X POST http://localhost:8000/assistant/users -d '{"name":"dana"}'
# -> 415 Unsupported Media Type

# Wrong method
curl -X GET http://localhost:8000/assistant/users -H "Content-Type: application/json" -d '{}'
# -> 405 Method Not Allowed

# A scoped endpoint, and no credential the agent can verify
curl http://localhost:8000/assistant/admin/metrics
# -> 401 {"error":{"code":"unauthorized","message":"This endpoint needs a caller this agent can verify, and the request carries none."}}
```

## Capability Discovery

Use `provides` (Python) / inferred capability (TypeScript via skill name) to declare what an endpoint provides:

```typescript tab="TypeScript"
@http({ path: '/export/pdf', method: 'POST', description: 'Export data as PDF' })
async exportPdf(req: Request): Promise<Response> {
  const data = await req.json();
  const pdf = await generatePdf(data);
  return new Response(pdf, { headers: { 'content-type': 'application/pdf' } });
}

@http({ path: '/api/search', method: 'GET', description: 'Search API endpoint' })
async search(req: Request): Promise<Response> {
  const query = new URL(req.url).searchParams.get('query') ?? '';
  return Response.json({ results: await performSearch(query) });
}
```

```python tab="Python"
@http("/export/pdf", method="post", provides="pdf_export")
def export_pdf(data: dict) -> bytes:
    """Export data as PDF."""
    return generate_pdf(data)

@http("/api/search", method="get", provides="search_api")
def search(query: str) -> dict:
    """Search API endpoint."""
    return {"results": perform_search(query)}
```

The `provides` value is included in the agent's capabilities for discovery.

## Access Control (Scopes)

Use `scopes` (TypeScript) / `scope` (Python) to restrict who can call an endpoint:

```typescript tab="TypeScript"
@http({ path: '/public', method: 'GET', scopes: ['all'] })
async publicEndpoint(_req: Request): Promise<Response> {
  return Response.json({ message: 'Public data' });
}

@http({ path: '/owner-info', method: 'GET', scopes: ['owner'] })
async ownerEndpoint(_req: Request): Promise<Response> {
  return Response.json({ private: 'owner data' });
}

@http({ path: '/admin/metrics', method: 'GET', scopes: ['admin'] })
async adminMetrics(_req: Request): Promise<Response> {
  return Response.json({ rps: 100, error_rate: 0.001 });
}
```

```python tab="Python"
@http("/public", method="get", scope="all")
def public_endpoint() -> dict:
    return {"message": "Public data"}

@http("/owner-info", method="get", scope="owner")
def owner_endpoint() -> dict:
    return {"private": "owner data"}

@http("/admin/metrics", method="get", scope="admin")
def admin_metrics() -> dict:
    return {"rps": 100, "error_rate": 0.001}
```

An endpoint with no scope, or `all`, is open to anyone who can reach the server, and nothing is asked of the caller.

A scoped endpoint is checked before its handler runs. The agent identifies the caller the way it does for a chat turn: its auth skill verifies a platform credential, and the agent file's [`access:` block](../guides/trust.md) places the caller in its groups. No payment is taken for this. Then the scope decides, with the same rule tools and prompts use:

| Scope | Who may call |
| --- | --- |
| `all` (or none) | Anyone |
| `user` | Any caller the agent verified |
| `owner` | The agent's owner, or an admin |
| `admin` | An admin |
| `group:<name>` | A member of that access group, the owner, or an admin |
| a list | Anyone any one entry lets in |

A caller the agent could not verify gets `401`, and a verified caller the scope does not include gets `403`:

```json
{"error": {"code": "unauthorized", "message": "This endpoint needs a caller this agent can verify, and the request carries none."}}
{"error": {"code": "forbidden", "message": "This endpoint is not open to this caller."}}
```

When identifying the caller refuses the request itself, that refusal is the answer: a Web Bot Auth signature that does not verify is `401`, and a caller the access block keeps out is `403`. An agent with no auth skill and no access block verifies no one, so its scoped endpoints refuse every caller. The handler of a scoped endpoint receives the verified caller on its context (`context.auth`).

A TypeScript `@http` endpoint can also declare an `auth` mode, which the SDK's servers apply before the handler runs, on top of its scopes:

| `auth` | Who may call |
| --- | --- |
| `public` (default), `signature` | Anyone; a `signature` endpoint checks its provider's signature itself |
| `session` | The agent's owner |
| `portal_token` | A caller the agent verified with its auth skill |
| `visitor_session` | Anyone: a caller that sends a credential is identified (and refused if it does not verify), and one that sends none reaches the handler with `context.auth.authenticated` false |

## WebSocket Endpoints

For bidirectional real-time communication, use the `@websocket` decorator:

```typescript tab="TypeScript"
import { Skill, websocket } from 'webagents';
import type { Context } from 'webagents';

class StreamSkill extends Skill {
  readonly name = 'stream';

  @websocket({ path: '/stream' })
  handleStream(ws: WebSocket, ctx: Context): void {
    ws.onmessage = async (ev) => {
      const message = JSON.parse(String(ev.data));
      const response = await this.process(message);
      ws.send(JSON.stringify(response));
    };
  }

  private async process(msg: unknown) { return { echo: msg }; }
}
```

```python tab="Python"
from webagents import BaseAgent, websocket
from starlette.websockets import WebSocketDisconnect

@websocket("/stream")
async def my_websocket(ws) -> None:
    """Bidirectional WebSocket handler"""
    await ws.accept()
    try:
        async for message in ws.iter_json():
            response = await process(message)
            await ws.send_json(response)
    except WebSocketDisconnect:
        pass

agent = BaseAgent(
    name="assistant",
    model="openai/gpt-4o-mini",
    capabilities=[my_websocket],
)
```

Both expose `WS /assistant/stream`.

A `@websocket` endpoint takes the same `scopes` (TypeScript) / `scope` (Python), checked the same way before the handshake completes, so a refused upgrade gets the HTTP status and JSON body above. A browser cannot set headers on a WebSocket upgrade, so a scoped socket also reads its credential from `?token=`.

### WebSocket with LLM Streaming

Combine WebSocket with handoffs for streaming chat:

```typescript tab="TypeScript"
import { Skill, websocket } from 'webagents';

class StreamingSkill extends Skill {
  readonly name = 'streaming';

  @websocket({ path: '/chat' })
  async handleChat(ws: WebSocket): Promise<void> {
    ws.onmessage = async (ev) => {
      const { messages = [] } = JSON.parse(String(ev.data));
      for await (const chunk of this.executeHandoff(messages)) {
        ws.send(JSON.stringify(chunk));
      }
    };
  }

  private async *executeHandoff(_: unknown[]) {
    yield { delta: 'streaming chunk' };
  }
}
```

```python tab="Python"
from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import websocket

class StreamingSkill(Skill):
    @websocket("/chat")
    async def chat_stream(self, ws) -> None:
        await ws.accept()

        async for msg in ws.iter_json():
            messages = msg.get("messages", [])

            async for chunk in self.execute_handoff(messages):
                await ws.send_json(chunk)
```

## SSE Streaming (Server-Sent Events)

Return an async iterable from an `@http` handler to stream as SSE:

```typescript tab="TypeScript"
import { Skill, http } from 'webagents';

class EventsSkill extends Skill {
  readonly name = 'events';

  @http({ path: '/events', method: 'GET', content_type: 'text/event-stream' })
  async streamEvents(_req: Request): Promise<Response> {
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        for (let i = 0; i < 5; i++) {
          controller.enqueue(encoder.encode(`data: {"count": ${i}}\n\n`));
          await new Promise((r) => setTimeout(r, 1000));
        }
        controller.enqueue(encoder.encode('data: [DONE]\n\n'));
        controller.close();
      },
    });
    return new Response(stream, {
      headers: {
        'content-type': 'text/event-stream',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
      },
    });
  }
}
```

```python tab="Python"
from webagents import http
from typing import AsyncGenerator
import asyncio

@http("/events", method="get")
async def stream_events() -> AsyncGenerator[str, None]:
    """SSE streaming endpoint"""
    for i in range(5):
        yield f"data: {{\"count\": {i}}}\n\n"
        await asyncio.sleep(1)
    yield "data: [DONE]\n\n"
```

The Python server automatically sets SSE headers (`Content-Type: text/event-stream`, `Cache-Control: no-cache`, `Connection: keep-alive`). In TypeScript, set them yourself on the `Response`.

## Auto-Registration via Transport Skills

Transport skills register endpoints automatically when added to an agent — no manual endpoint wiring needed:

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';
import { CompletionsTransportSkill } from 'webagents/skills/transport/completions';
import { A2ATransportSkill } from 'webagents/skills/transport/a2a';
import { UAMPTransportSkill } from 'webagents/skills/transport/uamp';

const agent = new BaseAgent({
  name: 'my-agent',
  skills: [
    new CompletionsTransportSkill(), // POST /v1/chat/completions, GET /v1/models
    new A2ATransportSkill(),         // POST /a2a, GET /.well-known/agent.json
    new UAMPTransportSkill(),        // WS /uamp
  ],
});
```

```python tab="Python"
# Transport endpoints are mounted automatically by the Python server (FastAPI):
# POST /{agent}/chat/completions, GET /.well-known/agent.json, etc.
# See webagents/python/webagents/server/core/app.py for the full route table.
```

## Tips

- Keep one responsibility per endpoint (CRUD-style patterns work well).
- Prefer `GET` for retrieval, `POST` for creation/processing.
- Validate inputs inside handlers; return JSON-serializable data.
- Register endpoints through skill classes alongside `@tool`, `@hook`, and `@handoff`.

## See Also

- **[Quickstart](../quickstart.md)** — serving agents
- **[Agent Skills](./skills.md)** — modular capabilities
- **[Tools](./tools.md)** — add executable functions
- **[Hooks](./hooks.md)** — lifecycle integration
