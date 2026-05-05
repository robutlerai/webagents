---
title: Transports
description: Transport skills bridge external protocols (Completions, A2A, Realtime, ACP, UAMP) to the agent's internal handoff system.
---

# Transports

Transports are skills that expose agent communication endpoints for different protocols. They bridge external protocols (OpenAI Completions, A2A, Realtime, ACP, UAMP) to the agent's internal handoff system.

## Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                     Client Request                           │
│    (HTTP, WebSocket, SSE)                                    │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Transport Skill                           │
│  ┌─────────────┐    ┌────────────┐    ┌──────────────┐      │
│  │ Parse       │ →  │ Convert to │ →  │ execute_     │      │
│  │ protocol    │    │ internal   │    │ handoff()    │      │
│  └─────────────┘    └────────────┘    └──────────────┘      │
│         ↑                                    │               │
│         │                                    ▼               │
│  ┌─────────────┐                    ┌──────────────┐         │
│  │ Format      │ ← ─ ─ ─ ─ ─ ─ ─ ─  │ LLM Response │         │
│  │ response    │                    │ (streaming)  │         │
│  └─────────────┘                    └──────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## Available Transports

| Transport | Protocol | Endpoints | Use Case |
|-----------|----------|-----------|----------|
| `CompletionsTransportSkill` | OpenAI API | `POST /chat/completions` | Standard LLM interaction |
| `A2ATransportSkill` | Google A2A | `GET /.well-known/agent.json`, `POST /a2a` | Agent-to-agent communication |
| `RealtimeTransportSkill` | OpenAI Realtime | `WS /realtime` | Voice / audio streaming |
| `ACPTransportSkill` | Agent Client Protocol | `POST /acp`, `WS /acp/stream` | IDE integration |
| `UAMPTransportSkill` | UAMP | `WS /uamp` | UAMP WebSocket (bidirectional) |
| `PortalConnectSkill` | UAMP (inbound) | Connects to platform WS | Daemon agents (no public URL) |

## Quick Start

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';
import { OpenAILLMSkill } from 'webagents/skills/llm';
import { CompletionsTransportSkill } from 'webagents/skills/transport/completions';
import { A2ATransportSkill } from 'webagents/skills/transport/a2a';
import { UAMPTransportSkill } from 'webagents/skills/transport/uamp';

const agent = new BaseAgent({
  name: 'multi-protocol-agent',
  skills: [
    new OpenAILLMSkill({ defaultModel: 'gpt-4o' }),
    new CompletionsTransportSkill(), // OpenAI-compatible HTTP
    new A2ATransportSkill(),         // Google A2A HTTP
    new UAMPTransportSkill(),        // UAMP WebSocket
  ],
});
```

```python tab="Python"
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.core.llm.openai import OpenAISkill
from webagents.agents.skills.core.transport import (
    CompletionsTransportSkill,
    A2ATransportSkill,
    RealtimeTransportSkill,
    ACPTransportSkill,
)

agent = BaseAgent(
    name="multi-protocol-agent",
    skills={
        "llm": OpenAISkill({"model": "gpt-4o"}),
        "completions": CompletionsTransportSkill(),
        "a2a": A2ATransportSkill(),
        "realtime": RealtimeTransportSkill(),
        "acp": ACPTransportSkill(),
    },
)
```

When a transport skill is added, the agent automatically wires it into the routing graph — no manual endpoint registration needed.

## Server Wiring

### Endpoint Registration

Transport skills use `@http` and `@websocket` decorators to register endpoints:

- **`httpRegistry`** — HTTP endpoints (e.g., `POST /v1/chat/completions`, `POST /a2a`, `GET /.well-known/agent.json`)
- **`wsRegistry`** — WebSocket endpoints (e.g., `/uamp`)

Servers read these registries to mount endpoints automatically.

### Node.js Single-Agent Server

`createAgentApp()` returns an `AgentServer` with both an HTTP app and a WebSocket upgrade handler:

```typescript tab="TypeScript"
import { createAgentApp, serve } from 'webagents';

const { app, handleUpgrade } = createAgentApp(agent);
// `app` is a Hono instance with httpRegistry routes mounted.
// `handleUpgrade` dispatches WS upgrades to wsRegistry handlers.

// Or use serve() which wires both automatically:
await serve(agent, { port: 3000 });
```

```python tab="Python"
from webagents.server.core.app import create_server
import uvicorn

server = create_server(agents=[agent])
uvicorn.run(server.app, host="0.0.0.0", port=3000)
```

> Breaking change in TypeScript v0.3+: `createAgentApp()` returns `AgentServer { app, handleUpgrade }` instead of a bare `Hono` instance. Use `.app` for HTTP-only access.

### Multi-Agent Server

The multi-agent server routes by name and consults `httpRegistry` before hardcoded fallback routes:

```typescript tab="TypeScript"
import { WebAgentsServer } from 'webagents';

const server = new WebAgentsServer({ agents: [] });
await server.addAgent('assistant', agent);
await server.listen({ port: 8080 });

// Requests to /agents/assistant/v1/chat/completions -> CompletionsTransportSkill
// Requests to /agents/assistant/a2a                 -> A2ATransportSkill
// WebSocket  to /agents/assistant/uamp              -> UAMPTransportSkill
```

```python tab="Python"
from webagents.server.core.app import create_server
import uvicorn

server = create_server(agents=[agent_a, agent_b])
uvicorn.run(server.app, host="0.0.0.0", port=8080)
```

### Portal Integration

The portal's custom `server.ts` dispatches `/agents/{name}/*` traffic directly to transport skill registries:

- **WS upgrades** — Smart router resolves the agent from the in-process runtime and calls the `wsRegistry` handler directly (no internal proxy loop).
- **HTTP requests** — Intercepted before Next.js, dispatched to `httpRegistry` handlers.
- **External agents** — Proxied to the agent's registered `agentUrl`.

Transport skills are added automatically via `PortalTransportFactory` in `factories.ts`.

## Completions Transport

OpenAI-compatible chat completions with SSE streaming.

### Endpoint

```
POST /agents/{name}/chat/completions
```

Agent names can include dots for namespace hierarchy. For example, `alice.my-bot.helper` routes to `/agents/alice.my-bot.helper/chat/completions` — dots are ordinary characters in URL path segments.

### Request

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": true,
  "model": "gpt-4o",
  "temperature": 0.7,
  "max_tokens": 1000,
  "tools": []
}
```

### Response (Streaming)

```
data: {"id":"chatcmpl-...","choices":[{"delta":{"role":"assistant"}}]}

data: {"id":"chatcmpl-...","choices":[{"delta":{"content":"Hello"}}]}

data: {"id":"chatcmpl-...","choices":[{"delta":{"content":"!"}}]}

data: [DONE]
```

---

## A2A Transport (Google Agent2Agent)

Implements the [A2A Protocol](https://google.github.io/A2A/) for agent-to-agent communication.

### Agent Card

```
GET /agents/{name}/.well-known/agent.json
```

Returns agent capabilities for discovery:

```json
{
  "name": "my-agent",
  "description": "A helpful assistant",
  "version": "0.2.1",
  "protocolVersion": "0.2.1",
  "capabilities": {
    "streaming": true,
    "pushNotifications": false
  },
  "defaultInputModes": ["text"],
  "defaultOutputModes": ["text"],
  "skills": []
}
```

### Create Task

```
POST /agents/{name}/tasks
```

Request (A2A format):

```json
{
  "message": {
    "role": "user",
    "parts": [
      {"type": "text", "text": "What is the weather?"}
    ]
  }
}
```

Response (SSE streaming):

```
event: task.started
data: {"id":"task-123","status":"running"}

event: task.message
data: {"role":"agent","parts":[{"type":"text","text":"The weather is..."}]}

event: task.completed
data: {"id":"task-123","status":"completed"}
```

### Get / Cancel Task

```
GET    /agents/{name}/tasks/{task_id}
DELETE /agents/{name}/tasks/{task_id}
```

---

## Realtime Transport (OpenAI Realtime API)

WebSocket-based real-time communication with audio support.

```
WS /agents/{name}/realtime
```

### Session Events

```json
// Sent on connection
{"type": "session.created", "session": {"id": "sess_...", "voice": "alloy"}}

// Update session
{"type": "session.update", "session": {"voice": "nova", "modalities": ["text", "audio"]}}

// Session updated confirmation
{"type": "session.updated", "session": {}}
```

### Audio Buffer Events

```json
// Append audio (base64 PCM16)
{"type": "input_audio_buffer.append", "audio": "base64..."}

// Commit buffer
{"type": "input_audio_buffer.commit"}

// Clear buffer
{"type": "input_audio_buffer.clear"}
```

### Conversation Events

```json
{"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": []}}
{"type": "conversation.item.delete", "item_id": "item_..."}
```

### Response Events

```json
{"type": "response.create"}
{"type": "response.text.delta", "delta": "Hello"}
{"type": "response.text.done", "text": "Hello world!"}
{"type": "response.done", "response": {"status": "completed"}, "signature": "eyJhbG..."}
{"type": "response.cancel"}
```

### Response Signing (Optional)

Agents with signing keys can attach an RS256 JWT to the `response.done` event via the optional `signature` field. The JWT contains `response_hash` (SHA-256 of the full response text) and `request_hash` (SHA-256 of the original request), enabling cryptographic non-repudiation.

- **UAMP transport** — `signature` is included in the `response.done` event.
- **Completions transport** (SSE) — after `data: [DONE]`, the agent emits an additional SSE event:

```
event: response_signature
data: {"signature": "eyJhbG..."}
```

Signing is optional. Agents that do not implement signing omit the field (UAMP) or the event (completions). Callers can verify signatures against the agent's JWKS endpoint.

---

## ACP Transport (Agent Client Protocol)

JSON-RPC 2.0 protocol for IDE integration (Cursor, Zed, JetBrains).

```
POST /agents/{name}/acp
WS   /agents/{name}/acp/stream
```

### Initialize

```json
{"jsonrpc": "2.0", "method": "initialize", "params": {}, "id": 1}

// Response
{"jsonrpc": "2.0", "id": 1, "result": {
  "protocolVersion": "1.0",
  "serverInfo": {"name": "my-agent", "version": "2.0.0"},
  "capabilities": {"streaming": true, "tools": true}
}}
```

### Chat / Submit

```json
{"jsonrpc": "2.0", "method": "prompt/submit", "params": {
  "messages": [{"role": "user", "content": "Hello"}]
}, "id": 2}

// Streaming notifications
{"jsonrpc": "2.0", "method": "prompt/started", "params": {"requestId": "2"}}
{"jsonrpc": "2.0", "method": "prompt/progress", "params": {"content": "Hello!", "role": "assistant"}}

// Final response
{"jsonrpc": "2.0", "id": 2, "result": {"status": "complete", "content": "Hello!"}}
```

### Tools

```json
{"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 3}

{"jsonrpc": "2.0", "method": "tools/call", "params": {
  "name": "search",
  "arguments": {"query": "weather"}
}, "id": 4}
```

---

## UAMP WebSocket Transport

[UAMP](../protocols/uamp.md) (Universal Agent Messaging Protocol) provides a unified event-based WebSocket transport with session multiplexing.

### Outbound (Agent Serves `/uamp`)

The `UAMPTransportSkill` exposes a `/uamp` WebSocket endpoint on the agent server. Clients (or the Roborum router) connect and exchange UAMP events.

```
WS /agents/{name}/uamp
```

| Direction | Event | Description |
|-----------|-------|-------------|
| Client → Agent | `session.create` | Create a new session |
| Agent → Client | `session.created` | Session confirmed |
| Client → Agent | `input.text` | Send text input |
| Agent → Client | `response.delta` | Streamed response chunk |
| Agent → Client | `response.done` | Response complete |
| Both | `ping` / `pong` | Keepalive |

### Inbound (Agent Connects to Platform)

The **PortalConnectSkill** reverses the direction: the agent connects to the Roborum platform's `/ws` endpoint. This is ideal for agents that don't have public URLs (e.g., hosted daemons, local development).

See [Portal Connect Skill](../skills/platform/portal-connect.md) for details.

### Session Multiplexing

A single UAMP WebSocket supports multiple concurrent sessions. Each event carries a `session_id` field for routing. This allows a daemon to register multiple agents on one connection.

```json
{"type": "session.create", "event_id": "evt_1", "session": {"agent": "agent-a", "token": "..."}}
{"type": "session.create", "event_id": "evt_2", "session": {"agent": "agent-b", "token": "..."}}
```

---

## Creating Custom Transports

Use `@http` and `@websocket` decorators with the agent's handoff API:

```typescript tab="TypeScript"
import { Skill, http, websocket } from 'webagents';

class MyCustomTransport extends Skill {
  readonly name = 'my-protocol';

  @http({ path: '/my-protocol', method: 'POST', content_type: 'text/event-stream' })
  async handleRequest(req: Request): Promise<Response> {
    const body = await req.json();
    const internalMessages = this.parseMyProtocol(body.messages);

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start: async (controller) => {
        for await (const chunk of this.executeHandoff(internalMessages)) {
          controller.enqueue(encoder.encode(this.formatMyProtocol(chunk)));
        }
        controller.close();
      },
    });
    return new Response(stream, {
      headers: { 'content-type': 'text/event-stream' },
    });
  }

  @websocket({ path: '/my-protocol/stream' })
  handleWebsocket(ws: WebSocket): void {
    ws.onmessage = async (ev) => {
      const message = JSON.parse(String(ev.data));
      const internalMessages = this.parseMyProtocol(message);
      for await (const chunk of this.executeHandoff(internalMessages)) {
        ws.send(JSON.stringify(this.formatMyProtocol(chunk)));
      }
    };
  }

  private parseMyProtocol(_: unknown) { return [] as unknown[]; }
  private formatMyProtocol(_: unknown) { return ''; }
  private async *executeHandoff(_: unknown[]) {
    yield { delta: 'chunk' } as const;
  }
}
```

```python tab="Python"
from typing import AsyncGenerator
from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import http, websocket

class MyCustomTransport(Skill):
    """Custom protocol transport"""

    @http("/my-protocol", method="post")
    async def handle_request(self, messages: list) -> AsyncGenerator[str, None]:
        """SSE streaming endpoint"""
        internal_messages = self._parse_my_protocol(messages)

        async for chunk in self.execute_handoff(internal_messages):
            yield self._format_my_protocol(chunk)

    @websocket("/my-protocol/stream")
    async def handle_websocket(self, ws) -> None:
        """WebSocket endpoint"""
        await ws.accept()

        async for message in ws.iter_json():
            internal_messages = self._parse_my_protocol(message)

            async for chunk in self.execute_handoff(internal_messages):
                await ws.send_json(self._format_my_protocol(chunk))
```

## Key Methods

### `execute_handoff()` / `executeHandoff()`

Route messages through the agent's handoff system:

```typescript tab="TypeScript"
for await (const chunk of this.executeHandoff(
  [{ role: 'user', content: 'Hello' }],
  { tools: undefined, handoffName: undefined },
)) {
  console.log(chunk);
}
```

```python tab="Python"
async for chunk in self.execute_handoff(
    messages=[{"role": "user", "content": "Hello"}],
    tools=None,
    handoff_name=None,
):
    print(chunk)
```

### SSE Streaming

```typescript tab="TypeScript"
@http({ path: '/stream', method: 'POST', content_type: 'text/event-stream' })
async streamResponse(_req: Request): Promise<Response> {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode('data: {"text": "hello"}\n\n'));
      controller.enqueue(encoder.encode('data: {"text": "world"}\n\n'));
      controller.close();
    },
  });
  return new Response(stream, { headers: { 'content-type': 'text/event-stream' } });
}
```

```python tab="Python"
@http("/stream", method="post")
async def stream_response(self) -> AsyncGenerator[str, None]:
    yield "data: {\"text\": \"hello\"}\n\n"
    yield "data: {\"text\": \"world\"}\n\n"
```

### WebSocket Handlers

```typescript tab="TypeScript"
@websocket({ path: '/chat' })
async chat(ws: WebSocket): Promise<void> {
  ws.onmessage = (ev) => {
    const msg = JSON.parse(String(ev.data));
    ws.send(JSON.stringify({ response: msg }));
  };
}
```

```python tab="Python"
@websocket("/chat")
async def chat(self, ws) -> None:
    await ws.accept()
    async for msg in ws.iter_json():
        await ws.send_json({"response": msg})
```

## Payment Handling

Each transport is responsible for catching `PaymentTokenRequiredError` from the payment skill and negotiating the payment token using the appropriate protocol mechanism.

| Transport | Error Signal | Token Delivery | Retry Mechanism |
|-----------|-------------|----------------|-----------------|
| **Completions** | HTTP 402 JSON (pre-flight) | `X-PAYMENT` header on retry | Client retries entire request |
| **UAMP** | `payment.required` event | `payment.submit` event or `session.update` | Transport retries internally |
| **A2A** | `task.failed` SSE with `code: "payment_required"` | `X-PAYMENT` header on new task | Client creates new task |
| **ACP** | JSON-RPC error `-32402` | `payment_token` in `session/prompt` params | Client retries prompt |
| **Realtime** | `payment.required` event | `payment.submit` event | Transport retries internally |

> As of x402 V2, all transports use the standardized `X-PAYMENT` header (replacing the earlier `X-Payment-Token`).

### Completions (HTTP)

The Completions transport performs a pre-flight check before committing to a streaming 200 response. If the first event from `process_uamp` raises `PaymentTokenRequiredError`, the transport returns 402 JSON instead of starting SSE:

```json
{"error": "Payment required", "status_code": 402, "context": {"accepts": []}}
```

The client retries with `X-PAYMENT: <jwt>` in the request headers.

### UAMP (WebSocket)

UAMP handles payment entirely over the WebSocket connection:

1. `payment.required` — server tells client what payment is needed.
2. `payment.submit` — client sends payment token back.
3. Transport sets `context.payment_token` and retries.
4. `payment.accepted` — server confirms payment after the successful response.

Clients can also pre-load tokens via `session.update { payment_token: "..." }`.

#### Mid-Stream Token Top-Up

When a lock's balance is insufficient during execution (e.g., an expensive tool call drains remaining funds), the UAMP transport triggers a **top-up** without aborting the turn:

1. Transport sends `payment.required` with `extra.action: "topup"` and the additional `amount` needed.
2. Client tops up the existing token via `POST /api/payments/tokens/{id}/topup`.
3. Client sends `payment.submit` with the refreshed token.
4. Transport resumes — no retry, streaming state is preserved.

#### UAMP Payment Event Reference

| Event | Direction | Key Fields | Description |
|-------|-----------|------------|-------------|
| `payment.required` | Server → Client | `requirements.amount`, `requirements.schemes`, `extra.action` | Payment needed; `extra.action='topup'` for mid-stream top-up |
| `payment.submit` | Client → Server | `payment.token`, `payment.scheme` | Client provides or refreshes a payment token |
| `payment.accepted` | Server → Client | `payment_id`, `balance_remaining` | Payment verified and accepted |
| `payment.balance` | Server → Client | `balance_remaining`, `threshold` | Low balance warning |
| `payment.error` | Server → Client | `code`, `message`, `can_retry` | Payment failed |

### A2A (Google Agent-to-Agent)

A2A returns payment requirements in the `task.failed` SSE event:

```json
{
  "id": "task-1",
  "status": "failed",
  "code": "payment_required",
  "status_code": 402,
  "accepts": [{"scheme": "token", "amount": "0.01"}]
}
```

### ACP (Agent Client Protocol)

ACP uses a custom JSON-RPC error code `-32402`:

```json
{
  "jsonrpc": "2.0",
  "id": "req-1",
  "error": {
    "code": -32402,
    "message": "Payment token required",
    "data": {"accepts": []}
  }
}
```

The client retries the `session/prompt` call with `payment_token` in params.

## See Also

- **[Handoffs](./handoffs.md)** — LLM routing
- **[Endpoints](./endpoints.md)** — HTTP API basics
- **[Skills](./skills.md)** — Skill development
- **[Payment Skill](../skills/platform/payments.md)** — Payment skill documentation
- **[x402 Payments](../skills/robutler/payments-x402.md)** — x402 protocol and UAMP payment flow
