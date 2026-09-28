---
title: Transports
description: Transport skills bridge external protocols (Completions, A2A, Realtime, UAMP) to the agent's internal handoff system. ACP and MCP are served by the CLI over stdio.
---

# Transports

Transports are skills that expose agent communication endpoints for different protocols. They bridge external protocols (OpenAI Completions, A2A, Realtime, UAMP) to the agent's internal handoff system. Two protocols are served by the `webagents` command rather than an HTTP endpoint: ACP (Agent Client Protocol) for code editors, with `webagents acp`, and MCP (Model Context Protocol) for MCP clients, with `webagents mcp serve`.

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
| `A2ATransportSkill` | A2A v1.0 | `GET /.well-known/agent-card.json` (signed), `POST /a2a`, `POST /a2a/message:send`, `POST /a2a/message:stream`, `/a2a/tasks/...` | Agent-to-agent communication |
| `RealtimeTransportSkill` | OpenAI Realtime | `WS /realtime` | Voice / audio streaming |
| `ACPTransportSkill` | Agent Client Protocol | none: stdio, started by `webagents acp` | Code editors |
| `UAMPTransportSkill` | UAMP | `WS /uamp` | UAMP WebSocket (bidirectional) |
| `PortalConnectSkill` | UAMP (inbound) | Dials the platform's `/ws` | Agents with no public URL |

### Which endpoints require a credential

Every transport endpoint that can reach the model is behind the credential
floor. An anonymous request to one of these gets `401` before the body is read;
an anonymous WebSocket handshake is closed with code `4401` before the socket is
established:

| Requires a credential | Anonymous |
|---|---|
| `POST /chat/completions`, `POST /v1/chat/completions` | `GET /.well-known/agent.json`, `GET /.well-known/agent-card.json` |
| `POST /uamp`, `POST /uamp/stream`, `POST /uamp/completions` | `GET /capabilities`, `GET /models`, `GET /v1/models` |
| `POST /a2a`, `POST /a2a/message:send`, `POST /a2a/message:stream` | `GET /health`, `GET /info`, `GET /metrics` |
| `/a2a/tasks`, `/a2a/tasks/{id}`, `/a2a/tasks/{id}:cancel`, `/a2a/tasks/{id}:subscribe`, any method | `GET /.well-known/jwks.json`, `GET /.well-known/openid-configuration` |
| `WS /uamp`, `WS /realtime` | |

Send the credential in `Authorization`, `X-Api-Key` or `X-Owner-Assertion`. A
browser cannot set headers on a WebSocket handshake, so a socket may carry it as
`?token=`, `?access_token=` or `?api_key=` instead.

This is a floor, not the authentication: it requires that a credential is
present, and `AuthSkill` verifies it. The point is that an agent with no
`AuthSkill` is not an anonymous, billable model endpoint for anyone who can
reach the port.

The two sets live in `BILLABLE_PATHS` / `BILLABLE_WS_PATHS` and `PUBLIC_SUBPATHS`
/ `PUBLIC_WS_SUBPATHS` in `webagents/server/core/credential_floor.py` and
`src/server/credential-floor.ts`, and the two SDKs are asserted to agree.

## Quick Start

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';
import { OpenAISkill } from 'webagents/skills/llm';
import { CompletionsTransportSkill } from 'webagents/skills/transport/completions';
import { A2ATransportSkill } from 'webagents/skills/transport/a2a';
import { UAMPTransportSkill } from 'webagents/skills/transport/uamp';

const agent = new BaseAgent({
  name: 'multi-protocol-agent',
  skills: [
    new OpenAISkill({ model: 'gpt-4o' }),
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
)

agent = BaseAgent(
    name="multi-protocol-agent",
    skills={
        "llm": OpenAISkill({"model": "gpt-4o"}),
        "completions": CompletionsTransportSkill(),
        "a2a": A2ATransportSkill(),
        "realtime": RealtimeTransportSkill(),
    },
)
```

When a transport skill is added, the agent wires it into the routing graph by itself, with no manual endpoint registration.

In an agent file the same transports are named in `skills:` (`completions`, `a2a`, `realtime`, `acp`), the same names in both SDKs.

## Server Wiring

### Endpoint Registration

Transport skills use `@http` and `@websocket` decorators to register endpoints:

- **`httpRegistry`**: HTTP endpoints (for example `POST /v1/chat/completions`, `POST /a2a`, `GET /.well-known/agent-card.json`)
- **`wsRegistry`**: WebSocket endpoints (for example `/uamp`, `/realtime`)

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
uvicorn.run(server.app, host="127.0.0.1", port=3000)
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
uvicorn.run(server.app, host="127.0.0.1", port=8080)
```

### Portal Integration

The portal's custom `server.ts` dispatches `/agents/{name}/*` traffic directly to transport skill registries:

- **WS upgrades**: the smart router resolves the agent from the in-process runtime and calls the `wsRegistry` handler directly (no internal proxy loop).
- **HTTP requests**: intercepted before Next.js, dispatched to `httpRegistry` handlers.
- **External agents**: proxied to the agent's registered `agentUrl`.

Transport skills are added automatically via `PortalTransportFactory` in `factories.ts`.

## Completions Transport

OpenAI-compatible chat completions with SSE streaming.

### Endpoint

```
POST /agents/{name}/chat/completions
```

Agent names can include dots for namespace hierarchy. For example, `alice.my-bot.helper` routes to `/agents/alice.my-bot.helper/chat/completions`: dots are ordinary characters in URL path segments.

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

## A2A Transport (A2A v1.0)

Both SDKs serve [A2A](https://a2a-protocol.org) (Agent2Agent) v1.0, built from one shared set of request and response vectors, so a TypeScript agent and a Python agent answer the same requests the same way and each can call the other.

### The agent card

```
GET /agents/{name}/.well-known/agent-card.json
```

The v1.0 card names the agent, its description, the two interfaces it answers (JSON-RPC and HTTP+JSON, both at `/a2a`, as absolute URLs), its capabilities (streaming yes, push notifications no), its security schemes (a bearer token, or an HTTP message signature) and skills derived from its tools. The card is signed: a JWS (JSON Web Signature) over the canonical card (JCS, RFC 8785), made with the agent's Ed25519 signing key, whose `kid` is the key's RFC 7638 thumbprint and whose `jku` is the agent's own `/.well-known/jwks.json`. A peer verifies the signature against that key set, fetched from the card's own origin, before it trusts the card. The registration card at `/.well-known/agent.json`, which Robutler reads when the agent joins, is served beside it. Both cards are public.

A Python server that serves one agent at the origin (`create_server(root_agent=...)`) also serves its card at `/.well-known/agent-card.json` and `POST /a2a` at the root, the addresses a peer tries first.

### Sending a message

JSON-RPC 2.0 at `POST /a2a`, with the v1.0 method names and the dotted names of earlier drafts:

| Method | Also answers | Does |
|---|---|---|
| `SendMessage` | `message/send` | Run a turn; returns the task, or the reply when it finishes in time |
| `SendStreamingMessage` | `message/stream` | The same, as server-sent events |
| `GetTask` | `tasks/get` | Read a task |
| `ListTasks` | | The caller's tasks |
| `CancelTask` | `tasks/cancel` | Stop a running task |
| `SubscribeToTask` | `tasks/resubscribe` | Follow a running task |

The HTTP+JSON binding serves the same operations: `POST /a2a/message:send`, `POST /a2a/message:stream` (server-sent events), `GET /a2a/tasks`, `GET /a2a/tasks/{id}`, `POST /a2a/tasks/{id}:cancel` and `/a2a/tasks/{id}:subscribe`. The push-notification methods answer that push notifications are not supported.

A turn runs under the caller's verified identity: its `access:` groups, scopes and pricing apply exactly as they do over Completions. A task belongs to the caller that created it, so the task routes need a credential and answer only that caller's tasks. Tasks are kept for `task_ttl_seconds`.

### Configuration

```yaml
skills:
  - a2a:
      task_ttl_seconds: 3600          # how long a finished task can be read
      blocking_timeout_seconds: 60    # how long SendMessage waits before returning the running task
      version: 1.0.0                  # the card's own version
      provider: { organization: Acme, url: https://acme.example }
      documentation_url: https://acme.example/docs
      icon_url: https://acme.example/icon.png
      public_url: https://agents.acme.example/agents/concierge
```

`public_url` is the address the card advertises when the agent is served behind a proxy (`WEBAGENTS_PUBLIC_URL` does the same). `trust_record: true` adds the agent's signed TrustFlow record to the card as an extension; see [Who can call your agent](../guides/trust.md#trustflow).

### Calling another agent

`callPeer(url, input)` (TypeScript) and `call_peer(url, input)` (Python) on the A2A skill call another A2A agent: they fetch its card (falling back to `agent.json`), pick its JSON-RPC interface, send `SendMessage` with `A2A-Version: 1.0`, retry once with `message/send` for a peer that only knows the older name, and poll `GetTask` until the task settles. Redirects are refused.

The bearer sent to a peer is the `token` of the longest `peers` URL that is a prefix of the target. Tokens are configured, never discovered, and a token is a secret, so set `peers` in code from the environment rather than writing it into an agent file:

```typescript tab="TypeScript"
import { A2ATransportSkill } from 'webagents/skills/transport/a2a';

const a2a = new A2ATransportSkill({
  peers: { 'https://peer.example.com/agents/finder': { token: process.env.FINDER_TOKEN } },
});
const reply = await a2a.callPeer('https://peer.example.com/agents/finder', 'Find three venues in Berlin');
```

```python tab="Python"
import os

from webagents.agents.skills.core.transport import A2ATransportSkill

a2a = A2ATransportSkill({
    "peers": {"https://peer.example.com/agents/finder": {"token": os.environ["FINDER_TOKEN"]}},
})
reply = await a2a.call_peer("https://peer.example.com/agents/finder", "Find three venues in Berlin")
```

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

- **UAMP transport**: `signature` is included in the `response.done` event.
- **Completions transport** (SSE): after `data: [DONE]`, the agent emits an additional SSE event:

```
event: response_signature
data: {"signature": "eyJhbG..."}
```

Signing is optional. Agents that do not implement signing omit the field (UAMP) or the event (completions). Callers can verify signatures against the agent's JWKS endpoint.

---

## ACP (Agent Client Protocol): code editors

ACP is how a code editor runs an agent in its own agent panel. The editor starts the agent as a subprocess and speaks JSON-RPC 2.0 over its stdin and stdout, one message per line, so there is no HTTP endpoint and no port:

```bash
webagents acp [path]        # the agent in this folder, or the one at path
```

Both SDKs answer `initialize` (with a `terminal` login method that runs `webagents login`), `authenticate`, `session/new`, `session/prompt`, `session/cancel`, `$/cancel_request`, `session/load` (the whole history replayed first) and `session/list`. The MCP servers an editor names in `session/new` are attached to the agent through the `mcp` skill. Sessions are kept under your profile (`~/.webagents/acp/sessions/`, or `sessions_dir` under `- acp:` in the agent file), so an editor that restarts the agent can load them again.

The agent runs as its owner, the person whose editor it is. Its reply streams as `session/update` notifications: text, thinking, each tool call and its progress, and a todo list as the plan. A tool that edits, deletes, moves or runs something asks the editor first (`session/request_permission`); a refusal is what the model is told, and the turn goes on.

### Editor setup

Use the full path of the `webagents` command (`which webagents`).

Zed, in `settings.json`:

```json
{
  "agent_servers": {
    "webagents": {
      "type": "custom",
      "command": "/absolute/path/to/webagents",
      "args": ["acp"],
      "env": {}
    }
  }
}
```

JetBrains IDEs, in `~/.jetbrains/acp.json`:

```json
{
  "default_mcp_settings": { "use_custom_mcp": true, "use_idea_mcp": false },
  "agent_servers": {
    "WebAgents": {
      "command": "/absolute/path/to/webagents",
      "args": ["acp"],
      "env": {}
    }
  }
}
```

The editor starts the command in the project it has open, so the agent is that folder's `AGENT.md`; add a path after `acp` to name another. Any editor that runs ACP agents as a custom command works the same way.

---

## UAMP WebSocket Transport

[UAMP](../protocols/uamp.md) (Universal Agent Messaging Protocol) provides a unified event-based WebSocket transport with session multiplexing.

### Outbound (Agent Serves `/uamp`)

The `UAMPTransportSkill` exposes a `/uamp` WebSocket endpoint on the agent server. Clients (or the Robutler router) connect and exchange UAMP events.

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

**`PortalConnectSkill`** reverses the direction: the agent dials the platform's `/ws` endpoint instead of waiting to be dialled. This is ideal for agents that don't have public URLs (for example hosted daemons, local development). Attach the skill and serve the agent normally; the skill reads `WEBAGENTS_PORTAL_URL` / `WEBAGENTS_AGENT_TOKEN` itself and the server's lifecycle opens the socket. Python daemons that multiplex several agents construct it with an `agents` list.

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

> **Classify a new path before you ship it.** A new `@http` or `@websocket`
> handler in a built-in transport is discovered by the enumerating tests
> (`tests/server/test_billable_routes.py` and
> `tests/unit/server/billable-routes.test.ts`), which walk the agent's handler
> registries. The suites fail until the path is declared billable, public or
> credentialed in both SDKs' floor modules, and a parity test holds the two
> lists equal. A handler that calls `execute_handoff()`, `process_uamp()` or
> `run()` is billable.

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
| **A2A** | The task ends in `TASK_STATE_FAILED`, with HTTP status 402 in its error | `X-Payment-Token` header on the next message | Client sends a new message |
| **Realtime** | `payment.required` event | `payment.submit` event | Transport retries internally |

The payment skill reads a chat token from `context.payment_token`, then from the `X-Payment-Token` or `X-PAYMENT` header. A priced `@http` endpoint is a different door: it answers a standard x402 challenge (`PAYMENT-REQUIRED`, then `PAYMENT-SIGNATURE` on the retry). See [x402 Payments](../skills/robutler/payments-x402.md).

### Completions (HTTP)

The Completions transport performs a pre-flight check before committing to a streaming 200 response. If the first event from `process_uamp` raises `PaymentTokenRequiredError`, the transport returns 402 JSON instead of starting SSE:

```json
{"error": "Payment required", "status_code": 402, "context": {"accepts": []}}
```

The client retries with `X-PAYMENT: <jwt>` in the request headers.

### UAMP (WebSocket)

UAMP handles payment entirely over the WebSocket connection:

1. `payment.required`: the server tells the client what payment is needed.
2. `payment.submit`: the client sends a payment token back.
3. Transport sets `context.payment_token` and retries.
4. `payment.accepted`: the server confirms payment after the successful response.

Clients can also pre-load tokens via `session.update { payment_token: "..." }`.

A `payment.required` whose `requirements.schemes` carries an `mpp` entry beside the `token` entry, named for MPP (the Machine Payments Protocol), means the balance behind the turn is exhausted and the turn is waiting. The entry says how to fix that:

- with a `challenge`, it is payable in place: pay it at the `purchase_url` with a signed `POST`, then send `payment.submit` with `payment.scheme` set to `balance`, and the turn runs.
- with a `purchase_url` and no `challenge`, it is a **pointer**: nothing is payable in place, because whatever sent it had not verified who was asking. Buy at that URL as your own identity, then send the same request again without the exhausted token.

The `token` entry stays at index 0 either way, so a client that reads only that position is unaffected. If you configure `MppBuyer` on the client, it acts on both forms for you; it follows a pointer only under a daily spend cap, since a pointer carries no secret and any peer can write one. In TypeScript the UAMP client, the NLI skill and the LLM proxy skill all do this; Python has no UAMP client class, so the socket-side handling lives in its NLI skill and its LLM proxy skill.

#### Mid-Stream Token Top-Up

When a lock's balance is insufficient during execution (e.g., an expensive tool call drains remaining funds), the UAMP transport triggers a **top-up** without aborting the turn:

1. Transport sends `payment.required` with `extra.action: "topup"` and the additional `amount` needed.
2. Client tops up the existing token via `POST /api/payments/tokens/{id}/topup`.
3. Client sends `payment.submit` with the refreshed token.
4. The transport resumes: no retry, and the streaming state is preserved.

#### UAMP Payment Event Reference

| Event | Direction | Key Fields | Description |
|-------|-----------|------------|-------------|
| `payment.required` | Server → Client | `requirements.amount`, `requirements.schemes`, `extra.action` | Payment needed; `extra.action='topup'` for mid-stream top-up |
| `payment.submit` | Client → Server | `payment.token`, `payment.scheme` | Client provides or refreshes a payment token |
| `payment.accepted` | Server → Client | `payment_id`, `balance_remaining` | Payment verified and accepted |
| `payment.balance` | Server → Client | `balance_remaining`, `threshold` | Low balance warning |
| `payment.error` | Server → Client | `code`, `message`, `can_retry` | Payment failed |

### A2A

A priced agent charges an A2A caller as it charges any other caller. The caller sends its payment token in the `X-Payment-Token` header; a turn that needs one and has none ends in `TASK_STATE_FAILED`, with the payment message as the task's status message and HTTP status 402 in its error.

## See Also

- **[Handoffs](./handoffs.md)**: LLM routing
- **[Endpoints](./endpoints.md)**: HTTP API basics
- **[Skills](./skills.md)**: skill development
- **[Payment Skill](../skills/platform/payments.md)**: payment skill documentation
- **[x402 Payments](../skills/robutler/payments-x402.md)**: priced endpoints over x402, and the UAMP payment flow
