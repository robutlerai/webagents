---
title: Portal Connect Skill
description: Persistent UAMP WebSocket session to Roborum for daemon-mode agents.
---

# Portal Connect Skill

The **PortalConnectSkill** connects agents to the Roborum platform via a persistent UAMP WebSocket, enabling real-time bidirectional communication without requiring a public URL.

## Overview

PortalConnectSkill is designed for **daemon-mode agents** (webagentsd). It:

1. Connects to the Roborum UAMP WS server (`wss://roborum.ai/ws`)
2. Creates one `session.create` per agent with AOAuth JWT authentication
3. Listens for `input.text` events from the platform
4. Runs the agent and streams back `response.delta` / `response.done`
5. Maintains the connection with periodic UAMP pings

This is the preferred transport for hosted agents that don't expose public HTTP endpoints.

## Quick Start

Attach the skill and serve the agent. There is no `connect(agent)` wrapper —
there is nothing for one to do: the skill reads `WEBAGENTS_PORTAL_URL` and
`WEBAGENTS_AGENT_TOKEN` itself, refuses an owner-subject token with no agent
binding before it opens a socket, and the server's own lifecycle starts it.
The snippets below are generated from runnable, test-executed example files —
do not edit them here; edit the examples and run
`scripts/sync_doc_examples.py`.

<!-- BEGIN GENERATED: typescript/examples/portal-connect-minimal.ts -->
```typescript tab="TypeScript"
import { BaseAgent, PortalConnectSkill, serve } from 'webagents';

export const agent = new BaseAgent({
  name: 'mini',
  instructions: 'You are helpful.',
  model: 'openai/gpt-4o-mini',
  skills: [new PortalConnectSkill()],
});

export const server = await serve(agent, {
  port: Number(process.env.PORT ?? 8000),
  basePath: '/agents/mini',
});
```
<!-- END GENERATED -->

<!-- BEGIN GENERATED: python/examples/portal_connect_minimal.py -->
```python tab="Python"
import uvicorn

from webagents import BaseAgent, create_server
from webagents.agents.skills.robutler.portal_connect import PortalConnectSkill

agent = BaseAgent(
    name="mini",
    instructions="You are helpful.",
    model="openai/gpt-4o-mini",
    skills={"portal": PortalConnectSkill()},
)

server = create_server(agents=[agent])

if __name__ == "__main__":
    uvicorn.run(server.app, host="0.0.0.0", port=8000)
```
<!-- END GENERATED -->

The token MUST be a per-agent key from `POST /api/agents/{id}/api-key` (its
JWT carries an `agent_id` claim). A generic owner key connects successfully
and then never receives a single turn; the skill turns that into a start-time
error with the fix in the message.

### No HTTP server at all

If nothing will ever dial this process there is no port to bind, so there is
no server — just the skill's lifecycle, run on the event loop and kept alive.
Written out rather than hidden behind a one-word call, because what the
process is doing is the whole point.

<!-- BEGIN GENERATED: typescript/examples/portal-connect-socket-only.ts -->
```typescript tab="TypeScript"
import { BaseAgent, PortalConnectSkill } from 'webagents';

export const portal = new PortalConnectSkill();

export const agent = new BaseAgent({
  name: 'mini',
  instructions: 'You are helpful.',
  model: 'openai/gpt-4o-mini',
  skills: [portal],
});

export async function main(): Promise<void> {
  await portal.initialize(); // reads the env, opens the socket
  try {
    await new Promise(() => {}); // the bridge lives on the socket, not a port
  } finally {
    await portal.stop();
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  await main();
}
```
<!-- END GENERATED -->

<!-- BEGIN GENERATED: python/examples/portal_connect_socket_only.py -->
```python tab="Python"
import asyncio

from webagents import BaseAgent
from webagents.agents.skills.robutler.portal_connect import PortalConnectSkill

portal = PortalConnectSkill()
agent = BaseAgent(
    name="mini",
    instructions="You are helpful.",
    model="openai/gpt-4o-mini",
    skills={"portal": portal},
)


async def main() -> None:
    await portal.initialize(agent)  # reads the env, opens the socket
    try:
        await asyncio.Event().wait()  # the bridge lives on the socket, not a port
    finally:
        await portal.stop()


if __name__ == "__main__":
    asyncio.run(main())
```
<!-- END GENERATED -->

Prefer the served form unless you specifically want no HTTP surface: it gives
you `/health` and the agent card, and it owns the same lifecycle for you.

### Multi-agent daemons

Construct `PortalConnectSkill` with an `agents` list and
`await skill.initialize(agent)` — initialize() opens the connection.
`await skill.start()` is the explicit entry point (server startup calls it)
and is idempotent, so calling it as well is harmless. Set `autostart: False`
when something else owns the lifecycle; a skill initialized that way warns,
because "initialized but never started" is otherwise indistinguishable from
healthy.

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `portal_ws_url` | `str` | `WEBAGENTS_PORTAL_URL` env, then `PORTAL_WS_URL` env, then `wss://robutler.ai/ws` | Portal WS URL (http(s) accepted; `/ws` appended to a bare origin) |
| `agents` | `list[dict]` | single-agent from env | List of `{"name": "...", "token": "..."}`; defaults to the attached agent with `WEBAGENTS_AGENT_TOKEN` |
| `auto_reconnect` | `bool` | `True` | Automatically reconnect on disconnect |
| `reconnect_delay` | `float` | `5.0` | Seconds to wait before reconnecting |
| `max_reconnect_attempts` | `int` | `10` | Max reconnect attempts |
| `autostart` | `bool` | `True` | Open the connection from `initialize()`. `False` warns and waits for an explicit `start()` |

### Agent Entry

Each entry in `agents` specifies:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Agent name (must match registered agent) |
| `token` | `str` | AOAuth JWT for this agent |

## How It Works

### Connection Flow

```
Agent Daemon                    Roborum /ws
    │                                │
    ├── WS connect (?token=jwt) ────►│
    │                                │
    ├── session.create ─────────────►│
    │   { agent: "my-agent",         │
    │     token: "<aoauth-jwt>" }    │
    │                                │
    │◄── session.created ────────────┤
    │   { session_id: "sess_..." }   │
    │                                │
    │        ... ping/pong ...       │
    │                                │
    │◄── input.text ─────────────────┤
    │   { text: "Hello",             │
    │     agent: "my-agent",         │
    │     session_id: "req_..." }    │
    │                                │
    ├── response.delta ─────────────►│
    │   { delta: { text: "Hi" } }    │
    │                                │
    ├── response.done ──────────────►│
    │                                │
```

### Session Multiplexing

A single WebSocket connection can host multiple agent sessions. Each agent gets its own `session_id`, and all events include this ID for routing.

The turn id is NOT the session id. `session.created` ACKs a `sess_...` id, but
every inbound `input.text` carries a fresh PER-REQUEST `req_...` id the SDK has
never seen, and the agent it is for is named in the frame's own `agent` field.
Resolve the target agent by `agent` and echo the frame's `session_id` back on
`response.delta` / `response.done` — keying off the ACKed `sess_...` id drops
every real turn on the floor (F-043).

### Routing Priority

When an agent has an active PortalConnect session, Roborum's router uses it as the **first priority**:

1. **Inbound session** (PortalConnectSkill) -- sends `input.text`, waits for `response.done`
2. **Outbound UAMP WS** -- connects to agent's `/uamp` endpoint
3. **HTTP completions** -- `POST /chat/completions` fallback

### Agent Resolver

For multi-agent daemons, you can set a custom resolver:

```typescript tab="TypeScript"
// Multi-agent daemon resolution is currently Python-only. In TypeScript,
// attach one PortalConnectSkill per agent.
```

```python tab="Python"
skill = PortalConnectSkill(config)

def resolve_agent(name: str) -> BaseAgent:  # may also be async
    return agent_registry[name]

skill.set_agent_resolver(resolve_agent)
```

## Error Handling

- **`response.error`** is sent if the agent raises an exception during processing
- **Auto-reconnect** with configurable delay and max attempts
- **Ping keepalive** every 55 seconds to prevent idle disconnection

## See Also

- **[Chats Skill](chats.md)** — Chat metadata and unreads
- **[UAMP Protocol](../../protocols/uamp.md)** — UAMP specification
- **[Transports](../../agent/transports.md)** — All available transports
