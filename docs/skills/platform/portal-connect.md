---
title: Portal Connect Skill
description: Persistent UAMP WebSocket session to Robutler for daemon-mode agents.
---

# Portal Connect Skill

The **PortalConnectSkill** connects agents to the Robutler platform via a persistent UAMP WebSocket, enabling real-time bidirectional communication without requiring a public URL.

## Overview

PortalConnectSkill is designed for **daemon-mode agents** (`webagents daemon`). It:

1. Connects to the Robutler UAMP WS server (`wss://robutler.ai/ws`)
2. Creates one `session.create` per agent, proving the agent's identity either
   by signing the handshake with the key the agent serves, or with its
   per-agent key
3. Listens for `input.text` events from the platform
4. Runs the agent and streams back `response.delta` / `response.done`
5. Maintains the connection with periodic UAMP pings

This is the preferred transport for hosted agents that don't expose public HTTP endpoints.

## Quick Start

Attach the skill and run the agent. No wrapper function is needed, because
there is nothing for one to do: the skill reads `WEBAGENTS_PORTAL_URL` and
`WEBAGENTS_AGENT_TOKEN` itself and refuses an owner-subject token with no
agent binding before it opens a socket. In TypeScript `serve()`'s lifecycle
starts it (the server binds loopback until an auth skill verifies callers); in
Python nothing needs to listen, so the skill's own lifecycle starts it and the
process keeps the event loop alive. Give the Python agent to
`create_server(agents=[agent])` as well when you want `/health`, the agent
card and a chat endpoint; the server's lifecycle then starts the skill.

<!-- Maintainers: generated from the example files; edit those and run scripts/sync_doc_examples.py. -->
<!-- BEGIN GENERATED: typescript/examples/portal-connect-minimal.ts,python/examples/portal_connect_minimal.py -->
```typescript tab="TypeScript"
import { BaseAgent, OpenAISkill, PortalConnectSkill, serve } from 'webagents';

export const agent = new BaseAgent({
  name: 'mini',
  instructions: 'You are helpful.',
  model: 'openai/gpt-4o-mini',
  // `PortalConnectSkill` carries the transport, not the model. In TypeScript
  // the language model is a skill you add: `model` above advertises which
  // input types this agent accepts, it does not choose a provider. Without a
  // provider skill every turn answers `No LLM skill available to process
  // request`.
  skills: [new OpenAISkill({ model: 'gpt-4o-mini' }), new PortalConnectSkill()],
});

export const server = await serve(agent, {
  port: Number(process.env.PORT ?? 8000),
  basePath: '/agents/mini',
});
```
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

A token is needed only where the platform cannot fetch the agent's key set:
a process with no HTTP server, or one served on a loopback or private address.
An agent served at a public https address signs its handshake instead.

Where a token is needed, the skill looks for it in this order: the `token`
option, `WEBAGENTS_AGENT_TOKEN`, then the key `webagents publish` stored for
the agent the working directory is linked to. So after one `publish` in the
agent's directory there is nothing
to configure; the variable is for containers and CI, where there is no
keystore.

When you do use a token, it MUST be a per-agent key: the one `webagents publish`
stores as `AGENT_KEY_<NAME>`, or one from
`POST /api/agents/{id}/api-key` (its JWT carries an `agent_id` claim). A generic
owner key connects successfully and then never receives a single turn; the
skill turns that into a start-time error with the fix in the message.

### No HTTP server at all

If nothing will ever dial this process there is no port to bind, so there is
no server: just the skill's lifecycle, run on the event loop and kept alive.
Written out rather than hidden behind a one-word call, because what the
process is doing is the whole point.

<!-- BEGIN GENERATED: typescript/examples/portal-connect-socket-only.ts,python/examples/portal_connect_socket_only.py -->
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

In TypeScript, prefer the served form unless you specifically want no HTTP
surface: it gives you `/health` and the agent card, and it owns the same
lifecycle for you. In Python the two examples on this page are the same shape;
add `create_server(agents=[agent])` when you want an HTTP surface.

### Multi-agent daemons

Construct `PortalConnectSkill` with an `agents` list and
`await skill.initialize(agent)`; `initialize()` opens the connection.
`await skill.start()` is the explicit entry point (server startup calls it)
and is idempotent, so calling it as well is harmless. Set `autostart: False`
when something else owns the lifecycle; a skill initialized that way warns,
because "initialized but never started" is otherwise indistinguishable from
healthy.

## Configuration

Python:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `portal_ws_url` | `str` | `WEBAGENTS_PORTAL_URL` env, then `PORTAL_WS_URL` env, then `wss://robutler.ai/ws` | Portal WS URL (http(s) accepted; `/ws` appended to a bare origin) |
| `agents` | `list[dict]` | single-agent from env | List of `{"name": "...", "token": "..."}`; defaults to the attached agent with `WEBAGENTS_AGENT_TOKEN` |
| `auto_reconnect` | `bool` | `True` | Automatically reconnect on disconnect |
| `reconnect_delay` | `float` | `5.0` | Seconds to wait before reconnecting |
| `max_reconnect_attempts` | `int` | `10` | Max reconnect attempts |
| `autostart` | `bool` | `True` | Open the connection from `initialize()`. `False` warns and waits for an explicit `start()` |

TypeScript (`new PortalConnectSkill({...})`), one agent per skill:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `portalUrl` | `string` | `WEBAGENTS_PORTAL_URL`, then `PORTAL_WS_URL`, then `wss://robutler.ai/ws` | Portal WS URL (http(s) accepted; `/ws` appended to a bare origin) |
| `token` | `string` | `WEBAGENTS_AGENT_TOKEN`, then the key `publish`/`deploy` stored for the linked directory | Per-agent key. Optional when the agent is served where the platform can fetch its key set |
| `identity` | `SigningIdentity` | the identity `serve()` hands the agent | Signs the handshake when there is no token |
| `autoReconnect` | `boolean` | `true` | Reconnect on an unexpected close |
| `reconnectDelayS` | `number` | `5` | Seconds between reconnect attempts |
| `maxReconnectAttempts` | `number` | `10` | Max consecutive reconnect attempts |
| `autostart` | `boolean` | `true` | Open the connection from `initialize()` |

### Agent Entry

Each entry in `agents` specifies:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Agent name (must match registered agent) |
| `token` | `str` | Platform token for this agent (`WEBAGENTS_AGENT_TOKEN`) |

## How It Works

### Connection Flow

```
Agent Daemon                    Robutler /ws
    │                                │
    ├── WS connect (?token=jwt) ────►│
    │                                │
    ├── session.create ─────────────►│
    │   { agent: "my-agent",         │
    │     token: "<platform-token>" }│
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
`response.delta` / `response.done`. Keying off the ACKed `sess_...` id drops
every real turn on the floor.

### Who the turn is from

The platform decides who a relayed turn is from, against the agent's owner,
and puts it on the `input.text` frame as `caller`:

```json
{ "user_id": "4b0a2f3e-...", "tier": "user", "username": "alice",
  "channel": { "type": "telegram", "sender_id": "8842" } }
```

Both SDKs read that field and nothing else: never the message text, and
never a `caller` a client put in its own frame, which the platform drops
before the frame reaches the agent. `tier: owner` runs the turn as the owner,
with the owner-only tools. `tier: user` runs it as a verified user, with the
principals `user:<id>` and, when there is a handle, `user:@<handle>`, which
an `access:` block can place in groups. A turn that came in through a
channel the owner connected on Robutler also carries `channel`, and its
sender is the principal `channel:<type>:<sender id>`, listed first (see
[Who can call your agent](../../guides/trust.md#channel-senders)). A frame
without `caller` runs anonymous.

### Routing Priority

When an agent has an active PortalConnect session, Robutler's router uses it as the **first priority**:

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

- **[Chats Skill](chats.md)**: chat metadata and unreads
- **[UAMP Protocol](../../protocols/uamp.md)**: the UAMP specification
- **[Transports](../../agent/transports.md)**: all available transports
