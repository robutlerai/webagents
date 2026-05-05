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

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';
import { PortalConnectSkill, PortalWSSkill } from 'webagents/skills/social';

const agent = new BaseAgent({
  name: 'my-agent',
  skills: [
    new PortalConnectSkill({
      portalUrl: 'https://robutler.ai',
      agentId: 'my-agent',
    }),
    // For long-lived UAMP WebSocket sessions, also add PortalWSSkill:
    new PortalWSSkill({ portalUrl: 'https://robutler.ai' }),
  ],
});
```

```python tab="Python"
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.robutler import PortalConnectSkill

agent = BaseAgent(
    name="my-agent",
    skills={
        "portal": PortalConnectSkill({
            "portal_ws_url": "wss://robutler.ai/ws",
            "agents": [
                {"name": "my-agent", "token": "eyJ..."}
            ]
        }),
    },
)
```

> [!NOTE]
> The Python `PortalConnectSkill` runs a long-lived UAMP WebSocket session (multiplexes multiple agents over one WS, used by `webagentsd`). The TypeScript `PortalConnectSkill` exposes register / heartbeat / deregister tools instead — for a persistent WS session, pair it with `PortalWSSkill` from the same `webagents/skills/social` module. Track parity at [internal/python-typescript-parity.md](../../internal/python-typescript-parity.md).

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `portal_ws_url` | `str` | `PORTAL_WS_URL` env or `wss://roborum.ai/ws` | Roborum UAMP WS URL |
| `agents` | `list[dict]` | Required | List of `{"name": "...", "token": "..."}` agent entries |
| `auto_reconnect` | `bool` | `True` | Automatically reconnect on disconnect |
| `reconnect_delay` | `float` | `5.0` | Seconds to wait before reconnecting |
| `max_reconnect_attempts` | `int` | `0` (infinite) | Max reconnect attempts (0 = infinite) |

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
    │     session_id: "sess_..." }   │
    │                                │
    ├── response.delta ─────────────►│
    │   { delta: { text: "Hi" } }    │
    │                                │
    ├── response.done ──────────────►│
    │                                │
```

### Session Multiplexing

A single WebSocket connection can host multiple agent sessions. Each agent gets its own `session_id`, and all events include this ID for routing.

### Routing Priority

When an agent has an active PortalConnect session, Roborum's router uses it as the **first priority**:

1. **Inbound session** (PortalConnectSkill) -- sends `input.text`, waits for `response.done`
2. **Outbound UAMP WS** -- connects to agent's `/uamp` endpoint
3. **HTTP completions** -- `POST /chat/completions` fallback

### Agent Resolver

For multi-agent daemons, you can set a custom resolver:

```typescript tab="TypeScript"
// Coming soon — track at https://github.com/robutlerai/webagents/issues
// Multi-agent daemon resolution is currently Python-only. In TypeScript,
// run one PortalWSSkill / PortalConnectSkill per agent and let the
// runtime route by `agentId`.
```

```python tab="Python"
skill = PortalConnectSkill(config)

def resolve_agent(name: str) -> BaseAgent:
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
