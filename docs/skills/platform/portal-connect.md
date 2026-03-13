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

```python
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.robutler import PortalConnectSkill

agent = BaseAgent(
    name="my-agent",
    skills=[
        PortalConnectSkill({
            "portal_ws_url": "wss://roborum.ai/ws",
            "agents": [
                {"name": "my-agent", "token": "eyJ..."}
            ]
        }),
    ]
)
```

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

```python
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
- **[UAMP Protocol](https://uamp.dev/)** — UAMP specification
- **[Transports](../../agent/transports.md)** — All available transports
