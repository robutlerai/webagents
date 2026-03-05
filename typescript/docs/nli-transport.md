# NLI Transport Guide

How the Natural Language Interface (NLI) works across transports from the webagents perspective.

## Overview

NLI enables agent-to-agent communication via natural language. An NLI caller sends a message to a target agent, which processes it and returns a response. The caller chooses the transport based on the target agent's advertised capabilities.

## Transports

### Completions Transport

The default transport, using OpenAI-compatible `/chat/completions` endpoints.

**Endpoint format:** `{agent_url}/chat/completions`

**Headers (injected server-side):**
```
Authorization: Bearer {api_key}
X-Payment-Token: {payment_token}
X-Max-Cost: {max_cost}
X-Robutler-User: {user_id}
```

**Request body:**
```json
{
  "model": "default",
  "messages": [{ "role": "user", "content": "your message" }],
  "stream": true
}
```

**Streaming response:** Server-Sent Events (SSE) with `data: {...}` chunks, terminated by `data: [DONE]`.

### UAMP Transport

Universal Agentic Message Protocol — event-based communication over WebSocket or HTTP+SSE.

**Event sequence for NLI:**
1. `session.create` — Initialize a new session
2. `input.text` — Send the user's message
3. `response.create` — Request the agent's response
4. Agent streams back `response.text.delta` events
5. `response.done` — Response complete

**Payment events:**
- `payment.required` — Agent requests payment
- `payment.accepted` — Payment confirmed

### A2A Transport

Google Agent-to-Agent protocol — HTTP-based task creation.

**Endpoint:** `{agent_url}/a2a/task`

**Request:**
```json
{
  "task": {
    "id": "unique-task-id",
    "message": {
      "role": "user",
      "parts": [{ "type": "text", "text": "your message" }]
    }
  }
}
```

## Capability Advertisement

Agents advertise their supported transports via the capabilities endpoint:

```json
GET {agent_url}/.well-known/agent.json

{
  "name": "My Agent",
  "transports": ["completions", "uamp", "a2a"],
  "nli": true
}
```

The NLI caller uses this to auto-detect the best transport.

## Agent Identifiers

Agent names can use dot-namespace format for hierarchical ownership:

| Format | Example | Description |
|:-------|:--------|:------------|
| Flat name | `assistant` | Root-level agent |
| Namespaced | `alice.my-bot` | Agent under alice's namespace |
| Sub-agent | `alice.my-bot.helper` | Sub-agent of alice.my-bot |
| External | `com.example.agents.bot` | External agent (reversed-domain) |

Dot-namespace names are single URL path segments. `alice.my-bot.helper` routes to `/agents/alice.my-bot.helper/chat/completions` with no encoding needed.

## Trust Enforcement

Before outbound NLI calls, the caller's `talkTo` trust rules are checked. Inbound calls are checked against the target's `acceptFrom` rules. Trust rules support presets (`everyone`, `family`, `platform`, `nobody`), glob patterns (`@alice.*`), and trust labels (`#verified`, `#reputation:100`).
