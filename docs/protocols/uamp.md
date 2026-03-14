---
title: UAMP
description: Universal Agentic Message Protocol — an open, event-based protocol that unifies agent communication across transports.
---

# UAMP

UAMP (Universal Agentic Message Protocol) is an open, event-based protocol designed to unify how AI agents communicate — across transports, providers, and frameworks.

Today's agent ecosystem is fragmented across incompatible protocols: OpenAI Completions, Google A2A, Realtime WebSocket, ACP, and more. Each has its own message format, session model, and capabilities negotiation. UAMP provides a single internal protocol that all of these map onto cleanly, so agent logic doesn't need to care which transport a request arrived on.

**Goals:**

- **Unify agentic transports** — One protocol that bridges OpenAI Completions, A2A, Realtime, ACP, and native WebSocket. Agents implement UAMP once and speak every protocol through transport adapters.
- **Flexible and extensible** — Supports text, audio, image, video, and file modalities. Custom event types and provider-specific extensions are first-class.
- **Open** — UAMP is an open protocol. The specification is available in the [Protocols](/docs/webagents/protocols) section of these docs.
- **Production-ready** — UAMP is the internal protocol of the [WebAgents SDK](../quickstart) (Python and TypeScript) and powers all agent communication on the Robutler platform.

## Transport Adapters

UAMP sits between your agent logic and the outside world. Transport skills handle protocol translation:

| External Protocol | Transport Skill | Conversion |
|-------------------|----------------|------------|
| OpenAI Completions | `CompletionsTransportSkill` | Messages → UAMP events → SSE chunks |
| Google A2A | `A2ATransportSkill` | A2A tasks → UAMP events → SSE task events |
| OpenAI Realtime | `RealtimeTransportSkill` | Realtime WS → UAMP events → Realtime WS |
| Agent Client Protocol | `ACPTransportSkill` | JSON-RPC → UAMP events → JSON-RPC |
| UAMP native | `UAMPTransportSkill` | Direct UAMP WebSocket |

Your agent receives UAMP events regardless of which transport the client connected through. See [Transports](../agent/transports) for implementation details.

## Protocol Overview

```
Client                          Agent
   |                              |
   |  session.create ────────►    |
   |                              |
   |  ◄──────── session.created   |
   |                              |
   |  input.text ────────────►    |
   |                              |
   |  response.create ───────►    |
   |                              |
   |  ◄─────────── response.delta |
   |  ◄─────────── response.delta |
   |  ◄─────────── response.done  |
   |                              |
```

## Core Concepts

### Events

All communication uses events with:
- `type` — Event type identifier
- `event_id` — Unique UUID
- `timestamp` — Unix milliseconds (optional)

### Sessions

Sessions maintain conversation state and configuration:
- Modalities (text, audio, image, video, file)
- System instructions
- Available tools
- Voice settings (for audio)

### Capabilities

Both clients and agents announce capabilities:
- Supported modalities
- Streaming support
- Tool support
- Provider-specific features

## Client → Server Events

### session.create

Create a new session.

```json
{
  "type": "session.create",
  "event_id": "uuid",
  "uamp_version": "1.0",
  "session": {
    "modalities": ["text"],
    "instructions": "You are a helpful assistant.",
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "search",
          "description": "Search the web",
          "parameters": {
            "type": "object",
            "properties": {
              "query": { "type": "string" }
            },
            "required": ["query"]
          }
        }
      }
    ]
  },
  "client_capabilities": {
    "id": "my-client",
    "provider": "webagents",
    "modalities": ["text"],
    "supports_streaming": true,
    "supports_thinking": false,
    "supports_caching": false
  }
}
```

### input.text

Send text input.

```json
{
  "type": "input.text",
  "event_id": "uuid",
  "text": "Hello, how are you?",
  "role": "user"
}
```

### input.audio

Send audio input (for voice conversations).

```json
{
  "type": "input.audio",
  "event_id": "uuid",
  "audio": "base64-encoded-audio",
  "format": "pcm16",
  "is_final": true
}
```

### input.image

Send image input.

```json
{
  "type": "input.image",
  "event_id": "uuid",
  "image": "base64-data-or-url",
  "format": "jpeg",
  "detail": "auto"
}
```

### response.create

Request a response from the agent.

```json
{
  "type": "response.create",
  "event_id": "uuid",
  "response": {
    "modalities": ["text"],
    "instructions": "Be concise."
  }
}
```

### response.cancel

Cancel an in-progress response.

```json
{
  "type": "response.cancel",
  "event_id": "uuid",
  "response_id": "optional-response-id"
}
```

### tool.result

Return tool execution result.

```json
{
  "type": "tool.result",
  "event_id": "uuid",
  "call_id": "tool-call-id",
  "result": "{\"data\": \"result\"}",
  "is_error": false
}
```

## Server → Client Events

### session.created

Confirm session creation.

```json
{
  "type": "session.created",
  "event_id": "uuid",
  "uamp_version": "1.0",
  "session": {
    "id": "session-uuid",
    "modalities": ["text"],
    "instructions": "...",
    "tools": []
  }
}
```

### response.delta

Stream response content.

```json
{
  "type": "response.delta",
  "event_id": "uuid",
  "response_id": "response-uuid",
  "delta": {
    "type": "text",
    "text": "Hello"
  }
}
```

Tool call delta:

```json
{
  "type": "response.delta",
  "event_id": "uuid",
  "response_id": "response-uuid",
  "delta": {
    "type": "tool_call",
    "tool_call": {
      "id": "call-uuid",
      "name": "search",
      "arguments": "{\"query\":"
    }
  }
}
```

### response.done

Complete response.

```json
{
  "type": "response.done",
  "event_id": "uuid",
  "response_id": "response-uuid",
  "response": {
    "id": "response-uuid",
    "status": "completed",
    "output": [
      { "type": "text", "text": "Hello! How can I help?" }
    ],
    "usage": {
      "input_tokens": 10,
      "output_tokens": 8,
      "total_tokens": 18
    }
  }
}
```

### response.error

Report an error.

```json
{
  "type": "response.error",
  "event_id": "uuid",
  "response_id": "optional-response-uuid",
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Too many requests",
    "details": { "retry_after": 60 }
  }
}
```

### tool.call

Request tool execution from client.

```json
{
  "type": "tool.call",
  "event_id": "uuid",
  "call_id": "call-uuid",
  "name": "search",
  "arguments": "{\"query\": \"AI news\"}"
}
```

### progress

Progress update for long operations.

```json
{
  "type": "progress",
  "event_id": "uuid",
  "target": "tool",
  "target_id": "call-uuid",
  "message": "Searching...",
  "percent": 50
}
```

### thinking

Reasoning/thinking content (for models that support it).

```json
{
  "type": "thinking",
  "event_id": "uuid",
  "content": "Let me analyze this step by step...",
  "stage": "analysis",
  "is_delta": true
}
```

## Content Items

Content items represent different types of content in messages:

| Type | Fields | Description |
|------|--------|-------------|
| `text` | `text` | Plain text |
| `image` | `url` or `data` + `mime_type` | Image content |
| `audio` | `data` + `mime_type` | Audio content |
| `file` | `data` + `mime_type` + `name` | File attachment |
| `tool_call` | `id`, `name`, `arguments` | Tool invocation |
| `tool_result` | `id`, `result` | Tool response |
| `thinking` | `text` | Reasoning trace |

## Capabilities

Capabilities describe what a client or agent supports:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier |
| `provider` | string | Provider name |
| `modalities` | string[] | `text`, `audio`, `image`, `video`, `file` |
| `supports_streaming` | boolean | Stream responses |
| `supports_thinking` | boolean | Extended thinking |
| `supports_caching` | boolean | Prompt caching |
| `provides` | string[] | What this agent/skill provides |
| `endpoints` | string[] | Available endpoints |

Optional sections for `image`, `audio`, and `tools` capabilities allow advertising format support, resolution limits, and parallel tool call support.

## Message Routing

UAMP supports capability-based message routing through a central router.

- **Handlers** declare `subscribes` (input types/patterns) and `produces` (output types)
- **Router** auto-wires handlers based on capabilities
- **Observers** can listen without consuming (for logging, analytics)
- **Default sink** (`*`) catches unhandled messages (typically transport reply)

See [Router](../agent/router) for detailed documentation.

## System Events

| Event | Description |
|-------|-------------|
| `system.error` | Error during processing |
| `system.stop` | Request to stop |
| `system.cancel` | Cancel and cleanup |
| `system.ping` / `system.pong` | Keep-alive |
| `system.unroutable` | No handler found |

## Version

Current UAMP version: `1.0`

The version is exchanged in `session.create` and `session.created` events to ensure compatibility.

## Further Reading

- [Transports](../agent/transports) — Protocol adapters for each external protocol
- [Router](../agent/router) — Message routing and capabilities
- [AOAuth](aoauth) — Agent authentication protocol
