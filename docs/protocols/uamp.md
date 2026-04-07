---
title: UAMP Protocol Specification
description: Universal Agentic Message Protocol — the open, event-based protocol that unifies agent communication across transports, providers, and frameworks.
---

# UAMP Protocol Specification

**Universal Agentic Message Protocol v1.0**

## 1. Introduction

### 1.1 Motivation

AI agents today speak dozens of incompatible languages. OpenAI Chat Completions, OpenAI Realtime, Google A2A, ACP — each defines its own message format, session model, and capability negotiation. Building an agent that works across all of them means writing and maintaining a separate integration for each protocol, and adding a new transport means touching every agent.

UAMP eliminates this fragmentation. It is a single, event-based internal protocol that all external protocols map onto cleanly through thin transport adapters. Agent logic implements UAMP once and automatically speaks every supported protocol. A new transport is one adapter — zero changes to your agent.

UAMP is an open protocol. When a request arrives over the OpenAI Completions API, it becomes UAMP events. When a Google A2A task comes in, it becomes UAMP events. The agent never knows or cares which wire format the client used. Reference implementations exist in Python and TypeScript.

### 1.2 Design Principles

1. **Event-based** — All communication is asynchronous events, not request/response. This works naturally for streaming, batch, and real-time voice.
2. **Multimodal native** — Text, audio, images, video, and files are first-class from day one. No bolted-on extensions.
3. **Transport agnostic** — Works over WebSocket, HTTP+SSE, or batch REST. The protocol does not assume a transport.
4. **Bidirectional** — Client and server events have clear, symmetric semantics.
5. **Session-aware** — Built-in conversation and session management, including multiplexed sessions over a single connection.
6. **Provider-agnostic** — No vendor lock-in. Works with any LLM backend through provider adapters.

### 1.3 Compatibility

UAMP is based on the event structure of OpenAI's Realtime API but is transport-independent and significantly extended for multimodal, multi-agent, and payment-enabled workflows.

| External Protocol | Compatibility | Conversion |
|---|---|---|
| OpenAI Chat Completions | Full | Messages → UAMP events → SSE chunks |
| OpenAI Realtime API | Near 1:1 mapping | Realtime WS → UAMP events → Realtime WS |
| Google A2A | Full via adapter | A2A tasks → UAMP events → SSE task events |
| Agent Communication Protocol | Full via adapter | JSON-RPC → UAMP events → JSON-RPC |
| UAMP Native | Direct | Native UAMP over WebSocket |

An agent receives UAMP events regardless of which transport the client connected through.

### 1.4 Specification Scope

This document defines the UAMP wire protocol: event structures, session lifecycle, capability negotiation, and transport mappings. Any conforming implementation — in any language or framework — can interoperate by following this specification.

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Universal Agentic Message Protocol             │
│          (Event-based, Multimodal, Bidirectional)           │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌───────────────────┐
│   WebSocket   │   │   HTTP + SSE    │   │   Batch/REST      │
│  Full Duplex  │   │  Server→Client  │   │  Request/Response │
└───────────────┘   └─────────────────┘   └───────────────────┘
```

Inbound transport adapters convert external protocol formats into UAMP events. The agent processes UAMP events through its internal processing pipeline. Outbound LLM adapters convert UAMP events into provider-specific API calls and stream the results back as UAMP events.

```
External Protocol          UAMP                   LLM Provider
     │                      │                          │
     ▼                      ▼                          ▼
┌──────────┐         ┌──────────┐              ┌──────────┐
│Transport │  toUAMP │  Agent   │   toProvider │   LLM    │
│ Adapter  │ ──────► │  Core    │ ───────────► │ Adapter  │
│          │         │          │              │          │
│          │fromUAMP │          │ fromProvider │          │
│          │ ◄────── │          │ ◄─────────── │          │
└──────────┘         └──────────┘              └──────────┘
```

## 3. Protocol Flow

### 3.1 Basic Text Chat

```
Client                              Server
   │                                   │
   │─── session.create ───────────────>│
   │<── session.created ───────────────│
   │<── capabilities ──────────────────│
   │                                   │
   │─── input.text ───────────────────>│
   │─── response.create ──────────────>│
   │                                   │
   │<── response.created ──────────────│
   │<── response.delta ────────────────│
   │<── response.delta ────────────────│
   │<── response.done ─────────────────│
   │                                   │
```

### 3.2 With Tool Calls

```
Client                              Server
   │                                   │
   │─── input.text ───────────────────>│
   │─── response.create ──────────────>│
   │                                   │
   │<── response.created ──────────────│
   │<── tool.call ─────────────────────│
   │                                   │
   │─── tool.result ──────────────────>│
   │                                   │
   │<── response.delta ────────────────│
   │<── response.done ─────────────────│
   │                                   │
```

### 3.3 With Payment Negotiation

```
Client                              Server
   │                                   │
   │─── input.text ───────────────────>│
   │─── response.create ──────────────>│
   │                                   │
   │<── payment.required ──────────────│
   │                                   │
   │─── payment.submit ───────────────>│
   │<── payment.accepted ──────────────│
   │                                   │
   │<── response.delta ────────────────│
   │<── response.done ─────────────────│
   │<── payment.balance ───────────────│
   │                                   │
```

## 4. Base Event Structure

All UAMP events share a common base structure:

```json
{
  "type": "event.type",
  "event_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": 1704067200000,
  "session_id": "sess_abc123"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `type` | string | Yes | Event type identifier (e.g. `session.create`, `response.delta`) |
| `event_id` | string | Yes | Unique event ID (UUID) |
| `timestamp` | number | No | Unix timestamp in milliseconds |
| `session_id` | string | No | Session scope. Required for multiplexed connections; omitted for single-session mode. |

## 5. Client → Server Events

### 5.1 session.create

Create a new session. This is always the first event a client sends.

```json
{
  "type": "session.create",
  "event_id": "evt_001",
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
    ],
    "voice": {
      "provider": "openai",
      "voice_id": "alloy"
    },
    "input_audio_format": "pcm16",
    "output_audio_format": "pcm16",
    "turn_detection": {
      "type": "server_vad",
      "threshold": 0.5,
      "silence_duration_ms": 500
    },
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "weather",
        "schema": { "type": "object", "properties": { "temp": { "type": "number" } } },
        "strict": true
      }
    },
    "extensions": {
      "openai": { "model": "gpt-4o", "temperature": 0.7 },
      "anthropic": { "thinking": true }
    }
  },
  "agent": "weather-bot",
  "chat": "chat_abc",
  "token": "eyJ...",
  "payment_token": "ptok_...",
  "client_capabilities": {
    "id": "web-app",
    "provider": "robutler",
    "modalities": ["text", "image", "audio"],
    "supports_streaming": true,
    "supports_thinking": false,
    "supports_caching": false,
    "widgets": ["chart", "table", "form"],
    "extensions": { "supports_html": true, "platform": "web" }
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `uamp_version` | string | Yes | Protocol version (e.g. `"1.0"`) |
| `session` | object | Yes | Session configuration (see [Session Config](#81-session-configuration)) |
| `agent` | string | No | Target agent name/ID for multiplexed connections |
| `chat` | string | No | Chat ID when session is chat-scoped |
| `token` | string | No | Per-session auth token (e.g. AOAuth JWT) |
| `payment_token` | string | No | Per-session payment token |
| `client_capabilities` | Capabilities | No | Client capability declaration (see [Capabilities](#9-capabilities)) |

### 5.2 session.update

Update session auth or payment context without reconnecting. Used for token refresh.

```json
{
  "type": "session.update",
  "event_id": "evt_010",
  "session_id": "sess_abc",
  "token": "eyJ_new...",
  "payment_token": "ptok_new..."
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `session_id` | string | No | Omit for connection-level update |
| `token` | string | No | New auth token |
| `payment_token` | string | No | New payment token |

### 5.3 session.end

End a session. Either side can send this.

```json
{
  "type": "session.end",
  "event_id": "evt_099",
  "reason": "user_left"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `reason` | string | No | `"user_left"`, `"timeout"`, `"error"`, or custom |

### 5.4 capabilities.query

Query server capabilities. The server responds with a `capabilities` event.

```json
{
  "type": "capabilities.query",
  "event_id": "evt_005",
  "model": "gpt-4o"
}
```

### 5.5 client.capabilities

Announce or update client capabilities mid-session.

```json
{
  "type": "client.capabilities",
  "event_id": "evt_006",
  "capabilities": {
    "id": "web-app",
    "provider": "robutler",
    "modalities": ["text", "image", "audio"],
    "supports_streaming": true,
    "widgets": ["chart", "table"]
  }
}
```

### 5.6 input.text

Send text input. Optionally carries full conversation history for stateless context passing.

```json
{
  "type": "input.text",
  "event_id": "evt_100",
  "text": "What's the weather in Paris?",
  "role": "user",
  "messages": [
    { "role": "system", "content": "You are a weather assistant." },
    { "role": "user", "content": "What's the weather in Paris?" }
  ],
  "payment_token": "ptok_...",
  "context": {
    "chat_id": "chat_abc",
    "sender_id": "user_123"
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | Yes | The text content |
| `role` | string | No | `"user"` (default), `"system"`, or `"assistant"` |
| `messages` | Message[] | No | Full conversation history for stateless context passing |
| `payment_token` | string | No | Payment token for this interaction |
| `context` | object | No | Routing and broadcast metadata (extensible) |

### 5.7 input.audio

Send audio input for voice conversations.

```json
{
  "type": "input.audio",
  "event_id": "evt_101",
  "audio": "base64-encoded-audio-data",
  "format": "pcm16",
  "is_final": true
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `audio` | string | Yes | Base64-encoded audio data |
| `format` | AudioFormat | Yes | Audio format (see [Audio Format](#82-audio-format)) |
| `is_final` | boolean | No | `true` marks the end of the audio stream |
| `content_id` | string | No | UUID for cross-agent content referencing (see [Content IDs](#854-content-ids)) |

### 5.8 input.image

Send image input.

```json
{
  "type": "input.image",
  "event_id": "evt_102",
  "image": "base64-data-or-url",
  "format": "jpeg",
  "detail": "auto"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | string \| `{ url: string }` | Yes | Base64-encoded data or URL |
| `format` | string | No | `"jpeg"`, `"png"`, `"webp"`, `"gif"` |
| `detail` | string | No | `"low"`, `"high"`, `"auto"` |
| `content_id` | string | No | UUID for cross-agent content referencing (see [Content IDs](#854-content-ids)) |

### 5.9 input.video

Send video input.

```json
{
  "type": "input.video",
  "event_id": "evt_103",
  "video": { "url": "https://example.com/video.mp4" },
  "format": "mp4",
  "content_id": "a1b2c3d4-..."
}
```

Optional `content_id` (string): UUID for cross-agent content referencing (see [Content IDs](#854-content-ids)).

### 5.10 input.file

Send file input.

```json
{
  "type": "input.file",
  "event_id": "evt_104",
  "file": "base64-encoded-file-data",
  "filename": "report.pdf",
  "mime_type": "application/pdf",
  "content_id": "e5f67890-..."
}
```

Optional `content_id` (string): UUID for cross-agent content referencing (see [Content IDs](#854-content-ids)).

### 5.11 input.typing

Indicate the user has started or stopped typing.

```json
{
  "type": "input.typing",
  "event_id": "evt_105",
  "is_typing": true,
  "chat_id": "chat_abc"
}
```

### 5.12 response.create

Request the agent to generate a response.

```json
{
  "type": "response.create",
  "event_id": "evt_200",
  "response": {
    "modalities": ["text"],
    "instructions": "Be concise.",
    "tools": []
  },
  "response_format": {
    "type": "json_object"
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `response` | object | No | Override session-level modalities, instructions, or tools for this response |
| `response_format` | ResponseFormat | No | Override output format for this response |

### 5.13 response.cancel

Cancel an in-progress response.

```json
{
  "type": "response.cancel",
  "event_id": "evt_201",
  "response_id": "resp_abc"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `response_id` | string | No | If omitted, cancels the current response |

### 5.14 tool.result

Return the result of a tool execution.

```json
{
  "type": "tool.result",
  "event_id": "evt_300",
  "call_id": "call_abc",
  "result": "{\"temperature\": 22, \"unit\": \"celsius\"}",
  "is_error": false
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `call_id` | string | Yes | The `call_id` from the corresponding `tool.call` event |
| `result` | string | Yes | JSON-serialized result |
| `is_error` | boolean | No | `true` if the tool execution failed |

### 5.15 payment.submit

Submit payment token or proof in response to a `payment.required` event.

```json
{
  "type": "payment.submit",
  "event_id": "evt_400",
  "payment": {
    "scheme": "token",
    "network": "robutler",
    "token": "tok_xxx",
    "amount": "10.00"
  }
}
```

See [Payment Events](#72-payment-events) for the full payment flow.

### 5.16 voice.invite / voice.accept / voice.decline / voice.end

Voice session lifecycle events. See [Voice Events](#73-voice-events).

### 5.17 ping

Connection keepalive.

```json
{
  "type": "ping",
  "event_id": "ping_001"
}
```

## 6. Server → Client Events

### 6.1 session.created

Confirm session creation.

```json
{
  "type": "session.created",
  "event_id": "evt_002",
  "uamp_version": "1.0",
  "session": {
    "id": "sess_abc123",
    "created_at": 1704067200,
    "config": {
      "modalities": ["text"],
      "instructions": "You are a helpful assistant.",
      "tools": []
    },
    "status": "active"
  },
  "session_id": "sess_abc123",
  "chat": "chat_abc",
  "agent": "weather-bot"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `uamp_version` | string | Yes | Server's supported UAMP version |
| `session` | Session | Yes | Full session object |
| `session_id` | string | No | For multiplexed sessions |
| `chat` | string | No | Echo of chat ID |
| `agent` | string | No | Echo of agent name |

### 6.2 session.updated

Confirm a session update (e.g. token refresh).

```json
{
  "type": "session.updated",
  "event_id": "evt_011",
  "session_id": "sess_abc"
}
```

### 6.3 session.error

Session-level error (distinct from response errors).

```json
{
  "type": "session.error",
  "event_id": "evt_012",
  "error": {
    "code": "agent_offline",
    "message": "The requested agent is currently unavailable.",
    "details": { "retry_after": 30 }
  }
}
```

| Error Code | Description |
|---|---|
| `agent_offline` | Target agent is unavailable |
| `rate_limited` | Too many requests |
| `unauthorized` | Authentication failed |
| `timeout` | Session timed out |

### 6.4 capabilities

Server announces model/agent capabilities. Sent after `session.created`, in response to `capabilities.query`, or when the backend model changes mid-session.

```json
{
  "type": "capabilities",
  "event_id": "evt_003",
  "capabilities": {
    "id": "gpt-4o",
    "provider": "openai",
    "modalities": ["text", "image"],
    "image": {
      "formats": ["jpeg", "png", "gif", "webp"],
      "detail_levels": ["auto", "low", "high"]
    },
    "file": { "supports_pdf": true },
    "tools": {
      "supports_tools": true,
      "supports_parallel_tools": true,
      "built_in_tools": ["web_search", "code_interpreter"]
    },
    "supports_streaming": true,
    "supports_thinking": false,
    "context_window": 128000,
    "max_output_tokens": 4096
  }
}
```

### 6.5 response.created

Confirms a response has started.

```json
{
  "type": "response.created",
  "event_id": "evt_210",
  "response_id": "resp_abc"
}
```

### 6.6 response.delta

Stream response content incrementally. The `delta.type` field indicates the content type.

**Text delta:**

```json
{
  "type": "response.delta",
  "event_id": "evt_211",
  "response_id": "resp_abc",
  "delta": {
    "type": "text",
    "text": "The weather in Paris is"
  }
}
```

**Tool call delta (streamed arguments):**

```json
{
  "type": "response.delta",
  "event_id": "evt_212",
  "response_id": "resp_abc",
  "delta": {
    "type": "tool_call",
    "tool_call": {
      "id": "call_abc",
      "name": "search",
      "arguments": "{\"query\":"
    }
  }
}
```

**Audio delta:**

```json
{
  "type": "response.delta",
  "event_id": "evt_213",
  "response_id": "resp_abc",
  "delta": {
    "type": "audio",
    "audio": "base64-encoded-audio-chunk"
  }
}
```

| Delta Type | Fields | Description |
|---|---|---|
| `text` | `text` | Incremental text content |
| `audio` | `audio` | Base64-encoded audio chunk |
| `tool_call` | `tool_call.id`, `tool_call.name`, `tool_call.arguments` | Streamed tool invocation |
| `tool_result` | `tool_result.call_id`, `tool_result.result`, `tool_result.status` | Tool result (server-side tools) |
| `tool_progress` | `tool_progress.call_id`, `tool_progress.text` | Tool execution progress |

### 6.7 response.done

Response completed. Contains the full output and usage statistics.

```json
{
  "type": "response.done",
  "event_id": "evt_220",
  "response_id": "resp_abc",
  "response": {
    "id": "resp_abc",
    "status": "completed",
    "output": [
      { "type": "text", "text": "The weather in Paris is 22°C and sunny." }
    ],
    "usage": {
      "input_tokens": 25,
      "output_tokens": 12,
      "total_tokens": 37,
      "cost": {
        "input_cost": 0.000025,
        "output_cost": 0.000036,
        "total_cost": 0.000061,
        "currency": "USD"
      }
    }
  },
  "signature": "eyJ..."
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `response.status` | string | Yes | `"completed"`, `"cancelled"`, or `"failed"` |
| `response.output` | ContentItem[] | Yes | Response content items |
| `response.usage` | UsageStats | No | Token and cost tracking |
| `signature` | string | No | RS256 JWT for cryptographic non-repudiation (contains `response_hash` and `request_hash` claims) |

### 6.8 response.error

Report a response-level error.

```json
{
  "type": "response.error",
  "event_id": "evt_230",
  "response_id": "resp_abc",
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Too many requests. Retry after 60 seconds.",
    "details": { "retry_after": 60 }
  }
}
```

### 6.9 response.cancelled

Confirms a response was cancelled (in response to `response.cancel`).

```json
{
  "type": "response.cancelled",
  "event_id": "evt_231",
  "response_id": "resp_abc",
  "partial_output": [
    { "type": "text", "text": "The weather in" }
  ]
}
```

### 6.10 tool.call

Request tool execution from the client.

```json
{
  "type": "tool.call",
  "event_id": "evt_310",
  "call_id": "call_abc",
  "name": "search",
  "arguments": "{\"query\": \"weather Paris\"}"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `call_id` | string | Yes | Unique call identifier. The client echoes this in `tool.result`. |
| `name` | string | Yes | Tool/function name |
| `arguments` | string | Yes | JSON-serialized arguments |

### 6.11 tool.call_done

Indicates a server-side tool call has completed.

```json
{
  "type": "tool.call_done",
  "event_id": "evt_311",
  "call_id": "call_abc"
}
```

### 6.12 progress

Progress update for long-running operations.

```json
{
  "type": "progress",
  "event_id": "evt_500",
  "target": "tool",
  "target_id": "call_abc",
  "stage": "searching",
  "message": "Searching the web...",
  "percent": 50,
  "step": 2,
  "total_steps": 4
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `target` | string | Yes | `"tool"`, `"response"`, `"upload"`, or `"reasoning"` |
| `target_id` | string | No | ID of the target (e.g. tool call ID or response ID) |
| `stage` | string | No | Current stage name |
| `message` | string | No | Human-readable status |
| `percent` | number | No | 0–100 |
| `step` / `total_steps` | number | No | Discrete step progress |

### 6.13 thinking

Reasoning/thinking content for models that support extended thinking. Supported providers:

- **Anthropic**: `thinking` content blocks (Claude extended thinking)
- **Google Gemini**: `thinkingConfig.includeThoughts` + `part.thought` (gemini-2.5+, gemini-3.x)
- **Fireworks / DeepSeek**: `delta.reasoning_content` (all OpenAI-compatible reasoning models)

```json
{
  "type": "thinking",
  "event_id": "evt_510",
  "content": "Let me analyze the weather data step by step...",
  "stage": "analysis",
  "redacted": false,
  "is_delta": true
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `content` | string | Yes | The reasoning text |
| `stage` | string | No | `"analyzing"`, `"planning"`, `"reflecting"` |
| `redacted` | boolean | No | `true` if content was redacted for safety |
| `is_delta` | boolean | No | `true` = append to previous thinking, `false` = complete thought |

**Persistence**: Thinking events are accumulated and persisted as `ThinkingContent` items in the message's `contentItems` array. Consecutive thinking deltas are merged into a single item. The inline position relative to text and tool calls is preserved.

**Delegate forwarding**: When a sub-agent emits thinking events during delegation, they are forwarded as `tool_progress` text to the parent agent. This behavior is controlled by the `DELEGATE_FORWARD_THINKING` environment variable (enabled by default, set to `'0'` to suppress).

### 6.14 audio.delta / audio.done

Streaming audio output and completion.

```json
{
  "type": "audio.delta",
  "event_id": "evt_520",
  "response_id": "resp_abc",
  "audio": "base64-encoded-audio-chunk"
}
```

```json
{
  "type": "audio.done",
  "event_id": "evt_521",
  "response_id": "resp_abc",
  "duration_ms": 3200
}
```

### 6.15 transcript.delta / transcript.done

Real-time transcription of audio content.

```json
{
  "type": "transcript.delta",
  "event_id": "evt_530",
  "response_id": "resp_abc",
  "transcript": "The weather"
}
```

### 6.16 usage.delta

Incremental usage statistics during streaming.

```json
{
  "type": "usage.delta",
  "event_id": "evt_540",
  "response_id": "resp_abc",
  "delta": {
    "output_tokens": 5
  }
}
```

### 6.17 rate_limit

Rate limit notification.

```json
{
  "type": "rate_limit",
  "event_id": "evt_550",
  "limit": 100,
  "remaining": 3,
  "reset_at": 1704067260
}
```

### 6.18 presence.typing

Broadcasts typing status from another participant (multi-user scenarios).

```json
{
  "type": "presence.typing",
  "event_id": "evt_560",
  "user_id": "user_a",
  "username": "alice",
  "is_typing": true,
  "chat_id": "chat_abc"
}
```

### 6.19 pong

Keepalive response.

```json
{
  "type": "pong",
  "event_id": "pong_001"
}
```

## 7. Extended Event Categories

### 7.1 Presence and Chat Events

For multi-user chat scenarios, UAMP defines presence and messaging events:

| Event | Direction | Description |
|---|---|---|
| `input.typing` | Client → Server | User started/stopped typing |
| `presence.typing` | Server → Client | Another participant is typing |
| `presence.online` | Server → Client | User/agent came online |
| `presence.offline` | Server → Client | User/agent went offline |
| `message.created` | Server → Client | New chat message (fan-out to subscribers) |
| `message.read` | Bidirectional | Read receipt |

### 7.2 Payment Events

Payment events enable real-time token balance management and payment negotiation during agent conversations.

| Event | Direction | Description |
|---|---|---|
| `payment.required` | Server → Client | Payment required to continue |
| `payment.submit` | Client → Server | Submit payment token/proof |
| `payment.accepted` | Server → Client | Payment accepted |
| `payment.balance` | Server → Client | Balance update notification |
| `payment.error` | Server → Client | Payment error |

**payment.required:**

```json
{
  "type": "payment.required",
  "event_id": "evt_410",
  "response_id": "resp_abc",
  "requirements": {
    "amount": "10.00",
    "currency": "USD",
    "schemes": [
      { "scheme": "token", "network": "robutler" },
      { "scheme": "crypto", "network": "base", "address": "0x..." }
    ],
    "expires_at": 1704067500,
    "reason": "llm_usage",
    "ap2": {
      "mandate_uri": "https://...",
      "credential_types": ["VerifiableCredential"],
      "checkout_session_uri": "https://..."
    }
  }
}
```

**payment.balance:**

```json
{
  "type": "payment.balance",
  "event_id": "evt_420",
  "balance": "9.50",
  "currency": "USD",
  "low_balance_warning": false,
  "estimated_remaining": 190,
  "expires_at": 1704153600
}
```

**payment.error:**

```json
{
  "type": "payment.error",
  "event_id": "evt_430",
  "code": "insufficient_balance",
  "message": "Insufficient balance to process request.",
  "balance_required": "0.05",
  "balance_current": "0.00",
  "can_retry": true
}
```

| Error Code | Description |
|---|---|
| `insufficient_balance` | Not enough funds |
| `token_expired` | Payment token has expired |
| `token_invalid` | Payment token is invalid |
| `payment_failed` | Payment processing failed |
| `rate_limited` | Payment rate limit hit |
| `mandate_revoked` | AP2 mandate was revoked |

### 7.3 Voice Events

Voice events manage real-time voice session lifecycle:

| Event | Direction | Description |
|---|---|---|
| `voice.invite` | Client → Server | Initiate voice session (with optional WebRTC SDP offer) |
| `voice.accept` | Server → Client | Accept voice session (with optional SDP answer) |
| `voice.decline` | Server → Client | Decline voice session |
| `voice.end` | Bidirectional | End voice session (with optional `duration_ms`) |

## 8. Type Definitions

### 8.1 Session Configuration

```json
{
  "modalities": ["text", "audio"],
  "instructions": "System prompt here.",
  "tools": [],
  "voice": { "provider": "openai", "voice_id": "alloy" },
  "input_audio_format": "pcm16",
  "output_audio_format": "pcm16",
  "turn_detection": { "type": "server_vad" },
  "response_format": { "type": "text" },
  "extensions": {}
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `modalities` | Modality[] | Yes | `"text"`, `"audio"`, `"image"`, `"video"`, `"file"`, or custom |
| `instructions` | string | No | System instructions |
| `tools` | ToolDefinition[] | No | Available tools |
| `voice` | VoiceConfig | No | Voice configuration |
| `input_audio_format` | AudioFormat | No | Expected input audio format |
| `output_audio_format` | AudioFormat | No | Output audio format |
| `turn_detection` | TurnDetectionConfig | No | Voice turn detection settings |
| `response_format` | ResponseFormat | No | Structured output format |
| `extensions` | object | No | Provider-specific extensions (keyed by provider name) |

### 8.2 Audio Format

```
"pcm16" | "g711_ulaw" | "g711_alaw" | "mp3" | "opus" | "wav" | "webm" | "aac" | string
```

### 8.3 Response Format

Controls structured output from the model.

| Type | Behavior |
|---|---|
| `text` | Default. Free-form text output. |
| `json_object` | Model returns valid JSON. No schema enforced. |
| `json_schema` | Model returns JSON conforming to the provided schema. |

Provider mapping:

| Provider | `json_schema` | `json_object` |
|---|---|---|
| OpenAI / LiteLLM | Passed through natively | Passed through natively |
| Google Gemini | `response_mime_type='application/json'` + `response_schema` | `response_mime_type='application/json'` |
| Anthropic | Forced tool use with `input_schema` + unwrap | System prompt instruction |

### 8.4 Content Items

Content items are a discriminated union on the `type` field. They appear in `response.done` output, `Message.content_items`, and `ToolResult.content_items`.

#### 8.4.1 Media Encoding Pattern

All media fields (`image`, `audio`, `video`, `file`) use the pattern:

```typescript
string | { url: string }
```

- **`{ url: string }`** (preferred): A URL reference. In the Robutler stack, this is typically `/api/content/<uuid>` — a signed content URL that the LLM proxy resolves to binary data at call time.
- **`string`** (fallback): Raw base64-encoded data. Accepted but avoided in inter-service transit due to payload size.

#### 8.4.2 ContentItem Types

**TextContent**

```typescript
{ type: 'text'; text: string }
```

**ImageContent**

```
{
  type: 'image';
  image: string | { url: string };  // base64 or URL
  format?: 'jpeg' | 'png' | 'webp' | 'gif';
  detail?: 'low' | 'high' | 'auto';
  alt_text?: string;
  content_id?: string;              // universal content handle (UUID)
}
```

**AudioContent**

```
{
  type: 'audio';
  audio: string | { url: string };  // base64 or URL
  format?: AudioFormat;              // 'pcm16' | 'mp3' | 'wav' | ...
  duration_ms?: number;
  content_id?: string;              // universal content handle (UUID)
}
```

**VideoContent**

```
{
  type: 'video';
  video: string | { url: string };  // base64 or URL
  format?: string;                   // 'mp4' | 'webm'
  duration_ms?: number;
  thumbnail?: string;
  content_id?: string;              // universal content handle (UUID)
}
```

**FileContent**

```
{
  type: 'file';
  file: string | { url: string };  // base64 or URL
  filename: string;                 // required
  mime_type: string;                // required
  size_bytes?: number;
  content_id?: string;             // universal content handle (UUID)
}
```

**ToolCallContent**

```
{
  type: 'tool_call';
  tool_call: {
    id: string;
    name: string;
    arguments: string;  // JSON string
  };
}
```

**ToolResultContent**

```
{
  type: 'tool_result';
  tool_result: {
    call_id: string;
    result: string;            // JSON string
    is_error?: boolean;
    content_items?: ContentItem[];  // multimodal content from tool execution
  };
}
```

The `content_items` on `ToolResult` allows tools to return rich media (e.g., a screenshot tool returning an image alongside text).

#### 8.4.3 Summary Table

| Type | Key Fields | Description |
|---|---|---|
| `text` | `text` | Plain text |
| `image` | `image`, `format?`, `detail?` | Image (URL preferred, base64 fallback) |
| `audio` | `audio`, `format?` | Audio (URL or base64) |
| `video` | `video`, `format?` | Video (URL or base64) |
| `file` | `file`, `filename`, `mime_type` | File attachment |
| `tool_call` | `tool_call.id`, `tool_call.name`, `tool_call.arguments` | Tool invocation |
| `tool_result` | `tool_result.call_id`, `tool_result.result`, `tool_result.content_items?` | Tool response (optionally multimodal) |

### 8.5 Message

Conversation message used in stateless context passing (the `messages` array on `input.text`):

```json
{
  "role": "user",
  "content": "Describe this image",
  "content_items": [
    { "type": "text", "text": "Describe this image" },
    { "type": "image", "image": { "url": "/api/content/550e8400-..." } }
  ],
  "name": "alice"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `role` | string | Yes | `"system"`, `"user"`, `"assistant"`, or `"tool"` |
| `content` | string | No | Text content (simple format) |
| `content_items` | ContentItem[] | No | Multimodal content items. When present, takes precedence for media; `content` carries the text portion for backward compatibility. |
| `name` | string | No | Participant name (multi-user contexts) |
| `tool_call_id` | string | No | For `tool` role: which call this responds to |
| `tool_calls` | ToolCall[] | No | For `assistant` role: tool calls made |

When both `content` and `content_items` are present, `content` is the text-only representation and `content_items` carries the full multimodal payload.

### 8.5.1 Transport Conversion Rules

Content items flow through transports unmodified. Each transport maps UAMP input events to `content_items` on the conversation message:

| UAMP Event | Resulting ContentItem |
|---|---|
| `input.text` | `{ type: 'text', text }` |
| `input.image` | `{ type: 'image', image, format?, detail?, content_id? }` |
| `input.audio` | `{ type: 'audio', audio, format, content_id? }` |
| `input.video` | `{ type: 'video', video, format?, content_id? }` |
| `input.file` | `{ type: 'file', file, filename, mime_type, content_id? }` |

The UAMP transport skill accumulates input events and assembles them into a single message with `content_items` when `response.create` is received. The Completions transport passes `content_items` through on message objects directly.

When `content_id` is present on an input event, it is propagated to the resulting ContentItem. If absent, the agent SDK assigns a new UUID automatically.

### 8.5.4 Content IDs

Media content items (image, audio, video, file) support an optional `content_id` field — a UUID that uniquely identifies the content. This enables cross-agent content referencing, especially for chained delegation scenarios.

**UUID derivation:**

- For `/api/content/<uuid>` URLs: the UUID is extracted from the URL path (reuses existing storage ID).
- For base64 or external URLs: a new random UUID is auto-generated by the agent SDK.

**Content Labels:**

Content producers (media generation tools, LLM skills) return `StructuredToolResult` with structured `content_items` carrying `content_id` and `description` fields. The `present` tool controls what content is displayed to the user. No `/api/content/` URLs are placed in text (text purity rule).

**Delegation:**

The `delegate` tool accepts an `attachments` array of content IDs. It resolves them by scanning conversation messages' `content_items` arrays for matching `content_id` values, with a UUID fallback for backward compatibility.

**Propagation:**

Content IDs propagate from input events through `_buildConversationFromEvents()` and survive cross-agent boundaries via UAMP transport. The conversation is the content registry — no separate in-memory registry is maintained.

`content_id` is optional. Text, tool_call, and tool_result items do not carry content IDs.

### 8.6 Tool Definition

Standard function definition (OpenAI-compatible):

```json
{
  "type": "function",
  "function": {
    "name": "search",
    "description": "Search the web for information",
    "parameters": {
      "type": "object",
      "properties": {
        "query": { "type": "string", "description": "Search query" }
      },
      "required": ["query"]
    }
  }
}
```

### 8.7 Usage Statistics

```json
{
  "input_tokens": 25,
  "output_tokens": 12,
  "total_tokens": 37,
  "cached_tokens": 15,
  "cost": {
    "input_cost": 0.000025,
    "output_cost": 0.000036,
    "total_cost": 0.000061,
    "currency": "USD"
  },
  "audio": {
    "input_seconds": 5.2,
    "output_seconds": 3.1
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `input_tokens` | number | No | Total input/prompt tokens (includes cached tokens for OpenAI/Google/Fireworks; excludes cached for Anthropic) |
| `output_tokens` | number | No | Total output/completion tokens |
| `total_tokens` | number | No | `input_tokens + output_tokens` |
| `cached_tokens` | number | No | Total cached tokens (reads + writes). Present when the provider reports cached token usage. |
| `cost` | object | No | Cost breakdown in USD |
| `audio` | object | No | Audio duration statistics |

**Provider caching behavior:**

| Provider | Caching Type | `input_tokens` semantics | `cached_tokens` includes |
|---|---|---|---|
| Anthropic | Automatic (top-level `cache_control`) | Excludes cached tokens | cache reads + cache writes |
| OpenAI | Automatic (prompts >= 1024 tokens) | Includes cached tokens | cache reads only |
| Google Gemini | Implicit (Gemini 2.5+, Gemini 3) | Includes cached tokens | cache reads only |
| Fireworks | Automatic (needs `x-session-affinity` header) | Includes cached tokens | cache reads only |

Cached tokens are billed at discounted rates (`cacheReadPer1k`, `cacheWritePer1k`) when available. If no cached token rate is configured for a model, cached tokens are billed at the normal `inputPer1k` rate.
```

### 8.8 Session

The session object returned by the server in `session.created`:

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique session identifier |
| `created_at` | number | Unix timestamp (seconds) |
| `config` | SessionConfig | Full session configuration |
| `status` | string | `"active"` or `"closed"` |

## 9. Capabilities

All capability declarations — model, client, and agent — use the **same unified structure**. This enables seamless negotiation between any participants.

```json
{
  "id": "gpt-4o",
  "provider": "openai",
  "modalities": ["text", "image"],
  "supports_streaming": true,
  "supports_thinking": false,
  "supports_caching": false,
  "context_window": 128000,
  "max_output_tokens": 4096,
  "image": {
    "formats": ["jpeg", "png", "gif", "webp"],
    "max_size_bytes": 20971520,
    "detail_levels": ["auto", "low", "high"],
    "max_images_per_request": 20
  },
  "audio": {
    "input_formats": ["pcm16", "wav"],
    "output_formats": ["pcm16", "mp3"],
    "sample_rates": [24000, 48000],
    "supports_realtime": true,
    "voices": ["alloy", "echo", "nova"]
  },
  "file": {
    "supported_mime_types": ["application/pdf", "text/plain"],
    "supports_pdf": true,
    "supports_code": true,
    "supports_structured_data": true
  },
  "tools": {
    "supports_tools": true,
    "supports_parallel_tools": true,
    "supports_streaming_tools": true,
    "max_tools_per_request": 128,
    "built_in_tools": ["web_search"]
  },
  "provides": ["web_search", "chart"],
  "widgets": ["chart", "table", "form"],
  "endpoints": ["/api/search", "/ws/stream"],
  "extensions": {}
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | Yes | Model, client, or agent identifier |
| `provider` | string | Yes | Provider name |
| `modalities` | Modality[] | Yes | Supported content types |
| `supports_streaming` | boolean | Yes | Streaming response support |
| `supports_thinking` | boolean | Yes | Extended thinking support |
| `supports_caching` | boolean | Yes | Context caching support |
| `context_window` | number | No | Maximum input tokens |
| `max_output_tokens` | number | No | Maximum output tokens |
| `image` | ImageCapabilities | No | Detailed image support |
| `audio` | AudioCapabilities | No | Detailed audio support |
| `file` | FileCapabilities | No | Detailed file support |
| `tools` | ToolCapabilities | No | Tool/function calling support |
| `provides` | string[] | No | Capabilities provided (for agents) |
| `widgets` | string[] | No | Available/supported UI widgets |
| `endpoints` | string[] | No | HTTP/WebSocket endpoints (for agents) |
| `extensions` | object | No | Provider- or context-specific extensions |

### 9.1 Capability Negotiation Flow

1. Client sends `session.create` with `client_capabilities`
2. Server responds with `session.created` followed by a `capabilities` event
3. Either side can update capabilities mid-session via `client.capabilities` or `capabilities` events
4. Client can query at any time with `capabilities.query`

### 9.2 Discovery Endpoints

Agents expose capabilities through transport-specific discovery:

| Transport | Discovery Method |
|---|---|
| Completions | `GET /capabilities` |
| A2A | `GET /.well-known/agent.json` → `modelCapabilities` |
| ACP | JSON-RPC `capabilities` method |
| Realtime | `session.created` event → `capabilities` |

## 10. Multiplexed Sessions

Multiplexing allows multiple concurrent sessions over a single transport connection. All events in multiplexed mode include a `session_id` field that scopes the event.

### 10.1 Single vs. Multiplexed Mode

- **Single-session (default):** Omit `session_id`. One connection = one session. Fully backwards-compatible.
- **Multiplexed:** Client sends multiple `session.create` events, each receiving a `session.created` with a unique `session_id`. All subsequent events include `session_id`.

### 10.2 Use Cases

- **Multi-agent daemons** — One daemon hosts N agents. One WebSocket, one session per agent. Each `session.create` includes a per-session `token`.
- **Browser chat UIs** — One WebSocket per user, one session per active chat. Joining a chat = `session.create { chat: "chat_abc" }`. Leaving = `session.end`.
- **Platform routing** — One WebSocket to an agent daemon, interaction-scoped `session_id` per request. The platform sends full conversation context in each `input.text`.

### 10.3 Per-Session Auth and Token Refresh

Each `session.create` can include its own `token` and `payment_token`. To refresh without reconnecting, send `session.update` with the new token:

```json
{ "type": "session.update", "session_id": "sess_1", "token": "new-jwt" }
```

The server responds with `session.updated`.

### 10.4 Example

```
// Create two sessions on one WebSocket
→ { "type": "session.create", "agent": "alice", "token": "jwt-alice", ... }
← { "type": "session.created", "session_id": "sess_1", "agent": "alice", ... }

→ { "type": "session.create", "agent": "bob", "token": "jwt-bob", ... }
← { "type": "session.created", "session_id": "sess_2", "agent": "bob", ... }

// Events scoped by session_id
→ { "type": "input.text", "session_id": "sess_1", "text": "Hi Alice" }
← { "type": "response.delta", "session_id": "sess_1", "delta": { "type": "text", "text": "Hello!" } }

// Token refresh (no reconnect)
→ { "type": "session.update", "session_id": "sess_1", "token": "new-jwt" }
← { "type": "session.updated", "session_id": "sess_1" }
```

## 11. System Events

System events are used for internal agent coordination and are not typically exposed on client-facing connections:

| Event | Description |
|---|---|
| `system.error` | Error during processing |
| `system.stop` | Request to stop current processing |
| `system.cancel` | Cancel and cleanup resources |
| `system.ping` / `system.pong` | Internal keep-alive |
| `system.unroutable` | No handler found for a message |

Implementations use these events for internal lifecycle management. They are distinct from the client-facing `ping`/`pong` and `response.error` events.

## 12. Message Routing (Informative)

UAMP's event-type namespace supports capability-based message routing within an agent implementation. This section describes a recommended pattern; implementations MAY use any internal dispatch mechanism.

In this pattern:

- **Handlers** declare which event types they accept (`subscribes`) and which they emit (`produces`)
- A **router** wires handlers together based on these declarations
- **Observers** can listen to events without consuming them (useful for logging and analytics)
- A **default sink** (`*`) catches unhandled events

| Property | Description |
|---|---|
| `subscribes` | Event types/patterns this handler accepts (string or regex) |
| `produces` | Event types this handler emits |
| `priority` | Higher priority handlers are preferred (default: 0) |

Custom event types (e.g. `analyze_emotion`, `translate.{lang}`) enable domain-specific routing between internal components.

## 13. Transport Mappings

### 13.1 WebSocket

Full duplex, persistent connection. Recommended for real-time and voice.

- Connect: `wss://agent/ws`
- Events: JSON-serialized, one event per WebSocket message
- Keepalive: `ping` / `pong` events
- Multiplexing: Supported (see [Section 10](#10-multiplexed-sessions))

### 13.2 HTTP + SSE

For environments where WebSocket is not available.

- **Client → Server:** `POST /events` with JSON body
- **Server → Client:** `GET /events/stream` with `text/event-stream` response
- Session ID in `X-Session-ID` header

### 13.3 Batch / REST

OpenAI Chat Completions compatible. Request/response pattern for simple use cases and serverless functions.

- `POST /chat/completions` with standard OpenAI format
- Internally converted to `session.create` → `input.text` → `response.create`
- Streaming via SSE when `stream: true`

| Use Case | Recommended Transport |
|---|---|
| Real-time voice | WebSocket |
| Chat with streaming | WebSocket or HTTP+SSE |
| Simple Q&A | Batch/REST |
| Browser without WS | HTTP+SSE |
| Serverless functions | Batch/REST |
| Mobile apps | WebSocket |

## 14. Versioning

### 14.1 Current Version

**UAMP 1.0**

### 14.2 Version Negotiation

Version is exchanged during session creation. The client sends `uamp_version` in `session.create`; the server echoes its supported version in `session.created`. On mismatch, the server sends a `response.error` with code `version_mismatch`.

### 14.3 Compatibility Rules

1. **Minor versions** (1.0 → 1.1) are additive and backward-compatible: new optional fields, new event types, new enum values
2. **Major versions** (1.x → 2.x) indicate breaking changes
3. **Clients** must ignore unknown fields (forward compatibility)
4. **Unknown event types** must be logged but must not cause errors
5. The `extensions` field allows experimentation without version bumps

### 14.4 Known Limitations (v1.0)

- ~33% overhead for audio due to base64 encoding (acceptable for most use cases)
- No automatic session recovery (client must track and replay)
- Service Workers cannot maintain WebSocket (use HTTP+SSE fallback)

## 15. Event Reference

### Client → Server Events

| Event | Description |
|---|---|
| `session.create` | Create new session |
| `session.update` | Update session config / refresh tokens |
| `session.end` | End a session |
| `capabilities.query` | Query server capabilities |
| `client.capabilities` | Announce client capabilities |
| `input.text` | Text input |
| `input.audio` | Audio input |
| `input.image` | Image input |
| `input.video` | Video input |
| `input.file` | File input |
| `input.typing` | Typing indicator |
| `response.create` | Request response |
| `response.cancel` | Cancel response |
| `tool.result` | Provide tool result |
| `payment.submit` | Submit payment |
| `voice.invite` | Initiate voice session |
| `voice.accept` | Accept voice session |
| `voice.decline` | Decline voice session |
| `voice.end` | End voice session |
| `ping` | Connection keepalive |

### Server → Client Events

| Event | Description |
|---|---|
| `session.created` | Session confirmed |
| `session.updated` | Session update confirmed |
| `session.error` | Session-level error |
| `capabilities` | Server capabilities |
| `response.created` | Response started |
| `response.delta` | Streaming content |
| `response.done` | Response complete |
| `response.error` | Response error |
| `response.cancelled` | Response cancelled |
| `tool.call` | Tool execution request |
| `tool.call_done` | Tool call completed |
| `progress` | Progress update |
| `thinking` | Reasoning content |
| `audio.delta` | Streaming audio output |
| `audio.done` | Audio stream complete |
| `transcript.delta` | Real-time transcription |
| `transcript.done` | Transcription complete |
| `usage.delta` | Usage statistics update |
| `rate_limit` | Rate limit notification |
| `presence.typing` | Typing indicator from another user |
| `presence.online` | User/agent came online |
| `presence.offline` | User/agent went offline |
| `message.created` | New chat message |
| `message.read` | Read receipt |
| `payment.required` | Payment required |
| `payment.accepted` | Payment accepted |
| `payment.balance` | Balance update |
| `payment.error` | Payment error |
| `voice.accept` | Voice session accepted |
| `voice.decline` | Voice session declined |
| `voice.end` | Voice session ended |
| `pong` | Keepalive response |

## 16. Further Reading

- [AOAuth](./aoauth.md) — Agent-to-agent authentication protocol
