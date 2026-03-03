# Universal Agentic Message Protocol

<span class="protocol-version">UAMP 1.0</span>

A language-agnostic protocol specification for AI agent communication.

---

## Overview

UAMP provides a unified internal protocol for all agent communication, enabling:

- **Cross-language compatibility** - Same protocol for any programming language
- **Transport agnosticism** - Works over WebSocket, HTTP+SSE, or batch REST
- **Multimodal support** - Text, audio, images, video, and files
- **Provider independence** - No vendor lock-in, works with any LLM backend

## Design Principles

1. **Event-based** - All communication is events (not request/response)
2. **Multimodal native** - Text, audio, images, video, files from day one
3. **Transport agnostic** - Works over any transport layer
4. **Bidirectional** - Client and server events with clear semantics
5. **Session-aware** - Built-in conversation/session management
6. **Provider-agnostic** - No vendor lock-in

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Universal Agentic Message Protocol             │
│         (Event-based, Multimodal, Bidirectional)            │
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

## Protocol Flow

### Basic Chat

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

### With Tool Calls

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

## Compatibility

UAMP is based on OpenAI's Realtime API event structure but transport-independent:

| Protocol | Compatibility |
|----------|--------------|
| OpenAI Realtime | Near 1:1 mapping |
| OpenAI Chat Completions | Via adapter |
| Google A2A | Via adapter |
| ACP | Via adapter |

## Documentation

| Section | Description |
|---------|-------------|
| [Events](events.md) | Client and server event definitions |
| [Types](types.md) | Type definitions and schemas |
| [Capabilities](capabilities.md) | Capability negotiation |
| [Transports](transports.md) | WebSocket, HTTP+SSE, batch mappings |
| [Adapters](adapters.md) | Protocol translation |
| [Versioning](versioning.md) | Protocol version negotiation |
| [Implementations](implementations.md) | SDKs and libraries |

## Quick Reference

### Client Events

| Event | Purpose |
|-------|---------|
| `session.create` | Create new session |
| `session.update` | Update session config |
| `capabilities.query` | Query server capabilities |
| `client.capabilities` | Announce client capabilities |
| `input.text` | Send text input |
| `input.audio` | Send audio input |
| `input.image` | Send image input |
| `response.create` | Request response |
| `response.cancel` | Cancel response |
| `tool.result` | Provide tool result |

### Server Events

| Event | Purpose |
|-------|---------|
| `session.created` | Session confirmed |
| `capabilities` | Server capabilities |
| `response.created` | Response started |
| `response.delta` | Streaming content |
| `response.done` | Response complete |
| `response.error` | Error occurred |
| `tool.call` | Tool execution request |
| `progress` | Progress update |
| `thinking` | Reasoning content |

## License

UAMP is licensed under the MIT License.
