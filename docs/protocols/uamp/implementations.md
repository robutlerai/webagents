# Implementations

UAMP is designed to be language-agnostic. This page lists known implementations.

## Roborum (Unified UAMP Server)

Roborum is a backend that speaks UAMP over raw WebSocket for all real-time communication.

- **UAMP WS server** at `/ws` — Browsers and agent daemons connect here. Single protocol, no Socket.IO.
- **Session multiplexing** — One WS per client, multiple sessions per connection (one per chat for browsers, one per agent for daemons). Per-session auth/payment tokens; `session.update` for token refresh without reconnect.
- **UAMP WS client** — Roborum opens outbound WS connections to external agents when no inbound session exists. Transport discovery via `capabilities.query` or `/.well-known/agent.json`; fallback to HTTP completions.
- **Fan-out** — Subscriber map + Redis pub/sub for multi-instance scaling.

## Official SDKs

| Language | Repository | Status |
|----------|------------|--------|
| Python | [webagents-python](https://github.com/robutlerai/webagents-python) | Reference |
| TypeScript | [webagents-ts](https://github.com/robutlerai/webagents-ts) | In Progress |

## Reference Implementation

The Python SDK ([webagents](https://github.com/robutlerai/webagents)) serves as the reference implementation for UAMP.

### Features

- Full UAMP event support
- Transport adapters (Completions, Realtime, A2A, ACP)
- LLM adapters (OpenAI, Anthropic, Google, xAI)
- Capability negotiation
- Streaming support

### Documentation

- [WebAgents Documentation](https://robutler.ai/webagents/)
- [UAMP Implementation Guide](https://robutler.ai/webagents/agent/capabilities/)

## Implementing UAMP

To implement UAMP in a new language:

### 1. Event Serialization

Implement JSON serialization for all [event types](events.md):

```
BaseEvent
├── Session Events (session.create, session.created, ...)
├── Input Events (input.text, input.audio, input.image, ...)
├── Response Events (response.create, response.delta, response.done, ...)
├── Tool Events (tool.call, tool.result, ...)
└── Utility Events (ping, pong, rate_limit, ...)
```

### 2. Type Definitions

Implement all [type definitions](types.md):

```
Types
├── Modality, AudioFormat
├── SessionConfig, Session
├── VoiceConfig, TurnDetectionConfig
├── ToolDefinition, ContentItem
├── UsageStats, Capabilities
└── ImageCapabilities, AudioCapabilities, FileCapabilities, ToolCapabilities
```

### 3. Transport Layer

Implement at least one [transport](transports.md):

- **WebSocket** - Full duplex, recommended for real-time
- **HTTP + SSE** - Server-sent events for streaming
- **Batch/REST** - Request/response for simple use cases

### 4. Adapters

Implement adapters for your target protocols and LLM providers:

- **Transport Adapters** - External protocol → UAMP
- **LLM Adapters** - UAMP → Provider API

See [Adapters](adapters.md) for the adapter architecture.

## Compliance Testing

To verify UAMP compliance:

### Event Compliance

1. All event types must serialize to valid JSON
2. Required fields must be present
3. Event IDs must be unique
4. Timestamps must be valid Unix milliseconds

### Transport Compliance

1. WebSocket must support bidirectional events
2. SSE must properly format `data:` lines
3. Batch mode must handle non-streaming correctly

### Capability Compliance

1. `capabilities.query` must return valid capabilities
2. `client.capabilities` must be accepted
3. Unknown fields must be ignored (forward compatibility)

## Contributing

To add your implementation:

1. Ensure it passes compliance tests
2. Submit a PR to update this page
3. Include link to repository and documentation
