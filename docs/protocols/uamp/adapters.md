# Adapters

Adapters translate between external protocols and UAMP. They are the bridge between transport-specific formats and the unified UAMP event model.

## Adapter Architecture

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

## Adapter Types

### Transport Adapters (Inbound)

Convert external protocol requests to UAMP events:

- **Completions Adapter** - OpenAI Chat Completions API → UAMP
- **Realtime Adapter** - OpenAI Realtime API → UAMP (near 1:1)
- **A2A Adapter** - Google Agent-to-Agent Protocol → UAMP
- **ACP Adapter** - Agent Communication Protocol → UAMP

### LLM Adapters (Outbound)

Convert UAMP events to LLM provider formats:

- **OpenAI Adapter** - UAMP → OpenAI API
- **Anthropic Adapter** - UAMP → Anthropic Messages API
- **Google Adapter** - UAMP → Google Gemini API
- **LiteLLM Adapter** - UAMP → LiteLLM unified format

## Transport Adapter Flow

### Completions to UAMP

Input (OpenAI Chat Completions format):
```json
{
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
  ],
  "tools": [...],
  "stream": true
}
```

Converts to UAMP events:
```json
[
  { "type": "session.create", "session": { "modalities": ["text"], "tools": [...] } },
  { "type": "input.text", "text": "You are helpful.", "role": "system" },
  { "type": "input.text", "text": "Hello!", "role": "user" },
  { "type": "response.create" }
]
```

### UAMP to Completions

UAMP events:
```json
{ "type": "response.delta", "delta": { "type": "text", "text": "Hi" } }
{ "type": "response.delta", "delta": { "type": "text", "text": " there!" } }
{ "type": "response.done", "response": { "status": "completed" } }
```

Convert to SSE stream:
```
data: {"choices":[{"delta":{"content":"Hi"}}]}

data: {"choices":[{"delta":{"content":" there!"}}]}

data: [DONE]
```

### Realtime Adapter

Near 1:1 mapping since UAMP is based on the Realtime API:

| Realtime Event | UAMP Event |
|----------------|------------|
| `session.create` | `session.create` |
| `input_audio_buffer.append` | `input.audio` |
| `response.text.delta` | `response.delta` |
| `response.done` | `response.done` |

### A2A Adapter

Google Agent-to-Agent protocol with task state management:

| A2A Concept | UAMP Mapping |
|-------------|--------------|
| Task creation | `session.create` + `input.text` |
| Task message | `input.text` / `input.file` |
| Task artifact | `response.done` → output |
| Task status | Managed by adapter, not UAMP |

The adapter manages task IDs and status transitions outside UAMP.

### ACP Adapter

Agent Communication Protocol (JSON-RPC based):

| ACP Method | UAMP Events |
|------------|-------------|
| `session/create` | `session.create` |
| `session/prompt` | `input.text` + `response.create` |
| `session/update` notification | `response.delta` |

## LLM Adapter Flow

### UAMP to OpenAI

UAMP events:
```json
[
  { "type": "session.create", "session": { "modalities": ["text"], "tools": [...] } },
  { "type": "input.text", "text": "System prompt", "role": "system" },
  { "type": "input.text", "text": "User message", "role": "user" },
  { "type": "response.create" }
]
```

Convert to OpenAI API call:
```json
{
  "model": "gpt-4o",
  "messages": [
    {"role": "system", "content": "System prompt"},
    {"role": "user", "content": "User message"}
  ],
  "tools": [...],
  "stream": true
}
```

### OpenAI to UAMP

OpenAI streaming chunk:
```json
{
  "choices": [{
    "delta": { "content": "Hello" }
  }]
}
```

Convert to UAMP event:
```json
{
  "type": "response.delta",
  "event_id": "evt_abc123",
  "delta": { "type": "text", "text": "Hello" }
}
```

### UAMP to Anthropic

Anthropic uses a different message format:

| UAMP | Anthropic |
|------|-----------|
| `input.text` (role=system) | `system` parameter |
| `input.text` (role=user) | `messages[].role="user"` |
| `input.image` | `messages[].content[].type="image"` |
| `response.delta` | `content_block_delta` event |

## Capability Discovery

Transport adapters expose UAMP capabilities through protocol-specific discovery:

| Transport | Discovery Endpoint | Capabilities Field |
|-----------|-------------------|-------------------|
| **Completions** | `GET /capabilities` | Root object |
| **A2A** | `GET /.well-known/agent.json` | `modelCapabilities` |
| **ACP** | JSON-RPC `capabilities` | `result.modelCapabilities` |
| **Realtime** | `session.created` event | `capabilities` |

## Adapter Guidelines

### Design Principles

1. **Keep adapters thin** - Only format conversion, no business logic
2. **Handle missing fields** - Use sensible defaults for optional fields
3. **Preserve semantics** - Don't lose information during conversion
4. **Log conversion errors** - But don't fail silently
5. **Test round-trips** - Ensure `fromUAMP(toUAMP(x)) ≈ x`

### Error Handling

- Invalid input → Return UAMP `response.error` event
- Provider error → Map to appropriate UAMP error code
- Timeout → Return error with `code: "timeout"`

### Forward Compatibility

- Ignore unknown fields in input
- Pass through unknown fields in `extensions`
- Don't fail on new event types

## Adapter Interfaces

Adapters should implement these conceptual interfaces:

### Transport Adapter

```
interface TransportAdapter {
  // Convert transport request to UAMP events
  toUAMP(request: TransportRequest): ClientEvent[]
  
  // Convert UAMP events to transport response
  fromUAMP(events: ServerEvent[]): TransportResponse
  
  // Convert streaming UAMP event to transport format
  fromUAMPStreaming(event: ServerEvent): TransportChunk | null
}
```

### LLM Adapter

```
interface LLMAdapter {
  // Convert UAMP events to provider API request
  toProvider(events: ClientEvent[]): ProviderRequest
  
  // Convert provider response to UAMP events
  fromProvider(response: ProviderResponse): ServerEvent[]
  
  // Convert streaming chunk to UAMP event
  fromProviderStreaming(chunk: ProviderChunk): ServerEvent
}
```
