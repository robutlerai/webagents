# UAMP Transport Mappings

UAMP events can flow over multiple transport layers.

## Transport Architecture

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
│               │   │  POST Client→   │   │                   │
└───────────────┘   └─────────────────┘   └───────────────────┘
```

## WebSocket Transport

Full duplex communication over persistent connection.

### Client → Server

```javascript
ws.send(JSON.stringify({
  type: 'input.text',
  event_id: 'evt_abc123',
  text: 'Hello, world!'
}));
```

### Server → Client

```javascript
ws.onmessage = (event) => {
  const uampEvent = JSON.parse(event.data);
  handleEvent(uampEvent);
};
```

### Connection Lifecycle

1. Client connects: `new WebSocket('wss://agent/ws')`
2. Client sends `session.create`
3. Server responds with `session.created`
4. Bidirectional event flow
5. Either side can close

### Keepalive

Use `ping`/`pong` events for application-level keepalive:

```javascript
// Client sends
{ type: 'ping', event_id: 'ping_123' }

// Server responds
{ type: 'pong', event_id: 'pong_456' }
```

### Multiplexed Sessions

WebSocket supports multiplexed sessions — multiple concurrent sessions over a single connection. See [Events: Multiplexed Sessions](events.md#multiplexed-sessions) for the full protocol-level specification including interaction-scoped sessions, per-session auth, and token refresh.

## HTTP + SSE Transport

For environments where WebSocket is not available.

### Client → Server (POST)

```http
POST /events HTTP/1.1
Content-Type: application/json

{
  "type": "input.text",
  "event_id": "evt_abc123",
  "text": "Hello, world!"
}
```

### Server → Client (SSE)

```http
GET /events/stream HTTP/1.1
Accept: text/event-stream

---

data: {"type":"response.delta","event_id":"evt_1","delta":{"type":"text","text":"Hello"}}

data: {"type":"response.delta","event_id":"evt_2","delta":{"type":"text","text":" there"}}

data: {"type":"response.done","event_id":"evt_3","response":{"status":"completed"}}
```

### Session Management

Include session ID in headers:

```http
POST /events HTTP/1.1
X-Session-ID: sess_abc123
```

## Batch/REST Transport

For simple request/response patterns (OpenAI Chat Completions compatible).

### Request

Maps to UAMP events internally:

```http
POST /chat/completions HTTP/1.1
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "stream": false
}
```

Internally becomes:

```typescript
[
  { type: 'session.create', session: { modalities: ['text'] } },
  { type: 'input.text', text: 'Hello', role: 'user' },
  { type: 'response.create' }
]
```

### Non-Streaming Response

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help?"
    }
  }],
  "usage": {
    "input_tokens": 5,
    "output_tokens": 8,
    "total_tokens": 13
  }
}
```

### Streaming Response

```http
POST /chat/completions HTTP/1.1
Content-Type: application/json

{"messages": [...], "stream": true}
```

Response (SSE):

```
data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"content":"!"}}]}

data: [DONE]
```

## Transport Selection Guidelines

| Use Case | Recommended Transport |
|----------|----------------------|
| Real-time voice | WebSocket |
| Chat with streaming | WebSocket or HTTP+SSE |
| Simple Q&A | Batch/REST |
| Browser without WS | HTTP+SSE |
| Serverless functions | Batch/REST |
| Mobile apps | WebSocket |

## Error Handling

### WebSocket Errors

```javascript
ws.onerror = (error) => {
  // Reconnect with exponential backoff
};

ws.onclose = (event) => {
  if (event.code !== 1000) {
    // Abnormal close, reconnect
  }
};
```

### HTTP Errors

| Status | Meaning |
|--------|---------|
| 400 | Invalid event format |
| 401 | Authentication required |
| 429 | Rate limited |
| 500 | Server error |

### UAMP Error Events

Errors within the protocol:

```typescript
{
  type: 'response.error',
  error: {
    code: 'invalid_request',
    message: 'Missing required field: text'
  }
}
```
