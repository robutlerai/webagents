# Transports

Transports are skills that expose agent communication endpoints for different protocols. They bridge external protocols (OpenAI Completions, A2A, Realtime, ACP) to the agent's internal handoff system.

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Request                           │
│    (HTTP, WebSocket, SSE)                                   │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Transport Skill                           │
│  ┌─────────────┐    ┌────────────┐    ┌──────────────┐     │
│  │ Parse       │ → │ Convert to │ → │ execute_     │     │
│  │ protocol    │    │ internal   │    │ handoff()    │     │
│  └─────────────┘    └────────────┘    └──────────────┘     │
│         ↑                                    │              │
│         │                                    ▼              │
│  ┌─────────────┐                    ┌──────────────┐       │
│  │ Format      │ ← ─ ─ ─ ─ ─ ─ ─ ─ │ LLM Response │       │
│  │ response    │                    │ (streaming)  │       │
│  └─────────────┘                    └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## Available Transports

| Transport | Protocol | Endpoints | Use Case |
|-----------|----------|-----------|----------|
| `CompletionsTransportSkill` | OpenAI API | `POST /chat/completions` | Standard LLM interaction |
| `A2ATransportSkill` | Google A2A | `GET /.well-known/agent.json`, `POST /tasks` | Agent-to-agent communication |
| `RealtimeTransportSkill` | OpenAI Realtime | `WS /realtime` | Voice/audio streaming |
| `ACPTransportSkill` | Agent Client Protocol | `POST /acp`, `WS /acp/stream` | IDE integration |

## Quick Start

```python
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.core.llm.litellm import LiteLLMSkill
from webagents.agents.skills.core.transport import (
    CompletionsTransportSkill,
    A2ATransportSkill,
    RealtimeTransportSkill,
    ACPTransportSkill,
)

agent = BaseAgent(
    name="multi-protocol-agent",
    skills=[
        LiteLLMSkill({"model": "gpt-4o"}),  # LLM provider
        CompletionsTransportSkill(),         # OpenAI-compatible
        A2ATransportSkill(),                 # Google A2A
        RealtimeTransportSkill(),            # Voice/audio
        ACPTransportSkill(),                 # IDE integration
    ]
)
```

## Completions Transport

OpenAI-compatible chat completions with SSE streaming.

### Endpoint

```
POST /agents/{name}/chat/completions
```

### Request

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": true,
  "model": "gpt-4o",
  "temperature": 0.7,
  "max_tokens": 1000,
  "tools": []
}
```

### Response (Streaming)

```
data: {"id":"chatcmpl-...","choices":[{"delta":{"role":"assistant"}}]}

data: {"id":"chatcmpl-...","choices":[{"delta":{"content":"Hello"}}]}

data: {"id":"chatcmpl-...","choices":[{"delta":{"content":"!"}}]}

data: [DONE]
```

---

## A2A Transport (Google Agent2Agent)

Implements the [A2A Protocol](https://google.github.io/A2A/) for agent-to-agent communication.

### Agent Card

```
GET /agents/{name}/.well-known/agent.json
```

Returns agent capabilities for discovery:

```json
{
  "name": "my-agent",
  "description": "A helpful assistant",
  "version": "0.2.1",
  "protocolVersion": "0.2.1",
  "capabilities": {
    "streaming": true,
    "pushNotifications": false
  },
  "defaultInputModes": ["text"],
  "defaultOutputModes": ["text"],
  "skills": [...]
}
```

### Create Task

```
POST /agents/{name}/tasks
```

Request (A2A format):

```json
{
  "message": {
    "role": "user",
    "parts": [
      {"type": "text", "text": "What is the weather?"}
    ]
  }
}
```

Response (SSE streaming):

```
event: task.started
data: {"id":"task-123","status":"running"}

event: task.message
data: {"role":"agent","parts":[{"type":"text","text":"The weather is..."}]}

event: task.completed
data: {"id":"task-123","status":"completed"}
```

### Get Task Status

```
GET /agents/{name}/tasks/{task_id}
```

### Cancel Task

```
DELETE /agents/{name}/tasks/{task_id}
```

---

## Realtime Transport (OpenAI Realtime API)

WebSocket-based real-time communication with audio support.

### Connect

```
WS /agents/{name}/realtime
```

### Session Events

```json
// Sent on connection
{"type": "session.created", "session": {"id": "sess_...", "voice": "alloy"}}

// Update session
{"type": "session.update", "session": {"voice": "nova", "modalities": ["text", "audio"]}}

// Session updated confirmation
{"type": "session.updated", "session": {...}}
```

### Audio Buffer Events

```json
// Append audio (base64 PCM16)
{"type": "input_audio_buffer.append", "audio": "base64..."}

// Commit buffer
{"type": "input_audio_buffer.commit"}

// Clear buffer
{"type": "input_audio_buffer.clear"}
```

### Conversation Events

```json
// Create item
{"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [...]}}

// Delete item
{"type": "conversation.item.delete", "item_id": "item_..."}
```

### Response Events

```json
// Request response
{"type": "response.create"}

// Response streaming
{"type": "response.text.delta", "delta": "Hello"}
{"type": "response.text.done", "text": "Hello world!"}
{"type": "response.done", "response": {"status": "completed"}}

// Cancel response
{"type": "response.cancel"}
```

---

## ACP Transport (Agent Client Protocol)

JSON-RPC 2.0 protocol for IDE integration (Cursor, Zed, JetBrains).

### HTTP Endpoint

```
POST /agents/{name}/acp
```

### WebSocket Endpoint

```
WS /agents/{name}/acp/stream
```

### Initialize

```json
{"jsonrpc": "2.0", "method": "initialize", "params": {}, "id": 1}

// Response
{"jsonrpc": "2.0", "id": 1, "result": {
  "protocolVersion": "1.0",
  "serverInfo": {"name": "my-agent", "version": "2.0.0"},
  "capabilities": {"streaming": true, "tools": true}
}}
```

### Chat/Submit

```json
{"jsonrpc": "2.0", "method": "prompt/submit", "params": {
  "messages": [{"role": "user", "content": "Hello"}]
}, "id": 2}

// Streaming notifications
{"jsonrpc": "2.0", "method": "prompt/started", "params": {"requestId": "2"}}
{"jsonrpc": "2.0", "method": "prompt/progress", "params": {"content": "Hello!", "role": "assistant"}}

// Final response
{"jsonrpc": "2.0", "id": 2, "result": {"status": "complete", "content": "Hello!"}}
```

### Tools

```json
// List tools
{"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 3}

// Call tool
{"jsonrpc": "2.0", "method": "tools/call", "params": {
  "name": "search",
  "arguments": {"query": "weather"}
}, "id": 4}
```

---

## Creating Custom Transports

Use `@http` and `@websocket` decorators with `execute_handoff()`:

```python
from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import http, websocket
from typing import AsyncGenerator

class MyCustomTransport(Skill):
    """Custom protocol transport"""
    
    @http("/my-protocol", method="post")
    async def handle_request(self, messages: list) -> AsyncGenerator[str, None]:
        """SSE streaming endpoint"""
        # Convert to internal format
        internal_messages = self._parse_my_protocol(messages)
        
        # Route through handoff system
        async for chunk in self.execute_handoff(internal_messages):
            # Convert to my protocol format
            yield self._format_my_protocol(chunk)
    
    @websocket("/my-protocol/stream")
    async def handle_websocket(self, ws) -> None:
        """WebSocket endpoint"""
        await ws.accept()
        
        async for message in ws.iter_json():
            # Parse and process
            internal_messages = self._parse_my_protocol(message)
            
            # Stream response
            async for chunk in self.execute_handoff(internal_messages):
                await ws.send_json(self._format_my_protocol(chunk))
```

## Key Methods

### `execute_handoff()`

Route messages through the agent's handoff system:

```python
async for chunk in self.execute_handoff(
    messages=[{"role": "user", "content": "Hello"}],
    tools=None,           # Optional tools
    handoff_name=None,    # Optional specific handoff
):
    # Process streaming chunk
    print(chunk)
```

### SSE Streaming

Return `AsyncGenerator[str, None]` from `@http` handlers for automatic SSE:

```python
@http("/stream", method="post")
async def stream_response(self) -> AsyncGenerator[str, None]:
    yield "data: {\"text\": \"hello\"}\n\n"
    yield "data: {\"text\": \"world\"}\n\n"
```

### WebSocket Handlers

Use `@websocket` for bidirectional communication:

```python
@websocket("/chat")
async def chat(self, ws) -> None:
    await ws.accept()
    async for msg in ws.iter_json():
        await ws.send_json({"response": msg})
```

## See Also

- **[Handoffs](handoffs.md)** — LLM routing
- **[Endpoints](endpoints.md)** — HTTP API basics
- **[Skills](skills.md)** — Skill development
