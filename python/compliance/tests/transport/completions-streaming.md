---
name: completions-streaming
version: 1.0
transport: completions
tags: [core, streaming]
---

# Streaming Chat Completion

Tests Server-Sent Events (SSE) streaming for the `/chat/completions` endpoint.

## Setup

Create an agent with the following configuration:
- Name: `verbose-agent`
- Instructions: "Provide detailed, multi-sentence responses to demonstrate streaming."

## Test Cases

### 1. Basic Streaming

**Request:**
POST `/chat/completions`
```json
{
  "model": "verbose-agent",
  "messages": [{"role": "user", "content": "Tell me a short story"}],
  "stream": true
}
```

**Assertions:**
- Response content-type is text/event-stream
- Response consists of SSE events
- Each data line contains valid JSON
- Multiple chunks are received before completion
- Final chunk has finish_reason
- Stream ends with [DONE] marker

**Strict:**
```yaml
format: sse
content_type: text/event-stream
chunks:
  - object: chat.completion.chunk
  - choices[0].delta: exists
final_event: "[DONE]"
```

### 2. Chunk Format

**Request:**
POST `/chat/completions`
```json
{
  "model": "verbose-agent",
  "messages": [{"role": "user", "content": "Count from 1 to 5"}],
  "stream": true
}
```

**Assertions:**
- Each chunk has `id` field
- Each chunk has `object` field set to "chat.completion.chunk"
- Each chunk has `choices` array
- Delta objects contain `content` or `role`
- First chunk may contain role, subsequent chunks contain content

**Strict:**
```yaml
chunks:
  - id: type(string)
  - object: chat.completion.chunk
  - choices: length(1)
  - choices[0].index: 0
  - choices[0].delta: exists
```

### 3. Streaming with Tool Calls

**Request:**
POST `/chat/completions`
```json
{
  "model": "tool-agent",
  "messages": [{"role": "user", "content": "What's the weather in NYC?"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          },
          "required": ["location"]
        }
      }
    }
  ],
  "stream": true
}
```

**Assertions:**
- Streaming includes tool call chunks
- Tool call is built incrementally across chunks
- Tool call has id, name, and arguments
- Final chunk has finish_reason of "tool_calls"

**Strict:**
```yaml
format: sse
chunks:
  - choices[0].delta.tool_calls: exists
final_chunk:
  choices[0].finish_reason: tool_calls
```

### 4. Concatenated Content

**Request:**
POST `/chat/completions`
```json
{
  "model": "verbose-agent",
  "messages": [{"role": "user", "content": "Say exactly: 'Hello World'"}],
  "stream": true
}
```

**Assertions:**
- Concatenating all delta.content produces complete message
- No content is lost between chunks
- Final message is coherent

**Strict:**
```yaml
concatenated_content: contains("Hello")
```

### 5. Stream Cancellation

**Request:**
POST `/chat/completions`
```json
{
  "model": "verbose-agent",
  "messages": [{"role": "user", "content": "Write a very long essay about history"}],
  "stream": true
}
```

**Behavior:**
Client disconnects after receiving 3 chunks.

**Assertions:**
- Server handles disconnection gracefully
- No server-side errors
- Resources are cleaned up

### 6. Empty Response Streaming

**Request:**
POST `/chat/completions`
```json
{
  "model": "minimal-agent",
  "messages": [{"role": "user", "content": ""}],
  "stream": true
}
```

**Assertions:**
- Stream completes even if content is minimal
- [DONE] marker is sent
- finish_reason is set

**Strict:**
```yaml
format: sse
final_event: "[DONE]"
```
