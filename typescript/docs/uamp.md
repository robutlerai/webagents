# UAMP Protocol

UAMP (Universal Agentic Message Protocol) is an event-based protocol for agent communication. It provides a standardized way for clients and agents to exchange messages, regardless of the underlying LLM provider.

## Overview

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
- `type` - Event type identifier
- `event_id` - Unique UUID
- `timestamp` - Unix milliseconds (optional)

### Sessions

Sessions maintain conversation state and configuration:
- Modalities (text, audio, image, video, file)
- System instructions
- Available tools
- Voice settings (for audio)

### Capabilities

Both clients and agents can announce capabilities:
- Supported modalities
- Streaming support
- Tool support
- Provider-specific features

## Event Types

### Client → Server Events

#### session.create

Create a new session.

```typescript
{
  type: 'session.create',
  event_id: 'uuid',
  uamp_version: '1.0',
  session: {
    modalities: ['text'],
    instructions: 'You are a helpful assistant.',
    tools: [
      {
        type: 'function',
        function: {
          name: 'search',
          description: 'Search the web',
          parameters: {
            type: 'object',
            properties: {
              query: { type: 'string' }
            },
            required: ['query']
          }
        }
      }
    ]
  },
  client_capabilities: {
    id: 'my-client',
    provider: 'webagents',
    modalities: ['text'],
    supports_streaming: true,
    supports_thinking: false,
    supports_caching: false
  }
}
```

#### input.text

Send text input.

```typescript
{
  type: 'input.text',
  event_id: 'uuid',
  text: 'Hello, how are you?',
  role: 'user'  // 'user' or 'system'
}
```

#### input.audio

Send audio input (for voice conversations).

```typescript
{
  type: 'input.audio',
  event_id: 'uuid',
  audio: 'base64-encoded-audio',
  format: 'pcm16',
  is_final: true
}
```

#### input.image

Send image input.

```typescript
{
  type: 'input.image',
  event_id: 'uuid',
  image: 'base64-data-or-url',
  format: 'jpeg',
  detail: 'auto'
}
```

#### response.create

Request a response from the agent.

```typescript
{
  type: 'response.create',
  event_id: 'uuid',
  response: {
    modalities: ['text'],  // Optional override
    instructions: 'Be concise.'  // Optional override
  }
}
```

#### response.cancel

Cancel an in-progress response.

```typescript
{
  type: 'response.cancel',
  event_id: 'uuid',
  response_id: 'optional-response-id'
}
```

#### tool.result

Return tool execution result.

```typescript
{
  type: 'tool.result',
  event_id: 'uuid',
  call_id: 'tool-call-id',
  result: '{"data": "result"}',  // JSON string
  is_error: false
}
```

### Server → Client Events

#### session.created

Confirm session creation.

```typescript
{
  type: 'session.created',
  event_id: 'uuid',
  uamp_version: '1.0',
  session: {
    id: 'session-uuid',
    modalities: ['text'],
    instructions: '...',
    tools: [...]
  }
}
```

#### response.delta

Stream response content.

```typescript
// Text delta
{
  type: 'response.delta',
  event_id: 'uuid',
  response_id: 'response-uuid',
  delta: {
    type: 'text',
    text: 'Hello'
  }
}

// Tool call delta
{
  type: 'response.delta',
  event_id: 'uuid',
  response_id: 'response-uuid',
  delta: {
    type: 'tool_call',
    tool_call: {
      id: 'call-uuid',
      name: 'search',
      arguments: '{"query":'  // Streamed
    }
  }
}
```

#### response.done

Complete response.

```typescript
{
  type: 'response.done',
  event_id: 'uuid',
  response_id: 'response-uuid',
  response: {
    id: 'response-uuid',
    status: 'completed',  // 'completed' | 'cancelled' | 'failed'
    output: [
      { type: 'text', text: 'Hello! How can I help?' }
    ],
    usage: {
      input_tokens: 10,
      output_tokens: 8,
      total_tokens: 18
    }
  }
}
```

#### response.error

Report an error.

```typescript
{
  type: 'response.error',
  event_id: 'uuid',
  response_id: 'optional-response-uuid',
  error: {
    code: 'rate_limit_exceeded',
    message: 'Too many requests',
    details: { retry_after: 60 }
  }
}
```

#### tool.call

Request tool execution from client.

```typescript
{
  type: 'tool.call',
  event_id: 'uuid',
  call_id: 'call-uuid',
  name: 'search',
  arguments: '{"query": "AI news"}'
}
```

#### progress

Progress update for long operations.

```typescript
{
  type: 'progress',
  event_id: 'uuid',
  target: 'tool',  // 'tool' | 'response' | 'upload' | 'reasoning'
  target_id: 'call-uuid',
  message: 'Searching...',
  percent: 50
}
```

#### thinking

Reasoning/thinking content (for models that support it).

```typescript
{
  type: 'thinking',
  event_id: 'uuid',
  content: 'Let me analyze this step by step...',
  stage: 'analysis',
  is_delta: true  // true = append, false = complete
}
```

## Content Items

Content items represent different types of content in messages:

```typescript
// Text
{ type: 'text', text: 'Hello' }

// Image (URL or base64 data URI)
{ type: 'image', image: { url: '/api/content/abc-123' }, content_id: 'abc-123' }
{ type: 'image', image: 'data:image/png;base64,...', content_id: 'def-456' }

// Video
{ type: 'video', video: { url: '/api/content/vid-789' }, content_id: 'vid-789' }

// Audio
{ type: 'audio', audio: { url: '/api/content/aud-012' }, content_id: 'aud-012' }

// File
{ type: 'file', file: { url: '/api/content/doc-345' }, filename: 'doc.pdf', mime_type: 'application/pdf', content_id: 'doc-345' }

// Tool call
{ type: 'tool_call', tool_call: { id: 'call-1', name: 'search', arguments: '{}' } }

// Tool result
{ type: 'tool_result', tool_result: { call_id: 'call-1', result: '{}' } }

// Thinking (reasoning)
{ type: 'thinking', text: 'Let me think...' }
```

### Content IDs and Delegation

Media content items support `content_id` (UUID) for cross-agent referencing. Content producers (tools, LLM skills) compose `[content:UUID]` labels in their result text. The delegate tool resolves attachments from conversation `content_items` by matching `content_id`.

```typescript
// Tool returns StructuredToolResult with content_items and labels
{
  text: 'Generated 1 image(s). [content:abc-123]\n{"_billing":{...}}',
  content_items: [
    { type: 'image', image: { url: '/api/content/abc-123' }, content_id: 'abc-123' }
  ]
}

// Delegate forwards attachment by content_id
delegate({ agent: 'editor', message: 'make it green', attachments: ['abc-123'] })
```

## Capabilities

Capabilities describe what a client or agent supports:

```typescript
interface Capabilities {
  // Required
  id: string;                    // Unique identifier
  provider: string;              // Provider name
  modalities: Modality[];        // ['text', 'audio', 'image', ...]
  supports_streaming: boolean;   // Stream responses
  supports_thinking: boolean;    // Extended thinking
  supports_caching: boolean;     // Prompt caching

  // Optional
  image?: {
    supported_formats: string[];
    max_resolution: string;
    max_file_size_mb: number;
    supports_url: boolean;
    supports_base64: boolean;
  };

  audio?: {
    supported_input_formats: string[];
    supported_output_formats: string[];
    sample_rates: number[];
    supports_voice_activity: boolean;
  };

  tools?: {
    supports_tools: boolean;
    supports_parallel_tools: boolean;
    supports_streaming_tools: boolean;
    built_in_tools: string[];
    max_tools: number;
  };

  provides?: string[];    // What this agent/skill provides
  endpoints?: string[];   // Available endpoints
  extensions?: Record<string, unknown>;  // Provider-specific
}
```

## Usage in TypeScript

### Creating Events

```typescript
import {
  createSessionCreateEvent,
  createInputTextEvent,
  createResponseCreateEvent,
  createToolResultEvent,
  createResponseDeltaEvent,
  createResponseDoneEvent,
  createResponseErrorEvent,
  createProgressEvent,
} from 'webagents/uamp';

// Create session
const sessionEvent = createSessionCreateEvent({
  modalities: ['text'],
  instructions: 'Be helpful',
});

// Send text
const inputEvent = createInputTextEvent('Hello!', 'user');

// Request response
const requestEvent = createResponseCreateEvent();

// Tool result
const resultEvent = createToolResultEvent(
  'call-123',
  { data: 'result' },
  false  // is_error
);
```

### Processing Events

```typescript
import { parseEvent, serializeEvent, isClientEvent, isServerEvent } from 'webagents/uamp';

// Parse from JSON
const event = parseEvent('{"type": "input.text", ...}');

// Serialize to JSON
const json = serializeEvent(event);

// Check event direction
if (isClientEvent(event)) {
  // Handle client event
}
if (isServerEvent(event)) {
  // Handle server event
}
```

### Agent Processing

```typescript
import { BaseAgent } from 'webagents';

const agent = new BaseAgent({ ... });

// Process UAMP events
const clientEvents = [
  createSessionCreateEvent({ modalities: ['text'] }),
  createInputTextEvent('Hello'),
  createResponseCreateEvent(),
];

for await (const serverEvent of agent.processUAMP(clientEvents)) {
  console.log(serverEvent.type, serverEvent);
}
```

## Transport Adapters

UAMP is the internal protocol. Transport skills adapt external protocols:

### OpenAI Completions → UAMP

```typescript
// Incoming OpenAI request
{
  model: 'gpt-4',
  messages: [
    { role: 'user', content: 'Hello' }
  ]
}

// Converted to UAMP events
[
  { type: 'session.create', ... },
  { type: 'input.text', text: 'Hello', role: 'user' },
  { type: 'response.create' }
]

// UAMP response events
[
  { type: 'response.delta', delta: { type: 'text', text: 'Hi' } },
  { type: 'response.done', response: { output: [...], usage: {...} } }
]

// Converted back to OpenAI format
{
  id: 'chatcmpl-...',
  choices: [{ message: { role: 'assistant', content: 'Hi' } }],
  usage: { ... }
}
```

## Message Routing

UAMP supports capability-based message routing through a central router.

### Routing Model

- **Handlers** declare `subscribes` (input types/patterns) and `produces` (output types)
- **Router** auto-wires handlers based on capabilities
- **Observers** can listen without consuming (for logging, analytics)
- **Default sink** (`*`) catches unhandled messages (typically transport reply)

### Handler Declaration

Handlers declare their capabilities:

| Property | Description |
|----------|-------------|
| `subscribes` | Event types/patterns this handler accepts (string or regex) |
| `produces` | Event types this handler emits |
| `priority` | Higher priority handlers are preferred (default: 0) |

Example:

```typescript
@handoff({
  name: 'speech-to-text',
  subscribes: ['input.audio'],
  produces: ['input.text'],
  priority: 100,
})
async *processAudio(events) { ... }
```

### System Events

| Event | Description |
|-------|-------------|
| `system.error` | Error occurred during processing |
| `system.stop` | Request to stop current processing |
| `system.cancel` | Cancel and cleanup resources |
| `system.ping` | Keep-alive request |
| `system.pong` | Keep-alive response |
| `system.unroutable` | No handler found for message |

### Custom Event Types

Agents can define custom event types for specialized routing:

- `analyze_emotion` - Route to emotion analysis handler
- `translate.{lang}` - Route to translation handler (regex support)
- Custom types enable NLI skills to expose capabilities

See [Router Guide](./router.md) for detailed documentation.

## Version

Current UAMP version: `1.0`

The version is exchanged in `session.create` and `session.created` events to ensure compatibility.

## Further Reading

- [UAMP Specification](https://uamp.dev) - Full protocol spec
- [Router Guide](./router.md) - Message routing and capabilities
- [Skills Guide](./skills.md) - How to use UAMP in skills
- [API Reference](./api.md) - TypeScript API details
