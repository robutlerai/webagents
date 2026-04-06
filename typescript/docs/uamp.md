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

## Structured Content Protocol

The Structured Content Protocol extends UAMP with rich media handling, LLM-controlled display, and content lifecycle management.

### Content Types

`HtmlContent` joins the existing content type family. The `html` field accepts either an inline HTML string or a URL pointing to hosted HTML (URL form preferred for large documents).

```typescript
interface HtmlContent {
  type: 'html';
  html: string;           // Inline HTML string or URL
  title?: string;
  sandbox?: boolean;      // Default: true
  dimensions?: { width: number; height: number };
  content_id: string;
  description?: string;
  display_hint?: 'inline' | 'attachment' | 'sandbox';
}
```

The `ContentItem` union type now includes `HtmlContent`:

```typescript
type ContentItem =
  | TextContent
  | ImageContent
  | VideoContent
  | AudioContent
  | FileContent
  | HtmlContent
  | ToolCallContent
  | ToolResultContent
  | ThinkingContent;
```

### Content Item Extensions

All media content types (`ImageContent`, `AudioContent`, `VideoContent`, `FileContent`, `HtmlContent`) gained two optional fields:

| Field | Type | Set By | Purpose |
|-------|------|--------|---------|
| `description` | `string` | Producing tool | Human-readable summary of the content |
| `display_hint` | `'inline' \| 'attachment' \| 'sandbox'` | `present` tool only | Controls how the client renders the item |

`content_id` (UUID) is required on all stored content items. Producers must assign it at creation time.

### `present` Tool

Built-in tool that gives the LLM explicit control over when and how content is displayed to the user.

**Schema:**

```typescript
{
  type: 'function',
  function: {
    name: 'present',
    description: 'Display a content item to the user',
    parameters: {
      type: 'object',
      properties: {
        content_id:  { type: 'string', description: 'ID of the content item to present' },
        display_as:  { type: 'string', enum: ['inline', 'attachment', 'sandbox'] },
        caption:     { type: 'string', description: 'Optional caption for the presented content' }
      },
      required: ['content_id']
    }
  }
}
```

**Capability gate:** Only injected when the client declares `supports_rich_display: true` in `client_capabilities`.

**Handler behavior:**

1. Looks up `content_id` in the conversation's `collectedContentItems` map.
2. Sets `display_hint` on the matched item (defaults to `inline` if `display_as` omitted).
3. Emits an immediate `response.delta` with the content item so the client can render it.
4. Returns a rich confirmation to the LLM:

```typescript
{
  text: 'Presented image (1024×768): A sunset over the ocean',
  content_items: [{ type: 'image', content_id: 'abc-123', /* ... */ }]
}
```

The confirmation includes `type`, `dimensions`, and `description` so the LLM knows what was shown.

**Error handling:** If `content_id` is not found, returns available IDs and a cross-reference to `save_content`:

```typescript
{
  text: 'Content not found: xyz-999. Available: [abc-123, def-456]. Use save_content to persist external media first.',
  is_error: true
}
```

**Re-present semantics:** Calling `present` on an already-presented item updates its `display_hint` and re-emits the delta — useful for showing edited content after a tool modifies it.

**No safety net:** Browser clients receive only explicitly presented items in `response.done` output. There is no fallback promotion for unpresented content.

### `save_content` Tool

Persists external content (URL or base64) to the user's media library, making it available as a `ContentItem` for downstream tools and `present`.

**Schema:**

```typescript
{
  type: 'function',
  function: {
    name: 'save_content',
    parameters: {
      type: 'object',
      properties: {
        url:         { type: 'string', description: 'URL of the content to save' },
        base64:      { type: 'string', description: 'Base64-encoded content' },
        mime_type:   { type: 'string' },
        description: { type: 'string', description: 'What this content depicts or contains' },
        filename:    { type: 'string' }
      },
      required: ['description']
    }
  }
}
```

**Capability gate:** Requires `StoreMediaSkill` on the agent (independent of `supports_rich_display`).

**Handler behavior:**

1. Validates that at least one of `url` or `base64` is provided.
2. Emits `tool_progress` events during download/upload with typed status (see [Typed Progress Events](#typed-progress-events)).
3. Returns a `StructuredToolResult` with the persisted `content_items`.

**Error handling:**

| Condition | Response |
|-----------|----------|
| Neither `url` nor `base64` provided | Error: "Provide either url or base64" |
| Download failure (HTTP error, DNS) | Error with status code and URL |
| Invalid base64 encoding | Error: "Invalid base64 data" |
| Download timeout (30s default) | Error: "Download timed out" |

### Typed Progress Events

The `tool_progress` delta type in `ContentDelta` gains structured fields for media-aware progress reporting:

```typescript
{
  type: 'response.delta',
  delta: {
    type: 'tool_progress',
    call_id: 'call-uuid',
    message: 'Generating image...',
    media_type: 'image',
    status: 'generating',           // See status values below
    progress_percent: 45,
    estimated_duration_ms: 8000,
    dimensions: { width: 1024, height: 768 },
    thumbnail_url: '/api/content/thumb-abc'
  }
}
```

**Status values:** `queued` | `generating` | `processing` | `uploading` | `downloading` | `complete` | `failed`

Clients use `media_type` and `status` to render typed skeleton placeholders (e.g., an image-shaped skeleton with a progress bar at 45%).

### Text Purity Rule

Internal content URLs (e.g., `/api/content/abc-123`) are prohibited in text output. The LLM must never embed platform URLs in prose — clients won't render them as media.

- **External URLs** (e.g., `https://example.com/article`) are allowed in text.
- **Media references** must use `StructuredToolResult` with `content_items`. Tools return `[content:UUID]` labels for LLM context, but these are stripped before display.

### Content Edits and Updates

The `content_updated` field on `ContentDelta` enables in-place content mutation:

```typescript
{
  type: 'response.delta',
  delta: {
    type: 'content_updated',
    content_id: 'abc-123',
    command: 'replace',       // 'replace' | 'patch'
    diff: '--- old\n+++ new\n@@ ...',  // Optional, for 'patch' command
    timestamp: 1719500000000
  }
}
```

Clients update already-rendered content in-place for items that have been presented. Updates targeting non-presented items are silently ignored.

### Delegation Content Forwarding

When delegating to child agents, all `ContentItem` types are forwarded — including `FileContent` and `HtmlContent`.

| Field | Forwarding Behavior |
|-------|-------------------|
| `content_id` | Preserved across hops |
| `description` | Preserved across hops |
| `display_hint` | Stripped — children cannot inherit parent display decisions |
| URLs | Re-signed per-hop (signed URLs are scoped to the delegating agent's session) |

### `response.done` Output Rules

Content promotion in `response.done` depends on the client type:

| Client Type | Output Behavior |
|-------------|----------------|
| **Browser** (`supports_rich_display: true`) | Only items in `presentedIds` appear in `output`. Unpresented items are omitted — no safety net. |
| **Non-browser** | All `collectedContentItems` are promoted to `output` regardless of presentation state. |

This design gives browser-capable LLMs full control over what the user sees, while ensuring non-interactive clients (API consumers, CLI tools) receive all generated content.

### UX Design Principles

1. **Zero-delay content visibility.** Three-stage rendering pipeline:
   - **Skeleton** — typed placeholder shown on first `tool_progress` (instant).
   - **Preview** — thumbnail or low-res version shown when `thumbnail_url` arrives.
   - **Presented** — full content rendered on `present` call.
2. **Encourage saving.** The LLM should proactively call `save_content` for externally-sourced media the user may want to keep.
3. **Minimize LLM round trips.** `present` returns enough metadata (type, dimensions, description) for the LLM to continue without re-inspecting content.
4. **Graceful degradation.** Clients without `supports_rich_display` receive all content automatically via non-browser output rules. No content is lost.

### Error Handling

| Scenario | Behavior |
|----------|----------|
| `present` with unknown `content_id` | Returns available IDs + cross-reference to `save_content` |
| `save_content` download failure | Returns HTTP status, URL, and retry guidance |
| `save_content` missing `url`/`base64` | Returns validation error |
| `save_content` timeout | Returns timeout error (default 30s) |
| Unknown delta type | Clients must ignore unrecognized `delta.type` values gracefully |

## Further Reading

- [UAMP Specification](https://robutler.ai/docs/webagents/protocols/uamp) - Full protocol spec
- [Router Guide](./router.md) - Message routing and capabilities
- [Skills Guide](./skills.md) - How to use UAMP in skills
- [API Reference](./api.md) - TypeScript API details
