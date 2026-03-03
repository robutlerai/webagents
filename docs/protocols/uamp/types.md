# Types

Type definitions for UAMP protocol elements. All types are defined using TypeScript-like notation but are language-agnostic.

## Modality

Extensible modality type:

```typescript
type Modality = 'text' | 'audio' | 'image' | 'video' | 'file' | string;
```

## Audio Format

Common audio formats:

```typescript
type AudioFormat = 
  | 'pcm16' 
  | 'g711_ulaw' 
  | 'g711_alaw' 
  | 'mp3' 
  | 'opus' 
  | 'wav' 
  | 'webm' 
  | 'aac' 
  | string;
```

## Voice Configuration

Provider-agnostic voice configuration:

```typescript
interface VoiceConfig {
  provider?: string;          // 'openai', 'elevenlabs', 'google', etc.
  voice_id?: string;          // Provider-specific voice ID
  name?: string;              // Human-readable voice name
  speed?: number;             // 0.5 - 2.0
  pitch?: number;             // Voice pitch adjustment
  language?: string;          // BCP-47 language tag
  extensions?: object;        // Provider-specific extensions
}
```

## Turn Detection Configuration

Generic turn detection for voice interfaces:

```typescript
interface TurnDetectionConfig {
  type: 'server_vad' | 'client_vad' | 'push_to_talk' | 'none';
  threshold?: number;         // VAD sensitivity (0.0 - 1.0)
  silence_duration_ms?: number;
  prefix_padding_ms?: number;
  extensions?: object;
}
```

## Response Format

Controls the output format of model responses (structured output). Compatible with OpenAI's `response_format` parameter and translated to provider-native equivalents.

```typescript
interface ResponseFormat {
  type: 'text' | 'json_object' | 'json_schema';
  json_schema?: {
    name: string;              // Schema name identifier
    description?: string;
    schema: JSONSchema;        // JSON Schema definition
    strict?: boolean;          // Enforce strict schema adherence
  };
}
```

| Type | Behavior |
|------|----------|
| `text` | Default. Model returns free-form text. |
| `json_object` | Model returns valid JSON. No schema enforced. |
| `json_schema` | Model returns JSON conforming to the provided schema. |

**Provider mapping:**

| Provider | `json_schema` | `json_object` |
|----------|--------------|---------------|
| OpenAI / LiteLLM | Passed through natively | Passed through natively |
| Google Gemini | `response_mime_type='application/json'` + `response_schema` | `response_mime_type='application/json'` |
| Anthropic | Forced tool use with `input_schema` + unwrap | System prompt instruction |

## Session Configuration

Full session configuration:

```typescript
interface SessionConfig {
  modalities: Modality[];
  instructions?: string;
  tools?: ToolDefinition[];
  voice?: VoiceConfig;
  input_audio_format?: AudioFormat;
  output_audio_format?: AudioFormat;
  turn_detection?: TurnDetectionConfig;
  response_format?: ResponseFormat;
  extensions?: object;        // Provider-specific extensions
}
```

## Tool Definition

Standard tool/function definition (OpenAI-compatible):

```typescript
interface ToolDefinition {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters: JSONSchema;
  };
}
```

## Content Item

Content in responses and messages:

```typescript
interface ContentItem {
  type: 'text' | 'audio' | 'image' | 'video' | 'file' | 'tool_call' | 'tool_result';
  text?: string;
  audio?: string;             // Base64 encoded
  image?: string;             // Base64 or URL
  video?: string | { url: string };   // Base64 or URL
  file?: string | { url: string };    // Base64 or URL
  filename?: string;          // Original filename (for file type)
  mime_type?: string;         // MIME type (for file type)
  tool_call?: {
    id: string;
    name: string;
    arguments: string;        // JSON string
  };
  tool_result?: {
    call_id: string;
    result: string;
    is_error?: boolean;
  };
}
```

## Message

Conversation message used in stateless context passing. Reuses `ContentItem` for multimodal content.

```typescript
interface Message {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | ContentItem[];
  name?: string;              // Participant name (for multi-user contexts)
  tool_call_id?: string;      // For tool role: which tool call this responds to
  tool_calls?: ToolCall[];    // For assistant role: tool calls made
}

interface ToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;        // JSON string
  };
}
```

## Usage Statistics

Token and cost tracking:

```typescript
interface UsageStats {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cached_tokens?: number;
  
  cost?: {
    input_cost: number;
    output_cost: number;
    total_cost: number;
    currency: string;         // 'USD', etc.
  };
  
  audio?: {
    input_seconds?: number;
    output_seconds?: number;
  };
  
  details?: object;           // Provider-specific details
}
```

## Session

Session object returned by server:

```typescript
interface Session {
  id: string;
  created_at: number;         // Unix timestamp (seconds)
  config: SessionConfig;
  status: 'active' | 'closed';
}
```

---

## Capability Types

Detailed capability declarations for feature negotiation.

### ImageCapabilities

```typescript
interface ImageCapabilities {
  formats: string[];           // ['jpeg', 'png', 'gif', 'webp']
  max_size_bytes?: number;     // e.g., 20971520 (20MB)
  max_pixels?: number;         // e.g., 20000000
  detail_levels?: string[];    // ['auto', 'low', 'high']
  max_images_per_request?: number;
}
```

### AudioCapabilities

```typescript
interface AudioCapabilities {
  input_formats: AudioFormat[];
  output_formats: AudioFormat[];
  sample_rates?: number[];     // [24000, 48000]
  max_duration_seconds?: number;
  supports_realtime: boolean;
  voices?: string[];           // Available voice IDs
}
```

### FileCapabilities

```typescript
interface FileCapabilities {
  supported_mime_types: string[];
  max_size_bytes?: number;
  supports_pdf: boolean;
  supports_code: boolean;
  supports_structured_data: boolean;  // JSON, CSV, etc.
}
```

### ToolCapabilities

```typescript
interface ToolCapabilities {
  supports_tools: boolean;
  supports_parallel_tools: boolean;
  supports_streaming_tools: boolean;
  max_tools_per_request?: number;
  built_in_tools: string[];    // ['web_search', 'code_interpreter', ...]
}
```

---

## Unified Capabilities

All capability declarations (model, client, agent) use the **same structure**. See [Capabilities](capabilities.md) for detailed documentation.

```typescript
interface Capabilities {
  // Identity
  id: string;                  // model_id, client_id, or agent_id
  provider: string;
  
  // Core modalities
  modalities: Modality[];
  
  // Detailed capabilities
  image?: ImageCapabilities;
  audio?: AudioCapabilities;
  file?: FileCapabilities;
  tools?: ToolCapabilities;
  
  // Features
  supports_streaming: boolean;
  supports_thinking: boolean;
  supports_caching: boolean;
  context_window?: number;
  max_output_tokens?: number;
  
  // Agent/client extensions
  provides?: string[];         // Capabilities provided
  widgets?: string[];          // Available widgets
  endpoints?: string[];        // HTTP/WebSocket endpoints
  
  // Custom extensions
  extensions?: object;
}
```

### Examples

**Model capabilities:**

```json
{
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
    "built_in_tools": ["web_search", "code_interpreter"] 
  },
  "supports_streaming": true,
  "supports_thinking": false,
  "context_window": 128000
}
```

**Client capabilities:**

```json
{
  "id": "web-app",
  "provider": "my-company",
  "modalities": ["text", "image", "audio"],
  "supports_streaming": true,
  "widgets": ["chart", "form", "table"],
  "extensions": { "supports_html": true, "platform": "web" }
}
```

**Agent capabilities:**

```json
{
  "id": "research-assistant",
  "provider": "webagents",
  "modalities": ["text", "image"],
  "tools": { 
    "supports_tools": true, 
    "built_in_tools": ["web_search", "render_chart"] 
  },
  "provides": ["web_search", "chart", "tts"],
  "widgets": ["chart", "table"],
  "endpoints": ["/api/search", "/ws/stream"],
  "supports_streaming": true
}
```
