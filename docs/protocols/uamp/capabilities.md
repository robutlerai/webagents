# Capabilities

UAMP uses a unified capability format for models, clients, and agents. This enables seamless capability negotiation between any participants in the protocol.

## Unified Capabilities

All capability declarations use the **same structure**, regardless of whether they describe a model, client, or agent:

```json
{
  "id": "identifier",
  "provider": "provider-name",
  "modalities": ["text", "image", "audio", "video"],
  "image": { ... },
  "audio": { ... },
  "file": { ... },
  "tools": { ... },
  "supports_streaming": true,
  "supports_thinking": false,
  "supports_caching": false,
  "context_window": 128000,
  "max_output_tokens": 4096,
  "provides": ["capability1", "capability2"],
  "widgets": ["widget1", "widget2"],
  "endpoints": ["/api/endpoint"],
  "extensions": { ... }
}
```

## Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Identifier (model_id, client_id, or agent_id) |
| `provider` | string | Provider name (e.g., "openai", "anthropic", "robutler") |
| `modalities` | string[] | Supported content types: text, image, audio, video, file |
| `image` | ImageCapabilities | Detailed image support (see [Types](types.md)) |
| `audio` | AudioCapabilities | Detailed audio support |
| `file` | FileCapabilities | Detailed file support |
| `tools` | ToolCapabilities | Tool/function calling support |
| `supports_streaming` | boolean | Whether streaming responses are supported |
| `supports_thinking` | boolean | Extended thinking/reasoning support |
| `supports_caching` | boolean | Context caching support |
| `context_window` | number | Maximum input tokens |
| `max_output_tokens` | number | Maximum output tokens |
| `provides` | string[] | Capabilities provided (for agents) |
| `widgets` | string[] | Available/supported widgets |
| `endpoints` | string[] | HTTP/WebSocket endpoints (for agents) |
| `extensions` | object | Context-specific extensions |

## Model Capabilities

Describes what an LLM can process:

```json
{
  "id": "gpt-4o",
  "provider": "openai",
  "modalities": ["text", "image"],
  "image": {
    "formats": ["jpeg", "png", "gif", "webp"],
    "detail_levels": ["auto", "low", "high"]
  },
  "file": {
    "supports_pdf": true
  },
  "tools": {
    "supports_tools": true,
    "built_in_tools": ["web_search", "code_interpreter"]
  },
  "supports_streaming": true,
  "supports_thinking": false,
  "context_window": 128000
}
```

## Client Capabilities

Describes what a client can render/handle:

```json
{
  "id": "web-app",
  "provider": "my-company",
  "modalities": ["text", "image", "audio"],
  "supports_streaming": true,
  "widgets": ["chart", "form", "table"],
  "extensions": {
    "supports_html": true,
    "platform": "web"
  }
}
```

Clients announce capabilities via:

1. `client_capabilities` field in `session.create` event
2. `client.capabilities` event sent anytime during session

## Agent Capabilities

Describes what an agent can do (model + tools + endpoints):

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

The `provides` field aggregates capabilities from the agent's tools, handoffs, and endpoints.

## Capability Negotiation

### Session Creation

Client announces capabilities when creating a session:

```json
{
  "type": "session.create",
  "uamp_version": "1.0",
  "session": {
    "modalities": ["text", "image"]
  },
  "client_capabilities": {
    "id": "web-app",
    "modalities": ["text", "image", "audio"],
    "widgets": ["chart"]
  }
}
```

### Capability Query

Client can query server capabilities at any time:

```json
{
  "type": "capabilities.query"
}
```

Server responds with:

```json
{
  "type": "capabilities",
  "capabilities": {
    "id": "gpt-4o",
    "modalities": ["text", "image"],
    "supports_streaming": true
  }
}
```

### Dynamic Updates

Either party can send capability updates during the session:

- **Client**: `client.capabilities` event
- **Server**: `capabilities` event

## Discovery Endpoints

Agents expose capabilities through transport-specific discovery mechanisms:

| Transport | Discovery Method |
|-----------|-----------------|
| **Completions** | `GET /capabilities` |
| **A2A** | `GET /.well-known/agent.json` → `modelCapabilities` |
| **ACP** | JSON-RPC `capabilities` method |
| **Realtime** | `session.created` event → `capabilities` |

## Best Practices

1. **Announce early** - Send client capabilities in `session.create`
2. **Query before assuming** - Check server capabilities before using features
3. **Handle missing gracefully** - Default to text-only if capabilities unknown
4. **Update on change** - Send capability events when features change
5. **Use extensions** - Put custom capabilities in the `extensions` field
