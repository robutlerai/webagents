---
title: Agent Capabilities
description: Capability declarations for agents, models, and clients — discovery, interoperability, and the `provides` field.
---

# Agent Capabilities

Capabilities enable discovery and interoperability between agents, clients, and models. WebAgents uses the [UAMP](../protocols/uamp.md) unified capabilities format.

## Unified Format

All capability declarations (model, client, agent) use the **same structure**:

```typescript tab="TypeScript"
import type { Capabilities } from 'webagents';

// Model capabilities
const modelCaps: Capabilities = {
  id: 'gpt-4o',
  provider: 'openai',
  modalities: ['text', 'image'],
  supports_streaming: true,
  context_window: 128_000,
};

// Client capabilities
const clientCaps: Capabilities = {
  id: 'web-app',
  provider: 'robutler',
  modalities: ['text', 'image', 'audio'],
  widgets: ['chart', 'form'],
  extensions: { supports_html: true },
};

// Agent capabilities
const agentCaps: Capabilities = {
  id: 'my-agent',
  provider: 'webagents',
  modalities: ['text', 'image'],
  provides: ['web_search', 'chart', 'tts'],
  endpoints: ['/api/search'],
};
```

```python tab="Python"
from webagents.uamp import Capabilities

# Model capabilities
model_caps = Capabilities(
    id="gpt-4o",
    provider="openai",
    modalities=["text", "image"],
    supports_streaming=True,
    context_window=128000,
)

# Client capabilities
client_caps = Capabilities(
    id="web-app",
    provider="robutler",
    modalities=["text", "image", "audio"],
    widgets=["chart", "form"],
    extensions={"supports_html": True},
)

# Agent capabilities
agent_caps = Capabilities(
    id="my-agent",
    provider="webagents",
    modalities=["text", "image"],
    provides=["web_search", "chart", "tts"],
    endpoints=["/api/search"],
)
```

## The `provides` Field

Decorators support a `provides` field to declare the capability they provide.

### Tools

```typescript tab="TypeScript"
@tool({ provides: 'web_search', description: 'Search the web for information' })
async searchWeb(params: { query: string }): Promise<string> { return ''; }

@tool({ provides: 'chart', description: 'Render data as a chart widget' })
async renderChart(params: { data: string }): Promise<string> { return ''; }

@tool({ provides: 'tts', description: 'Convert text to speech audio' })
async textToSpeech(params: { text: string }): Promise<Uint8Array> { return new Uint8Array(); }
```

```python tab="Python"
from webagents import tool

@tool(provides="web_search")
async def search_web(query: str) -> str:
    """Search the web for information."""
    ...

@tool(provides="chart")
async def render_chart(data: str) -> str:
    """Render data as a chart widget."""
    ...

@tool(provides="tts")
async def text_to_speech(text: str) -> bytes:
    """Convert text to speech audio."""
    ...
```

### Handoffs

```typescript tab="TypeScript"
@handoff({ name: 'gpt4', description: 'GPT-4 with extended thinking' })
async *gpt4Handoff(events) {
  yield { type: 'response.delta', delta: '...' } as const;
}

@handoff({ name: 'vision', description: 'Vision model for image analysis' })
async *visionHandoff(events) {
  yield { type: 'response.delta', delta: '...' } as const;
}
```

```python tab="Python"
from webagents import handoff

@handoff(name="gpt4", provides="thinking")
async def gpt4_handoff(messages, **kwargs):
    """GPT-4 with extended thinking."""
    ...

@handoff(name="vision", provides="image_analysis")
async def vision_handoff(messages, **kwargs):
    """Vision model for image analysis."""
    ...
```

> The TypeScript `@handoff` decorator does not currently accept a `provides` field; capabilities for handoffs are inferred from the skill's class name and `subscribes` / `produces`. Track this in the [parity matrix](../internal/python-typescript-parity.md).

### HTTP Endpoints

```typescript tab="TypeScript"
@http({ path: '/export/pdf', method: 'POST', description: 'Export data as PDF' })
async exportPdf(req: Request): Promise<Response> { return new Response('pdf'); }

@http({ path: '/api/search', method: 'GET', description: 'Search API endpoint' })
async searchApi(req: Request): Promise<Response> { return Response.json({}); }
```

```python tab="Python"
from webagents import http

@http("/export/pdf", method="post", provides="pdf_export")
def export_pdf(data: dict) -> bytes:
    """Export data as PDF."""
    ...

@http("/api/search", provides="search_api")
def search_api(query: str) -> dict:
    """Search API endpoint."""
    ...
```

### WebSockets

```typescript tab="TypeScript"
@websocket({ path: '/stream' })
realtimeStream(ws: WebSocket): void {
  ws.onmessage = () => ws.send('chunk');
}
```

```python tab="Python"
from webagents import websocket

@websocket("/stream", provides="realtime")
async def realtime_stream(ws):
    """Real-time streaming endpoint."""
    ...
```

### Widgets

```typescript tab="TypeScript"
// @widget is Python-only today. Return a <widget> envelope from a regular @tool:
@tool({ description: 'Interactive chart widget', provides: 'chart' })
async chartWidget(params: { data: string }): Promise<string> {
  return `<widget kind="webagents" id="chart"><div>${params.data}</div></widget>`;
}
```

```python tab="Python"
from webagents import widget

@widget(provides="chart")
def chart_widget(data: str) -> str:
    """Interactive chart widget."""
    ...
```

## Capability Aggregation

The agent automatically aggregates all `provides` values from:

- Tools (`@tool`)
- Handoffs (`@handoff`)
- HTTP endpoints (`@http`)
- WebSockets (`@websocket`)
- Widgets (`@widget`, Python only)

These are exposed via the `Capabilities.provides` field.

## Querying Capabilities

Agents expose capabilities through the `/capabilities` endpoint:

```bash
curl http://localhost:8000/my-agent/capabilities
```

Response:

```json
{
  "id": "my-agent",
  "provider": "webagents",
  "modalities": ["text", "image"],
  "provides": ["web_search", "chart", "tts", "pdf_export"],
  "endpoints": ["/api/search", "/export/pdf"],
  "widgets": ["chart"],
  "supports_streaming": true
}
```

## Client Capabilities

Clients announce their capabilities when creating a session:

```typescript tab="TypeScript"
import type { SessionCreateEvent, Capabilities } from 'webagents';

const event: SessionCreateEvent = {
  type: 'session.create',
  event_id: 'evt_1',
  session: { client_id: 'web-app' },
  client_capabilities: {
    id: 'web-app',
    provider: 'robutler',
    modalities: ['text', 'image', 'audio'],
    widgets: ['chart', 'form'],
    extensions: { supports_html: true },
  },
};
```

```python tab="Python"
from webagents.uamp import SessionCreateEvent, Capabilities

event = SessionCreateEvent(
    client_capabilities=Capabilities(
        id="web-app",
        provider="robutler",
        modalities=["text", "image", "audio"],
        widgets=["chart", "form"],
        extensions={"supports_html": True},
    )
)
```

This enables agents to adapt their responses based on client capabilities.

## UAMP Types

Import capability types from `webagents/uamp`:

```typescript tab="TypeScript"
import type {
  Capabilities,        // Unified capabilities (model, client, agent)
  ImageCapabilities,   // Detailed image support
  AudioCapabilities,   // Detailed audio support
  FileCapabilities,    // Detailed file support
  ToolCapabilities,    // Tool calling support
} from 'webagents';
```

```python tab="Python"
from webagents.uamp import (
    Capabilities,
    ImageCapabilities,
    AudioCapabilities,
    FileCapabilities,
    ToolCapabilities,
)
```

## Best Practices

1. **Use descriptive `provides` values** — make capabilities discoverable.
2. **Match client capabilities** — adapt output to what the client can render.
3. **Aggregate from skills** — let skills declare their capabilities.
4. **Query before calling** — check agent capabilities before making requests.
