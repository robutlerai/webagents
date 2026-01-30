# Agent Capabilities

Capabilities enable discovery and interoperability between agents, clients, and models. WebAgents uses the [UAMP](https://uamp.dev) unified capabilities format.

## Unified Format

All capability declarations (model, client, agent) use the **same structure**:

```python
from webagents.uamp import Capabilities

# Model capabilities
model_caps = Capabilities(
    id="gpt-4o",
    provider="openai",
    modalities=["text", "image"],
    supports_streaming=True,
    context_window=128000
)

# Client capabilities  
client_caps = Capabilities(
    id="web-app",
    provider="robutler",
    modalities=["text", "image", "audio"],
    widgets=["chart", "form"],
    extensions={"supports_html": True}
)

# Agent capabilities
agent_caps = Capabilities(
    id="my-agent",
    provider="webagents",
    modalities=["text", "image"],
    provides=["web_search", "chart", "tts"],
    endpoints=["/api/search"]
)
```

## The `provides` Parameter

Decorators support a `provides` parameter to declare what capability they provide:

### Tools

```python
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

```python
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

### HTTP Endpoints

```python
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

```python
from webagents import websocket

@websocket("/stream", provides="realtime")
async def realtime_stream(ws):
    """Real-time streaming endpoint."""
    ...
```

### Widgets

```python
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
- Widgets (`@widget`)

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

Clients can announce their capabilities when creating a session:

```python
from webagents.uamp import SessionCreateEvent, Capabilities

event = SessionCreateEvent(
    client_capabilities=Capabilities(
        id="web-app",
        provider="robutler",
        modalities=["text", "image", "audio"],
        widgets=["chart", "form"],
        extensions={"supports_html": True}
    )
)
```

This enables agents to adapt their responses based on client capabilities.

## UAMP Types

Import capability types from `webagents.uamp`:

```python
from webagents.uamp import (
    Capabilities,           # Unified capabilities (model, client, agent)
    ImageCapabilities,      # Detailed image support
    AudioCapabilities,      # Detailed audio support
    FileCapabilities,       # Detailed file support
    ToolCapabilities,       # Tool calling support
)
```

## Best Practices

1. **Use descriptive provides values** - Make capabilities discoverable
2. **Match client capabilities** - Adapt output to what client can render
3. **Aggregate from skills** - Let skills declare their capabilities
4. **Query before calling** - Check agent capabilities before making requests
