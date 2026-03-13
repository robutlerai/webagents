---
title: Python SDK Reference
description: Core classes, decorators, and server functions exported by the WebAgents Python SDK.
---

# Python SDK Reference

Install the SDK:

```bash
pip install webagents
```

---

## BaseAgent

The core agent class. Supports decorator-based registration, streaming, and scope-based access control.

```python
from webagents import BaseAgent, tool

agent = BaseAgent(
    name="my-agent",
    instructions="You are a helpful assistant.",
    model="gpt-4o",
    skills=[MySkill()],
)
```

### Constructor

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Agent display name |
| `instructions` | `str` | System prompt |
| `model` | `str \| None` | LLM model identifier |
| `skills` | `list[Skill]` | List of skill instances |
| `scopes` | `list[str]` | Required auth scopes |
| `tools` | `list` | Standalone tool functions |
| `hooks` | `list` | Standalone hook functions |
| `handoffs` | `list` | Standalone handoff functions |
| `capabilities` | `dict` | UAMP capabilities |

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `run` | `(input: str, **kwargs) -> RunResponse` | Single-turn execution |
| `run_streaming` | `(input: str, **kwargs) -> AsyncGenerator` | Streaming execution |

---

## Skill

Abstract base class for skills. Skills bundle tools, hooks, handoffs, and endpoints.

```python
from webagents import Skill, tool, hook

class WeatherSkill(Skill):
    @tool(description="Get weather for a city")
    async def get_weather(self, city: str) -> str:
        return f"Weather in {city}: sunny"

    @hook("before_run")
    async def on_before_run(self, data):
        pass
```

### Key Methods

| Method | Description |
|--------|-------------|
| `initialize()` | Called when the skill is registered with an agent |
| `register_tool(func)` | Programmatically register a tool |
| `register_hook(func)` | Programmatically register a hook |
| `register_handoff(func)` | Programmatically register a handoff |
| `get_context()` | Access the current execution context |

---

## Decorators

### @tool

Register a function as an LLM-callable tool.

```python
@tool(name=None, description=None, scope="all", provides=None)
def my_tool(query: str) -> str:
    ...
```

### @prompt

Register a system prompt provider. Multiple prompts are concatenated by priority.

```python
@prompt(priority=50, scope="all")
def system_context() -> str:
    return "Additional context..."
```

### @hook

Register a lifecycle hook.

```python
@hook("before_run", priority=50, scope="all")
async def on_before_run(data):
    ...
```

Events: `before_run`, `after_run`, `before_tool`, `after_tool`, `on_error`

### @handoff

Register a handoff handler for multi-agent delegation.

```python
@handoff(name=None, prompt=None, scope="all", subscribes=None, produces=None)
async def delegate_to_specialist(context, query: str):
    ...
```

### @command

Register a slash command (also creates an HTTP endpoint).

```python
@command("/search", description="Search the knowledge base")
async def search(query: str) -> str:
    ...
```

### @http

Register a custom HTTP endpoint on the agent server.

```python
@http("/webhook", method="post", scope="all")
async def handle_webhook(request):
    ...
```

### @websocket

Register a WebSocket endpoint.

```python
@websocket("/stream", scope="all")
async def handle_stream(ws):
    ...
```

### @widget

Register a widget that renders dynamic UI.

```python
@widget(name="chart", description="Render a chart", template="<div>{{ data }}</div>")
def render_chart(data: dict) -> dict:
    return {"data": data}
```

### @observe

Register a non-consuming event observer.

```python
@observe(subscribes=["tool_call", "response"])
async def log_events(event):
    ...
```

---

## WebAgentsServer

FastAPI-based server for hosting agents.

```python
from webagents.server.core.app import WebAgentsServer, create_server

server = create_server(
    title="My Agents",
    agents=[agent],
    enable_cors=True,
    url_prefix="/api",
)
server.run(port=8080)
```

### create_server

| Parameter | Type | Description |
|-----------|------|-------------|
| `agents` | `list[BaseAgent]` | Agents to serve |
| `dynamic_agents` | `callable` | Factory for dynamic agent creation |
| `enable_cors` | `bool` | Enable CORS (default: `True`) |
| `url_prefix` | `str` | URL prefix for all routes |
| `enable_file_watching` | `bool` | Hot-reload agent files |
| `enable_cron` | `bool` | Enable cron scheduling |
| `storage_backend` | `str` | `"json"` or `"sqlite"` |

---

## Context

Available in tools, hooks, and handoffs via `get_context()` or as an injected parameter.

```python
from webagents.server.context.context_vars import Context

ctx = self.get_context()
ctx.session  # SessionState
ctx.auth     # AuthInfo
ctx.payment  # PaymentInfo
```

---

## Agent Loader

Load agents from AGENT.md files or Python modules.

```python
from webagents import AgentLoader

loader = AgentLoader()
agents = loader.load_directory("./agents")
```

| Class | Description |
|-------|-------------|
| `AgentFile` | Parsed AGENT.md file |
| `AgentMetadata` | Agent metadata (name, model, skills) |
| `MergedAgent` | Agent assembled from file + code |

---

## Session Management

```python
from webagents import SessionManager, LocalState

manager = SessionManager()
state = manager.get_or_create("session-123")
```

| Class | Description |
|-------|-------------|
| `LocalState` | In-memory state store |
| `LocalRegistry` | Agent registry for local development |
| `SessionManager` | Session lifecycle management |
