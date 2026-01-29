# Agent Endpoints

Expose custom HTTP API endpoints for your agent using the `@http` decorator. Endpoints are mounted under the agent’s base path and are served by the same FastAPI app used for chat completions.

- Simple, declarative decorator: `@http("/path", method="get|post", scope="...")`
- Path parameters and query strings supported
- Scope-based access control (`all`, `owner`, `admin`)
- Plays nicely with skills, tools, and hooks

## Basic Usage

Define an endpoint and attach it to your agent via `capabilities` (auto-registration):

```python
from webagents import BaseAgent, http

@http("/status", method="get")
def get_status() -> dict:
    return {"status": "healthy"}

agent = BaseAgent(
    name="assistant",
    model="openai/gpt-4o-mini",
    capabilities=[get_status]
)
```

Serve it (same as in Quickstart):

```python
from webagents.server.core.app import create_server
import uvicorn

server = create_server(agents=[agent])
uvicorn.run(server.app, host="0.0.0.0", port=8000)
```

Available at:
- `GET /assistant/status`

## Methods, Path and Query

```python
from webagents import http

# GET collection
@http("/users", method="get")
def list_users() -> dict:
    return {"users": ["alice", "bob", "charlie"]}

# POST create with JSON body
@http("/users", method="post")
def create_user(data: dict) -> dict:
    return {"created": data.get("name"), "id": "user_123"}

# GET item with path param and optional query param
@http("/users/{user_id}", method="get")
def get_user(user_id: str, include_details: bool = False) -> dict:
    user = {"id": user_id, "name": f"User {user_id}"}
    if include_details:
        user["details"] = "Extended info"
    return user
```

Example requests:

```bash
# List users
curl http://localhost:8000/assistant/users

# Create user
curl -X POST http://localhost:8000/assistant/users \
  -H "Content-Type: application/json" \
  -d '{"name": "dana"}'

# Get user with query param
curl "http://localhost:8000/assistant/users/42?include_details=true"

# Missing or wrong Content-Type
curl -X POST http://localhost:8000/assistant/users -d '{"name":"dana"}'
# -> 415 Unsupported Media Type

# Wrong method
curl -X GET http://localhost:8000/assistant/users -H "Content-Type: application/json" -d '{}'
# -> 405 Method Not Allowed

# Unauthorized scope (example)
curl http://localhost:8000/assistant/admin/metrics
# -> 403 Forbidden
```

## Access Control (Scopes)

Use `scope` to restrict who can call an endpoint:

```python
@http("/public", method="get", scope="all")
def public_endpoint() -> dict:
    return {"message": "Public data"}

@http("/owner-info", method="get", scope="owner")
def owner_endpoint() -> dict:
    return {"private": "owner data"}

@http("/admin/metrics", method="get", scope="admin")
def admin_metrics() -> dict:
    return {"rps": 100, "error_rate": 0.001}
```

## WebSocket Endpoints

For bidirectional real-time communication, use the `@websocket` decorator:

```python
from webagents import BaseAgent, websocket

@websocket("/stream")
async def my_websocket(ws) -> None:
    """Bidirectional WebSocket handler"""
    await ws.accept()
    try:
        async for message in ws.iter_json():
            # Process incoming message
            response = await process(message)
            await ws.send_json(response)
    except WebSocketDisconnect:
        pass

agent = BaseAgent(
    name="assistant",
    model="openai/gpt-4o-mini",
    capabilities=[my_websocket]
)
```

Available at:
- `WS /assistant/stream`

### WebSocket with LLM Streaming

Combine WebSocket with `execute_handoff()` in a skill:

```python
from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import websocket

class StreamingSkill(Skill):
    @websocket("/chat")
    async def chat_stream(self, ws) -> None:
        await ws.accept()
        
        async for msg in ws.iter_json():
            messages = msg.get("messages", [])
            
            # Stream LLM response through WebSocket
            async for chunk in self.execute_handoff(messages):
                await ws.send_json(chunk)
```

## SSE Streaming (Server-Sent Events)

Return an `AsyncGenerator` from an `@http` handler to stream as SSE:

```python
from webagents import http
from typing import AsyncGenerator

@http("/events", method="get")
async def stream_events() -> AsyncGenerator[str, None]:
    """SSE streaming endpoint"""
    for i in range(5):
        yield f"data: {{\"count\": {i}}}\n\n"
        await asyncio.sleep(1)
    yield "data: [DONE]\n\n"
```

The server automatically sets SSE headers:
- `Content-Type: text/event-stream`
- `Cache-Control: no-cache`
- `Connection: keep-alive`

## Tips

- Keep one responsibility per endpoint (CRUD-style patterns work well)
- Prefer `get` for retrieval, `post` for creation/processing
- Validate inputs inside handlers; return JSON-serializable data
- Register endpoints through `capabilities=[...]` along with `@tool`/`@hook`/`@handoff`

## See Also

- **[Quickstart](../quickstart.md)** — serving agents
- **[Agent Skills](skills.md)** — modular capabilities
- **[Tools](tools.md)** — add executable functions
- **[Hooks](hooks.md)** — lifecycle integration
