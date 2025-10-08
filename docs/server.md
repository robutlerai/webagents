# Server Overview

Deploy agents as OpenAI-compatible API servers using the Robutler FastAPI server.

## Quick Start

### Basic Server

```python
from webagents.server.core.app import create_server
from webagents.agents import BaseAgent

# Create agent
agent = BaseAgent(
    name="assistant",
    instructions="You are a helpful assistant",
    model="openai/gpt-4o"
)

# Create and run server
server = create_server(agents=[agent])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(server.app, host="0.0.0.0", port=8000)
```

### Multiple Agents

```python
from webagents.agents.skills.core.memory import ShortTermMemorySkill

# Create multiple agents
agents = [
    BaseAgent(
        name="support",
        instructions="You are a customer service agent",
        model="openai/gpt-4o",
        skills={"memory": ShortTermMemorySkill()}
    ),
    BaseAgent(
        name="analyst",
        instructions="You are a data analyst",
        model="anthropic/claude-3-sonnet"
    )
]

# Create server with multiple agents
server = create_server(
    title="Multi-Agent Server",
    agents=agents
)
```

## Server Parameters

The `create_server()` function accepts these key parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `title` | str | "WebAgents Server" | Server title for OpenAPI docs |
| `description` | str | "AI Agent Server..." | Server description |
| `version` | str | "1.0.0" | API version |
| `agents` | List[BaseAgent] | [] | Static agents to serve |
| `dynamic_agents` | Callable | None | Dynamic agent resolver function |
| `url_prefix` | str | "" | URL prefix (e.g., "/agents") |

### Advanced Parameters

```python
server = create_server(
    title="Production Server",
    agents=agents,
    dynamic_agents=resolve_agent,
    url_prefix="/api/v1",
    enable_monitoring=True,
    enable_cors=True,
    request_timeout=300.0
)
```

## API Endpoints

The server automatically creates these endpoints for each agent:

```
GET  /                              # Server info
GET  /health                        # Health check
GET  /{agent_name}                  # Agent info
POST /{agent_name}/chat/completions # OpenAI-compatible chat
GET  /{agent_name}/health           # Agent health
```

With `url_prefix="/agents"`:
```
POST /agents/{agent_name}/chat/completions
```

## Client Usage

### OpenAI SDK Compatible

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/assistant",
    api_key="your-api-key"  # Optional
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Environment Variables

```bash
# LLM Provider Keys
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Optional Server Configuration
ROBUTLER_HOST=0.0.0.0
ROBUTLER_PORT=8000
```

## See Also

- **[Dynamic Agents](dynamic-agents.md)** - Runtime agent loading
- **[Agent Endpoints](agent/endpoints.md)** - Custom HTTP endpoints
- **[Agent Overview](agent/overview.md)** - Agent setup guide