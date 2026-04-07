# Getting Started

## Installation

```bash
pip install webagents
```

## Creating an Agent

```python
import asyncio
from webagents import BaseAgent

agent = BaseAgent(
    name="my-assistant",
    instructions="You are a helpful coding assistant.",
    model="openai/gpt-4o-mini",
)

async def main():
    response = await agent.run(messages=[
        {"role": "user", "content": "Hello!"}
    ])
    print(response["choices"][0]["message"]["content"])

asyncio.run(main())
```

## Serving as an API

```python
from webagents import BaseAgent
from webagents.server.core.app import create_server

agent = BaseAgent(
    name="my-assistant",
    instructions="You are a helpful coding assistant.",
    model="openai/gpt-4o-mini",
)

server = create_server(agents=[agent])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(server.app, host="0.0.0.0", port=8000)
```

Your agent is now available at `http://localhost:8000/my-assistant/chat/completions`.

```bash
curl -X POST http://localhost:8000/my-assistant/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

## Adding Skills

Bundle tools, hooks, and endpoints into reusable skills:

```python
from webagents import Skill, tool

class WeatherSkill(Skill):
    @tool(description="Get weather for a city")
    async def get_weather(self, city: str) -> str:
        return f"Weather in {city}: sunny, 72°F"

agent = BaseAgent(
    name="weather-agent",
    instructions="You help with weather queries.",
    model="openai/gpt-4o-mini",
    skills={"weather": WeatherSkill()},
)
```

## Environment Setup

```bash
export OPENAI_API_KEY="your-openai-key"
```

## Next Steps

- [Skills](./skills.md) -- Built-in and custom skills
- [Transports](./transports.md) -- Completions, UAMP, A2A, and more
- [UAMP](./uamp.md) -- Universal Agentic Message Protocol
- [Discovery](./discovery.md) -- Agent discovery and trust
