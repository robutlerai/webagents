# Getting Started

## Installation

```bash
pip install webagents
```

## Creating an Agent

```python
from webagents import WebAgent

agent = WebAgent(
    name="my-assistant",
    description="A helpful coding assistant",
)

@agent.on_message
async def handle(message: str) -> str:
    return f"I can help with: {message}"

agent.run(port=8080)
```

## Running Your Agent

```bash
python my_agent.py
```

Your agent is now available at `http://localhost:8080/chat/completions`.

## Publishing Intents

Register your agent's capabilities so it can be discovered:

```python
agent = WebAgent(
    name="coding-assistant",
    intents=[
        {"intent": "help with Python", "description": "Debug and write Python code"},
        {"intent": "code review", "description": "Review code quality and suggest improvements"},
    ],
)
```
