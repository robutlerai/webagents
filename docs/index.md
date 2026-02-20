# WebAgents Python SDK

Build AI agents that communicate via standard protocols.

## Quick Start

```bash
pip install webagents
```

```python
from webagents import WebAgent

agent = WebAgent(name="my-agent")

@agent.on_message
async def handle(message: str) -> str:
    return f"Hello! You said: {message}"

agent.run(port=8080)
```

## Features

- **Multiple transports**: Completions, UAMP, A2A, Realtime, ACP
- **NLI**: Natural Language Interface for agent-to-agent communication
- **Discovery**: Intent publishing for Roborum-based discovery
- **Skills**: Modular skill system for composable agent behaviors
- **UAMP**: Universal Agentic Message Protocol support
