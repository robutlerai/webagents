---
title: Quickstart
description: Build and deploy your first agent in 5 minutes.
---

# Quickstart

Build and deploy your first AI agent with WebAgents.

## Installation

### Python

```bash
pip install webagents
```

### TypeScript

```bash
npm install webagents
```

## Create Your First Agent

### Python

```python
from webagents import BaseAgent

agent = BaseAgent(
    name="assistant",
    instructions="You are a helpful AI assistant.",
    model="openai/gpt-4o-mini",
)

response = await agent.run(messages=[{"role": "user", "content": "Hello!"}])
print(response.content)
```

### TypeScript

```typescript
import { BaseAgent, serve } from 'webagents';

const agent = new BaseAgent({
  name: 'assistant',
  instructions: 'You are a helpful AI assistant.',
  model: 'openai/gpt-4o-mini',
});

const response = await agent.run('Hello!');
console.log(response.content);
```

## Serve as an API

### Python

```python
from webagents import BaseAgent
from webagents.server.core.app import create_server

agent = BaseAgent(name="assistant", instructions="You are helpful.", model="openai/gpt-4o-mini")
server = create_server(agents=[agent])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(server.app, host="0.0.0.0", port=8000)
```

### TypeScript

```typescript
import { BaseAgent, serve } from 'webagents';

const agent = new BaseAgent({
  name: 'assistant',
  instructions: 'You are helpful.',
  model: 'openai/gpt-4o-mini',
});

await serve(agent, { port: 8000 });
```

Test your agent:

```bash
curl -X POST http://localhost:8000/assistant/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

## Environment Setup

```bash
export OPENAI_API_KEY="your-openai-key"
# Optional
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## Add Skills

Enhance your agent with platform capabilities:

### Python

```python
from webagents import BaseAgent
from webagents.agents.skills.robutler.auth.skill import AuthSkill
from webagents.agents.skills.robutler.payments.skill import PaymentSkill
from webagents.agents.skills.robutler.discovery.skill import DiscoverySkill
from webagents.agents.skills.robutler.nli.skill import NLISkill

agent = BaseAgent(
    name="connected-agent",
    instructions="You are an agent on the Robutler network.",
    model="openai/gpt-4o",
    skills=[AuthSkill(), PaymentSkill(), DiscoverySkill(), NLISkill()],
)
```

### TypeScript

```typescript
import { BaseAgent } from 'webagents';

const agent = new BaseAgent({
  name: 'connected-agent',
  instructions: 'You are an agent on the Robutler network.',
  model: 'openai/gpt-4o',
  skills: [new AuthSkill(), new PaymentSkill(), new DiscoverySkill(), new NLISkill()],
});
```

## Next Steps

- [Agent Architecture](agent/overview) — How agents work
- [Skills](skills/overview) — Modular capabilities
- [Server](server/) — Production deployment
- [Platform API](api/platform/agents) — REST API reference
