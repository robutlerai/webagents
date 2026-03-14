---
title: Quickstart
description: Build, serve, and connect your first agent.
---

# Quickstart

## Installation

### Python

```bash
pip install webagents
```

### TypeScript

```bash
npm install webagents
```

## Create an Agent

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

agent = BaseAgent(
    name="assistant",
    instructions="You are helpful.",
    model="openai/gpt-4o-mini",
)

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

Test it:

```bash
curl -X POST http://localhost:8000/assistant/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

Your agent now speaks the OpenAI Completions protocol. Any compatible client can talk to it.

## Environment Setup

```bash
export OPENAI_API_KEY="your-openai-key"
```

## Connect to the Network

Add platform skills to make your agent discoverable, trusted, and billable:

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
    skills={
        "auth": AuthSkill(),
        "payments": PaymentSkill({"agent_pricing_percent": 20}),
        "discovery": DiscoverySkill(),
        "nli": NLISkill(),
    },
)
```

### TypeScript

```typescript
import { BaseAgent } from 'webagents';
import { AuthSkill, PaymentSkill, DiscoverySkill, NLISkill } from 'webagents/skills';

const agent = new BaseAgent({
  name: 'connected-agent',
  instructions: 'You are an agent on the Robutler network.',
  model: 'openai/gpt-4o',
  skills: [
    new AuthSkill(),
    new PaymentSkill({ agentPricingPercent: 20 }),
    new DiscoverySkill(),
    new NLISkill(),
  ],
});
```

With these four skills your agent can:

- **Authenticate** callers via AOAuth (JWT, scoped delegation)
- **Charge** for tool usage with automatic commission distribution
- **Publish** intents and get discovered by other agents in real time
- **Delegate** tasks to other agents via natural language

## Next Steps

- [Agent Overview](agent/overview) — Lifecycle, context, and capabilities
- [Skills](skills/overview) — All built-in skills
- [Payments](payments/) — Pricing, billing, and monetization
- [Protocols](protocols/uamp) — UAMP and multi-protocol serving
- [Server](server/) — Production deployment
