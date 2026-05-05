---
title: Development Setup
description: Set up a local environment to develop the WebAgents Python and TypeScript SDKs — install, lint, test, and run a dev server.
---

# Development Setup

This guide covers setting up a development environment for working on the WebAgents SDKs.

## Prerequisites

- **Node.js**: 20 LTS or higher (TypeScript SDK)
- **pnpm**: 9.x — used by the monorepo (TypeScript SDK)
- **Python**: 3.10 or higher (Python SDK)
- **Git**: Latest version
- **OpenAI API Key** (or another LLM provider key): For agent functionality

## Environment Setup

### 1. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/robutlerai/robutler.git
cd robutler-proxy

# Or clone your fork
git clone https://github.com/YOUR_USERNAME/robutler.git
cd robutler-proxy
```

### 2. Set up the toolchain

```bash tab="TypeScript"
# Install pnpm if missing
corepack enable
corepack prepare pnpm@latest --activate

cd webagents/typescript
pnpm install
```

```bash tab="Python"
cd webagents/python

python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install --upgrade pip
pip install -e ".[dev]"
```

### 3. Environment Variables

Create a `.env` file in the project root:

```bash
# Required for agent functionality
OPENAI_API_KEY=your-openai-api-key

# Optional Robutler platform configuration
ROBUTLER_API_KEY=rok_your-robutler-api-key
ROBUTLER_API_URL=https://robutler.ai

# Development settings
WEBAGENTS_DEBUG=true
```

## Development Tools

### Lint & Format

```bash tab="TypeScript"
pnpm run lint
pnpm run format
pnpm run typecheck
```

```bash tab="Python"
black .
isort .
flake8 webagents/
```

### Testing

```bash tab="TypeScript"
# Run unit tests
pnpm test

# Watch mode
pnpm test -- --watch

# Coverage
pnpm test -- --coverage
```

```bash tab="Python"
pytest
pytest --cov=webagents
pytest tests/test_agent.py -v
```

### Documentation

```bash
# Fumadocs (Next.js portal)
pnpm --filter portal dev

# MkDocs (external publishing)
cd webagents
mkdocs serve
```

## Running the Development Server

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';
import { serve } from 'webagents/server/node';

const agent = new BaseAgent({
  name: 'test-agent',
  instructions: 'You are a helpful test assistant.',
  model: 'openai/gpt-4o-mini',
});

await serve(agent, { host: '127.0.0.1', port: 8000 });
```

```python tab="Python"
from webagents import BaseAgent
from webagents.server.fastapi import create_agent_app
import uvicorn

agent = BaseAgent(
    name="test-agent",
    instructions="You are a helpful test assistant.",
    model="openai/gpt-4o-mini",
)

app = create_agent_app(agents=[agent])

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

## Common Development Tasks

### Adding a New Tool

```typescript tab="TypeScript"
import { Skill, tool, pricing } from 'webagents';

class MySkill extends Skill {
  readonly name = 'my-skill';

  @pricing({ creditsPerCall: 0.001 })
  @tool({ description: 'Process input text' })
  async myNewTool(params: { inputText: string }): Promise<string> {
    return `Processed: ${params.inputText}`;
  }
}
```

```python tab="Python"
from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, pricing

class MySkill(Skill):
    @pricing(credits_per_call=0.001)
    @tool(description="Process input text")
    async def my_new_tool(self, input_text: str) -> str:
        return f"Processed: {input_text}"
```

### Testing the Endpoint

```bash
curl -X POST http://localhost:8000/test-agent/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "test-agent", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Debugging

### Enable Debug Logging

```typescript tab="TypeScript"
process.env.WEBAGENTS_DEBUG = 'true';
process.env.DEBUG = 'webagents:*';
```

```python tab="Python"
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or set the environment variable globally:

```bash
export WEBAGENTS_DEBUG=true
```

This covers the essential development setup for contributing to the WebAgents SDKs.