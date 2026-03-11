# Migration Guide: Python to TypeScript

## Overview

The TypeScript `webagents` SDK has **full feature parity** with the Python `webagentsd` package. This guide helps you migrate agents from Python to TypeScript.

## Key Differences

| Feature | Python | TypeScript |
|---------|--------|------------|
| Agent class | `Agent` | `BaseAgent` |
| Skill base | `Skill` | `Skill` (identical) |
| Tool decorator | `@tool` | `@tool` (identical) |
| Hook decorator | `@hook` | `@hook` (identical) |
| Server | FastAPI | Hono |
| CLI | `webagentsd` | `webagents` |
| Package manager | pip | npm |

## Migration Steps

### 1. Agent Definition

**Python:**
```python
from webagentsd import Agent, OpenAISkill

agent = Agent(
    name="my-agent",
    instructions="You are helpful.",
    skills=[OpenAISkill(model="gpt-4o")],
)
```

**TypeScript:**
```typescript
import { BaseAgent, OpenAISkill } from 'webagents';

const agent = new BaseAgent({
  name: 'my-agent',
  instructions: 'You are helpful.',
  skills: [new OpenAISkill({ model: 'gpt-4o' })],
});
```

### 2. Custom Skills

**Python:**
```python
from webagentsd import Skill, tool

class SearchSkill(Skill):
    @tool(description="Search the web")
    async def search(self, query: str) -> dict:
        results = await web_search(query)
        return {"results": results}
```

**TypeScript:**
```typescript
import { Skill, tool } from 'webagents';

class SearchSkill extends Skill {
  @tool({ name: 'search', description: 'Search the web' })
  async search(params: { query: string }) {
    const results = await webSearch(params.query);
    return { results };
  }
}
```

### 3. Server

**Python:**
```python
from webagentsd import serve
serve(agent, port=3000)
```

**TypeScript:**
```typescript
import { serve } from 'webagents';
await serve(agent, { port: 3000 });
```

### 4. UAMP Events

The UAMP event types are identical between Python and TypeScript. Both use the same JSON wire format:

```json
{
  "type": "input.text",
  "event_id": "evt_abc123",
  "text": "Hello!",
  "role": "user"
}
```

### 5. Portal Integration

The portal supports **dual-mode execution**. Each agent has a `runtimeEngine` field (`python` or `typescript`) that controls which runtime handles it. You can migrate agents one at a time:

1. Set `runtimeEngine: 'typescript'` on the agent config
2. The portal will route to the in-process TypeScript runtime
3. If issues arise, set back to `'python'` to fall back

Use the migration script:
```bash
npx tsx scripts/migrate-to-typescript-runtime.ts --agent my-agent --dry-run
npx tsx scripts/migrate-to-typescript-runtime.ts --agent my-agent
```

## Interoperability

TypeScript and Python agents can communicate seamlessly:

- **UAMP over HTTP**: Same event format, same endpoints
- **NLI**: Both support `@agent-name` resolution and streaming
- **Completions API**: Both serve `/chat/completions` (OpenAI-compatible)
- **Payment tokens**: JWT tokens are cross-runtime compatible
- **A2A Protocol**: Both support `.well-known/agent.json` and `/a2a`
