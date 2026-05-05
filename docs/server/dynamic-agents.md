---
title: Dynamic Agents
description: Resolve agents on demand from databases, files, or external APIs.
---

# Dynamic Agents

Load agents at runtime using the `dynamic_agents` parameter and resolver functions.

## Overview

Dynamic agents enable runtime agent loading without pre-registration:

- **On-Demand Creation** - Agents created when first requested
- **Configuration-Driven** - Load from external sources (DB, API, files)
- **Flexible Updates** - Change agent behavior without redeployment
- **Memory Efficient** - Only create agents that are actually used

## Dynamic Agent Resolver

The `dynamic_agents` parameter accepts a resolver function that creates agents by name:

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';
import { createAgentApp } from 'webagents/server';
import { serve } from 'webagents/server/node';

async function resolveAgent(agentName: string): Promise<BaseAgent | null> {
  const config = await loadConfig(agentName);
  if (!config) return null;
  return new BaseAgent({
    name: config.name,
    instructions: config.instructions,
    model: config.model,
  });
}

const app = createAgentApp({
  title: 'Dynamic Server',
  dynamicAgents: resolveAgent,
});
await serve(app, { host: '0.0.0.0', port: 8000 });
```

```python tab="Python"
from webagents.server.core.app import create_server
from webagents.agents import BaseAgent

async def resolve_agent(agent_name: str):
    """Resolver function - return BaseAgent or None"""
    config = await load_config(agent_name)
    if not config:
        return None
    return BaseAgent(
        name=config["name"],
        instructions=config["instructions"],
        model=config["model"],
    )

server = create_server(
    title="Dynamic Server",
    dynamic_agents=resolve_agent,
)
```

## Resolver Function Signature

The resolver function must match this signature:

```typescript tab="TypeScript"
type AgentResolver = (agentName: string) => Promise<BaseAgent | null> | BaseAgent | null;
```

```python tab="Python"
async def resolve_agent(agent_name: str) -> Optional[BaseAgent]: ...
def resolve_agent(agent_name: str) -> Optional[BaseAgent]: ...
```

**Parameters:**
- `agent_name`: The agent name from the URL path
- **Returns:** `BaseAgent` instance or `None` if not found

## Resolution Flow

1. **Request** arrives for `/agent-name/chat/completions`
2. **Static Check** - Look for pre-registered agents first
3. **Dynamic Call** - Call `dynamic_agents(agent_name)` if not found
4. **Agent Creation** - Resolver creates and returns BaseAgent
5. **Request Processing** - Server uses the resolved agent

## Configuration Sources

### Database Resolver

```typescript tab="TypeScript"
async function dbResolver(agentName: string): Promise<BaseAgent | null> {
  const row = await db.query(
    'SELECT * FROM agents WHERE name = $1',
    [agentName],
  );
  if (!row) return null;
  return new BaseAgent({
    name: row.name,
    instructions: row.instructions,
    model: row.model,
  });
}
```

```python tab="Python"
async def db_resolver(agent_name: str):
    """Load agent from database"""
    query = "SELECT * FROM agents WHERE name = $1"
    row = await db.fetchrow(query, agent_name)

    if not row:
        return None

    return BaseAgent(
        name=row["name"],
        instructions=row["instructions"],
        model=row["model"],
    )
```

### File-Based Resolver

```typescript tab="TypeScript"
import { readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';

async function fileResolver(agentName: string): Promise<BaseAgent | null> {
  const path = `agents/${agentName}.json`;
  if (!existsSync(path)) return null;
  const config = JSON.parse(await readFile(path, 'utf8'));
  return new BaseAgent(config);
}
```

```python tab="Python"
import json
import os

async def file_resolver(agent_name: str):
    """Load agent from JSON files"""
    config_path = f"agents/{agent_name}.json"

    if not os.path.exists(config_path):
        return None

    with open(config_path) as f:
        config = json.load(f)

    return BaseAgent(**config)
```

### API Resolver

```typescript tab="TypeScript"
async function apiResolver(agentName: string): Promise<BaseAgent | null> {
  const res = await fetch(`https://api.example.com/agents/${agentName}`);
  if (!res.ok) return null;
  const config = await res.json();
  return new BaseAgent(config);
}
```

```python tab="Python"
import aiohttp

async def api_resolver(agent_name: str):
    """Load agent from external API"""
    url = f"https://api.example.com/agents/{agent_name}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                return None

            config = await resp.json()
            return BaseAgent(**config)
```



## Combined Static and Dynamic

Use both static agents and dynamic resolution:

```typescript tab="TypeScript"
const staticAgents = [
  new BaseAgent({ name: 'assistant', model: 'openai/gpt-4o' }),
  new BaseAgent({ name: 'support', model: 'openai/gpt-4o' }),
];

async function dynamicResolver(agentName: string): Promise<BaseAgent | null> {
  return loadFromDatabase(agentName);
}

const app = createAgentApp({
  agents: staticAgents,
  dynamicAgents: dynamicResolver,
});
```

```python tab="Python"
static_agents = [
    BaseAgent(name="assistant", model="openai/gpt-4o"),
    BaseAgent(name="support", model="openai/gpt-4o"),
]

async def dynamic_resolver(agent_name: str):
    return await load_from_database(agent_name)

server = create_server(
    agents=static_agents,
    dynamic_agents=dynamic_resolver,
)
```

## Error Handling

Handle errors gracefully in resolvers:

```typescript tab="TypeScript"
async function safeResolver(agentName: string): Promise<BaseAgent | null> {
  try {
    const config = await loadConfig(agentName);
    if (!config) {
      console.info(`Agent '${agentName}' not found`);
      return null;
    }
    const agent = new BaseAgent(config);
    console.info(`Created agent '${agentName}'`);
    return agent;
  } catch (err) {
    console.error(`Failed to resolve agent '${agentName}':`, err);
    return null;
  }
}
```

```python tab="Python"
import logging

async def safe_resolver(agent_name: str):
    """Resolver with error handling"""
    try:
        config = await load_config(agent_name)
        if not config:
            logging.info(f"Agent '{agent_name}' not found")
            return None

        agent = BaseAgent(**config)
        logging.info(f"Created agent '{agent_name}'")
        return agent

    except Exception as e:
        logging.error(f"Failed to resolve agent '{agent_name}': {e}")
        return None
```

## See Also

- **[Server Overview](./index.md)** - Basic server setup
- **[Agent Overview](../agent/overview.md)** - Agent setup options
- **[Server Architecture](./architecture.md)** - Production deployment