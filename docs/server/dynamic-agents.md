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

```python
from webagents.server.core.app import create_server
from webagents.agents import BaseAgent

async def resolve_agent(agent_name: str):
    """Resolver function - return BaseAgent or None"""
    
    # Load configuration from your source
    config = await load_config(agent_name)
    if not config:
        return None
    
    # Create and return agent
    return BaseAgent(
        name=config["name"],
        instructions=config["instructions"],
        model=config["model"]
    )

# Pass resolver to server
server = create_server(
    title="Dynamic Server",
    dynamic_agents=resolve_agent  # Resolver function
)
```

## Resolver Function Signature

The resolver function must match this signature:

```python
# Async resolver (recommended)
async def resolve_agent(agent_name: str) -> Optional[BaseAgent]:
    pass

# Sync resolver (also supported)
def resolve_agent(agent_name: str) -> Optional[BaseAgent]:
    pass
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

```python
async def db_resolver(agent_name: str):
    """Load agent from database"""
    query = "SELECT * FROM agents WHERE name = $1"
    row = await db.fetchrow(query, agent_name)
    
    if not row:
        return None
    
    return BaseAgent(
        name=row["name"],
        instructions=row["instructions"],
        model=row["model"]
    )
```

### File-Based Resolver

```python
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

```python
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

```python
# Static agents (always available)
static_agents = [
    BaseAgent(name="assistant", model="openai/gpt-4o"),
    BaseAgent(name="support", model="openai/gpt-4o")
]

# Dynamic resolver for additional agents
async def dynamic_resolver(agent_name: str):
    return await load_from_database(agent_name)

server = create_server(
    agents=static_agents,        # Pre-registered agents
    dynamic_agents=dynamic_resolver  # Runtime resolution
)
```

## Error Handling

Handle errors gracefully in resolvers:

```python
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

- **[Server Overview](server.md)** - Basic server setup
- **[Agent Overview](agent/overview.md)** - Agent setup options
- **[Server Architecture](server-architecture.md)** - Production deployment