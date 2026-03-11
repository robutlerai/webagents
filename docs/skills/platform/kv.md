# KV Storage Skill

Simple per-agent key-value storage for persistent data and configuration.

## Overview

The `KVSkill` provides owner-scoped key-value storage capabilities, allowing agents to persistently store and retrieve simple string data via the Robutler portal `/api/kv` endpoint.

## Features

- **Owner-Only Access**: All operations are restricted to the agent owner using `scope="owner"`
- **Per-Agent Storage**: Each agent has its own isolated key-value namespace
- **Namespace Support**: Optional namespacing for organizing keys
- **Simple String Storage**: Store and retrieve string values by key
- **Automatic Authentication**: Uses agent API keys for secure access

## Usage

### Basic Setup

```python
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.robutler.kv.skill import KVSkill

agent = BaseAgent(
    name="kv-agent",
    model="openai/gpt-4o-mini",
    skills={
        "kv": KVSkill()
    }
)
```

### Storing and Retrieving Data

The skill provides simple key-value operations:

```python
# Store configuration
response = await agent.run(messages=[
    {"role": "user", "content": "Store my API key as 'openai_key' with value 'sk-...'"}
])

# Retrieve configuration
response = await agent.run(messages=[
    {"role": "user", "content": "Get my stored API key from 'openai_key'"}
])
```

## Tool Reference

### `memory`

Unified key-value storage tool. Replaces `kv_get`, `kv_set`, `kv_delete`, and `kv_list`. Use the `action` parameter to specify the operation.

**Parameters:**

- `action` (str, required): One of `get`, `set`, `delete`, `list`
- `key` (str, required for get/set/delete): The key to operate on
- `value` (str, required for set): The string value to store
- `namespace` (str, optional): Optional namespace for organizing keys

**Actions:**

| Action | Description |
|--------|-------------|
| `get` | Retrieve value by key |
| `set` | Store value under key |
| `delete` | Remove key and value |
| `list` | List keys (optionally in namespace) |

!!! note "Backward compatibility"
    The legacy tools (`kv_get`, `kv_set`, `kv_delete`, `kv_list`) remain available for backward compatibility. New agents should use `memory` with the `action` parameter.

### Legacy tools (backward compatibility)

**`kv_set`** — Set a key to a string value.

**`kv_get`** — Get a string value by key.

**`kv_delete`** — Delete a key and its value.

**`kv_list`** — List keys in namespace.

**Scope:** `owner` - Only the agent owner can access the key-value store

## Configuration

The skill requires no additional configuration beyond adding it to your agent. It automatically:

- Resolves the agent and user context from the current request
- Uses the agent's API key for authentication
- Connects to the appropriate Robutler portal API endpoint

## Use Cases

Perfect for storing:

- **API Keys and Tokens**: Securely store third-party API credentials
- **Configuration Settings**: Agent-specific configuration values
- **State Information**: Simple state data between agent interactions
- **User Preferences**: Store user-specific settings and preferences

## Example Integration

```python
from webagents import Skill, tool

class WeatherSkill(Skill):
    @tool
    async def get_weather(self, location: str) -> str:
        # Retrieve stored API key
        api_key = await self.discover_and_call("kv", "get", "weather_api_key")
        
        if not api_key:
            return "❌ Weather API key not configured"
        
        # Use API key to fetch weather data
        # ... weather API logic here
        
        return f"Weather in {location}: Sunny, 72°F"
    
    @tool
    async def configure_weather_api(self, api_key: str) -> str:
        # Store API key for future use
        result = await self.discover_and_call("kv", "set", "weather_api_key", api_key)
        return f"Weather API configured: {result}"
```

## Security

- **Owner-Only Access**: All KV operations are scoped to `owner` only
- **Agent Isolation**: Each agent has its own isolated key-value store
- **API Key Authentication**: Uses secure agent API keys for portal communication
- **Context Resolution**: Automatically resolves agent and user context for proper isolation

## Error Handling

The skill handles common error scenarios:

- **Missing Context**: Returns error messages if agent/user context cannot be resolved
- **API Authentication Failures**: Handles missing or invalid API keys
- **Network Issues**: Returns empty strings or error messages for connection problems
- **Portal API Errors**: Surfaces API error responses for debugging

## Limitations

- **String Values Only**: Only supports string values (use JSON encoding for complex data)
- **Owner Scope**: Only the agent owner can access the key-value store
- **No Bulk Operations**: Operations are performed one key at a time
- **Simple Querying**: No advanced querying or pattern matching capabilities

## Dependencies

- **Agent API Key**: Requires valid agent API key for portal authentication
- **Agent Context**: Requires agent to be properly initialized with context
- **Portal Connectivity**: Requires network access to Robutler portal API endpoints
