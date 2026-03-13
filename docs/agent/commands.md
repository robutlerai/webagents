---
title: Commands
---
# Commands

WebAgents provides a structured command system that exposes functionality as both CLI slash commands and HTTP endpoints. This allows agents to define actions that can be invoked from the terminal or via the REST API.

## The `@command` Decorator

Use the `@command` decorator to define commands in your skills:

```python
from webagents.agents.tools.decorators import command

class MySkill(Skill):
    
    @command("/mycommand/action", description="Do something", scope="all")
    async def my_action(self, param: str = "") -> Dict[str, Any]:
        """Perform the action."""
        return {"status": "done", "param": param}
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | str | Command path (e.g., "/checkpoint/create"). Defaults to "/" + function name. |
| `alias` | str | Optional alias for the command (e.g., "/checkpoint"). |
| `description` | str | Command description (defaults to function docstring). |
| `scope` | str | Access scope - "all", "owner", or "admin". |

## Command Hierarchy

Commands support hierarchical paths for organization:

```
/session
  /session/save
  /session/load
  /session/new
  /session/history
  /session/clear

/checkpoint
  /checkpoint/create
  /checkpoint/restore
  /checkpoint/list
```

## CLI Usage

Commands are available as slash commands in the CLI:

```bash
# Execute a command
/checkpoint create

# With arguments
/session load abc123

# Show subcommands for a group
/checkpoint
```

## HTTP API

Commands are also exposed as HTTP endpoints:

### List Commands

```http
GET /agents/{agent_name}/command
```

Returns a list of all available commands:

```json
{
  "commands": [
    {
      "path": "/checkpoint/create",
      "alias": "/checkpoint",
      "description": "Create a new checkpoint",
      "scope": "owner",
      "parameters": {},
      "required": []
    }
  ]
}
```

### Execute Command

```http
POST /agents/{agent_name}/command/checkpoint/create
Content-Type: application/json

{
  "description": "Before major refactoring"
}
```

### Get Command Documentation

```http
GET /agents/{agent_name}/command/checkpoint/create
```

Returns command details including parameters and description.

## Scopes

Commands support scope-based access control:

| Scope | Description |
|-------|-------------|
| `all` | Available to everyone |
| `owner` | Only available to the agent owner |
| `admin` | Only available to administrators |

Example with restricted scope:

```python
@command("/admin/reset", description="Reset everything", scope="admin")
async def reset(self) -> Dict[str, Any]:
    # Only admins can call this
    return {"status": "reset"}
```

## Built-in Commands

WebAgents includes several built-in commands:

### Session Commands

| Command | Description |
|---------|-------------|
| `/session/save` | Save current session |
| `/session/load` | Load a session by ID |
| `/session/new` | Start a new session |
| `/session/history` | List all sessions |
| `/session/clear` | Clear all sessions (owner only) |

### Checkpoint Commands

| Command | Description |
|---------|-------------|
| `/checkpoint/create` | Create a new checkpoint (alias: `/checkpoint`) |
| `/checkpoint/restore` | Restore to a previous checkpoint |
| `/checkpoint/list` | List all checkpoints |
| `/checkpoint/info` | Get checkpoint details |
| `/checkpoint/delete` | Delete a checkpoint |

## Calling Commands from NLI

Commands can be invoked from the Natural Language Interface skill, allowing agents to call commands programmatically:

```python
# In another skill or agent
result = await self.agent.execute_command("/checkpoint/create", {
    "description": "Before changes"
})
```
