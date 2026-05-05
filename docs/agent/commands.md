---
title: Commands
description: Slash commands and command HTTP endpoints — Python-first today, TypeScript on the roadmap.
---

# Commands

WebAgents provides a structured command system that exposes functionality as both CLI slash commands and HTTP endpoints. This allows agents to define actions that can be invoked from the terminal or via the REST API.

> **TypeScript: Coming soon.** The `@command` decorator currently only ships in the Python SDK. Track parity in the [Python ↔ TypeScript Parity Matrix](../internal/python-typescript-parity.md). The TypeScript SDK can model commands today as `@http` POST endpoints — see the [TypeScript stub](#typescript-equivalent) below.

## The `@command` Decorator

```typescript tab="TypeScript"
// @command is not yet available in the TypeScript SDK.
// Until it lands, expose commands as HTTP endpoints. The agent server
// will register them under POST /agents/{name}/command/<path>.

import { Skill, http } from 'webagents';

class MySkill extends Skill {
  readonly name = 'my-skill';

  @http({
    path: '/command/mycommand/action',
    method: 'POST',
    auth: 'session',
    description: 'Do something',
  })
  async myAction(req: Request): Promise<Response> {
    const { param = '' } = await req.json().catch(() => ({}));
    return Response.json({ status: 'done', param });
  }
}
```

```python tab="Python"
from typing import Any, Dict
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
| `path` | `str` | Command path (e.g., `/checkpoint/create`). Defaults to `/` + function name. |
| `alias` | `str` | Optional alias for the command (e.g., `/checkpoint`). |
| `description` | `str` | Command description (defaults to function docstring). |
| `scope` | `str` | Access scope — `all`, `owner`, or `admin`. |

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

Commands are also exposed as HTTP endpoints.

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

```typescript tab="TypeScript"
// HTTP-endpoint workaround until @command lands
import { Skill, http } from 'webagents';

class AdminSkill extends Skill {
  readonly name = 'admin';

  @http({
    path: '/command/admin/reset',
    method: 'POST',
    scopes: ['admin'],
    description: 'Reset everything',
  })
  async reset(_req: Request): Promise<Response> {
    return Response.json({ status: 'reset' });
  }
}
```

```python tab="Python"
@command("/admin/reset", description="Reset everything", scope="admin")
async def reset(self) -> Dict[str, Any]:
    # Only admins can call this
    return {"status": "reset"}
```

## Built-in Commands

WebAgents (Python) includes several built-in commands:

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

```typescript tab="TypeScript"
// Until @command lands, dispatch to HTTP endpoints directly.
const res = await fetch(`${baseUrl}/agents/${agent.name}/command/checkpoint/create`, {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ description: 'Before changes' }),
});
const result = await res.json();
```

```python tab="Python"
result = await self.agent.execute_command("/checkpoint/create", {
    "description": "Before changes",
})
```

## TypeScript Equivalent

Until the dedicated `@command` decorator ships in TypeScript, model commands as scoped HTTP endpoints under a `/command/` path. This preserves URL parity with the Python implementation, so REST clients can target the same routes regardless of the agent's language.
