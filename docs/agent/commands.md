---
title: Commands
description: Slash commands and command HTTP endpoints — Python-first today, TypeScript on the roadmap.
---

# Commands

WebAgents provides a structured command system that exposes functionality as both CLI slash commands and HTTP endpoints. This allows agents to define actions that can be invoked from the terminal or via the REST API.

> **TypeScript: Coming soon.** The `@command` decorator currently only ships in the Python SDK. The TypeScript SDK can model commands today as `@http` POST endpoints — see the [TypeScript stub](#typescript-equivalent) below.

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
| `path` | `str` | Command path (e.g., `/notes/save`). Defaults to `/` + function name. |
| `alias` | `str` | Optional alias for the command (e.g., `/save`). |
| `description` | `str` | Command description (defaults to function docstring). |
| `scope` | `str` | Access scope — `all`, `owner`, or `admin`. |

## Command Hierarchy

Commands support hierarchical paths for organization:

```
/notes
  /notes/save
  /notes/list
  /notes/clear
```

## The Chat's Own Commands

The chat's slash commands (`/help`, `/resume`, `/undo` and the rest) are the
chat's, the same in both CLIs; `/help` lists them. An agent's commands are not
among them: they are reached over HTTP, as below, when the agent is served.

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
      "path": "/notes/save",
      "alias": "/save",
      "description": "Save a note",
      "scope": "owner",
      "parameters": {},
      "required": []
    }
  ]
}
```

### Execute Command

```http
POST /agents/{agent_name}/command/notes/save
Content-Type: application/json

{
  "text": "Call the venue on Monday"
}
```

### Get Command Documentation

```http
GET /agents/{agent_name}/command/notes/save
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

## Calling Commands from NLI

Commands can be invoked from the Natural Language Interface skill, allowing agents to call commands programmatically:

```typescript tab="TypeScript"
// Until @command lands, dispatch to HTTP endpoints directly.
const res = await fetch(`${baseUrl}/agents/${agent.name}/command/notes/save`, {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ text: 'Call the venue on Monday' }),
});
const result = await res.json();
```

```python tab="Python"
result = await self.agent.execute_command("/notes/save", {
    "text": "Call the venue on Monday",
})
```

## TypeScript Equivalent

Until the dedicated `@command` decorator ships in TypeScript, model commands as scoped HTTP endpoints under a `/command/` path. This preserves URL parity with the Python implementation, so REST clients can target the same routes regardless of the agent's language.
