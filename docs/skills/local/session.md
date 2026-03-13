---
title: Session Manager Skill
---
# Session Manager Skill

The Session Manager Skill provides conversation persistence for agents, enabling session save/restore functionality.

## Overview

Sessions are stored in the agent's local `.webagents/` directory:

- **Sessions**: `<agent_path>/.webagents/sessions/` (JSON files)
- **A2A Sessions**: `<agent_path>/.webagents/sessions/a2a/<conversation_id>/`

## Commands

### Save Session

Save the current session.

```
/session/save [name]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Optional session name (uses session_id if not provided) |

### Load Session

Load a session by ID.

```
/session/load [session_id]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | str | Session ID to load (loads latest if not provided) |

### New Session

Start a new session, discarding the current one.

```
/session/new
```

Or use the alias:

```
/new
```

### List Sessions

List all sessions.

```
/session/history [limit]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 20 | Maximum number of sessions to return |

### Clear Sessions

Clear all sessions. Requires `owner` scope.

```
/session/clear confirm=true
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `confirm` | bool | Must be true to actually clear sessions |

## Configuration

Enable the session skill in your `AGENT.md`:

```yaml
---
name: my-agent
skills:
  - session
---
```

## Auto-Resume

By default, the CLI automatically resumes the latest session when connecting to an agent. This can be disabled:

```python
session._auto_resume_enabled = False
```

## Auto-Save

Sessions are automatically saved after each interaction in the CLI.

## A2A Sessions

Agent-to-Agent (A2A) conversations are stored separately using a conversation ID:

```python
# Load session for a specific A2A conversation
session = session_manager.load_latest(conversation_id="abc123")
```

This allows each A2A conversation to maintain its own history.

## API Endpoints

### Save Session

```http
POST /agents/{name}/command/session/save
Content-Type: application/json

{
  "name": "my-session"
}
```

### Load Session

```http
POST /agents/{name}/command/session/load
Content-Type: application/json

{
  "session_id": "abc123-def456-..."
}
```

### New Session

```http
POST /agents/{name}/command/session/new
```

### List Sessions

```http
GET /agents/{name}/command/session/history
```

## Storage Structure

```
.webagents/
└── sessions/
    ├── .latest              # Pointer to latest session
    ├── abc123-def456.json   # Session file
    ├── xyz789-uvw012.json
    └── a2a/                 # A2A sessions
        └── conversation-id/
            ├── .latest
            └── session-id.json
```

Each session JSON file contains:

```json
{
  "session_id": "abc123-def456-...",
  "agent_name": "my-agent",
  "created_at": "2024-01-15T10:30:00.000000",
  "updated_at": "2024-01-15T11:45:00.000000",
  "messages": [
    {
      "role": "user",
      "content": "Hello",
      "timestamp": "2024-01-15T10:30:00.000000"
    },
    {
      "role": "assistant", 
      "content": "Hi there!",
      "timestamp": "2024-01-15T10:30:05.000000"
    }
  ],
  "metadata": {},
  "input_tokens": 100,
  "output_tokens": 50
}
```

## Context Window Management

When loading sessions, the `max_messages` parameter limits the number of messages to prevent context window overflow:

```python
# Load with message limit
session = session_manager.load_latest(max_messages=100)
```

This keeps the system prompt and the most recent messages within the limit.
