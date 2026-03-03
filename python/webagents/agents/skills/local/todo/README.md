# Todo Skill

The Todo Skill helps agents manage complex tasks by maintaining a structured list of subtasks.

## Features

- **Task Management**: Create, update, and track status of multiple items.
- **UI Integration**: In the CLI, the current "in progress" task is displayed prominently.
- **State Management**: Todos persist during the session.

## Configuration

In your `AGENT.md`:

```yaml
skills:
  - todo
```

## Tools

### `write_todos`
Updates the agent's todo list.
- `todos`: List of task objects or JSON string.
  - Each item: `{"description": "Task name", "status": "pending|in_progress|completed"}`
