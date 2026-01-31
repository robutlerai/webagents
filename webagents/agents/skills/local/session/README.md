# Session Skill

The Session Skill manages conversation state and history.

## Features

- **Checkpoints**: Save and restore conversation state to disk.
- **Context Management**: Handles loading history within token limits.

## Configuration

In your `AGENT.md`:

```yaml
skills:
  - session
```

## Tools

### `save_checkpoint`
Saves current session state.
- `name`: Name for the checkpoint.

### `load_checkpoint`
Restores session state.
- `name`: Name of checkpoint to load.
