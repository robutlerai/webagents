# Checkpoint Skill

The Checkpoint Skill provides Git-based file snapshots for agents, enabling version control of the agent's working directory.

## Overview

Checkpoints are stored in the agent's local `.webagents/` directory:

- **Git History**: `<agent_path>/.webagents/history/` (Git repository)
- **Metadata**: `<agent_path>/.webagents/checkpoints/` (JSON files)

## Commands

All checkpoint commands require `owner` scope by default.

### Create Checkpoint

Create a new checkpoint snapshot.

```
/checkpoint/create [description] [files]
```

Or use the alias:

```
/checkpoint
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `description` | str | Description of the checkpoint |
| `files` | str | Comma-separated list of files to snapshot (all files if not specified) |

**Example:**

```bash
/checkpoint create "Before refactoring auth module"
```

### Restore Checkpoint

Restore files from a previous checkpoint.

```
/checkpoint/restore <checkpoint_id>
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `checkpoint_id` | str | ID of the checkpoint to restore |

**Example:**

```bash
/checkpoint restore 20240115_143022_abc12345
```

### List Checkpoints

List recent checkpoints.

```
/checkpoint/list [limit]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 20 | Maximum number of checkpoints to return |

### Get Checkpoint Info

Get detailed information about a checkpoint.

```
/checkpoint/info <checkpoint_id>
```

### Delete Checkpoint

Delete a checkpoint (removes metadata only, not Git history).

```
/checkpoint/delete <checkpoint_id>
```

## Configuration

Enable the checkpoint skill in your `AGENT.md`:

```yaml
---
name: my-agent
skills:
  - checkpoint
  - filesystem  # Required for file operations
---
```

## Auto-Checkpointing

The FilesystemSkill can be configured to automatically create checkpoints before file modifications:

```python
# In skill initialization
filesystem_skill.set_checkpoint_manager(checkpoint_manager)
```

When enabled, a checkpoint is created before any `write_file` or `replace` operation.

## API Endpoints

### List Checkpoints

```http
GET /agents/{name}/command/checkpoint/list
```

### Create Checkpoint

```http
POST /agents/{name}/command/checkpoint/create
Content-Type: application/json

{
  "description": "My checkpoint"
}
```

### Restore Checkpoint

```http
POST /agents/{name}/command/checkpoint/restore
Content-Type: application/json

{
  "checkpoint_id": "20240115_143022_abc12345"
}
```

## Storage Structure

```
.webagents/
├── history/           # Git repository
│   ├── .git/
│   ├── .gitignore
│   └── (snapshotted files)
└── checkpoints/       # Checkpoint metadata
    ├── 20240115_143022_abc12345.json
    └── 20240115_150000_def67890.json
```

Each checkpoint JSON file contains:

```json
{
  "checkpoint_id": "20240115_143022_abc12345",
  "created_at": "2024-01-15T14:30:22.123456",
  "description": "Before refactoring",
  "commit_hash": "abc123def456...",
  "files_changed": ["src/auth.py", "config.json"],
  "session_id": "uuid-of-session",
  "metadata": {}
}
```
