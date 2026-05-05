---
title: Session Management
description: Save, load, and resume CLI conversation state with checkpoints — Python REPL/CLI feature.
---

# Session Management

> [!NOTE]
> `webagents session …` and `/session …` slash commands are **Python-only** today. The TypeScript [`SessionSkill`](../skills/local/session.md) provides a programmatic K/V store, but no CLI subcommand yet — track in [internal/python-typescript-parity.md](../internal/python-typescript-parity.md).

The CLI maintains stateful sessions for your interactions with agents.

## Checkpoints

You can save and load the state of your conversation (messages history) using checkpoints.

- **Save Checkpoint**: `save_checkpoint(name="my-save")`
- **Load Checkpoint**: `load_checkpoint(name="my-save")`

Checkpoints are stored in `~/.webagents/agents/{agent_name}/checkpoints/`.

## History

The CLI automatically maintains command history (accessible via Up/Down arrows).
Conversation history is maintained in memory during the session and can be persisted via checkpoints.
