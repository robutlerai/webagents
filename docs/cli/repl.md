---
title: Interactive REPL Guide
---
# Interactive REPL Guide

The WebAgents REPL provides a premium terminal experience for interacting with AI agents.

## Starting a Session

```bash
# Default - connects to AGENT.md in current directory
webagents

# Explicit connect command
webagents connect

# Connect to specific agent
webagents connect planner
webagents connect ./AGENT-writer.md
```

## The Interface

```
█   █ █▀▀ █▀▄ █▀█ █▀▀ █▀▀ █▄ █ ▀█▀ █▀
█ █ █ █▀  █▀▄ █▀█ █ █ █▀  █ ▀█  █  ▀█
▀ ▀ ▀ ▀▀▀ ▀▀  ▀ ▀ ▀▀▀ ▀▀▀ ▀  ▀  ▀  ▀▀

Tips for getting started:
1. Ask questions, edit files, or run commands.
2. Be specific for the best results.
3. /help for more information.

❯ 
```

### Components

- **Banner**: Colorful ASCII art logo
- **Tips**: Getting started hints
- **Prompt**: `❯` indicates ready for input
- **Status bar**: Shows directory, agent, sandbox status

## Slash Commands

Type `/` followed by a command:

### Session Management

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/exit` or `/quit` | Exit the session |
| `/clear` | Clear the screen |

### Checkpoints

| Command | Description |
|---------|-------------|
| `/save [name]` | Save session checkpoint |
| `/load [name]` | Load session checkpoint |

Checkpoints save your conversation history and context, allowing you to resume later.

```
❯ /save my-project
Saving checkpoint: my-project
Checkpoint saved

❯ /load my-project
Loading checkpoint: my-project
Checkpoint loaded
```

### Agent Management

| Command | Description |
|---------|-------------|
| `/agent` | Show current agent info |
| `/agent <name>` | Switch to different agent |

```
❯ /agent
┌─ Agent ────────────────────────────┐
│ Current Agent: planner             │
│ Path: ./AGENT-planner.md           │
│ Use /agent <name> to switch        │
└────────────────────────────────────┘

❯ /agent writer
Switching to agent: writer
```

### Discovery

| Command | Description |
|---------|-------------|
| `/discover <intent>` | Find agents by intent |

```
❯ /discover summarize documents
Searching for: summarize documents
Found 3 agents...
```

### Tools and Context

| Command | Description |
|---------|-------------|
| `/mcp` | Show MCP server status |
| `/tokens` | Show token usage statistics |
| `/history` | Show conversation history |
| `/config` | Show configuration |

## File References

Reference files in your prompts using `@`:

```
❯ Summarize @README.md

❯ Compare @src/old.py with @src/new.py

❯ What's in the @docs/ folder?
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+C` | Cancel current operation |
| `Ctrl+D` | Exit session |
| `Up/Down` | Navigate history |
| `Tab` | Autocomplete |

## History

Command history is saved to `~/.webagents/history` and persists across sessions.

- Use Up/Down arrows to navigate
- History is searchable
- Auto-suggestions from history

## Streaming Responses

Responses stream in real-time with Markdown rendering:

```
❯ Write a Python function to sort a list

  Responding with gpt-4o...

Here's a function to sort a list:

```python
def sort_list(items, reverse=False):
    """Sort a list with optional reverse order."""
    return sorted(items, reverse=reverse)
```

This uses Python's built-in `sorted()` function...
```

## Tool Execution Display

When the agent uses tools, you'll see execution panels:

```
┌─ WriteFile Writing to utils.py ────┐
│ 1 def helper():                    │
│ 2     return "hello"               │
└────────────────────────────────────┘
```

## Token Usage

Track token consumption with `/tokens`:

```
❯ /tokens
┌─ Token Usage ──────────────────────┐
│ Input tokens:  1,234               │
│ Output tokens: 567                 │
│ Total:         1,801               │
└────────────────────────────────────┘
```

## Session State

Your session includes:

- Conversation history
- Agent context
- Checkpoint data
- Token statistics

All stored in `.webagents/sessions/` (gitignored).

## Tips for Best Results

1. **Be specific** - Clear prompts get better responses
2. **Use context** - Reference files with `@path/to/file`
3. **Save often** - Use `/save` before complex tasks
4. **Check tokens** - Monitor usage with `/tokens`
5. **Switch agents** - Use `/agent` to access specialized agents
