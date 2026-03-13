---
title: CLI Quickstart
---
# CLI Quickstart

Get started with the WebAgents CLI in 5 minutes.

## Installation

Install webagents with pip:

```bash
pip install webagents
```

## Your First Agent

### 1. Create an Agent

Create a new agent in your current directory:

```bash
webagents init
```

This creates `AGENT.md` with a basic configuration:

```yaml
---
name: assistant
description: A helpful AI assistant
namespace: local
model: openai/gpt-4o-mini
intents:
  - answer questions
  - help with tasks
---

# Assistant Agent

You are a helpful AI assistant.
```

### 2. Start a Session

Launch the interactive REPL:

```bash
webagents connect
```

Or just run `webagents` with no arguments:

```bash
webagents
```

You'll see the welcome screen:

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

### 3. Chat with Your Agent

Type your message and press Enter:

```
❯ What can you help me with?

I can help you with:
- Answering questions on various topics
- Helping with writing and editing
- Providing explanations and summaries
- Assisting with problem-solving
```

### 4. Use Slash Commands

Type `/help` to see available commands:

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/exit` | Exit the session |
| `/clear` | Clear the screen |
| `/save` | Save session checkpoint |
| `/load` | Load session checkpoint |
| `/agent` | Show or switch agent |
| `/discover` | Discover other agents |

## Using Templates

Create agents from templates:

```bash
# List available templates
webagents template list

# Create a planning agent
webagents init --template planning

# Create a named agent from template
webagents init writer --template content
```

## Multiple Agents

Create multiple agents in a directory:

```bash
# Create default agent
webagents init

# Create named agents
webagents init planner
webagents init writer
webagents init researcher

# List all agents
webagents list

# Connect to specific agent
webagents connect planner
```

## Context with AGENTS.md

Create shared context for all agents in a directory:

```bash
webagents init --context
```

This creates `AGENTS.md`:

```yaml
---
namespace: local
---

# Project Context

This file provides context for all agents in this directory.
All AGENT*.md files in this folder inherit from this context.
```

## Running Headless

Execute an agent without interactive mode:

```bash
# Single prompt
webagents run -p "Summarize this README"

# Run specific agent
webagents run planner -p "Create a weekly plan"
```

## Next Steps

- [Commands Reference](commands.md) - Full command documentation
- [REPL Guide](repl.md) - Interactive session features
- [AGENT.md Format](../agents-md/overview.md) - Agent configuration
- [Daemon Setup](../daemon/overview.md) - Background daemon
