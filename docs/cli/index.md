---
title: WebAgents CLI
---
# WebAgents CLI

The WebAgents CLI provides a powerful terminal interface for interacting with and managing your AI agents.

## Overview

The CLI allows you to:

- Chat with agents in a rich, interactive REPL
- Manage agent sessions and checkpoints
- Execute tools and skills securely (including Docker sandboxing)
- Configure agent behaviors and providers
- Publish agents to the WebAgents platform
- Discover and connect with other agents

## Installation

```bash
pip install webagents
```

## Quick Start

### Start Interactive REPL

```bash
# Start with default agent (auto-detected from current directory)
webagents

# Start with a specific agent
webagents connect my-agent

# Start with a specific agent file
webagents connect /path/to/AGENT.md
```

### Common Operations

```bash
# Create a new agent
webagents init my-agent

# List registered agents
webagents list

# Run agent headlessly
webagents run my-agent --prompt "Summarize this document"

# Manage sessions
webagents session new
webagents session save my-session

# Manage checkpoints
webagents checkpoint create "Before refactor"
webagents checkpoint list
```

## CLI Structure

The CLI follows a hierarchical command structure:

```
webagents
├── connect [agent]        # Start interactive REPL (default)
├── init [name]            # Create new agent
├── list                   # List agents
├── run [agent]            # Run headlessly
├── login                  # Authenticate with platform
├── version                # Show version
│
├── session                # Session management
│   ├── new               # Start new session
│   ├── save [id]         # Save session
│   ├── load <id>         # Load session
│   ├── list              # List sessions
│   └── history           # Show history
│
├── checkpoint             # Checkpoint management
│   ├── create [desc]     # Create checkpoint
│   ├── restore <id>      # Restore checkpoint
│   ├── list              # List checkpoints
│   └── info <id>         # Show details
│
├── skill                  # Skill management
│   ├── list              # List skills
│   ├── add <name>        # Add skill
│   └── remove <name>     # Remove skill
│
├── daemon                 # Daemon management
│   ├── start             # Start daemon
│   ├── stop              # Stop daemon
│   └── status            # Show status
│
└── auth                   # Authentication
    ├── login             # Login to platform
    └── logout            # Logout
```

## REPL Commands

Inside the interactive REPL, use `/` commands:

- `/help` - Show available commands
- `/new` - Start a fresh session
- `/agent list` - List available agents
- `/skill list` - List active skills
- `/checkpoint create` - Create a checkpoint
- `/exit` - Exit the REPL

See [Commands](commands.md) for the complete list.

## Features

### Interactive REPL

- Rich text editing with syntax highlighting
- Command history with search (↑/↓ arrows)
- Tab completion for slash commands
- Multi-line editing (Alt+Enter)

### Streaming Responses

- Real-time response streaming
- Visible "thinking" blocks for reasoning models
- Inline tool call indicators

### Session Management

- Automatic session persistence
- Named session save/load
- Cross-session history

### Checkpointing

- Git-based file snapshots
- Automatic checkpointing on file changes (optional)
- Easy restore to any checkpoint

### Secure Sandboxing

- Optional Docker-based execution
- File system isolation
- Resource limits

## Configuration

The CLI reads configuration from:

1. `AGENT.md` YAML frontmatter
2. `~/.webagents/config.yaml`
3. Environment variables

See [Configuration](./configuration.md) for details.
