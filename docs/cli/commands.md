# Commands

The WebAgents CLI supports slash commands for controlling the environment and managing agents.

## CLI Commands

The `webagents` CLI provides command-line access to agent management:

```bash
# Lifecycle
webagents init [name]          # Create new agent
webagents connect [agent]      # Start interactive REPL
webagents run [agent]          # Run agent headlessly
webagents list                 # List registered agents

# System
webagents login                # Authenticate with platform
webagents daemon start|stop|status  # Manage background daemon
webagents version              # Show version info

# Session Management
webagents session new          # Start fresh session
webagents session history      # Show conversation logs
webagents session save [id]    # Save current session
webagents session load <id>    # Load previous session
webagents session list         # List saved sessions

# Checkpoint Management
webagents checkpoint create [desc]  # Create file snapshot
webagents checkpoint restore <id>   # Restore checkpoint
webagents checkpoint list           # List checkpoints
webagents checkpoint info <id>      # Show checkpoint details

# Skill Management
webagents skill list           # List active skills
webagents skill add <name>     # Add skill to agent
webagents skill remove <name>  # Remove skill from agent
```

## REPL Slash Commands

Inside the interactive REPL, use `/` commands:

### System Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/exit`, `/quit` | Exit the CLI |
| `/clear` | Clear screen and conversation |
| `/cls` | Clear screen only |
| `/status` | Show daemon status |
| `/config` | Show configuration |

### Agent Commands

| Command | Description |
|---------|-------------|
| `/agent` | Show current agent info |
| `/agent list` | List registered agents |
| `/agent connect <name>` | Switch to another agent |
| `/agent info` | Show current agent config |
| `/list` | List registered agents (shortcut) |
| `/register [path]` | Register agent with daemon |
| `/run <agent>` | Run a registered agent |

### Skill Commands

| Command | Description |
|---------|-------------|
| `/skill` | List active skills |
| `/skill list` | List active skills |
| `/skill add <name>` | Add a skill |
| `/skill remove <name>` | Remove a skill |

### Session Commands

These commands are provided by the SessionSkill:

| Command | Description |
|---------|-------------|
| `/new` | Start a new session |
| `/session new` | Start a new session |
| `/session save [id]` | Save current session |
| `/session load <id>` | Load a previous session |
| `/session history` | Show conversation history |
| `/session clear` | Clear session history |

### Checkpoint Commands

These commands are provided by the CheckpointSkill:

| Command | Description |
|---------|-------------|
| `/checkpoint` | Show checkpoint subcommands |
| `/checkpoint create [desc]` | Create a new checkpoint |
| `/checkpoint restore <id>` | Restore a checkpoint |
| `/checkpoint list` | List available checkpoints |
| `/checkpoint info <id>` | Show checkpoint details |
| `/checkpoint delete <id>` | Delete a checkpoint |

### Intent Commands

These commands are provided by the DiscoverySkill:

| Command | Description |
|---------|-------------|
| `/intent discover <query>` | Discover agents by intent |
| `/intent publish` | Publish agent intents to platform |
| `/intent delete [intent]` | Delete published intents |
| `/intent update` | Update published intents |
| `/intent list` | List current agent intents |

### Namespace Commands

These commands are provided by the NamespaceSkill:

| Command | Description |
|---------|-------------|
| `/namespace`, `/ns` | Show current namespace info |
| `/namespace list` | List available namespaces |
| `/namespace create <name>` | Create a new namespace |
| `/namespace join <name>` | Join an existing namespace |
| `/namespace leave` | Leave current namespace |
| `/namespace delete <name>` | Delete a namespace |

### Publish Commands

These commands are provided by the PublishSkill:

| Command | Description |
|---------|-------------|
| `/publish [visibility]` | Publish agent to platform |
| `/publish status` | Check publication status |
| `/publish unpublish`, `/unpublish` | Remove agent from platform |

### Utility Commands

| Command | Description |
|---------|-------------|
| `/tokens` | Show token usage |
| `/model [name]` | Show or change model |
| `/history` | Show conversation history |
| `/discover <intent>` | Discover agents by intent |
| `/mcp` | MCP server management |

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Enter` | Submit command |
| `Alt+Enter` or `Esc+Enter` | Insert newline |
| `Ctrl+C` | Interrupt current generation |
| `Ctrl+D` | Exit |
| `Ctrl+T` | Toggle Todo list visibility |
| `↑/↓` | Navigate command history |

## Dynamic Agent Commands

Agents can expose their own commands via the `@command` decorator. These commands are automatically available when connected to the agent. See [Agent Commands](../agent/commands.md) for details on creating custom commands.
