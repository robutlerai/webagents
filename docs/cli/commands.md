# Commands

The CLI supports special slash commands to control the environment.

## Available Commands

### General
- `/help`: Show available commands.
- `/exit` or `/quit`: Exit the CLI.
- `/clear`: Clear the screen and conversation.
- `/config`: Show/edit configuration.

### Session & Environment
- `/new`: Start a new session.
- `/history`: View conversation history.
- `/tokens`: Show token usage.
- `/model [name]`: Switch the active model (e.g., `/model gpt-4o`).

### Agent Management
- `/agent`: Switch or show current agent.
- `/discover`: Discover agents.
- `/mcp`: MCP server management.

### Daemon Management
- `/status`: Show daemon status.
- `/list`: List registered agents.
- `/register`: Register agent with daemon.
- `/run`: Run a registered agent.

### Dynamic Agent Commands

Agents can expose their own commands (e.g., `/checkpoint`, `/session`). See [Agent Commands](../agent/commands.md) for details.

## Keyboard Shortcuts

- `Enter`: Submit command.
- `Alt+Enter` (or `Esc+Enter`): Insert newline.
- `Ctrl+C`: Interrupt current generation.
- `Ctrl+D`: Exit.
- `Ctrl+T`: Toggle Todo list visibility.
