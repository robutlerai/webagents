# Shell Skill

The Shell Skill allows agents to execute shell commands securely. It is designed to work in tandem with the Sandbox Skill for maximum isolation.

## Features

- **Sandboxing**: 
  - If `SandboxSkill` is available, commands are automatically executed inside a Docker container.
  - If running locally, restricts execution to the agent's directory and enforces a strict whitelist of allowed commands.
- **Safety**: Blocks dangerous commands (`rm`, `sudo`, etc.) when running locally.

## Configuration

In your `AGENT.md`:

```yaml
skills:
  - shell
  # - sandbox # Recommended for full isolation
```

## Tools

### `run_command`
Executes a shell command.
- `command`: The command string to execute.
- `timeout`: Execution timeout in seconds (default: 30).

## Allowed Commands (Local Mode)
When running locally (without Docker), only specific commands are allowed by default:
- `ls`, `cat`, `grep`, `find`, `head`, `tail`, `wc`
- `echo`, `date`, `pwd`, `which`, `whereis`
- `git`, `npm`, `pip`, `python`, `node`, `uvx`, `curl`, `wget`

## Sandbox Mode
If the `sandbox` skill is enabled, all commands are executed inside the container, bypassing the local whitelist but ensuring full isolation from the host system.
