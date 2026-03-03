# Sandbox Skill

The Sandbox Skill provides an isolated Docker environment for agent execution. It is the foundation for secure shell and tool execution.

## Features

- **Docker Isolation**: runs commands inside a secure container.
- **Persistence**: Mounts the agent's directory to `/workspace` inside the container.
- **Automatic Integration**: Used by `ShellSkill` and `McpSkill`.

## Setup (Recommended)

The default `python:3.11-slim` image is lightweight but lacks `npx` (Node.js) and `uvx` (Python tools), which are commonly used by MCP servers.

**We strongly recommend building the custom sandbox image:**

1.  Build the image from the provided Dockerfile:
    ```bash
    docker build -t webagents-sandbox - < webagents/agents/skills/local/sandbox/Dockerfile
    ```

2.  Configure your agent to use it:
    ```yaml
    sandbox:
      image: "webagents-sandbox"
    ```

## Configuration

In your `AGENT.md`:

```yaml
skills:
  - sandbox

sandbox:
  image: "webagents-sandbox" # Or "python:3.11-slim" (minimal)
```

## Tools

### `reset_sandbox`
Restarts the container, clearing any temporary state (installed packages, `/tmp` files) but preserving `/workspace`.

## Requirements
- Docker must be installed and running on the host system.
