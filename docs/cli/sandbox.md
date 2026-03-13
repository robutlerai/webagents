---
title: Sandbox
---
# Sandbox

The CLI integrates a secure Docker-based sandbox for executing untrusted code and tools.

## Enabling Sandbox

Add the `sandbox` skill to your `AGENT.md`:

```yaml
skills:
  - sandbox
  - shell
  - mcp
```

## How it Works

1.  **Docker Container**: A `python:3.11-slim` container is started in the background.
2.  **Mounting**: The agent's directory is mounted to `/workspace` inside the container.
3.  **Tool Routing**:
    *   **Shell**: `run_command` automatically executes inside the container.
    *   **MCP**: MCP servers (like `sqlite`) configured with `uvx` or `npx` run inside the container.

## Security

*   **Isolation**: Commands cannot access your host filesystem (except the mounted agent dir).
*   **Networking**: Container has network access (unless restricted by custom Docker config), but is isolated from host services.
*   **Persistence**: Changes to `/workspace` persist; system changes (apt-get install) are lost on restart.
