# MCP Skill

The MCP (Model Context Protocol) Skill enables agents to connect to and use external tools provided by MCP servers.

## Features

- **Protocol Support**: Connects via Stdio (CLI) or SSE (HTTP).
- **Tool Discovery**: Automatically discovers and registers tools from connected servers.
- **Sandbox Integration**: Can run MCP servers inside the Docker sandbox for security.

## Configuration

In your `AGENT.md` YAML header:

```yaml
skills:
  - mcp

mcp:
  # Map of server name to config
  sqlite:
    command: uvx
    args:
      - mcp-server-sqlite
      - --db-path
      - data.db
  
  filesystem:
    command: npx
    args:
      - -y
      - @modelcontextprotocol/server-filesystem
      - /path/to/allowed/dir
```

### Sandbox Support
If the `sandbox` skill is enabled for the agent, MCP servers configured with CLI commands will automatically execute inside the Docker container. Paths to files in the agent's directory are automatically mapped to `/workspace` paths.

## Tools

### `list_mcp_servers`
Lists connected servers and their available tools.

### Dynamic Tools
Tools provided by MCP servers (e.g., `sqlite__read_query`) are automatically registered as agent tools.
