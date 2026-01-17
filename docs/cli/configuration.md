# Configuration

Configure the CLI and agents via `AGENT.md` files.

## Global Settings

Global CLI settings are stored in `~/.webagents/config.yaml` (coming soon). Currently, configuration is primarily per-agent.

## Agent Configuration (`AGENT.md`)

```yaml
---
name: my-assistant
model: google/gemini-2.5-flash
skills:
  - filesystem
  - shell
  - sandbox
  - mcp

filesystem:
  whitelist:
    - ./data

mcp:
  sqlite:
    command: uvx
    args: ["mcp-server-sqlite", "--db-path", "./data.db"]
---

# Instructions

You are a helpful assistant...
```

## Environment Variables

- `GOOGLE_API_KEY`: For Google Gemini models.
- `OPENAI_API_KEY`: For OpenAI models.
- `ANTHROPIC_API_KEY`: For Claude models.
