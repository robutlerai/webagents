# xAI (Grok) Skill

This skill provides integration with xAI's Grok models, using the OpenAI-compatible API.

## Features

- **Grok Support**: Access `grok-beta`, `grok-2`, etc.
- **Streaming Support**: Full streaming support.
- **Tool Use**: Supports standard function calling tools.
- **Handoffs**: Registers as a handoff target (`xai_grok_beta...`).
- **Configurable**: Fully configurable via `AGENT.md` metadata.

## Configuration

Add to your `AGENT.md`:

```yaml
skills:
  - xai

xai:
  model: "grok-beta" # Default
  api_key: "xai-..." # Optional if set in env vars
```

## Environment Variables

- `XAI_API_KEY`: Your xAI API key.
