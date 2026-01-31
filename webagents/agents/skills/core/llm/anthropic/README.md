# Anthropic Skill

This skill provides native integration with Anthropic's Claude models via the official SDK.

## Features

- **Native SDK Integration**: Uses `anthropic` Python SDK for robust performance.
- **Streaming Support**: Full streaming support with real-time token output.
- **Tool Use**: Supports dynamic tools and built-in Anthropic tools (Computer Use, Bash, Text Editor).
- **Handoffs**: Registers as a handoff target (`anthropic_claude_3_5_sonnet...`) for seamless model switching.
- **Configurable**: Fully configurable via `AGENT.md` metadata.

## Configuration

Add to your `AGENT.md`:

```yaml
skills:
  - anthropic

anthropic:
  model: "claude-3-5-sonnet-20241022" # Default
  api_key: "sk-ant-..." # Optional if set in env vars
  anthropic_tools: # Enable built-in tools
    - "computer_use"
    - "bash"
    - "text_editor"
```

## Environment Variables

- `ANTHROPIC_API_KEY`: Your Anthropic API key.
