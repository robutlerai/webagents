# WebAgents CLI

The WebAgents CLI provides a powerful terminal interface for interacting with and managing your AI agents.

## Overview

The CLI allows you to:
- Chat with agents in a rich, interactive REPL.
- Manage agent sessions and checkpoints.
- Execute tools and skills securely (including Docker sandboxing).
- Configure agent behaviors and providers.

## Getting Started

To start the CLI with the default agent:

```bash
webagents
```

To start with a specific agent:

```bash
webagents /path/to/AGENT.md
```

## Features

- **Interactive REPL**: Rich text editing, history, and autocomplete.
- **Streaming**: Real-time response streaming with visible "thinking" blocks.
- **Tool Execution**: See tool calls and results as they happen.
- **Sandboxing**: Secure execution environment.
