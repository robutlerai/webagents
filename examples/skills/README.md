# Skill Examples

This directory contains example agents demonstrating various WebAgents skills.

## Agents

| Agent | Skill | Description |
|-------|-------|-------------|
| `auth-demo` | AuthSkill | AOAuth authentication (self-issued JWT) |
| `plugin-demo` | PluginSkill | Plugin marketplace and management |
| `webui-demo` | WebUISkill | Browser-based chat interface |
| `lsp-demo` | LSPSkill | Code intelligence (requires multilspy) |

## Quick Start

### Run All Demos

```bash
cd /Users/vs/dev/webagents
python examples/skills/run_skill_demos.py
```

### Individual Agent Files

The AGENT-*.md files can be loaded by webagentsd:

```bash
cd examples/skills
webagentsd start
```

## Skills Overview

### AuthSkill (AOAuth)

Provides OAuth 2.0 authentication for agent-to-agent communication:

- **Self-issued mode**: Agent generates its own JWT tokens
- **Portal mode**: Uses centralized Robutler Portal for auth

Endpoints:
- `/.well-known/jwks.json` - Public keys for token verification
- `/oauth/token` - Token generation endpoint

### PluginSkill

Extends agent capabilities with plugins from the marketplace:

Commands:
- `/plugin list` - Show installed plugins
- `/plugin search <query>` - Search marketplace
- `/plugin install <name>` - Install a plugin

### WebUISkill

Serves a React-based chat interface:

- Mounts at `/ui` endpoint
- Supports streaming responses
- Requires building the UI first: `pnpm build` in `cli/webui/`

### LSPSkill

Provides code intelligence using Language Server Protocol:

Tools:
- `goto_definition` - Jump to symbol definition
- `find_references` - Find all usages
- `get_hover` - Show documentation
- `get_completions` - Autocomplete
- `get_document_symbols` - List symbols

Requires: `pip install multilspy`

## Testing

Run the integration tests:

```bash
pytest tests/test_skill_integration.py -v
```
