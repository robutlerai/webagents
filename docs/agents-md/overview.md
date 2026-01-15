# AGENT.md Format

Define agents using Markdown files with YAML frontmatter.

## File Naming

| File | Purpose |
|------|---------|
| `AGENT.md` | Default agent in a directory |
| `AGENT-<name>.md` | Named agent (e.g., `AGENT-planner.md`) |
| `AGENTS.md` | Context inherited by all agents |

## Basic Structure

An agent file has two parts:

1. **YAML Frontmatter** - Configuration and metadata
2. **Markdown Body** - Instructions for the agent

```markdown
---
name: my-agent
description: A helpful assistant
namespace: local
model: openai/gpt-4o-mini
intents:
  - answer questions
  - help with tasks
skills: []
visibility: local
---

# My Agent

You are a helpful AI assistant.

## Capabilities

- Answer questions
- Help with tasks

## Guidelines

- Be helpful and concise
- Ask for clarification when needed
```

## YAML Schema

### Identity

```yaml
name: string          # Agent name (required)
description: string   # Human-readable description
namespace: string     # Namespace (default: "local")
```

### Discovery

```yaml
intents:              # What this agent can do
  - summarize documents
  - create reports
  - parse PDFs
```

### Configuration

```yaml
model: string         # LLM model (default: "openai/gpt-4o-mini")
skills:               # Installed skills
  - cron
  - folder-index
  - mcp
tools: []             # Additional tools
mcp_servers: []       # MCP server configurations
```

### Triggers

```yaml
cron: string          # Cron schedule (e.g., "0 9 * * 1-5")
watch:                # File patterns to watch
  - "*.md"
  - "data/*.json"
```

### Visibility

```yaml
visibility: string    # local | namespace | public
```

### Sandbox

```yaml
sandbox:
  preset: development  # strict | development | unrestricted
  allowed_folders:
    - "."
    - "./data"
  allowed_commands:
    - "git status"
```

## Full Example

```yaml
---
name: weekly-reporter
description: Generates weekly status reports
namespace: ai.myorg.planning
model: openai/gpt-4o
intents:
  - generate weekly reports
  - summarize progress
  - create status updates
skills:
  - cron
  - folder-index
  - memory
cron: "0 18 * * 5"
watch:
  - "logs/*.md"
visibility: namespace
sandbox:
  preset: development
  allowed_folders:
    - "."
    - "../shared"
---

# Weekly Reporter

You are a weekly status report generator.

## Responsibilities

1. Read progress logs from `logs/` directory
2. Summarize key accomplishments
3. Identify blockers and risks
4. Generate formatted report

## Report Format

Use this structure:

### This Week's Accomplishments
- [list items]

### In Progress
- [list items]

### Blockers
- [list items]

### Next Week's Goals
- [list items]

## Guidelines

- Be concise but comprehensive
- Highlight important achievements
- Flag any concerns early
- Include relevant metrics when available
```

## Default Agent Resolution

When you run `webagents connect` without specifying an agent:

1. If `AGENT.md` exists → use it
2. If only one `AGENT-*.md` exists → use it
3. If multiple `AGENT-*.md` files → error (must specify)

## Skills Reference

Common skills to add:

| Skill | Description |
|-------|-------------|
| `cron` | Scheduled execution |
| `folder-index` | Vector indexing (sqlite-vec) |
| `llm` | LLM provider integration |
| `mcp` | Model Context Protocol |
| `memory` | Short/long-term memory |
| `discovery` | Agent discovery |
| `web` | Web fetch and search |

Install skills with:

```bash
webagents skill install cron folder-index
```

## Next Steps

- [Context Hierarchy](hierarchy.md) - AGENTS.md inheritance
- [Templates](templates.md) - Create agents from templates
- [Skills](../skills/overview.md) - Available skills
