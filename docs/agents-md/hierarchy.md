# Context Hierarchy

AGENTS.md files provide inherited context for agents.

## Overview

```
project/
├── AGENTS.md              # Root context
├── AGENT.md               # Inherits from root AGENTS.md
├── AGENT-planner.md       # Inherits from root AGENTS.md
└── subproject/
    ├── AGENTS.md          # Inherits from parent AGENTS.md
    └── AGENT.md           # Inherits from both AGENTS.md files
```

## How Inheritance Works

### AGENTS.md (Context Files)

- Provide shared context for all agents in a directory
- Settings cascade down the directory tree
- Child directories can override parent settings

### AGENT*.md (Agent Definitions)

- Define individual agents
- Inherit from AGENTS.md in same and parent directories
- Agent-specific settings take precedence

## Inheritance Rules

### Namespace

Overridden by child (most local wins):

```yaml
# /project/AGENTS.md
namespace: ai.myorg

# /project/subproject/AGENTS.md  
namespace: ai.myorg.team  # This is used for subproject agents
```

### Model

Overridden by child:

```yaml
# AGENTS.md
model: openai/gpt-4o

# AGENT.md
model: anthropic/claude-3-opus  # Agent uses Claude
```

### Skills, Tools, MCP Servers

Accumulated (combined from all levels):

```yaml
# /project/AGENTS.md
skills:
  - memory

# /project/sub/AGENTS.md
skills:
  - cron

# /project/sub/AGENT.md
skills:
  - mcp

# Result: agent has [memory, cron, mcp]
```

### Instructions

Combined with context first, agent second:

```
Result:
1. Root AGENTS.md content
2. Sub AGENTS.md content  
3. AGENT.md instructions
```

## Example

### Directory Structure

```
myproject/
├── AGENTS.md
├── AGENT.md
└── tools/
    ├── AGENTS.md
    └── AGENT-linter.md
```

### /myproject/AGENTS.md

```yaml
---
namespace: ai.myorg.myproject
model: openai/gpt-4o
skills:
  - memory
---

# Project Context

This is a TypeScript project using React.
All agents should follow our coding standards.
```

### /myproject/AGENT.md

```yaml
---
name: assistant
description: General project assistant
intents:
  - help with the project
  - answer questions
skills:
  - mcp
---

# Project Assistant

You help developers work on this project.
```

**Merged Result:**
- namespace: `ai.myorg.myproject`
- model: `openai/gpt-4o`
- skills: `[memory, mcp]`
- instructions: Project context + assistant instructions

### /myproject/tools/AGENTS.md

```yaml
---
namespace: ai.myorg.myproject.tools
skills:
  - cron
---

# Tools Context

These are automated tools that run on schedules.
```

### /myproject/tools/AGENT-linter.md

```yaml
---
name: linter
description: Automated code linter
cron: "0 * * * *"
intents:
  - lint code
  - fix style issues
---

# Linter Agent

Run ESLint and fix issues automatically.
```

**Merged Result:**
- namespace: `ai.myorg.myproject.tools` (from tools/AGENTS.md)
- model: `openai/gpt-4o` (from root AGENTS.md)
- skills: `[memory, cron]` (accumulated)
- instructions: All three context files + linter

## Creating Context Files

```bash
# Create AGENTS.md in current directory
webagents init --context
```

## Best Practices

1. **Use root AGENTS.md for project-wide settings**
   - Default model
   - Shared namespace
   - Common skills

2. **Use subdirectory AGENTS.md for team/feature context**
   - Team-specific namespace
   - Feature-specific skills

3. **Keep agent-specific details in AGENT*.md**
   - Individual intents
   - Specific instructions
   - Cron schedules

4. **Don't repeat shared context in agent files**
   - Let inheritance handle it
   - Override only when needed

## Debugging Hierarchy

See what an agent inherits:

```bash
# View agent with merged context
webagents info agent-name
```

Or load programmatically:

```python
from webagents.loader import AgentLoader

loader = AgentLoader()
merged = loader.load(Path("./AGENT.md"))

print(f"Namespace: {merged.metadata.namespace}")
print(f"Skills: {merged.metadata.skills}")
print(f"Context files: {[str(c.path) for c in merged.context_files]}")
```
