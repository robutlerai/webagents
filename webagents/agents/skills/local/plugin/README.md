# Plugin Skill

Claude Code compatible plugin system with marketplace discovery, fuzzy search, and dynamic tool registration.

## Overview

The Plugin Skill enables WebAgents to discover, install, and manage plugins from the Claude Marketplaces ecosystem. It provides:

- **Marketplace Discovery** - Search and browse plugins from claudemarketplaces.com
- **Fuzzy Search** - Find plugins by name, description, or keywords with typo tolerance
- **GitHub Star Ranking** - Results ranked by popularity and relevance
- **Claude Code Compatibility** - Works with existing Claude Code plugins
- **Dynamic Tool Registration** - Plugin tools automatically registered with agent

## Quick Start

### Enable Plugin Skill

```python
from webagents.agents.skills.local.plugin import PluginSkill

# Add to agent skills
agent = BaseAgent(
    skills=[PluginSkill()],
    # ... other config
)
```

### Using Plugin Commands

```bash
# Search for plugins
/plugin/search code review

# Install a plugin
/plugin/install code-review

# List installed plugins
/plugin/list

# Get plugin info
/plugin/info code-review

# Enable/disable plugins
/plugin/disable code-review
/plugin/enable code-review

# Refresh marketplace index
/plugin/refresh
```

## Plugin Format

Plugins use the Claude Code `plugin.json` format:

```json
{
  "name": "my-plugin",
  "version": "1.0.0",
  "description": "My awesome plugin",
  "author": "Your Name",
  "license": "MIT",
  "commands": "./commands/",
  "skills": "./skills/",
  "agents": "./agents/",
  "hooks": "./hooks/hooks.json",
  "mcpServers": "./.mcp.json",
  "dependencies": ["requests>=2.28"],
  "keywords": ["utility", "automation"],
  "repository": "https://github.com/user/my-plugin"
}
```

### Directory Structure

```
my-plugin/
├── plugin.json          # Plugin manifest
├── commands/            # Python command scripts
│   ├── analyze.py
│   └── report.py
├── skills/              # SKILL.md files
│   └── review.md
├── hooks/               # Hook configurations
│   ├── hooks.json
│   └── on_message.py
└── .mcp.json           # MCP server configs (optional)
```

## SKILL.md Format

Skills use Markdown with YAML frontmatter:

```markdown
---
name: code-review
description: Review code for issues and improvements
disable-model-invocation: false
allowed-tools: [read_file, write_file]
context: inline
---

# Code Review Skill

Review the file at `$ARGUMENTS.file_path` for:

1. Code quality issues
2. Security vulnerabilities  
3. Performance improvements

Focus on: $ARGUMENTS.focus_areas
```

### Frontmatter Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `name` | string | filename | Skill identifier |
| `description` | string | "" | Human-readable description |
| `disable-model-invocation` | bool | false | Run without LLM |
| `allowed-tools` | list | null | Tool whitelist for forked execution |
| `context` | string | "inline" | Execution mode: "inline" or "fork" |

### Argument Substitution

Use `$ARGUMENTS.key` or `$ARGUMENTS['key']` for dynamic values:

```markdown
Review the repository at $ARGUMENTS.repo_path

Options:
- Include tests: $ARGUMENTS['include_tests']
- Max depth: $ARGUMENTS.depth
```

## Commands

Plugin commands are Python scripts in the `commands/` directory:

```python
# commands/analyze.py
"""Analyze code for issues."""

import json
import os

def run(arguments):
    """Main entry point. Receives arguments dict."""
    file_path = arguments.get("file_path")
    
    # Do analysis...
    results = analyze_file(file_path)
    
    # Return JSON for structured output
    return {
        "issues": results,
        "file": file_path
    }

if __name__ == "__main__":
    # For subprocess execution
    args = json.loads(os.environ.get("PLUGIN_ARGUMENTS", "{}"))
    result = run(args)
    print(json.dumps(result))
```

## Hooks

Hooks allow plugins to respond to agent lifecycle events:

```json
// hooks/hooks.json
{
  "hooks": [
    {
      "event": "on_message",
      "handler": "./on_message.py",
      "priority": 50,
      "enabled": true
    }
  ]
}
```

```python
# hooks/on_message.py
async def run(event_data):
    """Handle message event."""
    message = event_data.get("message")
    
    # Process message...
    
    return {
        "processed": True,
        "modified_message": message
    }
```

## Marketplace API

The skill fetches plugins from `https://claudemarketplaces.com/api/marketplaces`.

### Search Algorithm

1. **Fuzzy Match** - Uses rapidfuzz WRatio for typo tolerance
2. **Star Boost** - GitHub stars provide log-scale ranking boost
3. **Combined Score** - `rank = match_score + log10(stars + 1) * 10`

### Caching

- Index cached to `~/.webagents/plugin_cache/marketplace_index.json`
- Refreshes every 6 hours in background
- Manual refresh with `/plugin/refresh`

## Configuration

```python
PluginSkill(config={
    # GitHub token for higher API rate limits
    "github_token": "ghp_...",
    
    # Custom plugins directory
    "plugins_dir": "/path/to/plugins",
    
    # Disable background refresh
    "auto_refresh": False,
})
```

## API Reference

### PluginSkill

Main skill class with commands:

| Command | Description | Scope |
|---------|-------------|-------|
| `/plugin` | Show help | all |
| `/plugin/list` | List installed plugins | all |
| `/plugin/search <query>` | Search marketplace | all |
| `/plugin/install <name>` | Install plugin | owner |
| `/plugin/uninstall <name>` | Uninstall plugin | owner |
| `/plugin/enable <name>` | Enable plugin | owner |
| `/plugin/disable <name>` | Disable plugin | owner |
| `/plugin/info <name>` | Show plugin info | all |
| `/plugin/refresh` | Refresh marketplace | all |

### PluginLoader

```python
from webagents.agents.skills.local.plugin import PluginLoader

loader = PluginLoader()

# Load from local path
plugin = loader.load_local(Path("./my-plugin"))

# Install from Git
plugin = await loader.install_from_repo("https://github.com/user/plugin")

# List installed
plugins = loader.list_installed()

# Uninstall
loader.uninstall("plugin-name")
```

### MarketplaceClient

```python
from webagents.agents.skills.local.plugin import MarketplaceClient

client = MarketplaceClient(github_token="...")

# Load cached index
client.load_cached_index()

# Refresh from API
await client.refresh_index()

# Search plugins
results = client.search("code review", limit=10)

# Get by name
plugin = client.get("code-review")

# Get completions for autocomplete
names = client.get_completions()
```

### SkillRunner

```python
from webagents.agents.skills.local.plugin.components import SkillRunner

runner = SkillRunner()

# Parse SKILL.md
skill = runner.parse(Path("./skill.md"))

# Substitute arguments
content = runner.substitute_arguments(
    skill.content,
    {"file_path": "/src/main.py", "focus_areas": "security"}
)

# Execute skill
result = await runner.execute(skill, arguments, agent)
```

## Dependencies

```
rapidfuzz>=3.0     # Fuzzy search
pyyaml>=6.0        # YAML frontmatter parsing
gitpython>=3.1     # Git repository installation
httpx>=0.25        # HTTP client for marketplace API
```

## Examples

### Creating a Plugin

```bash
mkdir my-plugin && cd my-plugin

# Create manifest
cat > plugin.json << 'EOF'
{
  "name": "my-plugin",
  "version": "1.0.0",
  "description": "My first plugin",
  "commands": "./commands/",
  "skills": "./skills/"
}
EOF

# Create a command
mkdir commands
cat > commands/hello.py << 'EOF'
"""Say hello."""
def run(args):
    name = args.get("name", "World")
    return {"message": f"Hello, {name}!"}
EOF

# Create a skill
mkdir skills
cat > skills/greet.md << 'EOF'
---
name: greet
description: Greet the user
---
Please greet $ARGUMENTS.name warmly.
EOF
```

### Installing from Local Path

```python
plugin = await loader.install_from_path(
    Path("./my-plugin"),
    copy=True  # Copy to plugins directory
)
```

### Custom Marketplace

```python
# Point to alternative marketplace
client = MarketplaceClient()
# Modify MARKETPLACE_API constant or subclass
```
