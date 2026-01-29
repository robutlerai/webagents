# WebUI Skill

Serves the compiled React web UI for agent interaction in the browser.

## Overview

The WebUI skill mounts a React single-page application at `/ui` that provides:
- Agent list sidebar
- Chat interface with markdown rendering
- Agent details panel
- Real-time communication with the daemon

## Requirements

The React app must be built before the skill can serve it:

```bash
# Build the WebUI
webagents ui --build

# Or manually
cd webagents/cli/webui
pnpm install
pnpm build
```

## Usage

### Adding to an Agent

```python
from webagents.agents.skills.local.webui import WebUISkill

agent = Agent(
    name="my-agent",
    skills=[
        WebUISkill(config={"title": "My Agent Dashboard"})
    ]
)
```

### Commands

| Command | Description |
|---------|-------------|
| `/ui` | Get the URL for the web UI |
| `/ui/status` | Check WebUI build and mount status |

### Accessing the UI

Once the daemon is running with the WebUI skill:

```bash
# Start daemon
webagents daemon start

# Open browser to
http://localhost:8765/ui
```

## Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `title` | string | "WebAgents Dashboard" | Display title for the UI |

## Architecture

```
webagents/
├── cli/
│   └── webui/           # React application source
│       ├── src/         # React components, hooks, etc.
│       ├── dist/        # Built output (gitignored)
│       └── package.json
└── agents/
    └── skills/
        └── local/
            └── webui/
                └── skill.py  # This skill
```

The skill serves:
- `/ui/assets/*` - Static files (JS, CSS, images)
- `/ui/*` - React SPA (all routes return index.html)

## Development

For development with hot reload:

```bash
# Terminal 1: Start daemon
webagents daemon start --dev

# Terminal 2: Start Vite dev server
webagents ui --port 5173

# Or use the script
./scripts/dev-webui.sh
```

The Vite dev server proxies API requests to the daemon at localhost:8765.

## Troubleshooting

### "WebUI dist not found"

Build the React app:

```bash
webagents ui --build
```

### "Agent has no app attribute"

The skill requires the agent to have a Starlette `app` attribute. This is 
automatically provided when running through webagentsd.

### Assets 404

Ensure the build completed successfully and `dist/assets/` exists:

```bash
ls webagents/cli/webui/dist/assets/
```
