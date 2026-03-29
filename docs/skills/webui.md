---
title: WebUI Skill
---
# WebUI Skill

Serves a compiled React web UI for agent interaction in the browser.

## Overview

The WebUI skill mounts a React single-page application at `/ui` that provides:

- **Agent List Sidebar** - Browse and switch between available agents
- **Chat Interface** - Conversational UI with markdown rendering
- **Agent Details** - View agent configuration and status
- **Real-time Communication** - Live updates via daemon connection

## Requirements

The React app must be built before the skill can serve it:

```bash
# Build using CLI
webagents ui --build

# Or manually
cd webagents/cli/webui
pnpm install
pnpm build
```

## Configuration

```python
from webagents.agents.skills.local.webui import WebUISkill

agent = BaseAgent(
    name="my-agent",
    skills={
        "webui": WebUISkill(config={"title": "My Agent Dashboard"}),
    },
)
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `title` | string | "WebAgents Dashboard" | Display title for the UI |

## Commands

| Command | Description |
|---------|-------------|
| `/ui` | Get the URL for the web UI |
| `/ui/status` | Check WebUI build and mount status |

### /ui

Returns the URL to access the web interface:

```
/ui

**WebAgents Dashboard:** http://localhost:8765/ui
```

### /ui/status

Returns detailed status information:

```json
{
  "mounted": true,
  "dist_path": "/path/to/webagents/cli/webui/dist",
  "dist_exists": true,
  "index_exists": true,
  "assets_exist": true,
  "build_time": "2024-01-15T10:30:00"
}
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `webagents ui` | Start development server |
| `webagents ui --build` | Build production assets |
| `webagents ui --port 5173` | Use custom port for dev server |

### Development Mode

```bash
# Terminal 1: Start daemon
webagents daemon start --dev

# Terminal 2: Start Vite dev server with hot reload
webagents ui --port 5173
```

The Vite dev server proxies API requests to the daemon at `localhost:8765`.

### Production Build

```bash
# Build assets
webagents ui --build

# Restart daemon to serve built assets
webagents daemon restart
```

## Accessing the UI

Once the daemon is running with built assets:

```bash
# Start daemon (if not running)
webagents daemon start

# Open browser
open http://localhost:8765/ui
```

Or from the REPL:

```
> /ui
**WebAgents Dashboard:** http://localhost:8765/ui
```

## Architecture

```
webagents/
├── cli/
│   └── webui/           # React application source
│       ├── src/         # React components, hooks, etc.
│       │   ├── components/
│       │   ├── hooks/
│       │   └── App.tsx
│       ├── dist/        # Built output (gitignored)
│       └── package.json
└── agents/
    └── skills/
        └── local/
            └── webui/
                └── skill.py  # WebUI skill
```

### Routes Served

| Route | Content |
|-------|---------|
| `/ui` | React SPA (index.html) |
| `/ui/*` | React SPA (SPA routing) |
| `/ui/assets/*` | Static files (JS, CSS, images) |

All `/ui/*` routes return `index.html` to support client-side routing.

## React Application

The web UI is built with:

- **React 18** - UI framework
- **Vite** - Build tool and dev server
- **TailwindCSS** - Styling
- **React Query** - Data fetching
- **React Router** - Client-side routing

### Features

- Markdown rendering with syntax highlighting
- Dark/light theme support
- Responsive design
- Keyboard shortcuts
- Session persistence

## Development

### Prerequisites

```bash
# Install pnpm if needed
npm install -g pnpm

# Install dependencies
cd webagents/cli/webui
pnpm install
```

### Development Workflow

```bash
# Start Vite dev server (hot reload)
pnpm dev

# Run type checking
pnpm typecheck

# Run linting
pnpm lint

# Build for production
pnpm build

# Preview production build
pnpm preview
```

### Environment Variables

Create `.env.local` for development:

```bash
# API endpoint (defaults to localhost:8765)
VITE_API_URL=http://localhost:8765
```

## Troubleshooting

### "WebUI dist not found"

Build the React app:

```bash
webagents ui --build
```

### "Agent has no app attribute"

The skill requires the agent to have a Starlette `app` attribute. This is automatically provided when running through webagentsd or using:

```python
agent = BaseAgent(name="agent", enable_http=True)
```

### Assets 404

Ensure the build completed successfully:

```bash
ls webagents/cli/webui/dist/assets/
```

Should show JavaScript and CSS files.

### Blank Page

Check browser console for errors. Common issues:
- API endpoint not accessible
- CORS configuration
- Build errors

### Port Already in Use

```bash
# Check what's using the port
lsof -i :8765

# Use different port
webagents daemon start --port 8766
```

## API Integration

The WebUI communicates with the daemon via:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/agents` | GET | List agents |
| `/api/agents/{id}` | GET | Get agent details |
| `/api/agents/{id}/chat` | POST | Send message |
| `/api/agents/{id}/stream` | SSE | Stream responses |

See the [Server API documentation](../server/index.md) for details.
