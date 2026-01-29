#!/bin/bash
# WebAgents Web UI Development Server
#
# Starts the Vite development server for the React web UI.
# The dev server proxies API requests to the webagentsd daemon.
#
# Usage:
#   ./scripts/dev-webui.sh
#   ./scripts/dev-webui.sh 3000  # Custom port

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UI_DIR="$SCRIPT_DIR/../webagents/cli/webui"
PORT="${1:-5173}"

cd "$UI_DIR"

# Check for pnpm
if ! command -v pnpm &> /dev/null; then
    echo "Error: pnpm is not installed"
    echo "Install with: npm install -g pnpm"
    exit 1
fi

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    pnpm install
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           WebAgents Web UI Development Server                ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
echo "║  Local:   http://localhost:$PORT/ui                           ║"
echo "║  API:     http://localhost:8765 (proxy)                      ║"
echo "║                                                              ║"
echo "║  Press Ctrl+C to stop                                        ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Start dev server
pnpm dev --port "$PORT"
