#!/bin/bash
# WebAgents Web UI Production Build
#
# Builds the React web UI for production deployment.
# Output goes to webagents/cli/webui/dist/
#
# Usage:
#   ./scripts/build-webui.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UI_DIR="$SCRIPT_DIR/../webagents/cli/webui"

cd "$UI_DIR"

# Check for pnpm
if ! command -v pnpm &> /dev/null; then
    echo "Error: pnpm is not installed"
    echo "Install with: npm install -g pnpm"
    exit 1
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           WebAgents Web UI Production Build                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Install dependencies
echo "Installing dependencies..."
pnpm install

# Build
echo ""
echo "Building for production..."
pnpm build

# Show results
echo ""
echo "Build complete!"
echo ""
echo "Output directory: $UI_DIR/dist"
echo ""
echo "Contents:"
ls -la dist/

if [ -d "dist/assets" ]; then
    echo ""
    echo "Assets:"
    ls -la dist/assets/
fi

echo ""
echo "The WebUI will be served at /ui when webagentsd is running."
echo "Start the daemon with: webagents daemon start"
