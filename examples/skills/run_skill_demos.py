#!/usr/bin/env python3
"""
Skill Demo Server

Runs a server with all skill demo agents.
"""

import os
from pathlib import Path

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.local.auth import AuthSkill
from webagents.agents.skills.local.plugin import PluginSkill
from webagents.agents.skills.local.webui import WebUISkill
from webagents.server.core.app import WebAgentsServer


# Check for optional LSP
try:
    from webagents.agents.skills.local.lsp import LSPSkill
    HAS_LSP = True
except ImportError:
    HAS_LSP = False
    print("Note: LSP skill not available (install multilspy)")


def create_agents():
    """Create demo agents with various skills."""
    agents = []
    
    # Auth demo agent
    auth_agent = BaseAgent(
        name="auth-demo",
        instructions="You demonstrate authentication capabilities using AOAuth.",
        skills={"AuthSkill": AuthSkill(config={"mode": "self-issued"})},
        scopes=["all"]
    )
    agents.append(auth_agent)
    print(f"  ✅ auth-demo: AOAuth skill (self-issued mode)")
    
    # Plugin demo agent
    plugin_agent = BaseAgent(
        name="plugin-demo",
        instructions="You help users discover and manage plugins.",
        skills={"PluginSkill": PluginSkill(config={})},
        scopes=["all"]
    )
    agents.append(plugin_agent)
    print(f"  ✅ plugin-demo: Plugin skill (marketplace access)")
    
    # WebUI demo agent
    webui_agent = BaseAgent(
        name="webui-demo",
        instructions="You provide a web-based chat interface.",
        skills={"WebUISkill": WebUISkill(config={})},
        scopes=["all"]
    )
    agents.append(webui_agent)
    print(f"  ✅ webui-demo: WebUI skill (browser interface)")
    
    # LSP demo agent (if available)
    if HAS_LSP:
        lsp_agent = BaseAgent(
            name="lsp-demo",
            instructions="You provide code intelligence using LSP.",
            skills={"LSPSkill": LSPSkill(config={"workspace": str(Path.cwd())})},
            scopes=["all"]
        )
        agents.append(lsp_agent)
        print(f"  ✅ lsp-demo: LSP skill (code intelligence)")
    
    return agents


def main():
    """Run the skill demo server."""
    print("🎯 Skill Demo Server")
    print("=" * 50)
    
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY not set")
        print()
    
    print("📋 Creating demo agents...")
    agents = create_agents()
    
    print(f"\n🌐 Creating server with {len(agents)} agents...")
    server = WebAgentsServer(
        agents=agents,
        title="Skill Demo Server",
        description="Demonstrates AOAuth, Plugin, WebUI, and LSP skills"
    )
    
    print("\n✨ Available Agents:")
    for agent in agents:
        print(f"   http://localhost:8000/{agent.name}")
    
    print("\n💡 Example Requests:")
    print("   # Auth - get JWKS")
    print("   curl http://localhost:8000/auth-demo/.well-known/jwks.json")
    print()
    print("   # WebUI - open in browser")
    print("   open http://localhost:8000/webui-demo/ui")
    print()
    
    print("🔥 Starting server on http://localhost:8000")
    print("   Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        import uvicorn
        uvicorn.run(server.app, host="0.0.0.0", port=8000, log_level="info")
    except KeyboardInterrupt:
        print("\n👋 Server stopped!")


if __name__ == "__main__":
    main()
