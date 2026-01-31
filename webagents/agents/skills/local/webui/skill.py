"""
WebUI Skill

Serves compiled React web UI for agent dashboard.
The React app is built from cli/webui/ and served as static files.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ...base import Skill
from webagents.agents.tools.decorators import command

# Path to compiled React app (relative to this file)
# webagents/agents/skills/local/webui/skill.py -> webagents/cli/webui/dist/
DIST_DIR = Path(__file__).parent.parent.parent.parent / "cli" / "webui" / "dist"

logger = logging.getLogger(__name__)


class WebUISkill(Skill):
    """Serve compiled React web UI for the agent.
    
    This skill mounts the built React application at /ui and serves
    static assets. It provides a browser-based interface for agent
    interaction with markdown rendering, chat history, and agent details.
    
    The React app must be built first with:
        cd webagents/cli/webui && pnpm install && pnpm build
    
    Or use the CLI:
        webagents ui --build
    
    Configuration:
        title: Display title for the UI (default: "WebAgents Dashboard")
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {}, scope="all")
        self.title = self.config.get("title", "WebAgents Dashboard")
        self._mounted = False
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self, agent) -> None:
        """Initialize skill and mount static files if dist exists."""
        await super().initialize(agent)
        
        # Check if agent has a Starlette app to mount routes on
        if not hasattr(agent, "app"):
            self.logger.debug("Agent has no app attribute, skipping WebUI mount")
            return
        
        # Check if dist directory exists
        if not DIST_DIR.exists():
            self.logger.warning(
                f"WebUI dist not found at {DIST_DIR}. "
                "Run 'pnpm build' in cli/webui/ or 'webagents ui --build' to build."
            )
            return
        
        assets_dir = DIST_DIR / "assets"
        index_file = DIST_DIR / "index.html"
        
        if not index_file.exists():
            self.logger.warning(f"index.html not found in {DIST_DIR}")
            return
        
        try:
            from starlette.staticfiles import StaticFiles
            from starlette.responses import FileResponse
            
            # Mount static assets (JS, CSS, images)
            if assets_dir.exists():
                agent.app.mount(
                    "/ui/assets",
                    StaticFiles(directory=str(assets_dir)),
                    name="webui_assets"
                )
                self.logger.debug(f"Mounted WebUI assets from {assets_dir}")
            
            # Serve index.html for all /ui/* routes (SPA routing)
            @agent.app.get("/ui")
            @agent.app.get("/ui/{path:path}")
            async def serve_ui(path: str = ""):
                """Serve React SPA - all routes return index.html."""
                return FileResponse(
                    str(index_file),
                    media_type="text/html"
                )
            
            self._mounted = True
            self.logger.info(f"WebUI mounted at /ui (serving from {DIST_DIR})")
            
        except ImportError as e:
            self.logger.error(f"Failed to import Starlette components: {e}")
        except Exception as e:
            self.logger.error(f"Failed to mount WebUI: {e}")
    
    @command("/ui", description="Open agent web UI in browser")
    async def open_ui(self) -> Dict[str, Any]:
        """Get the URL for the agent's web UI.
        
        Returns:
            URL to the web UI and display message.
        """
        # Try to get base URL from agent
        base_url = getattr(self.agent, "base_url", None)
        if not base_url:
            # Try to get from context
            context = self.get_context()
            if context and hasattr(context, "request"):
                request = context.request
                if request:
                    scheme = request.url.scheme
                    host = request.url.netloc
                    base_url = f"{scheme}://{host}"
        
        # Fallback to localhost
        if not base_url:
            base_url = "http://localhost:8765"
        
        ui_url = f"{base_url}/ui"
        
        if not self._mounted:
            return {
                "url": ui_url,
                "mounted": False,
                "display": (
                    f"**Web UI not available**\n\n"
                    f"The React app has not been built. Run:\n"
                    f"```bash\n"
                    f"webagents ui --build\n"
                    f"```\n\n"
                    f"Then restart the daemon."
                )
            }
        
        return {
            "url": ui_url,
            "mounted": True,
            "display": f"**{self.title}:** {ui_url}"
        }
    
    @command("/ui/status", description="Check WebUI status")
    async def ui_status(self) -> Dict[str, Any]:
        """Get WebUI status including build state.
        
        Returns:
            Status information about the WebUI.
        """
        has_dist = DIST_DIR.exists()
        has_index = (DIST_DIR / "index.html").exists() if has_dist else False
        has_assets = (DIST_DIR / "assets").exists() if has_dist else False
        
        status = {
            "mounted": self._mounted,
            "dist_path": str(DIST_DIR),
            "dist_exists": has_dist,
            "index_exists": has_index,
            "assets_exist": has_assets,
        }
        
        if has_dist and has_index:
            # Get build info
            try:
                index_stat = (DIST_DIR / "index.html").stat()
                from datetime import datetime
                status["build_time"] = datetime.fromtimestamp(index_stat.st_mtime).isoformat()
            except Exception:
                pass
        
        return status
