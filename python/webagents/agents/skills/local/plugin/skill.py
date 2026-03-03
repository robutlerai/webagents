"""
Plugin Skill - Plugin Management Commands

Provides slash commands for plugin management:
- /plugin - Show help
- /plugin/list - List installed plugins
- /plugin/search - Search marketplace
- /plugin/install - Install from marketplace or URL
- /plugin/enable - Enable plugin
- /plugin/disable - Disable plugin
- /plugin/info - Show plugin details
- /plugin/refresh - Refresh marketplace index
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import command
from .loader import PluginLoader, Plugin
from .marketplace import MarketplaceClient
from .executor import PluginExecutor

logger = logging.getLogger(__name__)


class PluginSkill(Skill):
    """Plugin management with Claude Code compatibility.
    
    Provides commands for:
    - Discovering plugins from marketplace
    - Installing/uninstalling plugins
    - Managing plugin state (enable/disable)
    - Viewing plugin information
    
    Features:
    - Fuzzy search with rapidfuzz
    - GitHub star ranking
    - Background marketplace refresh
    - Claude Code plugin.json compatibility
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize plugin skill.
        
        Args:
            config: Configuration with optional keys:
                - github_token: GitHub token for API access
                - plugins_dir: Custom plugins directory
                - auto_refresh: Enable background refresh (default: True)
        """
        super().__init__(config)
        self.marketplace = MarketplaceClient(
            github_token=self.config.get("github_token")
        )
        self.loader = PluginLoader(self.config)
        self.executor = PluginExecutor(self.config)
        self._plugins: Dict[str, Plugin] = {}
        self._refresh_task: Optional[asyncio.Task] = None
        self._auto_refresh = self.config.get("auto_refresh", True)
    
    async def initialize(self, agent) -> None:
        """Initialize skill with agent reference.
        
        Loads cached marketplace index, starts background refresh,
        and loads installed plugins.
        """
        await super().initialize(agent)
        
        # Load cached marketplace index
        self.marketplace.load_cached_index()
        
        # Start background refresh if enabled
        if self._auto_refresh:
            self._refresh_task = asyncio.create_task(self._periodic_refresh())
        
        # Load installed plugins
        await self._load_installed_plugins()
        
        logger.info(f"PluginSkill initialized: {len(self._plugins)} plugins loaded")
    
    async def _periodic_refresh(self) -> None:
        """Periodically refresh marketplace index in background."""
        while True:
            if self.marketplace.needs_refresh():
                try:
                    await self.marketplace.refresh_index()
                    logger.info("Marketplace index refreshed in background")
                except Exception as e:
                    logger.warning(f"Background marketplace refresh failed: {e}")
            
            # Sleep for 1 hour between checks
            await asyncio.sleep(3600)
    
    async def _load_installed_plugins(self) -> None:
        """Load all installed plugins from disk."""
        for plugin in self.loader.list_installed():
            self._plugins[plugin.name] = plugin
            
            # Register plugin tools with agent
            for tool in plugin.get_tools():
                logger.debug(f"Registered plugin tool: {tool['name']}")
    
    def _get_subcommand_completions(self) -> Dict[str, List[str]]:
        """Return subcommands for autocomplete."""
        return {"subcommand": ["list", "search", "install", "enable", "disable", "info", "refresh", "uninstall"]}
    
    @command("/plugin", description="Plugin management commands", scope="all",
             completions=lambda self: self._get_subcommand_completions())
    async def plugin_help(self, subcommand: str = None) -> Dict[str, Any]:
        """Show plugin help or subcommand info.
        
        Args:
            subcommand: Optional subcommand for specific help
            
        Returns:
            Help information
        """
        subcommands = {
            "list": {"description": "List installed plugins", "usage": "/plugin/list"},
            "search": {"description": "Search marketplace for plugins", "usage": "/plugin/search <query>"},
            "install": {"description": "Install plugin from marketplace or URL", "usage": "/plugin/install <name|url>"},
            "enable": {"description": "Enable a disabled plugin", "usage": "/plugin/enable <name>"},
            "disable": {"description": "Disable an enabled plugin", "usage": "/plugin/disable <name>"},
            "info": {"description": "Show plugin details", "usage": "/plugin/info <name>"},
            "refresh": {"description": "Refresh marketplace index", "usage": "/plugin/refresh"},
            "uninstall": {"description": "Uninstall a plugin", "usage": "/plugin/uninstall <name>"},
        }
        
        if not subcommand:
            lines = ["**Plugin Commands**", ""]
            for name, info in subcommands.items():
                lines.append(f"- `/plugin/{name}` - {info['description']}")
            
            stats = self.marketplace.get_index_stats()
            lines.append("")
            lines.append(f"*Marketplace: {stats['total_plugins']} plugins indexed*")
            
            return {
                "command": "/plugin",
                "description": "Plugin management commands",
                "subcommands": subcommands,
                "marketplace_stats": stats,
                "display": "\n".join(lines),
            }
        
        if subcommand in subcommands:
            info = subcommands[subcommand]
            return {
                "command": f"/plugin/{subcommand}",
                **info,
                "display": f"`{info['usage']}`\n\n{info['description']}",
            }
        
        return {
            "error": f"Unknown subcommand: {subcommand}",
            "available": list(subcommands.keys()),
            "display": f"[red]Error:[/red] Unknown subcommand: {subcommand}. Available: {', '.join(subcommands.keys())}",
        }
    
    @command("/plugin/list", description="List installed plugins", scope="all")
    async def list_plugins(self) -> Dict[str, Any]:
        """List all installed plugins with their status.
        
        Returns:
            List of installed plugins
        """
        plugins = list(self._plugins.values())
        
        if not plugins:
            return {
                "plugins": [],
                "total": 0,
                "display": "No plugins installed. Use `/plugin/search <query>` to find plugins.",
            }
        
        lines = ["**Installed Plugins**", "", "| Plugin | Version | Status | Tools |", "|--------|---------|--------|-------|"]
        
        for p in sorted(plugins, key=lambda x: x.name):
            status = "✅ Enabled" if p.enabled else "⏸️ Disabled"
            tool_count = len(p.get_tools())
            lines.append(f"| {p.name} | {p.version} | {status} | {tool_count} |")
        
        return {
            "plugins": [
                {
                    "name": p.name,
                    "version": p.version,
                    "enabled": p.enabled,
                    "tools": len(p.get_tools()),
                }
                for p in plugins
            ],
            "total": len(plugins),
            "display": "\n".join(lines),
        }
    
    @command("/plugin/search", description="Search marketplace for plugins", scope="all",
             completions=lambda self: {"query": []})
    async def search_plugins(self, query: str) -> Dict[str, Any]:
        """Fuzzy search plugins from marketplace.
        
        Args:
            query: Search query string
            
        Returns:
            Search results with plugin info
        """
        if not query:
            return {
                "error": "Query is required",
                "display": "[red]Error:[/red] Please provide a search query.",
            }
        
        # Check if marketplace needs refresh
        if self.marketplace.needs_refresh() and len(self.marketplace._index) == 0:
            await self.marketplace.refresh_index()
        
        results = self.marketplace.search(query)
        
        if not results:
            return {
                "plugins": [],
                "total": 0,
                "query": query,
                "display": f"No plugins found for '{query}'.\n\nTry `/plugin/refresh` to update the marketplace index.",
            }
        
        lines = [f"**Plugin Search: {query}**", "", "| Plugin | Stars | Description |", "|--------|-------|-------------|"]
        
        for p in results:
            stars = f"⭐ {p.get('stars', 0):,}" if p.get('stars') else "-"
            desc = p.get("description", "")[:50]
            if len(p.get("description", "")) > 50:
                desc += "..."
            
            repo = p.get("repo") or p.get("repository", "")
            name_cell = f"[{p['name']}]({repo})" if repo else p['name']
            
            # Mark if already installed
            installed = p['name'] in self._plugins
            if installed:
                name_cell += " ✓"
            
            lines.append(f"| {name_cell} | {stars} | {desc} |")
        
        lines.append("")
        lines.append(f"*Found {len(results)} plugins. Use `/plugin/install <name>` to install.*")
        
        return {
            "plugins": results,
            "total": len(results),
            "query": query,
            "display": "\n".join(lines),
        }
    
    @command("/plugin/refresh", description="Refresh marketplace plugin index", scope="all")
    async def refresh_marketplace(self) -> Dict[str, Any]:
        """Refresh marketplace index from API.
        
        Returns:
            Refresh status
        """
        try:
            await self.marketplace.refresh_index()
            stats = self.marketplace.get_index_stats()
            
            return {
                "status": "refreshed",
                "plugins_indexed": stats["total_plugins"],
                "display": f"✅ Marketplace refreshed. Indexed {stats['total_plugins']} plugins.",
            }
        except Exception as e:
            return {
                "error": str(e),
                "display": f"[red]Error:[/red] Failed to refresh marketplace: {e}",
            }
    
    def _get_marketplace_completions(self) -> Dict[str, List[str]]:
        """Return marketplace plugin names for autocomplete."""
        return {"name": self.marketplace.get_completions()}
    
    @command("/plugin/install", description="Install plugin from marketplace or URL", scope="owner",
             completions=lambda self: self._get_marketplace_completions())
    async def install_plugin(self, name: str) -> Dict[str, Any]:
        """Install a plugin from marketplace or Git URL.
        
        Args:
            name: Plugin name or Git repository URL
            
        Returns:
            Installation result
        """
        if not name:
            return {
                "error": "Plugin name or URL is required",
                "display": "[red]Error:[/red] Please provide a plugin name or URL.",
            }
        
        # Check if already installed
        if name in self._plugins:
            p = self._plugins[name]
            return {
                "status": "already_installed",
                "plugin": name,
                "version": p.version,
                "display": f"Plugin **{name}** v{p.version} is already installed.",
            }
        
        # Determine source
        if name.startswith("http://") or name.startswith("https://") or name.startswith("git@"):
            repo_url = name
        else:
            # Look up in marketplace
            plugin_info = self.marketplace.get(name)
            if not plugin_info:
                return {
                    "error": f"Plugin not found: {name}",
                    "display": f"[red]Error:[/red] Plugin '{name}' not found in marketplace.\n\nUse `/plugin/search {name}` to find similar plugins.",
                }
            
            repo_url = plugin_info.get("repo") or plugin_info.get("repository")
            if not repo_url:
                return {
                    "error": f"Plugin {name} has no repository URL",
                    "display": f"[red]Error:[/red] Plugin '{name}' has no repository URL in marketplace.",
                }
        
        # Install from repository
        try:
            plugin = await self.loader.install_from_repo(repo_url)
            self._plugins[plugin.name] = plugin
            
            tools = plugin.get_tools()
            hooks = plugin.get_hooks()
            
            lines = [f"✅ Installed **{plugin.name}** v{plugin.version}"]
            if plugin.description:
                lines.append(f"> {plugin.description}")
            lines.append("")
            lines.append(f"- {len(tools)} tools registered")
            lines.append(f"- {len(hooks)} hooks registered")
            lines.append(f"- Path: `{plugin.path}`")
            
            return {
                "status": "installed",
                "plugin": plugin.name,
                "version": plugin.version,
                "tools": len(tools),
                "hooks": len(hooks),
                "path": str(plugin.path),
                "display": "\n".join(lines),
            }
        except ImportError as e:
            return {
                "error": str(e),
                "display": f"[red]Error:[/red] {e}\n\nInstall GitPython: `pip install gitpython`",
            }
        except Exception as e:
            logger.error(f"Plugin installation failed: {e}")
            return {
                "error": str(e),
                "display": f"[red]Error:[/red] Failed to install plugin: {e}",
            }
    
    def _get_installed_completions(self) -> Dict[str, List[str]]:
        """Return installed plugin names for autocomplete."""
        return {"name": list(self._plugins.keys())}
    
    @command("/plugin/uninstall", description="Uninstall a plugin", scope="owner",
             completions=lambda self: self._get_installed_completions())
    async def uninstall_plugin(self, name: str) -> Dict[str, Any]:
        """Uninstall a plugin.
        
        Args:
            name: Plugin name to uninstall
            
        Returns:
            Uninstall result
        """
        if not name:
            return {
                "error": "Plugin name is required",
                "display": "[red]Error:[/red] Please provide a plugin name.",
            }
        
        if name not in self._plugins:
            return {
                "error": f"Plugin not installed: {name}",
                "display": f"[red]Error:[/red] Plugin '{name}' is not installed.",
            }
        
        try:
            success = self.loader.uninstall(name)
            if success:
                del self._plugins[name]
                return {
                    "status": "uninstalled",
                    "plugin": name,
                    "display": f"✅ Uninstalled **{name}**",
                }
            else:
                return {
                    "error": f"Failed to uninstall: {name}",
                    "display": f"[red]Error:[/red] Failed to uninstall plugin '{name}'.",
                }
        except Exception as e:
            return {
                "error": str(e),
                "display": f"[red]Error:[/red] Failed to uninstall: {e}",
            }
    
    @command("/plugin/enable", description="Enable a disabled plugin", scope="owner",
             completions=lambda self: self._get_installed_completions())
    async def enable_plugin(self, name: str) -> Dict[str, Any]:
        """Enable a disabled plugin.
        
        Args:
            name: Plugin name to enable
            
        Returns:
            Enable result
        """
        if not name:
            return {
                "error": "Plugin name is required",
                "display": "[red]Error:[/red] Please provide a plugin name.",
            }
        
        if name not in self._plugins:
            return {
                "error": f"Plugin not installed: {name}",
                "display": f"[red]Error:[/red] Plugin '{name}' is not installed.",
            }
        
        plugin = self._plugins[name]
        if plugin.enabled:
            return {
                "status": "already_enabled",
                "plugin": name,
                "display": f"Plugin **{name}** is already enabled.",
            }
        
        plugin.enabled = True
        
        return {
            "status": "enabled",
            "plugin": name,
            "display": f"✅ Enabled **{name}**",
        }
    
    @command("/plugin/disable", description="Disable an enabled plugin", scope="owner",
             completions=lambda self: self._get_installed_completions())
    async def disable_plugin(self, name: str) -> Dict[str, Any]:
        """Disable an enabled plugin.
        
        Args:
            name: Plugin name to disable
            
        Returns:
            Disable result
        """
        if not name:
            return {
                "error": "Plugin name is required",
                "display": "[red]Error:[/red] Please provide a plugin name.",
            }
        
        if name not in self._plugins:
            return {
                "error": f"Plugin not installed: {name}",
                "display": f"[red]Error:[/red] Plugin '{name}' is not installed.",
            }
        
        plugin = self._plugins[name]
        if not plugin.enabled:
            return {
                "status": "already_disabled",
                "plugin": name,
                "display": f"Plugin **{name}** is already disabled.",
            }
        
        plugin.enabled = False
        
        return {
            "status": "disabled",
            "plugin": name,
            "display": f"⏸️ Disabled **{name}**",
        }
    
    @command("/plugin/info", description="Show plugin details", scope="all",
             completions=lambda self: {"name": list(self._plugins.keys()) + self.marketplace.get_completions()})
    async def plugin_info(self, name: str) -> Dict[str, Any]:
        """Show detailed information about a plugin.
        
        Args:
            name: Plugin name (installed or from marketplace)
            
        Returns:
            Plugin information
        """
        if not name:
            return {
                "error": "Plugin name is required",
                "display": "[red]Error:[/red] Please provide a plugin name.",
            }
        
        # Check installed plugins first
        if name in self._plugins:
            p = self._plugins[name]
            tools = p.get_tools()
            hooks = p.get_hooks()
            mcp_servers = p.get_mcp_servers()
            
            lines = [f"**{p.name}** v{p.version}"]
            if p.description:
                lines.append(f"> {p.description}")
            lines.append("")
            lines.append(f"- **Status:** {'✅ Enabled' if p.enabled else '⏸️ Disabled'}")
            lines.append(f"- **Path:** `{p.path}`")
            lines.append(f"- **Tools:** {len(tools)}")
            lines.append(f"- **Hooks:** {len(hooks)}")
            lines.append(f"- **MCP Servers:** {len(mcp_servers)}")
            
            if p.manifest.repository:
                lines.append(f"- **Repository:** {p.manifest.repository}")
            if p.manifest.author:
                lines.append(f"- **Author:** {p.manifest.author}")
            if p.manifest.license:
                lines.append(f"- **License:** {p.manifest.license}")
            
            if tools:
                lines.append("")
                lines.append("**Tools:**")
                for tool in tools[:10]:
                    lines.append(f"  - `{tool['name']}` ({tool['type']})")
                if len(tools) > 10:
                    lines.append(f"  - ... and {len(tools) - 10} more")
            
            return {
                "name": p.name,
                "version": p.version,
                "description": p.description,
                "installed": True,
                "enabled": p.enabled,
                "path": str(p.path),
                "tools": tools,
                "hooks": hooks,
                "mcp_servers": list(mcp_servers.keys()),
                "manifest": p.manifest.to_dict(),
                "display": "\n".join(lines),
            }
        
        # Check marketplace
        info = self.marketplace.get(name)
        if info:
            lines = [f"**{info['name']}**"]
            if info.get("description"):
                lines.append(f"> {info['description']}")
            lines.append("")
            lines.append(f"- **Stars:** ⭐ {info.get('stars', 0):,}")
            lines.append("- **Status:** 📦 Not installed")
            
            if info.get("repo") or info.get("repository"):
                repo = info.get("repo") or info.get("repository")
                lines.append(f"- **Repository:** [{repo}]({repo})")
            if info.get("author"):
                lines.append(f"- **Author:** {info['author']}")
            if info.get("keywords"):
                lines.append(f"- **Keywords:** {', '.join(info['keywords'])}")
            
            lines.append("")
            lines.append(f"Use `/plugin/install {name}` to install.")
            
            return {
                **info,
                "installed": False,
                "display": "\n".join(lines),
            }
        
        return {
            "error": f"Plugin not found: {name}",
            "display": f"[red]Error:[/red] Plugin '{name}' not found.\n\nUse `/plugin/search {name}` to search the marketplace.",
        }
    
    def get_enabled_plugins(self) -> List[Plugin]:
        """Get list of enabled plugins.
        
        Returns:
            List of enabled Plugin instances
        """
        return [p for p in self._plugins.values() if p.enabled]
    
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None
        """
        return self._plugins.get(name)
