"""
Plugin Loader - Load and Install Plugins

Handles plugin loading from local directories and Git repositories.
Supports Claude Code plugin.json format.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from .schema import PluginManifest, validate_manifest, ManifestValidationError, load_manifest

logger = logging.getLogger(__name__)


@dataclass
class Plugin:
    """Loaded plugin instance.
    
    Attributes:
        name: Plugin identifier
        version: Plugin version
        description: Human-readable description
        path: Local filesystem path
        manifest: Full plugin manifest
        enabled: Whether plugin is active
    """
    name: str
    version: str
    description: str
    path: Path
    manifest: PluginManifest
    enabled: bool = True
    _tools_cache: Optional[List[Dict]] = field(default=None, repr=False)
    
    def get_tools(self) -> List[Dict]:
        """Get tools defined by this plugin.
        
        Discovers:
        - Commands from commands/ directory (Python files)
        - Skills from skills/ directory (SKILL.md files)
        
        Returns:
            List of tool descriptors
        """
        if self._tools_cache is not None:
            return self._tools_cache
        
        tools = []
        
        # From commands/
        commands_dir = self.path / self.manifest.commands.lstrip("./")
        if commands_dir.exists():
            for cmd_file in commands_dir.glob("*.py"):
                if cmd_file.stem.startswith("_"):
                    continue
                tools.append({
                    "name": f"{self.name}:{cmd_file.stem}",
                    "type": "command",
                    "path": str(cmd_file),
                    "plugin": self.name,
                })
        
        # From skills/
        skills_dir = self.path / self.manifest.skills.lstrip("./")
        if skills_dir.exists():
            for skill_file in skills_dir.glob("*.md"):
                if skill_file.stem.startswith("_"):
                    continue
                tools.append({
                    "name": f"{self.name}:{skill_file.stem}",
                    "type": "skill",
                    "path": str(skill_file),
                    "plugin": self.name,
                })
        
        self._tools_cache = tools
        return tools
    
    def get_hooks(self) -> List[Dict]:
        """Get hooks defined by this plugin.
        
        Returns:
            List of hook configurations
        """
        if not self.manifest.hooks:
            return []
        
        hooks_path = self.path / self.manifest.hooks.lstrip("./")
        if not hooks_path.exists():
            return []
        
        try:
            data = json.loads(hooks_path.read_text())
            return data.get("hooks", [])
        except Exception as e:
            logger.warning(f"Failed to load hooks from {hooks_path}: {e}")
            return []
    
    def get_mcp_servers(self) -> Dict[str, Dict]:
        """Get MCP server configurations.
        
        Returns:
            Dict mapping server name to configuration
        """
        if not self.manifest.mcpServers:
            return {}
        
        mcp_path = self.path / self.manifest.mcpServers.lstrip("./")
        if not mcp_path.exists():
            return {}
        
        try:
            data = json.loads(mcp_path.read_text())
            return data.get("mcpServers", data)
        except Exception as e:
            logger.warning(f"Failed to load MCP servers from {mcp_path}: {e}")
            return {}


class PluginLoader:
    """Load and validate plugins from filesystem.
    
    Handles:
    - Loading from local directories
    - Installing from Git repositories
    - Listing installed plugins
    - Plugin validation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize plugin loader.
        
        Args:
            config: Configuration dict with optional keys:
                - plugins_dir: Custom plugins directory path
        """
        self.config = config or {}
        self.plugins_dir = Path(self.config.get(
            "plugins_dir", 
            Path.home() / ".webagents" / "plugins"
        ))
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        self._plugins_cache: Dict[str, Plugin] = {}
    
    def load_local(self, path: Path) -> Plugin:
        """Load plugin from local directory.
        
        Args:
            path: Path to plugin directory containing plugin.json
            
        Returns:
            Loaded Plugin instance
            
        Raises:
            FileNotFoundError: If plugin.json not found
            ManifestValidationError: If manifest is invalid
        """
        path = Path(path).resolve()
        manifest_path = path / "plugin.json"
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"No plugin.json found in {path}")
        
        manifest = load_manifest(manifest_path)
        
        plugin = Plugin(
            name=manifest.name,
            version=manifest.version,
            description=manifest.description,
            path=path,
            manifest=manifest,
        )
        
        self._plugins_cache[manifest.name] = plugin
        return plugin
    
    async def install_from_repo(self, repo_url: str, ref: str = None) -> Plugin:
        """Clone and install plugin from Git repository.
        
        Args:
            repo_url: Git repository URL
            ref: Optional branch, tag, or commit to checkout
            
        Returns:
            Installed Plugin instance
            
        Raises:
            ImportError: If GitPython is not installed
            Exception: If clone/pull fails
        """
        try:
            import git
        except ImportError:
            raise ImportError(
                "GitPython is required for Git installation. "
                "Install with: pip install gitpython"
            )
        
        # Determine plugin name from URL
        name = repo_url.rstrip("/").split("/")[-1]
        if name.endswith(".git"):
            name = name[:-4]
        
        install_path = self.plugins_dir / name
        
        if install_path.exists():
            # Update existing plugin
            logger.info(f"Updating existing plugin: {name}")
            try:
                repo = git.Repo(install_path)
                # Fetch and reset to ensure clean state
                repo.remotes.origin.fetch()
                if ref:
                    repo.git.checkout(ref)
                else:
                    repo.remotes.origin.pull()
            except git.GitCommandError as e:
                logger.warning(f"Git pull failed, attempting fresh clone: {e}")
                import shutil
                shutil.rmtree(install_path)
                git.Repo.clone_from(repo_url, install_path)
                if ref:
                    repo = git.Repo(install_path)
                    repo.git.checkout(ref)
        else:
            # Clone new plugin
            logger.info(f"Cloning plugin from {repo_url}")
            repo = git.Repo.clone_from(repo_url, install_path)
            if ref:
                repo.git.checkout(ref)
        
        return self.load_local(install_path)
    
    def install_from_path(self, source_path: Path, copy: bool = True) -> Plugin:
        """Install plugin from local path.
        
        Args:
            source_path: Source directory containing plugin
            copy: If True, copy to plugins directory. If False, symlink.
            
        Returns:
            Installed Plugin instance
        """
        source_path = Path(source_path).resolve()
        manifest_path = source_path / "plugin.json"
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"No plugin.json found in {source_path}")
        
        manifest = load_manifest(manifest_path)
        install_path = self.plugins_dir / manifest.name
        
        if install_path.exists():
            import shutil
            shutil.rmtree(install_path)
        
        if copy:
            import shutil
            shutil.copytree(source_path, install_path)
        else:
            install_path.symlink_to(source_path)
        
        return self.load_local(install_path)
    
    def uninstall(self, name: str) -> bool:
        """Uninstall a plugin.
        
        Args:
            name: Plugin name to uninstall
            
        Returns:
            True if uninstalled, False if not found
        """
        install_path = self.plugins_dir / name
        
        if not install_path.exists():
            return False
        
        import shutil
        if install_path.is_symlink():
            install_path.unlink()
        else:
            shutil.rmtree(install_path)
        
        # Remove from cache
        self._plugins_cache.pop(name, None)
        
        return True
    
    def list_installed(self) -> List[Plugin]:
        """List all installed plugins.
        
        Returns:
            List of installed Plugin instances
        """
        plugins = []
        
        for path in self.plugins_dir.iterdir():
            if path.is_dir() and (path / "plugin.json").exists():
                try:
                    # Check cache first
                    if path.name in self._plugins_cache:
                        plugins.append(self._plugins_cache[path.name])
                    else:
                        plugins.append(self.load_local(path))
                except Exception as e:
                    logger.warning(f"Failed to load plugin at {path}: {e}")
        
        return plugins
    
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None if not found
        """
        # Check cache
        if name in self._plugins_cache:
            return self._plugins_cache[name]
        
        # Try to load from disk
        path = self.plugins_dir / name
        if path.exists() and (path / "plugin.json").exists():
            try:
                return self.load_local(path)
            except Exception:
                pass
        
        return None
    
    def refresh_plugin(self, name: str) -> Optional[Plugin]:
        """Reload a plugin from disk.
        
        Args:
            name: Plugin name
            
        Returns:
            Reloaded Plugin or None if not found
        """
        # Clear from cache
        self._plugins_cache.pop(name, None)
        
        return self.get_plugin(name)
