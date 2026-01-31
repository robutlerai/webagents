"""
Local State Management

Manage .webagents/ directory structure for both global and project state.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class LocalState:
    """Manage local state in .webagents/ directories.
    
    Two locations:
    - Global: ~/.webagents/ (user-level settings, credentials)
    - Project: .webagents/ (project-specific state, sessions)
    
    Project takes precedence over global.
    """
    
    def __init__(self, project_dir: Optional[Path] = None):
        """Initialize state manager.
        
        Args:
            project_dir: Project directory (defaults to cwd)
        """
        # Global state
        self.global_dir = Path.home() / ".webagents"
        
        # Project state
        self.project_dir = (project_dir or Path.cwd()) / ".webagents"
        
        # Ensure directories exist
        self._init_directories()
    
    def _init_directories(self):
        """Create state directory structure."""
        # Global directories
        global_dirs = [
            self.global_dir,
            self.global_dir / "templates",
        ]
        
        # Project directories
        project_dirs = [
            self.project_dir,
            self.project_dir / "sessions",
            self.project_dir / "history",
            self.project_dir / "logs",
            self.project_dir / "cache",
            self.project_dir / "cache" / "tokens",
            self.project_dir / "cache" / "embeddings",
            self.project_dir / "vectors",
        ]
        
        for d in global_dirs + project_dirs:
            d.mkdir(parents=True, exist_ok=True)
        
        # Create .gitignore in project state dir
        gitignore = self.project_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("*\n")
    
    # Config management
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get config value (project > global > default)."""
        # Check project config
        project_config = self._load_json(self.project_dir / "config.json")
        if key in project_config:
            return project_config[key]
        
        # Check global config
        global_config = self._load_json(self.global_dir / "config.json")
        if key in global_config:
            return global_config[key]
        
        return default
    
    def set_config(self, key: str, value: Any, scope: str = "project"):
        """Set config value."""
        if scope == "global":
            config_file = self.global_dir / "config.json"
        else:
            config_file = self.project_dir / "config.json"
        
        config = self._load_json(config_file)
        config[key] = value
        self._save_json(config_file, config)
    
    # Credentials management
    
    def get_credentials(self) -> Dict:
        """Get stored credentials."""
        creds_file = self.global_dir / "credentials.json"
        return self._load_json(creds_file)
    
    def set_credentials(self, **kwargs):
        """Store credentials."""
        creds_file = self.global_dir / "credentials.json"
        creds = self._load_json(creds_file)
        creds.update(kwargs)
        self._save_json(creds_file, creds)
    
    def clear_credentials(self):
        """Clear stored credentials."""
        creds_file = self.global_dir / "credentials.json"
        if creds_file.exists():
            creds_file.unlink()
    
    # State file management
    
    def get_state(self) -> Dict:
        """Get current agent runtime state."""
        return self._load_json(self.project_dir / "state.json")
    
    def set_state(self, **kwargs):
        """Update runtime state."""
        state_file = self.project_dir / "state.json"
        state = self._load_json(state_file)
        state.update(kwargs)
        state["updated_at"] = datetime.utcnow().isoformat()
        self._save_json(state_file, state)
    
    # Registry access
    
    def get_registry_path(self, scope: str = "project") -> Path:
        """Get registry file path."""
        if scope == "global":
            return self.global_dir / "registry.json"
        return self.project_dir / "registry.json"
    
    # Session paths
    
    def get_sessions_dir(self) -> Path:
        """Get sessions directory."""
        return self.project_dir / "sessions"
    
    def get_history_dir(self, agent_name: str) -> Path:
        """Get history directory for an agent."""
        history_dir = self.project_dir / "history" / agent_name
        history_dir.mkdir(parents=True, exist_ok=True)
        return history_dir
    
    # Log paths
    
    def get_logs_dir(self, agent_name: Optional[str] = None) -> Path:
        """Get logs directory."""
        if agent_name:
            logs_dir = self.project_dir / "logs" / agent_name
        else:
            logs_dir = self.project_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir
    
    # Cache paths
    
    def get_cache_dir(self, cache_type: str = "general") -> Path:
        """Get cache directory."""
        cache_dir = self.project_dir / "cache" / cache_type
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    # Vectors path
    
    def get_vectors_dir(self) -> Path:
        """Get vectors directory for sqlite-vec indexes."""
        return self.project_dir / "vectors"
    
    # Templates path (global)
    
    def get_templates_dir(self) -> Path:
        """Get global templates cache directory."""
        return self.global_dir / "templates"
    
    # Helper methods
    
    def _load_json(self, path: Path) -> Dict:
        """Load JSON file or return empty dict."""
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                return {}
        return {}
    
    def _save_json(self, path: Path, data: Dict):
        """Save data to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str))


# Global state instance
_state: Optional[LocalState] = None


def get_state(project_dir: Optional[Path] = None) -> LocalState:
    """Get global state instance."""
    global _state
    if _state is None or project_dir:
        _state = LocalState(project_dir)
    return _state
