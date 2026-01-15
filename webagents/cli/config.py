"""
CLI Configuration Management

Handle global and project configuration.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import json


class Config:
    """WebAgents configuration management."""
    
    def __init__(self):
        # Global config in home directory
        self.global_dir = Path.home() / ".webagents"
        self.global_config_file = self.global_dir / "config.json"
        
        # Project config in current directory
        self.project_dir = Path.cwd() / ".webagents"
        self.project_config_file = self.project_dir / "config.json"
        
        # Default configuration
        self.defaults = {
            "model": "openai/gpt-4o-mini",
            "sandbox": {
                "preset": "development",
                "allowed_folders": ["."],
                "allowed_commands": [
                    "ls", "cat", "head", "tail", "grep",
                    "git status", "git log", "git diff",
                ],
            },
            "daemon": {
                "port": 8765,
                "auto_start": False,
            },
            "platform": {
                "url": "https://robutler.ai",
            },
            "ui": {
                "splash_style": "block",
                "theme": "default",
            },
        }
        
        self._global_config: Dict = {}
        self._project_config: Dict = {}
        self._load()
    
    def _load(self):
        """Load configuration files."""
        # Load global config
        if self.global_config_file.exists():
            try:
                self._global_config = json.loads(self.global_config_file.read_text())
            except Exception:
                self._global_config = {}
        
        # Load project config
        if self.project_config_file.exists():
            try:
                self._project_config = json.loads(self.project_config_file.read_text())
            except Exception:
                self._project_config = {}
    
    def _save_global(self):
        """Save global configuration."""
        self.global_dir.mkdir(parents=True, exist_ok=True)
        self.global_config_file.write_text(json.dumps(self._global_config, indent=2))
    
    def _save_project(self):
        """Save project configuration."""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.project_config_file.write_text(json.dumps(self._project_config, indent=2))
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Precedence: project > global > defaults
        """
        # Check project config first
        if key in self._project_config:
            return self._project_config[key]
        
        # Then global config
        if key in self._global_config:
            return self._global_config[key]
        
        # Then defaults
        if key in self.defaults:
            return self.defaults[key]
        
        return default
    
    def set(self, key: str, value: Any, scope: str = "project"):
        """Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
            scope: "global" or "project"
        """
        if scope == "global":
            self._global_config[key] = value
            self._save_global()
        else:
            self._project_config[key] = value
            self._save_project()
    
    def delete(self, key: str, scope: str = "project"):
        """Delete configuration value."""
        if scope == "global":
            if key in self._global_config:
                del self._global_config[key]
                self._save_global()
        else:
            if key in self._project_config:
                del self._project_config[key]
                self._save_project()
    
    def reset(self, scope: str = "project"):
        """Reset configuration to defaults."""
        if scope == "global":
            self._global_config = {}
            self._save_global()
        else:
            self._project_config = {}
            self._save_project()
    
    def all(self) -> Dict:
        """Get all configuration (merged)."""
        result = dict(self.defaults)
        result.update(self._global_config)
        result.update(self._project_config)
        return result
    
    # Convenience accessors
    
    @property
    def model(self) -> str:
        return self.get("model", "openai/gpt-4o-mini")
    
    @property
    def sandbox_preset(self) -> str:
        sandbox = self.get("sandbox", {})
        return sandbox.get("preset", "development")
    
    @property
    def daemon_port(self) -> int:
        daemon = self.get("daemon", {})
        return daemon.get("port", 8765)
    
    @property
    def platform_url(self) -> str:
        platform = self.get("platform", {})
        return platform.get("url", "https://robutler.ai")


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config
