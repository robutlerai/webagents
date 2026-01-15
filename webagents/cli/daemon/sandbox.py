"""
Sandbox Configuration

Manage sandbox security settings for agent execution.
"""

from typing import List, Optional, Set
from pathlib import Path
from pydantic import BaseModel
import fnmatch
import os


class SandboxConfig(BaseModel):
    """Sandbox security configuration."""
    
    # Preset
    preset: str = "development"  # strict, development, unrestricted
    
    # Folder access control
    allowed_folders: List[str] = ["."]
    denied_folders: List[str] = [
        "/etc", "/var", "/usr", "/bin", "/sbin",
        "~/.ssh", "~/.gnupg", "~/.aws", "~/.config"
    ]
    
    # Command execution control
    allowed_commands: List[str] = [
        "ls", "cat", "head", "tail", "grep", "find", "wc", "sort", "uniq",
        "git status", "git log", "git diff", "git show",
        "python --version", "node --version", "npm --version",
    ]
    denied_commands: List[str] = [
        "rm -rf *", "sudo *", "chmod 777 *",
        "curl * | sh", "wget * | sh",
        "dd *", "mkfs *", "fdisk *",
    ]
    
    # Python import control
    allowed_imports: List[str] = [
        "json", "re", "datetime", "pathlib",
        "collections", "itertools", "functools",
        "typing", "dataclasses", "pydantic",
        "os.path", "shutil",
    ]
    denied_imports: List[str] = [
        "os.system", "subprocess.call", "subprocess.run",
        "eval", "exec", "compile", "__import__",
    ]
    
    @classmethod
    def from_preset(cls, preset: str) -> "SandboxConfig":
        """Create config from preset.
        
        Args:
            preset: strict, development, or unrestricted
            
        Returns:
            SandboxConfig
        """
        if preset == "strict":
            return cls(
                preset="strict",
                allowed_folders=["."],
                allowed_commands=["ls", "cat", "head", "tail"],
                allowed_imports=["json", "re", "datetime"],
            )
        elif preset == "unrestricted":
            return cls(
                preset="unrestricted",
                allowed_folders=["*"],
                allowed_commands=["*"],
                allowed_imports=["*"],
            )
        else:
            # development (default)
            return cls(preset="development")


class SandboxEnforcer:
    """Enforce sandbox rules."""
    
    def __init__(self, config: SandboxConfig):
        """Initialize enforcer.
        
        Args:
            config: Sandbox configuration
        """
        self.config = config
        
        # Expand paths
        self._allowed_paths: Set[Path] = set()
        self._denied_paths: Set[Path] = set()
        self._expand_paths()
    
    def _expand_paths(self):
        """Expand folder patterns to absolute paths."""
        cwd = Path.cwd()
        
        for folder in self.config.allowed_folders:
            if folder == "*":
                self._allowed_paths.add(Path("/"))
            elif folder.startswith("~"):
                self._allowed_paths.add(Path(os.path.expanduser(folder)).resolve())
            elif folder.startswith("/"):
                self._allowed_paths.add(Path(folder).resolve())
            else:
                self._allowed_paths.add((cwd / folder).resolve())
        
        for folder in self.config.denied_folders:
            if folder.startswith("~"):
                self._denied_paths.add(Path(os.path.expanduser(folder)).resolve())
            elif folder.startswith("/"):
                self._denied_paths.add(Path(folder).resolve())
    
    def can_access_path(self, path: str) -> bool:
        """Check if path access is allowed.
        
        Args:
            path: Path to check
            
        Returns:
            True if allowed
        """
        if self.config.preset == "unrestricted":
            return True
        
        target = Path(path).resolve()
        
        # Check denied first
        for denied in self._denied_paths:
            try:
                target.relative_to(denied)
                return False
            except ValueError:
                continue
        
        # Check allowed
        for allowed in self._allowed_paths:
            try:
                target.relative_to(allowed)
                return True
            except ValueError:
                continue
        
        return False
    
    def can_run_command(self, command: str) -> bool:
        """Check if command execution is allowed.
        
        Args:
            command: Command to check
            
        Returns:
            True if allowed
        """
        if self.config.preset == "unrestricted":
            return True
        
        # Check denied patterns
        for denied in self.config.denied_commands:
            if fnmatch.fnmatch(command, denied):
                return False
        
        # Check allowed patterns
        if "*" in self.config.allowed_commands:
            return True
        
        for allowed in self.config.allowed_commands:
            if fnmatch.fnmatch(command, allowed):
                return True
            # Also check if command starts with allowed
            if command.split()[0] == allowed.split()[0]:
                return True
        
        return False
    
    def can_import(self, module: str) -> bool:
        """Check if import is allowed.
        
        Args:
            module: Module to import
            
        Returns:
            True if allowed
        """
        if self.config.preset == "unrestricted":
            return True
        
        # Check denied
        for denied in self.config.denied_imports:
            if module == denied or module.startswith(denied + "."):
                return False
        
        # Check allowed
        if "*" in self.config.allowed_imports:
            return True
        
        for allowed in self.config.allowed_imports:
            if module == allowed or module.startswith(allowed + "."):
                return True
        
        return False


def get_sandbox_config() -> SandboxConfig:
    """Get sandbox config from state.
    
    Returns:
        SandboxConfig
    """
    from ..state.local import get_state
    
    state = get_state()
    sandbox_data = state.get_config("sandbox", {})
    
    if sandbox_data:
        return SandboxConfig(**sandbox_data)
    
    return SandboxConfig()


def create_enforcer() -> SandboxEnforcer:
    """Create sandbox enforcer from config.
    
    Returns:
        SandboxEnforcer
    """
    config = get_sandbox_config()
    return SandboxEnforcer(config)
