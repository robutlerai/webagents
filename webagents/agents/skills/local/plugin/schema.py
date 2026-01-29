"""
Plugin Manifest Schema - Claude Code Compatible

Defines the plugin.json schema for validation and parsing.
Compatible with Claude Code plugin format.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import json
from pathlib import Path


@dataclass
class McpServerConfig:
    """MCP server configuration within a plugin."""
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)


@dataclass
class HookConfig:
    """Hook configuration within a plugin."""
    event: str
    handler: str
    priority: int = 50


@dataclass
class PluginManifest:
    """Claude Code compatible plugin manifest (plugin.json).
    
    Attributes:
        name: Plugin identifier (required)
        version: Semantic version string
        description: Human-readable description
        author: Plugin author
        license: License identifier
        commands: Path to commands directory (default: ./commands/)
        skills: Path to skills directory (default: ./skills/)
        agents: Path to agents directory (default: ./agents/)
        hooks: Path to hooks configuration file
        mcpServers: Path to MCP servers config or inline config
        dependencies: Python package dependencies
        repository: Git repository URL
        keywords: Searchable keywords
        homepage: Plugin homepage URL
    """
    name: str
    version: str = "0.0.0"
    description: str = ""
    author: str = ""
    license: str = ""
    commands: str = "./commands/"
    skills: str = "./skills/"
    agents: str = "./agents/"
    hooks: Optional[str] = None
    mcpServers: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    repository: str = ""
    keywords: List[str] = field(default_factory=list)
    homepage: str = ""
    
    # Additional fields for marketplace
    stars: int = 0
    downloads: int = 0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginManifest":
        """Create manifest from dictionary."""
        return cls(
            name=data.get("name", ""),
            version=data.get("version", "0.0.0"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            license=data.get("license", ""),
            commands=data.get("commands", "./commands/"),
            skills=data.get("skills", "./skills/"),
            agents=data.get("agents", "./agents/"),
            hooks=data.get("hooks"),
            mcpServers=data.get("mcpServers"),
            dependencies=data.get("dependencies", []),
            repository=data.get("repository", ""),
            keywords=data.get("keywords", []),
            homepage=data.get("homepage", ""),
            stars=data.get("stars", 0),
            downloads=data.get("downloads", 0),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "license": self.license,
            "commands": self.commands,
            "skills": self.skills,
            "agents": self.agents,
            "hooks": self.hooks,
            "mcpServers": self.mcpServers,
            "dependencies": self.dependencies,
            "repository": self.repository,
            "keywords": self.keywords,
            "homepage": self.homepage,
            "stars": self.stars,
            "downloads": self.downloads,
        }


class ManifestValidationError(Exception):
    """Raised when plugin manifest validation fails."""
    pass


def validate_manifest(data: Dict[str, Any]) -> PluginManifest:
    """Validate and parse a plugin.json manifest.
    
    Args:
        data: Raw manifest dictionary
        
    Returns:
        Validated PluginManifest instance
        
    Raises:
        ManifestValidationError: If validation fails
    """
    # Required fields
    if "name" not in data or not data["name"]:
        raise ManifestValidationError("Missing required field: name")
    
    # Name format validation
    name = data["name"]
    if not isinstance(name, str):
        raise ManifestValidationError("Field 'name' must be a string")
    
    # Check for valid name characters (alphanumeric, hyphen, underscore)
    import re
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', name):
        raise ManifestValidationError(
            f"Invalid plugin name '{name}'. Must start with letter and contain only "
            "alphanumeric characters, hyphens, and underscores."
        )
    
    # Version format validation (if provided)
    version = data.get("version", "0.0.0")
    if version:
        # Loose semver validation
        if not re.match(r'^\d+\.\d+\.\d+', version):
            raise ManifestValidationError(
                f"Invalid version '{version}'. Must follow semver format (e.g., 1.0.0)"
            )
    
    # Validate dependencies format
    dependencies = data.get("dependencies", [])
    if not isinstance(dependencies, list):
        raise ManifestValidationError("Field 'dependencies' must be an array")
    for dep in dependencies:
        if not isinstance(dep, str):
            raise ManifestValidationError(f"Invalid dependency: {dep}. Must be a string.")
    
    # Validate keywords format
    keywords = data.get("keywords", [])
    if not isinstance(keywords, list):
        raise ManifestValidationError("Field 'keywords' must be an array")
    
    return PluginManifest.from_dict(data)


def load_manifest(path: Path) -> PluginManifest:
    """Load and validate manifest from file.
    
    Args:
        path: Path to plugin.json file
        
    Returns:
        Validated PluginManifest
        
    Raises:
        FileNotFoundError: If manifest file doesn't exist
        ManifestValidationError: If validation fails
    """
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ManifestValidationError(f"Invalid JSON in manifest: {e}")
    
    return validate_manifest(data)
