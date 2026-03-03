"""
AGENT.md Parser

Parse AGENT.md and AGENT-*.md files with YAML frontmatter.
"""

import re
from pathlib import Path
from typing import Optional, Tuple
import yaml

from .schema import AgentMetadata


class AgentFile:
    """Represents a parsed AGENT.md or AGENT-*.md file."""
    
    def __init__(self, path: Path):
        self.path = Path(path)
        self.metadata: AgentMetadata
        self.instructions: str
        self._raw_yaml: dict = {}
        self._parse()
    
    def _parse(self):
        """Parse the file into metadata and instructions."""
        content = self.path.read_text()
        yaml_data, body = parse_frontmatter(content)
        
        self._raw_yaml = yaml_data
        
        # Parse metadata with defaults
        if yaml_data:
            self.metadata = AgentMetadata(**yaml_data)
        else:
            # No frontmatter, use defaults
            self.metadata = AgentMetadata()
            
            # Try to infer name from filename
            if self.path.name == "AGENT.md":
                self.metadata.name = "default"
            elif self.path.name.startswith("AGENT-"):
                self.metadata.name = self.path.stem.replace("AGENT-", "")
        
        self.instructions = body.strip() if body else ""
    
    @property
    def name(self) -> str:
        """Get agent name."""
        return self.metadata.name
    
    @property
    def is_named(self) -> bool:
        """Check if this is a named agent (AGENT-<name>.md)."""
        return self.path.name.startswith("AGENT-")
    
    def update_metadata(self, **kwargs):
        """Update metadata fields and save."""
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)
        self._save()
    
    def add_skill(self, skill: str):
        """Add a skill to the agent."""
        # Simple check if skill name exists in list (handling strings or dicts)
        exists = False
        for s in self.metadata.skills:
            if isinstance(s, str) and s == skill:
                exists = True
                break
            elif isinstance(s, dict) and skill in s:
                exists = True
                break
        
        if not exists:
            self.metadata.skills.append(skill)
            self._save()
    
    def remove_skill(self, skill: str):
        """Remove a skill from the agent."""
        to_remove = None
        for s in self.metadata.skills:
            if isinstance(s, str) and s == skill:
                to_remove = s
                break
            elif isinstance(s, dict) and skill in s:
                to_remove = s
                break
        
        if to_remove:
            self.metadata.skills.remove(to_remove)
            self._save()
    
    def _save(self):
        """Save changes back to file."""
        # Rebuild YAML frontmatter
        yaml_data = self.metadata.model_dump(exclude_none=True)
        
        # Remove default values to keep file clean
        defaults = AgentMetadata().model_dump()
        for key in list(yaml_data.keys()):
            if yaml_data[key] == defaults.get(key):
                if key not in ['name', 'description']:  # Keep essential fields
                    del yaml_data[key]
        
        # Build content
        yaml_str = yaml.dump(yaml_data, default_flow_style=False, sort_keys=False)
        content = f"---\n{yaml_str}---\n\n{self.instructions}"
        
        self.path.write_text(content)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "path": str(self.path),
            "name": self.name,
            "metadata": self.metadata.model_dump(),
            "instructions": self.instructions,
        }


def parse_frontmatter(content: str) -> Tuple[dict, str]:
    """Parse YAML frontmatter from markdown content.
    
    Args:
        content: Full file content
        
    Returns:
        Tuple of (yaml_dict, body_content)
    """
    # Match YAML frontmatter: ---\n...\n---(\n...)?
    # Allow leading whitespace/newlines and optional body
    pattern = r'^\s*---\s*\n(.*?)\n---\s*(?:\n(.*))?$'
    match = re.match(pattern, content, re.DOTALL)
    
    if match:
        yaml_str = match.group(1)
        body = match.group(2)
        
        try:
            yaml_data = yaml.safe_load(yaml_str) or {}
        except yaml.YAMLError:
            yaml_data = {}
        
        return yaml_data, body
    
    # No frontmatter
    return {}, content


def parse_agent_file(path: Path) -> AgentFile:
    """Parse an agent file.
    
    Args:
        path: Path to AGENT.md or AGENT-*.md file
        
    Returns:
        Parsed AgentFile
    """
    return AgentFile(path)


def find_agent_files(directory: Path, recursive: bool = False) -> list:
    """Find all agent files in a directory.
    
    Args:
        directory: Directory to search
        recursive: Search subdirectories
        
    Returns:
        List of AgentFile instances
    """
    pattern = "**/" if recursive else ""
    agent_files = []
    
    # Find AGENT.md
    for f in directory.glob(f"{pattern}AGENT.md"):
        agent_files.append(AgentFile(f))
    
    # Find AGENT-*.md
    for f in directory.glob(f"{pattern}AGENT-*.md"):
        agent_files.append(AgentFile(f))
    
    return agent_files


def create_agent_file(
    directory: Path,
    name: Optional[str] = None,
    metadata: Optional[dict] = None,
    instructions: str = "",
) -> AgentFile:
    """Create a new agent file.
    
    Args:
        directory: Directory to create file in
        name: Agent name (creates AGENT-<name>.md, or AGENT.md if None)
        metadata: YAML frontmatter data
        instructions: Markdown instructions
        
    Returns:
        Created AgentFile
    """
    if name:
        filename = f"AGENT-{name}.md"
    else:
        filename = "AGENT.md"
    
    path = directory / filename
    
    # Build metadata
    meta = AgentMetadata(**(metadata or {}))
    if name and not metadata:
        meta.name = name
    
    # Build content
    yaml_data = meta.model_dump(exclude_none=True)
    yaml_str = yaml.dump(yaml_data, default_flow_style=False, sort_keys=False)
    
    if not instructions:
        instructions = f"# {meta.name.title()} Agent\n\nDescribe your agent's purpose and behavior here."
    
    content = f"---\n{yaml_str}---\n\n{instructions}"
    
    path.write_text(content)
    return AgentFile(path)
