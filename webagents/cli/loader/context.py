"""
AGENTS.md Context Parser

Parse and manage AGENTS.md context files.
"""

import re
from pathlib import Path
from typing import Optional, List
import yaml

from .schema import ContextMetadata


class ContextFile:
    """Represents a parsed AGENTS.md context file."""
    
    def __init__(self, path: Path):
        self.path = Path(path)
        self.metadata: ContextMetadata
        self.content: str
        self._raw_yaml: dict = {}
        self._parse()
    
    def _parse(self):
        """Parse the file."""
        content = self.path.read_text()
        
        # Match YAML frontmatter
        pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
        match = re.match(pattern, content, re.DOTALL)
        
        if match:
            yaml_str = match.group(1)
            body = match.group(2)
            
            try:
                self._raw_yaml = yaml.safe_load(yaml_str) or {}
            except yaml.YAMLError:
                self._raw_yaml = {}
            
            self.metadata = ContextMetadata(**self._raw_yaml)
            self.content = body.strip()
        else:
            # No frontmatter
            self.metadata = ContextMetadata()
            self.content = content.strip()
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "path": str(self.path),
            "metadata": self.metadata.model_dump(),
            "content": self.content,
        }


class ContextHierarchy:
    """Manage AGENTS.md context hierarchy.
    
    Context files inherit from parent directories:
    - project/AGENTS.md
      - project/subdir/AGENTS.md (inherits from parent)
        - project/subdir/AGENT.md (inherits from both)
    """
    
    def __init__(self, stop_at: Optional[Path] = None):
        """Initialize hierarchy resolver.
        
        Args:
            stop_at: Stop traversing at this directory (e.g., project root)
        """
        self.stop_at = stop_at
    
    def resolve(self, start_path: Path) -> List[ContextFile]:
        """Resolve context hierarchy from a starting path.
        
        Args:
            start_path: Path to start from (usually directory containing AGENT.md)
            
        Returns:
            List of ContextFile instances, ordered from root to local
            (i.e., outermost first, so local can override)
        """
        context_files = []
        current = start_path if start_path.is_dir() else start_path.parent
        
        while current != current.parent:
            agents_md = current / "AGENTS.md"
            if agents_md.exists():
                context_files.append(ContextFile(agents_md))
            
            # Stop at specified directory
            if self.stop_at and current == self.stop_at:
                break
            
            current = current.parent
        
        # Reverse so root is first, local is last (for override order)
        return list(reversed(context_files))
    
    def merge_contexts(self, contexts: List[ContextFile]) -> dict:
        """Merge multiple context files.
        
        Later contexts override earlier ones.
        
        Args:
            contexts: List of ContextFile instances (root first, local last)
            
        Returns:
            Merged context dictionary
        """
        merged = {
            "namespace": "local",
            "skills": [],
            "tools": [],
            "mcp_servers": [],
            "instructions": [],
        }
        
        for ctx in contexts:
            # Namespace is overridden
            if ctx.metadata.namespace:
                merged["namespace"] = ctx.metadata.namespace
            
            # Model is overridden
            if ctx.metadata.model:
                merged["model"] = ctx.metadata.model
            
            # Skills, tools, mcp_servers are accumulated
            merged["skills"].extend(ctx.metadata.skills)
            merged["tools"].extend(ctx.metadata.tools)
            merged["mcp_servers"].extend(ctx.metadata.mcp_servers)
            
            # Visibility is overridden
            if ctx.metadata.visibility:
                merged["visibility"] = ctx.metadata.visibility
            
            # Sandbox is merged (later overrides)
            if ctx.metadata.sandbox:
                merged["sandbox"] = ctx.metadata.sandbox.model_dump()
            
            # Instructions are accumulated
            if ctx.content:
                merged["instructions"].append(ctx.content)
        
        # Deduplicate lists
        merged["skills"] = list(dict.fromkeys(merged["skills"]))
        merged["tools"] = list(dict.fromkeys(merged["tools"]))
        merged["mcp_servers"] = list(dict.fromkeys(merged["mcp_servers"]))
        
        return merged


def find_context_file(directory: Path) -> Optional[ContextFile]:
    """Find AGENTS.md in a directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        ContextFile if found, None otherwise
    """
    agents_md = directory / "AGENTS.md"
    if agents_md.exists():
        return ContextFile(agents_md)
    return None


def create_context_file(
    directory: Path,
    metadata: Optional[dict] = None,
    content: str = "",
) -> ContextFile:
    """Create a new AGENTS.md context file.
    
    Args:
        directory: Directory to create file in
        metadata: YAML frontmatter data
        content: Markdown content
        
    Returns:
        Created ContextFile
    """
    path = directory / "AGENTS.md"
    
    meta = ContextMetadata(**(metadata or {}))
    
    yaml_data = meta.model_dump(exclude_none=True)
    yaml_str = yaml.dump(yaml_data, default_flow_style=False, sort_keys=False)
    
    if not content:
        content = "# Project Context\n\nThis file provides context for all agents in this directory."
    
    file_content = f"---\n{yaml_str}---\n\n{content}"
    
    path.write_text(file_content)
    return ContextFile(path)
