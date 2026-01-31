"""
Shell Autocomplete Providers

Dynamic completions for Typer CLI commands.
"""

from typing import List
from pathlib import Path


def complete_agent_names(incomplete: str) -> List[str]:
    """Autocomplete agent names from local registry and current directory."""
    agent_names = []
    
    # Scan current directory
    cwd = Path.cwd()
    
    # Check for AGENT.md
    if (cwd / "AGENT.md").exists():
        agent_names.append("default")
    
    # Check for AGENT-*.md files
    for f in cwd.glob("AGENT-*.md"):
        name = f.stem.replace("AGENT-", "")
        agent_names.append(name)
    
    # TODO: Also scan registry from .webagents/registry.json
    
    return [n for n in agent_names if n.startswith(incomplete)]


def complete_skill_names(incomplete: str) -> List[str]:
    """Autocomplete available skill names."""
    skills = [
        "cron",
        "folder-index", 
        "llm",
        "mcp",
        "memory",
        "discovery",
        "web",
        "filesystem",
        "database",
    ]
    return [s for s in skills if s.startswith(incomplete)]


def complete_template_names(incomplete: str) -> List[str]:
    """Autocomplete template names."""
    # Bundled templates
    templates = [
        "assistant",
        "planning",
        "marketing",
        "content",
        "code-review",
        "research",
    ]
    
    # TODO: Also include cached templates from ~/.webagents/templates/
    
    return [t for t in templates if t.startswith(incomplete)]


def complete_namespace_names(incomplete: str) -> List[str]:
    """Autocomplete namespace names."""
    # Default local namespace
    namespaces = ["local"]
    
    # TODO: Get namespaces from platform API or local cache
    # Try to read from ~/.webagents/namespaces.json
    
    return [n for n in namespaces if n.startswith(incomplete)]


def complete_index_names(incomplete: str) -> List[str]:
    """Autocomplete vector index names."""
    index_names = []
    
    # Look in .webagents/vectors/
    vectors_dir = Path.cwd() / ".webagents" / "vectors"
    if vectors_dir.exists():
        for f in vectors_dir.iterdir():
            if f.is_dir():
                index_names.append(f.name)
    
    return [n for n in index_names if n.startswith(incomplete)]


def complete_job_ids(incomplete: str) -> List[str]:
    """Autocomplete cron job IDs."""
    # TODO: Get job IDs from daemon
    return []


def complete_checkpoint_names(incomplete: str) -> List[str]:
    """Autocomplete checkpoint names."""
    checkpoints = []
    
    # Look in ~/.webagents/sessions/
    sessions_dir = Path.home() / ".webagents" / "sessions"
    if sessions_dir.exists():
        for f in sessions_dir.glob("*.json"):
            if f.name != "latest.json":
                checkpoints.append(f.stem)
    
    return [c for c in checkpoints if c.startswith(incomplete)]


def complete_file_paths(incomplete: str) -> List[str]:
    """Autocomplete file paths for AGENT*.md files."""
    completions = []
    
    try:
        if "/" in incomplete:
            parent = Path(incomplete).parent
            prefix = incomplete.rsplit("/", 1)[1]
        else:
            parent = Path(".")
            prefix = incomplete
        
        if parent.exists():
            for item in parent.iterdir():
                name = item.name
                if name.startswith(prefix):
                    # Prioritize AGENT*.md files
                    if name.startswith("AGENT") and name.endswith(".md"):
                        completions.insert(0, str(item))
                    elif item.is_dir():
                        completions.append(str(item) + "/")
    except Exception:
        pass
    
    return completions
