"""
REPL Completions

Auto-completions for the interactive REPL.
"""

from typing import List, Iterable
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from pathlib import Path


class WebAgentsCompleter(Completer):
    """Custom completer for WebAgents REPL."""
    
    def __init__(self):
        # Slash commands
        self.slash_commands = [
            "help", "exit", "quit", "clear",
            "save", "load", "agent", "discover",
            "mcp", "history", "tokens", "config",
        ]
    
    def get_completions(
        self, 
        document: Document, 
        complete_event
    ) -> Iterable[Completion]:
        """Generate completions."""
        text = document.text_before_cursor
        
        # Slash command completion
        if text.startswith("/"):
            cmd_text = text[1:]
            for cmd in self.slash_commands:
                if cmd.startswith(cmd_text):
                    yield Completion(
                        cmd,
                        start_position=-len(cmd_text),
                        display_meta="command"
                    )
        
        # @ file path completion
        elif "@" in text:
            at_pos = text.rfind("@")
            path_text = text[at_pos + 1:]
            
            # Complete file paths
            for completion in self._complete_path(path_text):
                yield completion
    
    def _complete_path(self, partial: str) -> Iterable[Completion]:
        """Complete file paths."""
        try:
            if "/" in partial:
                parent = Path(partial).parent
                prefix = partial.rsplit("/", 1)[1]
            else:
                parent = Path(".")
                prefix = partial
            
            if parent.exists():
                for item in parent.iterdir():
                    name = item.name
                    if name.startswith(prefix) and not name.startswith("."):
                        display = name + ("/" if item.is_dir() else "")
                        yield Completion(
                            str(item),
                            start_position=-len(partial),
                            display=display,
                            display_meta="dir" if item.is_dir() else "file"
                        )
        except Exception:
            pass


def complete_agent_names(incomplete: str) -> List[str]:
    """Complete agent names from local registry."""
    # Scan current directory for agent files
    agent_names = []
    
    cwd = Path.cwd()
    for f in cwd.glob("AGENT*.md"):
        if f.name == "AGENT.md":
            agent_names.append("default")
        elif f.name.startswith("AGENT-"):
            name = f.stem.replace("AGENT-", "")
            agent_names.append(name)
    
    return [n for n in agent_names if n.startswith(incomplete)]


def complete_skill_names(incomplete: str) -> List[str]:
    """Complete available skill names."""
    skills = [
        "cron", "folder-index", "llm", "mcp", 
        "memory", "discovery", "web", "filesystem",
        "database"
    ]
    return [s for s in skills if s.startswith(incomplete)]


def complete_template_names(incomplete: str) -> List[str]:
    """Complete template names."""
    templates = [
        "assistant", "planning", "marketing", 
        "content", "code-review", "research"
    ]
    return [t for t in templates if t.startswith(incomplete)]


def complete_namespace_names(incomplete: str) -> List[str]:
    """Complete namespace names."""
    # For now, just return local
    namespaces = ["local"]
    return [n for n in namespaces if n.startswith(incomplete)]
