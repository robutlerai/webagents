"""
WEBAGENTS.md Context Parser

Parse and manage WEBAGENTS.md context files: directory-scoped defaults and
background text inherited by every agent beneath them.

WHY THE FILE IS NOT CALLED AGENTS.md (renamed 2026-09-23). It used to be.
`AGENTS.md` has since become a cross-vendor standard stewarded by the Agentic
AI Foundation, and what people put in it is instructions for a CODING agent
working on the repository: build commands, test invocations, house style. This
loader merges whatever it finds into the SYSTEM PROMPT of the agent being run,
under a `## Background Context` heading. So on any repo that already had one
for Codex, Cursor or Claude Code, webagents silently absorbed a file written
for a different tool and a different purpose.

There is deliberately NO back-compat read of `AGENTS.md`. A fallback would keep
exactly the behaviour the rename exists to stop.
"""

import re
from pathlib import Path
from typing import Any, List, Optional
import yaml

from .schema import AgentFormatError, ContextMetadata

#: The one filename this loader reads. See the module docstring.
CONTEXT_FILENAME = "WEBAGENTS.md"

#: Entries that mark a directory as the top of a project. The upward walk stops
#: here (inclusive: the marked directory's own context file IS read).
#:
#: `.git` covers repositories, including worktrees where `.git` is a file rather
#: than a directory, which is why this tests `exists()` and not `is_dir()`.
#: `.webagents` is the escape hatch for a project that is not a git checkout:
#: creating it declares "the walk stops here".
#:
#: Deliberately NOT `pyproject.toml` or `package.json`: in a monorepo those sit
#: on every package, and stopping at one would silently drop the repo-level
#: context that the layout exists to share.
PROJECT_MARKERS = (".git", ".webagents")


def entry_identity(item: Any) -> str:
    """A hashable identity for a skill / tool / mcp_server entry.

    These lists are `List[Union[str, Dict[str, Any]]]`, so the obvious
    `dict.fromkeys(...)` deduplication raised `TypeError: unhashable type:
    'dict'` on any context file using the documented dict form
    (`- mcp: {...}`) (fixed 2026-09-23). `mcp_servers` hit it hardest, since a
    server entry is almost always a mapping.

    A dict is identified by its FIRST key, which is the skill or server name in
    the dict form. `merge_entries` below is the only caller, and
    `hierarchy._merge_metadata` goes through it too, so there is one answer to
    "is this the same entry" across the whole loader.
    """
    if isinstance(item, str):
        return item
    if isinstance(item, dict) and item:
        return str(next(iter(item)))
    return repr(item)


def merge_entries(*lists: List[Any]) -> List[Any]:
    """Combine entry lists: first POSITION wins, last VALUE wins.

    One rule for every skills / tools / mcp_servers merge in the loader, used
    both for stacking context files on each other and for stacking the agent on
    top of them. Two entries collide when `entry_identity` matches.

    WHY LAST VALUE WINS. The lists arrive outermost-first, so the nearer file
    is the later one, and "nearer overrides" is the inheritance model the rest
    of the merge already follows for `namespace` and `model`. The previous
    behaviour kept the FIRST value, so a leaf directory could add a skill it
    already inherited but could never reconfigure one, and an agent's own
    configuration lost to an ancestor's (fixed 2026-09-23).

    Position comes from the first appearance so inherited entries keep their
    order and genuinely new ones append.
    """
    order: List[str] = []
    by_key: dict = {}
    for items in lists:
        for item in items:
            key = entry_identity(item)
            if key not in by_key:
                order.append(key)
            by_key[key] = item
    return [by_key[key] for key in order]


class ContextFile:
    """Represents a parsed WEBAGENTS.md context file."""
    
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
            
            try:
                self.metadata = ContextMetadata(**self._raw_yaml)
            except AgentFormatError as exc:
                raise AgentFormatError(f"{self.path}: {exc}") from None
            self.content = body.strip()
        else:
            # No frontmatter
            self.metadata = ContextMetadata()
            self.content = content.strip()

    def declares(self, field: str) -> bool:
        """Whether the file actually SET this field, as opposed to inheriting
        the schema default.

        `namespace` and `visibility` both default to `"local"`, so a plain
        truthiness test cannot tell "this file says local" from "this file says
        nothing". Pydantic tracks the difference in `model_fields_set`, so the
        merge can stop treating silence as an override (2026-09-23).
        """
        return field in self.metadata.model_fields_set
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "path": str(self.path),
            "metadata": self.metadata.model_dump(),
            "content": self.content,
        }


class ContextHierarchy:
    """Manage WEBAGENTS.md context hierarchy.
    
    Context files inherit from parent directories:
    - project/WEBAGENTS.md
      - project/subdir/WEBAGENTS.md (inherits from parent)
        - project/subdir/AGENT.md (inherits from both)
    """
    
    def __init__(self, stop_at: Optional[Path] = None):
        """Initialize hierarchy resolver.
        
        Args:
            stop_at: Stop traversing at this directory (e.g., project root).
                Given explicitly, it REPLACES the default bounds below rather
                than adding to them: the caller has said where the top is.
        """
        self.stop_at = Path(stop_at).resolve() if stop_at else None
    
    def resolve(self, start_path: Path) -> List[ContextFile]:
        """Resolve context hierarchy from a starting path.

        THE WALK IS BOUNDED (2026-09-23). It used to climb to the filesystem
        root, so running an agent from `~/dev/proj/sub` read `~/dev`, `~`,
        `/Users` and `/`, merging anything it found there into the system
        prompt. With no `stop_at`, it now stops at the first project marker
        (inclusive) and never reaches or passes the home directory.
        
        Args:
            start_path: Path to start from (usually directory containing AGENT.md)
            
        Returns:
            List of ContextFile instances, ordered from root to local
            (i.e., outermost first, so local can override)
        """
        context_files = []
        start = Path(start_path).resolve()
        current = start if start.is_dir() else start.parent

        try:
            home = Path.home().resolve()
        except (RuntimeError, OSError):
            home = None

        while True:
            # The home directory is a boundary, not a project. Checked BEFORE
            # reading, so `~/WEBAGENTS.md` is never merged: a dotfiles repo puts
            # a `.git` right there, and inheriting from it would give every
            # agent on the machine the same silent preamble.
            if self.stop_at is None and home is not None and current == home:
                break

            candidate = current / CONTEXT_FILENAME
            if candidate.exists():
                context_files.append(ContextFile(candidate))

            # Explicit stop wins outright.
            if self.stop_at is not None and current == self.stop_at:
                break

            if self.stop_at is None and self._is_project_root(current):
                break

            if current == current.parent:  # filesystem root
                break

            current = current.parent
        
        # Reverse so root is first, local is last (for override order)
        return list(reversed(context_files))

    @staticmethod
    def _is_project_root(directory: Path) -> bool:
        """Whether this directory is the top of a project. See PROJECT_MARKERS."""
        return any((directory / marker).exists() for marker in PROJECT_MARKERS)
    
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
        
        per_file: dict = {"skills": [], "tools": [], "mcp_servers": []}

        for ctx in contexts:
            # Namespace is overridden, but only by a file that actually SAYS a
            # namespace. `namespace` defaults to "local", so testing
            # truthiness made every silent child reset its parent's namespace
            # back to local (2026-09-23).
            if ctx.declares("namespace"):
                merged["namespace"] = ctx.metadata.namespace
            
            # Model is overridden
            if ctx.metadata.model:
                merged["model"] = ctx.metadata.model
            
            # Skills, tools, mcp_servers are accumulated, per file, so that
            # `merge_entries` below can tell which file an entry came from and
            # let the nearer one win.
            per_file["skills"].append(list(ctx.metadata.skills))
            per_file["tools"].append(list(ctx.metadata.tools))
            per_file["mcp_servers"].append(list(ctx.metadata.mcp_servers))
            
            # Visibility is overridden, same rule as namespace above.
            if ctx.declares("visibility"):
                merged["visibility"] = ctx.metadata.visibility
            
            # Sandbox is merged (later overrides)
            if ctx.metadata.sandbox:
                merged["sandbox"] = ctx.metadata.sandbox.model_dump()
            
            # Instructions are accumulated
            if ctx.content:
                merged["instructions"].append(ctx.content)
        
        # Deduplicate. `merge_entries` rather than `dict.fromkeys`: these lists
        # can hold dicts, which are unhashable, and the nearer file's version
        # of a repeated entry is the one to keep.
        for field in ("skills", "tools", "mcp_servers"):
            merged[field] = merge_entries(*per_file[field])
        
        return merged


def find_context_file(directory: Path) -> Optional[ContextFile]:
    """Find WEBAGENTS.md in a directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        ContextFile if found, None otherwise
    """
    candidate = Path(directory) / CONTEXT_FILENAME
    if candidate.exists():
        return ContextFile(candidate)
    return None


def create_context_file(
    directory: Path,
    metadata: Optional[dict] = None,
    content: str = "",
) -> ContextFile:
    """Create a new WEBAGENTS.md context file.
    
    Args:
        directory: Directory to create file in
        metadata: YAML frontmatter data
        content: Markdown content
        
    Returns:
        Created ContextFile
    """
    path = Path(directory) / CONTEXT_FILENAME
    
    meta = ContextMetadata(**(metadata or {}))
    
    yaml_data = meta.model_dump(exclude_none=True)
    yaml_str = yaml.dump(yaml_data, default_flow_style=False, sort_keys=False)
    
    if not content:
        content = "# Project Context\n\nThis file provides context for all agents in this directory."
    
    file_content = f"---\n{yaml_str}---\n\n{content}"
    
    path.write_text(file_content)
    return ContextFile(path)
