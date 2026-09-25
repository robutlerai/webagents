"""
AGENT.md Parser

Parse AGENT.md and AGENT-*.md files with YAML frontmatter.
"""

import re
from pathlib import Path
from typing import Optional, Tuple
import yaml

from .schema import AgentFormatError, AgentMetadata


class AgentFile:
    """Represents a parsed AGENT.md or AGENT-*.md file."""
    
    def __init__(self, path: Path):
        self.path = Path(path)
        self.metadata: AgentMetadata
        self.instructions: str
        self._raw_yaml: dict = {}
        self._snapshot: dict = {}
        self._parse()
    
    def _parse(self):
        """Parse the file into metadata and instructions."""
        content = self.path.read_text()
        yaml_data, body = parse_frontmatter(content)
        
        self._raw_yaml = yaml_data
        
        # Parse metadata with defaults
        if yaml_data:
            # The schema rejects unknown keys, so the message has to say WHICH
            # file: a scan reports several and "unknown key 'skils'" on its own
            # sends the reader hunting.
            try:
                self.metadata = AgentMetadata(**yaml_data)
            except AgentFormatError as exc:
                raise AgentFormatError(f"{self.path}: {exc}") from None
        else:
            # No frontmatter, use defaults
            self.metadata = AgentMetadata()
            
            # Try to infer name from filename
            if self.path.name == "AGENT.md":
                self.metadata.name = "default"
            elif self.path.name.startswith("AGENT-"):
                self.metadata.name = self.path.stem.replace("AGENT-", "")
        
        self.instructions = body.strip() if body else ""

        # What the file said, AFTER validation and default-filling. `_save`
        # diffs against this to find genuine edits; see `_pending_updates`.
        self._snapshot = self.metadata.model_dump()
    
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
        """Write changed keys back into the existing frontmatter, in place.

        THIS USED TO REBUILD THE WHOLE BLOCK with `yaml.dump`, and the cost was
        silent and unrecoverable (fixed 2026-09-23):

          * COMMENTS WERE DESTROYED. `yaml.dump` does not round-trip them, so
            a line like `# Production agent. Do not change the namespace
            without telling ops.` vanished on the first `/skill add`. Only the
            author can restore that.
          * KEYS EQUAL TO THE SCHEMA DEFAULT WERE DELETED, which stopped being
            merely cosmetic once the loader learned to distinguish "declared"
            from "defaulted": deleting an explicit `namespace: local` changes
            the agent from pinned-to-local into inherits-from-context.
          * Key order and formatting were rewritten for no reason.

        So nothing is re-serialised. Each changed key's span is located in the
        original text and replaced; every other byte, including every comment
        outside a changed key, survives. `ruamel.yaml` would do this more
        generally, but it is a new hard dependency on a published SDK for the
        sake of three mutators, and this needs none.

        The one residual loss is a comment INSIDE the block of a key that is
        being rewritten. That is inherent to changing that value, and it is
        bounded to the key you asked to change rather than the whole file.
        """
        updates = self._pending_updates()
        if not updates:
            return

        # Read as BYTES and restore the file's own line endings on the way out.
        # `read_text` applies universal newlines and `write_text` emits the
        # platform's, so a CRLF file came back entirely LF: every line changed,
        # which is precisely the whole-file diff this method exists to avoid.
        raw = self.path.read_bytes()
        crlf = b"\r\n" in raw
        text = raw.decode("utf-8").replace("\r\n", "\n")

        edited = _apply_frontmatter_updates(text, updates, self.instructions)
        if crlf:
            edited = edited.replace("\n", "\r\n")
        self.path.write_bytes(edited.encode("utf-8"))

        self._raw_yaml = {**self._raw_yaml, **updates}
        self._snapshot = self.metadata.model_dump()

    def _pending_updates(self) -> dict:
        """Keys a mutator actually changed since the file was parsed.

        Compared against a SNAPSHOT taken at parse time, not against the raw
        YAML. Comparing to the raw YAML looks obvious and is wrong: `model_dump`
        normalises and fills in defaults, so a file writing

            sandbox:
              preset: strict

        dumps as four fields, never equals its own two-field source, and was
        therefore rewritten on every save with three defaults the author never
        typed. Snapshot and current are both dumps, so normalisation appears on
        both sides and cancels, leaving only real edits.

        Keys are only ever added or updated, never removed: none of the
        mutators above needs to delete a top-level key, and "write only what
        changed" is what keeps the rest of the file untouched.
        """
        current = self.metadata.model_dump()
        return {
            key: value
            for key, value in current.items()
            if value != self._snapshot.get(key)
        }
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "path": str(self.path),
            "name": self.name,
            "metadata": self.metadata.model_dump(),
            "instructions": self.instructions,
        }


def _frontmatter_key_span(lines: list, key: str) -> Optional[Tuple[int, int]]:
    """Locate a TOP-LEVEL key's lines within frontmatter, as a half-open range.

    Top level means column zero, which is the only shape these files use. The
    span runs from the key line through its indented continuation (block
    mappings, block sequences, and sequences written unindented at column
    zero), stopping at the next column-zero key. Trailing blank lines and
    trailing column-zero comments are left OUT of the span so that replacing a
    value cannot swallow the comment introducing the key after it.
    """
    key_line = re.compile(rf"^{re.escape(key)}\s*:")
    for start in range(len(lines)):
        if not key_line.match(lines[start]):
            continue

        end = start + 1
        while end < len(lines):
            line = lines[end]
            if line.strip() == "" or line.startswith((" ", "\t", "-")):
                end += 1
                continue
            break

        # Do not absorb trailing blanks or a comment that introduces what follows.
        while end - 1 > start and (
            lines[end - 1].strip() == "" or lines[end - 1].lstrip().startswith("#")
        ):
            end -= 1
        return start, end
    return None


def _match_sequence_indent(original: list, replacement: list) -> list:
    """Re-indent a rendered block sequence to match the lines it replaces.

    `yaml.dump` writes sequence items flush against the key (`- item`), while
    these files are usually written indented (`  - item`). Mixing both styles
    in one file reads as damage even though both parse, so the replacement
    borrows whatever the original used.
    """
    indents = {
        len(line) - len(line.lstrip(" "))
        for line in original
        if line.lstrip(" ").startswith("- ") or line.strip() == "-"
    }
    if len(indents) != 1:
        return replacement

    indent = " " * indents.pop()
    if not indent:
        return replacement
    return [
        indent + line if line.startswith("- ") or line == "-" else line
        for line in replacement
    ]


def _apply_frontmatter_updates(text: str, updates: dict, instructions: str) -> str:
    """Return `text` with `updates` applied to its frontmatter, byte-preserving.

    A file with no frontmatter gets one, built from the updates alone.
    """
    match = re.match(r"^(\s*---\s*\n)(.*?)(\n---\s*\n?)(.*)$", text, re.DOTALL)
    if not match:
        # No frontmatter to preserve, so there is nothing to lose by writing a
        # fresh block.
        rendered = yaml.dump(updates, default_flow_style=False, sort_keys=False)
        return f"---\n{rendered}---\n\n{instructions}"

    open_fence, block, close_fence, body = match.groups()
    lines = block.split("\n")

    for key, value in updates.items():
        rendered = yaml.dump({key: value}, default_flow_style=False, sort_keys=False)
        replacement = rendered.rstrip("\n").split("\n")

        span = _frontmatter_key_span(lines, key)
        if span is None:
            lines.extend(replacement)
        else:
            start, end = span
            replacement = _match_sequence_indent(lines[start:end], replacement)
            lines[start:end] = replacement

    return open_fence + "\n".join(lines) + close_fence + body


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
    files, _ = scan_agent_files(directory, recursive=recursive)
    return files


def scan_agent_files(directory: Path, recursive: bool = False) -> Tuple[list, list]:
    """Find agent files, separating the ones that would not parse.

    ONE BAD FILE MUST NOT HIDE THE REST (2026-09-23). `find_agent_files`
    constructed every `AgentFile` eagerly, which was harmless while the schema
    accepted anything and became a directory-wide failure the moment it started
    rejecting unknown keys. A scan is a survey, so it reports what it found AND
    what it could not read, and the caller decides which matters.

    Args:
        directory: Directory to search
        recursive: Search subdirectories

    Returns:
        `(files, errors)`, where `errors` is a list of `(path, message)` for
        each file that could not be parsed.
    """
    pattern = "**/" if recursive else ""
    agent_files = []
    errors = []

    paths = sorted(directory.glob(f"{pattern}AGENT.md")) + sorted(
        directory.glob(f"{pattern}AGENT-*.md")
    )
    for f in paths:
        try:
            agent_files.append(AgentFile(f))
        except AgentFormatError as exc:
            errors.append((f, str(exc)))
        except Exception as exc:  # unreadable file, bad encoding, bad types
            errors.append((f, f"{f}: {exc}"))

    return agent_files, errors


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
