"""
Filesystem Skill

Local file operations with proper sandboxing using whitelist/blacklist.
Matches Gemini CLI file system tools specification.
"""

import base64
import mimetypes
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Dict, Any, Union

from ...base import Skill
from webagents.agents.tools.decorators import tool


class FilesystemSkill(Skill):
    """Filesystem operations with sandboxing matching Gemini CLI specs"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Sandboxing config
        self.base_dir = Path(config.get("base_dir", Path.cwd())) if config else Path.cwd()
        self.whitelist: Set[Path] = self._load_whitelist()
        self.blacklist: Set[Path] = self._load_blacklist()
        
        # Optional checkpoint manager for auto-snapshots before modifications
        self._checkpoint_manager = None
        self._auto_checkpoint = config.get("auto_checkpoint", False) if config else False
    
    def set_checkpoint_manager(self, manager) -> None:
        """Set checkpoint manager for auto-snapshots.
        
        Args:
            manager: CheckpointManager instance
        """
        self._checkpoint_manager = manager
        self._auto_checkpoint = True
    
    def _trigger_checkpoint(self, file_path: Path, operation: str = "modify") -> None:
        """Trigger a checkpoint before file modification if auto_checkpoint is enabled."""
        if self._auto_checkpoint and self._checkpoint_manager:
            try:
                description = f"Auto-checkpoint before {operation}: {file_path.name}"
                self._checkpoint_manager.create(description=description, files=[file_path])
            except Exception:
                pass  # Don't fail the operation if checkpoint fails
    
    def _load_whitelist(self) -> Set[Path]:
        """Load whitelisted directories from config"""
        whitelist = {self.base_dir}  # Always allow agent's directory
        
        if self.config and "whitelist" in self.config:
            whitelist.update(Path(p).resolve() for p in self.config["whitelist"])
        
        return whitelist
    
    def _load_blacklist(self) -> Set[Path]:
        """Load blacklisted paths from config"""
        # Default sensitive directories
        blacklist = {
            Path.home() / ".ssh",
            Path.home() / ".aws",
            Path.home() / ".config" / "gcloud",
            Path.home() / ".gnupg",
        }
        
        if self.config and "blacklist" in self.config:
            blacklist.update(Path(p).resolve() for p in self.config["blacklist"])
        
        return blacklist
    
    def _check_access(self, path: Path) -> bool:
        """Check if path is allowed"""
        try:
            resolved = path.resolve()
        except Exception:
            # If path doesn't exist, check parent
            try:
                resolved = path.parent.resolve() / path.name
            except Exception:
                # If parent doesn't exist either, rely on absolute path check against whitelist
                resolved = path.absolute()

        # Check blacklist first
        for blocked in self.blacklist:
            try:
                # Check if resolved path is relative to blocked path
                if resolved == blocked or blocked in resolved.parents:
                    return False
            except Exception:
                pass
        
        # Check whitelist
        for allowed in self.whitelist:
            try:
                if resolved == allowed or allowed in resolved.parents:
                    return True
            except Exception:
                pass
        
        return False
    
    def _resolve_path(self, path_str: str, base_dir: Optional[Path] = None) -> Path:
        """Resolve path relative to base_dir if relative"""
        path = Path(path_str).expanduser()
        root = base_dir if base_dir else self.base_dir
        
        if not path.is_absolute():
            return (root / path).resolve()
        return path.resolve()

    def _is_binary(self, path: Path) -> bool:
        """Check if file is binary"""
        # Simple check: read first chunk and look for null bytes
        try:
            with open(path, 'rb') as f:
                chunk = f.read(1024)
                return b'\0' in chunk
        except Exception:
            return False

    def _get_mime_type(self, path: Path) -> str:
        mime, _ = mimetypes.guess_type(path)
        return mime or "application/octet-stream"

    def _is_git_ignored(self, path: Path) -> bool:
        """Check if path is git ignored using git check-ignore"""
        if not shutil.which("git"):
            return False
            
        try:
            # Check if inside git repo
            cwd = path.parent if path.exists() else self.base_dir
            subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=cwd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            
            result = subprocess.run(
                ["git", "check-ignore", str(path)],
                cwd=cwd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return result.returncode == 0
        except Exception:
            return False

    @tool
    async def list_directory(self, path: str, ignore: Optional[List[str]] = None, respect_git_ignore: bool = True) -> str:
        """Lists files and subdirectories in a directory.
        
        Args:
            path: The absolute path to the directory to list.
            ignore: Optional list of glob patterns to exclude.
            respect_git_ignore: Whether to respect .gitignore patterns.
            
        Returns:
            Directory listing with [DIR] prefix for directories.
        """
        dir_path = self._resolve_path(path)
        
        if not self._check_access(dir_path):
            return f"Access denied: {path} is outside allowed directories"
        
        if not dir_path.exists():
            return f"Directory not found: {path}"
        
        if not dir_path.is_dir():
            return f"Not a directory: {path}"
            
        entries = []
        try:
            for item in dir_path.iterdir():
                name = item.name
                
                # Check ignore patterns
                if ignore:
                    should_ignore = False
                    for pattern in ignore:
                        if item.match(pattern):
                            should_ignore = True
                            break
                    if should_ignore:
                        continue
                
                # Check gitignore
                if respect_git_ignore and self._is_git_ignored(item):
                    continue
                
                prefix = "[DIR] " if item.is_dir() else ""
                entries.append((item.is_dir(), f"{prefix}{name}"))
                
            # Sort: Directories first, then alphabetical
            entries.sort(key=lambda x: (not x[0], x[1]))
            
            result = [f"Directory listing for {dir_path}:"]
            result.extend([e[1] for e in entries])
            
            return "\n".join(result)
            
        except Exception as e:
            return f"Error listing directory: {e}"

    @tool
    async def read_file(self, path: str, offset: Optional[int] = None, limit: Optional[int] = None) -> Union[str, Dict[str, Any]]:
        """Reads and returns the content of a specified file.
        
        Args:
            path: The absolute path to the file to read.
            offset: Start line number (0-based, requires limit).
            limit: Maximum number of lines to read.
            
        Returns:
            File content or object with inlineData for binaries.
        """
        file_path = self._resolve_path(path)
        
        if not self._check_access(file_path):
            return f"Access denied: {path} is outside allowed directories"
        
        if not file_path.exists():
            return f"File not found: {path}"
            
        if not file_path.is_file():
            return f"Not a file: {path}"
            
        mime_type = self._get_mime_type(file_path)
        is_media = any(t in mime_type for t in ['image/', 'audio/', 'application/pdf'])
        
        if is_media:
            try:
                data = base64.b64encode(file_path.read_bytes()).decode('utf-8')
                return {
                    "inlineData": {
                        "mimeType": mime_type,
                        "data": data
                    }
                }
            except Exception as e:
                return f"Error reading media file: {e}"
        
        # Check if binary but not supported media
        if self._is_binary(file_path):
            return f"Cannot display content of binary file: {path}"
            
        # Text file
        try:
            lines = file_path.read_text(encoding='utf-8', errors='replace').splitlines()
            total_lines = len(lines)
            
            start = offset if offset is not None else 0
            end = total_lines
            
            if limit is not None:
                end = min(start + limit, total_lines)
            elif offset is None and total_lines > 2000:
                # Default limit if not specified and file is large
                end = 2000
                
            content_lines = lines[start:end]
            content = "\n".join(content_lines)
            
            if start > 0 or end < total_lines:
                return f"[File content truncated: showing lines {start+1}-{end} of {total_lines} total lines...]\n{content}"
            
            return content
            
        except Exception as e:
            return f"Error reading file: {e}"

    @tool
    async def write_file(self, file_path: str, content: str) -> str:
        """Writes content to a specified file.
        
        Args:
            file_path: The absolute path to the file.
            content: The content to write.
            
        Returns:
            Success message.
        """
        path = self._resolve_path(file_path)
        
        if not self._check_access(path):
            return f"Access denied: {file_path} is outside allowed directories"
            
        try:
            exists = path.exists()
            
            # Trigger checkpoint before modification if enabled
            if exists:
                self._trigger_checkpoint(path, "overwrite")
            
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
            
            if exists:
                return f"Successfully overwrote file: {file_path}"
            else:
                return f"Successfully created and wrote to new file: {file_path}"
        except Exception as e:
            return f"Error writing file: {e}"

    @tool
    async def glob(self, pattern: str, path: Optional[str] = None, case_sensitive: bool = False, respect_git_ignore: bool = True) -> str:
        """Finds files matching specific glob patterns.
        
        Args:
            pattern: Glob pattern (e.g., "*.ts").
            path: Directory to search in (default: root).
            case_sensitive: Whether search is case sensitive.
            respect_git_ignore: Whether to respect .gitignore.
            
        Returns:
            List of absolute paths sorted by modification time (newest first).
        """
        search_dir = self._resolve_path(path) if path else self.base_dir
        
        if not self._check_access(search_dir):
            return f"Access denied: {path} is outside allowed directories"
            
        if not search_dir.exists():
            return f"Directory not found: {search_dir}"
            
        try:
            # Note: pathlib glob matches case-sensitively on POSIX.
            
            # Recursive glob if pattern contains **
            matches = []
            
            # If pattern is relative, we use glob.
            all_files = list(search_dir.glob(pattern))
            
            results = []
            for p in all_files:
                if not p.is_file():
                    continue
                    
                if respect_git_ignore and self._is_git_ignored(p):
                    continue
                
                results.append(p)
                
            # Sort by modification time (newest first)
            results.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            if not results:
                return f"Found 0 file(s) matching \"{pattern}\" within {search_dir}"
                
            output = [f"Found {len(results)} file(s) matching \"{pattern}\" within {search_dir}, sorted by modification time (newest first):"]
            output.extend([str(p) for p in results])
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Error in glob: {e}"

    @tool
    async def search_file_content(self, pattern: str, path: Optional[str] = None, include: Optional[str] = None) -> str:
        """Searches for a regex pattern within files.
        
        Args:
            pattern: Regex pattern to search for.
            path: Directory to search in.
            include: Glob pattern to filter files.
            
        Returns:
            Matches with line numbers.
        """
        search_dir = self._resolve_path(path) if path else self.base_dir
        
        if not self._check_access(search_dir):
            return f"Access denied: {path} is outside allowed directories"
            
        results = []
        try:
            # Walk files
            files_to_search = []
            if include:
                files_to_search = list(search_dir.glob(include))
            else:
                files_to_search = list(search_dir.rglob("*"))
                
            regex = re.compile(pattern)
            
            count = 0
            file_matches = []
            
            for p in files_to_search:
                if not p.is_file():
                    continue
                if self._is_binary(p):
                    continue
                # Skip .git
                if ".git" in p.parts:
                    continue
                    
                try:
                    lines = p.read_text(encoding='utf-8', errors='ignore').splitlines()
                    matches_in_file = []
                    for i, line in enumerate(lines):
                        if regex.search(line):
                            matches_in_file.append(f"L{i+1}: {line}")
                            count += 1
                            
                    if matches_in_file:
                        rel_path = p.relative_to(search_dir)
                        file_matches.append(f"File: {rel_path}\n" + "\n".join(matches_in_file))
                        
                except Exception:
                    continue
            
            if not file_matches:
                return f"Found 0 matches for pattern \"{pattern}\" in {search_dir}"
                
            header = f"Found {count} matches for pattern \"{pattern}\" in {search_dir}"
            if include:
                header += f" (filter: \"{include}\")"
            header += ":"
            
            return header + "\n---\n" + "\n---\n".join(file_matches) + "\n---"
            
        except Exception as e:
            return f"Error searching content: {e}"

    @tool
    async def replace(self, file_path: str, old_string: str, new_string: str, expected_replacements: int = 1) -> str:
        """Replaces text within a file.
        
        Args:
            file_path: Absolute path to the file.
            old_string: Exact text to replace. Empty to create new file.
            new_string: New text.
            expected_replacements: Number of occurrences to replace.
            
        Returns:
            Success or failure message.
        """
        path = self._resolve_path(file_path)
        
        if not self._check_access(path):
            return f"Access denied: {file_path} is outside allowed directories"
            
        # Case: Create new file
        if not old_string:
            if path.exists():
                return f"Failed to edit: old_string is empty but file {file_path} already exists."
            
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(new_string, encoding='utf-8')
                return f"Created new file: {file_path} with provided content."
            except Exception as e:
                return f"Error creating file: {e}"
        
        # Case: Replace in existing file
        if not path.exists():
            return f"Failed to edit: file {file_path} does not exist."
            
        # Trigger checkpoint before modification if enabled
        self._trigger_checkpoint(path, "replace")
            
        try:
            content = path.read_text(encoding='utf-8')
            
            # Check occurrences
            count = content.count(old_string)
            
            if count == 0:
                return f"Failed to edit, 0 occurrences found of old_string. Please ensure exact match including whitespace."
                
            if count != expected_replacements:
                 if count < expected_replacements:
                     return f"Failed to edit, expected {expected_replacements} occurrences but found {count}."
                 
                 if count > expected_replacements and expected_replacements == 1:
                     return f"Failed to edit, expected 1 occurrence but found {count}. Please provide more context to disambiguate."
            
            new_content = content.replace(old_string, new_string, expected_replacements)
            
            path.write_text(new_content, encoding='utf-8')
            
            return f"Successfully modified file: {file_path} ({expected_replacements} replacements)."
            
        except Exception as e:
            return f"Error replacing text: {e}"
