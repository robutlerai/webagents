"""
Filesystem Skill - WebAgents V2.0
Local file operations with optional sandbox enforcement.
"""

import os
import glob
from typing import Dict, List, Any, Optional
from pathlib import Path

from ...base import Skill
from webagents.agents.tools.decorators import tool

class FilesystemSkill(Skill):
    """
    Skill for local filesystem operations.
    
    Tools:
    - list_files: List files in a directory
    - read_file: Read content of a file
    - write_file: Write content to a file
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.base_dir = Path(config.get('base_dir', os.getcwd())).resolve() if config else Path(os.getcwd()).resolve()

    @tool
    async def list_files(self, path: str = ".", pattern: str = "*") -> List[str]:
        """
        List files in the current working directory or a specified subdirectory. 
        Use this tool when the user asks to "list files", "show files", or "ls".
        
        Args:
            path: Directory path (defaults to current directory ".")
            pattern: Glob pattern to filter files (e.g. "*.py")
        """
        target_dir = (self.base_dir / path).resolve()
        
        # Security check: stay within base_dir
        if not str(target_dir).startswith(str(self.base_dir)):
            return ["Error: Access denied (outside of base directory)"]
            
        if not target_dir.exists() or not target_dir.is_dir():
            return [f"Error: Directory {path} does not exist"]
            
        files = glob.glob(os.path.join(target_dir, pattern))
        return [os.path.relpath(f, self.base_dir) for f in files]

    @tool
    async def read_file(self, path: str) -> str:
        """
        Read the content of a file.
        
        Args:
            path: Path to the file
        """
        target_file = (self.base_dir / path).resolve()
        
        if not str(target_file).startswith(str(self.base_dir)):
            return "Error: Access denied"
            
        if not target_file.exists() or not target_file.is_file():
            return f"Error: File {path} not found"
            
        try:
            return target_file.read_text(encoding='utf-8')
        except Exception as e:
            return f"Error reading file: {str(e)}"

    @tool
    async def write_file(self, path: str, content: str) -> str:
        """
        Write content to a file.
        
        Args:
            path: Path to the file
            content: Content to write
        """
        target_file = (self.base_dir / path).resolve()
        
        if not str(target_file).startswith(str(self.base_dir)):
            return "Error: Access denied"
            
        try:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(content, encoding='utf-8')
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"
