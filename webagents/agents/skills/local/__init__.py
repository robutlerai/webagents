"""
Local Skills - Skills for Local Agent Operations

Provides skills for local development and CLI-based agent operations:
- CheckpointSkill: Git-based file checkpointing
- CLISkill: Session management and CLI operations
- FilesystemSkill: File operations
- PluginSkill: Plugin management with marketplace discovery
- RAGSkill: Retrieval-augmented generation
- SessionSkill: Session state management
- ShellSkill: Shell command execution
- WebUISkill: Web-based dashboard UI
"""

from .checkpoint.skill import CheckpointSkill
from .cli.skill import CLISkill
from .plugin import PluginSkill
from .webui import WebUISkill

__all__ = [
    "CheckpointSkill",
    "CLISkill",
    "PluginSkill",
    "WebUISkill",
]
