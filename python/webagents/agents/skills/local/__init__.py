"""
Local Skills - Skills for Local Agent Operations

Provides skills for local development and CLI-based agent operations:
- CLISkill: Session management and CLI operations
- FilesystemSkill: File operations
- PluginSkill: Plugin management with marketplace discovery
- RAGSkill: Retrieval-augmented generation
- SecretsSkill: Named credentials in the OS keystore
- SessionSkill: An agent keeps its callers' conversations (`session/`)
- ShellSkill: Shell command execution
- WebUISkill: Web-based dashboard UI
"""

from .cli.skill import CLISkill
from .plugin import PluginSkill
from .secrets import SecretsSkill
from .webui import WebUISkill

__all__ = [
    "CLISkill",
    "PluginSkill",
    "SecretsSkill",
    "WebUISkill",
]
