"""
WebAgents Loader Module

Parse and load AGENT.md, AGENT-*.md, and AGENTS.md files.
"""

from .schema import AgentMetadata, ContextMetadata, TemplateMetadata, SandboxConfig
from .agent_md import AgentFile, parse_agent_file, parse_frontmatter, find_agent_files, create_agent_file
from .context import ContextFile, ContextHierarchy, find_context_file, create_context_file
from .hierarchy import AgentLoader, MergedAgent, find_default_agent, load_agent

__all__ = [
    # Schema
    "AgentMetadata",
    "ContextMetadata", 
    "TemplateMetadata",
    "SandboxConfig",
    # Agent files
    "AgentFile",
    "parse_agent_file",
    "parse_frontmatter",
    "find_agent_files",
    "create_agent_file",
    # Context files
    "ContextFile",
    "ContextHierarchy",
    "find_context_file",
    "create_context_file",
    # Loading
    "AgentLoader",
    "MergedAgent",
    "find_default_agent",
    "load_agent",
]
