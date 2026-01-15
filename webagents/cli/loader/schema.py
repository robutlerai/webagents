"""
YAML Schema Definitions

Pydantic models for AGENT.md and AGENTS.md YAML frontmatter.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SandboxConfig(BaseModel):
    """Sandbox configuration in YAML frontmatter."""
    preset: str = "development"  # strict, development, unrestricted
    allowed_folders: List[str] = Field(default_factory=lambda: ["."])
    allowed_commands: List[str] = Field(default_factory=list)
    allowed_imports: List[str] = Field(default_factory=list)


class AgentMetadata(BaseModel):
    """YAML frontmatter schema for AGENT.md and AGENT-*.md files."""
    
    # Identity (required)
    name: str = "assistant"
    description: str = "An AI assistant"
    
    # Namespace
    namespace: str = "local"
    
    # Discovery - intents (unified discovery mechanism)
    intents: List[str] = Field(default_factory=list)
    # Example: ["summarize documents", "create weekly reports", "parse PDFs"]
    
    # Configuration
    model: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    
    # Triggers
    cron: Optional[str] = None  # Cron expression: "0 18 * * 1-5"
    watch: Optional[List[str]] = None  # File patterns to watch
    
    # Visibility
    visibility: str = "local"  # local | namespace | public
    
    # Sandbox overrides
    sandbox: Optional[SandboxConfig] = None
    
    # Additional metadata
    version: str = "1.0.0"
    author: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    # MCP servers
    mcp_servers: List[str] = Field(default_factory=list)
    
    class Config:
        extra = "allow"  # Allow additional fields


class ContextMetadata(BaseModel):
    """YAML frontmatter schema for AGENTS.md context files."""
    
    # Namespace for all agents in this directory
    namespace: str = "local"
    
    # Default configuration for agents
    model: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    
    # Sandbox defaults
    sandbox: Optional[SandboxConfig] = None
    
    # Default visibility
    visibility: str = "local"
    
    # MCP servers available to all agents
    mcp_servers: List[str] = Field(default_factory=list)
    
    # Additional context
    guidelines: Optional[str] = None
    
    class Config:
        extra = "allow"  # Allow additional fields


class TemplateMetadata(BaseModel):
    """YAML frontmatter schema for TEMPLATE.md files."""
    
    name: str
    description: str
    
    # Template configuration
    output_name: Optional[str] = None  # Default output filename
    keep_template: bool = False  # Keep TEMPLATE.md after applying
    
    # Default values for generated agent
    defaults: Dict[str, Any] = Field(default_factory=dict)
    
    # Template variables
    variables: List[str] = Field(default_factory=list)
    
    class Config:
        extra = "allow"
