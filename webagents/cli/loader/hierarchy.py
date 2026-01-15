"""
Agent Loading with Context Hierarchy

Load agents with inherited context from AGENTS.md files.
"""

from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

from .agent_md import AgentFile
from .context import ContextFile, ContextHierarchy
from .schema import AgentMetadata


@dataclass
class MergedAgent:
    """An agent with merged context from AGENTS.md hierarchy."""
    
    # Original agent file
    agent_file: AgentFile
    
    # Merged metadata (agent + context)
    metadata: AgentMetadata
    
    # Combined instructions
    instructions: str
    
    # Context files that were merged
    context_files: List[ContextFile]
    
    @property
    def name(self) -> str:
        return self.metadata.name
    
    @property
    def path(self) -> Path:
        return self.agent_file.path
    
    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "name": self.name,
            "metadata": self.metadata.model_dump(),
            "instructions": self.instructions,
            "context_files": [str(c.path) for c in self.context_files],
        }


class AgentLoader:
    """Load agents with context inheritance."""
    
    def __init__(self, stop_at: Optional[Path] = None):
        """Initialize loader.
        
        Args:
            stop_at: Stop context traversal at this directory
        """
        self.hierarchy = ContextHierarchy(stop_at=stop_at)
    
    def load(self, agent_path: Path) -> MergedAgent:
        """Load an agent with merged context.
        
        Args:
            agent_path: Path to AGENT.md or AGENT-*.md file
            
        Returns:
            MergedAgent with inherited context
        """
        # Parse the agent file
        agent = AgentFile(agent_path)
        
        # Get context hierarchy
        context_files = self.hierarchy.resolve(agent_path.parent)
        
        # Merge contexts
        merged_context = self.hierarchy.merge_contexts(context_files)
        
        # Merge agent metadata with context
        merged_metadata = self._merge_metadata(agent.metadata, merged_context)
        
        # Combine instructions
        instructions = self._merge_instructions(
            [ctx.content for ctx in context_files],
            agent.instructions
        )
        
        return MergedAgent(
            agent_file=agent,
            metadata=merged_metadata,
            instructions=instructions,
            context_files=context_files,
        )
    
    def _merge_metadata(self, agent_meta: AgentMetadata, context: dict) -> AgentMetadata:
        """Merge agent metadata with context.
        
        Agent values take precedence over context.
        
        Args:
            agent_meta: Agent's own metadata
            context: Merged context from AGENTS.md files
            
        Returns:
            Merged AgentMetadata
        """
        # Start with agent's metadata
        data = agent_meta.model_dump()
        
        # Apply context defaults (only if agent doesn't have a value)
        if not data.get("namespace") or data["namespace"] == "local":
            data["namespace"] = context.get("namespace", "local")
        
        if not data.get("model"):
            data["model"] = context.get("model", None)
        
        # Merge skills (unique)
        context_skills = context.get("skills", [])
        data["skills"] = list(dict.fromkeys(context_skills + data.get("skills", [])))
        
        # Merge tools (unique)
        context_tools = context.get("tools", [])
        data["tools"] = list(dict.fromkeys(context_tools + data.get("tools", [])))
        
        # Merge MCP servers (unique)
        context_mcp = context.get("mcp_servers", [])
        data["mcp_servers"] = list(dict.fromkeys(context_mcp + data.get("mcp_servers", [])))
        
        # Apply sandbox from context if not set
        if not data.get("sandbox") and context.get("sandbox"):
            data["sandbox"] = context["sandbox"]
        
        return AgentMetadata(**data)
    
    def _merge_instructions(
        self, 
        context_instructions: List[str], 
        agent_instructions: str
    ) -> str:
        """Merge context instructions with agent instructions.
        
        Context provides background, agent instructions are primary.
        
        Args:
            context_instructions: Instructions from AGENTS.md files
            agent_instructions: Agent's own instructions
            
        Returns:
            Combined instructions
        """
        parts = []
        
        # Add context as background (if any)
        context_content = "\n\n---\n\n".join(
            instr for instr in context_instructions if instr.strip()
        )
        if context_content:
            parts.append(f"## Background Context\n\n{context_content}")
        
        # Add agent's instructions
        if agent_instructions.strip():
            parts.append(agent_instructions)
        
        return "\n\n".join(parts)
    
    def load_all(self, directory: Path, recursive: bool = False) -> List[MergedAgent]:
        """Load all agents in a directory.
        
        Args:
            directory: Directory to search
            recursive: Search subdirectories
            
        Returns:
            List of MergedAgent instances
        """
        from .agent_md import find_agent_files
        
        agent_files = find_agent_files(directory, recursive=recursive)
        return [self.load(af.path) for af in agent_files]


def find_default_agent(directory: Path) -> Optional[MergedAgent]:
    """Find and load the default agent in a directory.
    
    Resolution rules:
    1. If AGENT.md exists, use it
    2. If single AGENT-*.md exists, use it
    3. If multiple AGENT-*.md, return None (ambiguous)
    
    Args:
        directory: Directory to search
        
    Returns:
        MergedAgent if found, None if ambiguous or not found
    """
    loader = AgentLoader()
    
    # Check for AGENT.md
    agent_md = directory / "AGENT.md"
    if agent_md.exists():
        return loader.load(agent_md)
    
    # Check for single AGENT-*.md
    agent_files = list(directory.glob("AGENT-*.md"))
    if len(agent_files) == 1:
        return loader.load(agent_files[0])
    
    # Multiple or none
    return None


def load_agent(path: Path) -> MergedAgent:
    """Load an agent from path.
    
    Args:
        path: Path to AGENT.md or AGENT-*.md
        
    Returns:
        MergedAgent
    """
    loader = AgentLoader()
    return loader.load(path)
