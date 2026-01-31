"""
Dynamic Agent Resolver

Resolve agents on-demand using the dynamic_agents pattern.
"""

from typing import Optional
from pathlib import Path

from webagents.agents.core.base_agent import BaseAgent


async def local_agent_resolver(agent_name: str) -> Optional[BaseAgent]:
    """Resolve agent from AGENT*.md files.
    
    This function follows the webagents dynamic_agents pattern
    for on-demand agent creation.
    
    Args:
        agent_name: Name of the agent to resolve
        
    Returns:
        BaseAgent instance or None if not found
    """
    from ..loader import AgentLoader
    from ..state.registry import LocalRegistry
    
    # Get registry
    registry = LocalRegistry()
    
    # Find agent in registry
    registered = registry.find_by_name(agent_name)
    if not registered:
        return None
    
    # Load agent with context inheritance
    loader = AgentLoader()
    merged = loader.load(Path(registered.source_path))
    
    # Build skills
    skills = _build_skills(merged.metadata.skills)
    
    # Create BaseAgent
    return BaseAgent(
        name=merged.metadata.name,
        instructions=merged.instructions,
        model=merged.metadata.model,
        skills=skills,
    )


def _build_skills(skill_names: list) -> list:
    """Build skill instances from names.
    
    Args:
        skill_names: List of skill names
        
    Returns:
        List of Skill instances
    """
    skills = []
    
    for skill_name in skill_names:
        skill = _get_skill(skill_name)
        if skill:
            skills.append(skill)
    
    return skills


def _get_skill(name: str):
    """Get skill instance by name.
    
    Args:
        name: Skill name
        
    Returns:
        Skill instance or None
    """
    # Map skill names to imports
    skill_map = {
        "llm": "webagents.agents.skills.core.llm.litellm.skill:LiteLLMSkill",
        "mcp": "webagents.agents.skills.core.mcp.skill:MCPSkill",
        "memory": "webagents.agents.skills.core.memory.short_term_memory.skill:ShortTermMemorySkill",
        # Add more as needed
    }
    
    if name not in skill_map:
        return None
    
    try:
        module_path, class_name = skill_map[name].rsplit(":", 1)
        import importlib
        module = importlib.import_module(module_path)
        skill_class = getattr(module, class_name)
        return skill_class()
    except Exception:
        return None


class AgentResolver:
    """Resolver class for more complex resolution logic."""
    
    def __init__(self, registry=None, loader=None):
        """Initialize resolver.
        
        Args:
            registry: LocalRegistry instance
            loader: AgentLoader instance
        """
        from ..state.registry import LocalRegistry
        from ..loader import AgentLoader
        
        self.registry = registry or LocalRegistry()
        self.loader = loader or AgentLoader()
        self._cache = {}
    
    async def resolve(self, agent_name: str, use_cache: bool = True) -> Optional[BaseAgent]:
        """Resolve an agent.
        
        Args:
            agent_name: Agent name
            use_cache: Use cached agent if available
            
        Returns:
            BaseAgent or None
        """
        # Check cache
        if use_cache and agent_name in self._cache:
            return self._cache[agent_name]
        
        # Find in registry
        registered = self.registry.find_by_name(agent_name)
        if not registered:
            return None
        
        # Load and create
        merged = self.loader.load(Path(registered.source_path))
        
        agent = BaseAgent(
            name=merged.metadata.name,
            instructions=merged.instructions,
            model=merged.metadata.model,
            skills=_build_skills(merged.metadata.skills),
        )
        
        # Cache
        if use_cache:
            self._cache[agent_name] = agent
        
        return agent
    
    def invalidate(self, agent_name: str = None):
        """Invalidate cache.
        
        Args:
            agent_name: Specific agent or all if None
        """
        if agent_name:
            self._cache.pop(agent_name, None)
        else:
            self._cache.clear()
