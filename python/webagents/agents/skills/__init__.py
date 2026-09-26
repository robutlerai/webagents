"""
WebAgents Agents Skills

This module provides the skill system for WebAgents agents, including core skills,
platform skills, and ecosystem skills. Skills are modular components that provide
specific functionality to agents.
"""

from .base import Skill, Handoff, HandoffResult

# Core LLM skills - native provider integrations.
#
# Route through the `.core.llm` facade, which already wraps every provider
# import in a graceful guard and binds the name to None when the provider's
# optional dependency is missing. Importing the provider modules directly
# here was unguarded, so a missing optional dependency (e.g. google-genai)
# broke `import webagents` outright (F-040).
from .core.llm import (
    OpenAISkill,
    AnthropicSkill,
    GoogleAISkill,
    XAISkill,
    FireworksAISkill,
)

CORE_SKILLS = {
    name: skill_cls
    for name, skill_cls in {
        "openai": OpenAISkill,
        "anthropic": AnthropicSkill,
        "google": GoogleAISkill,
        "xai": XAISkill,
        "fireworks": FireworksAISkill,
    }.items()
    if skill_cls is not None  # provider's optional dependency not installed
}

# WebAgents platform skills. Empty since 2026-09-25: it held only the CRM
# skill (as `crm` and `analytics`), which called `/api/crm/*` routes the
# platform never had and was retired, and importing it here loaded that module
# and aiohttp with this package. Agent files name platform skills through the
# CLI's registry (`cli/agent_builder.py`), and code imports them lazily from
# `.robutler`.
ROBUTLER_SKILLS: dict = {}

# Ecosystem skills - these integrate with external services
ECOSYSTEM_SKILLS = {
    # Will be populated as ecosystem integrations are implemented
}

# Combined skills registry
ALL_SKILLS = {
    **CORE_SKILLS,
    **ROBUTLER_SKILLS, 
    **ECOSYSTEM_SKILLS
}

__all__ = [
    # Base classes
    'Skill',
    'Handoff', 
    'HandoffResult',
    
    # Skill registries
    'CORE_SKILLS',
    'ROBUTLER_SKILLS',
    'ECOSYSTEM_SKILLS', 
    'ALL_SKILLS',
    
    # Core LLM skills
    'OpenAISkill',
    'AnthropicSkill',
    'GoogleAISkill',
    'XAISkill',
    'FireworksAISkill',
]

# Lazy loading for ecosystem skills to avoid heavy imports
def get_skill(skill_name: str):
    """Get a skill class by name, with lazy loading for ecosystem skills
    
    Args:
        skill_name: Name of the skill to load
        
    Returns:
        Skill class if found, None otherwise
    """
    # Check core skills first
    if skill_name in CORE_SKILLS:
        return CORE_SKILLS[skill_name]
    
    # Check platform skills
    if skill_name in ROBUTLER_SKILLS:
        return ROBUTLER_SKILLS[skill_name]
    
    # Lazy load ecosystem skills
    ecosystem_imports = {
        "google": ("webagents.agents.skills.ecosystem.google", "GoogleSkill"),
        "database": ("webagents.agents.skills.ecosystem.database", "DatabaseSkill"),
        "filesystem": ("webagents.agents.skills.ecosystem.filesystem", "FilesystemSkill"),
        "web": ("webagents.agents.skills.ecosystem.web", "WebSkill"),
        "crewai": ("webagents.agents.skills.ecosystem.crewai", "CrewAISkill"),
        "n8n": ("webagents.agents.skills.ecosystem.n8n", "N8nSkill"),
        "zapier": ("webagents.agents.skills.ecosystem.zapier", "ZapierSkill"),
    }
    
    if skill_name in ecosystem_imports:
        try:
            module_path, class_name = ecosystem_imports[skill_name]
            module = __import__(module_path, fromlist=[class_name])
            skill_class = getattr(module, class_name)
            ECOSYSTEM_SKILLS[skill_name] = skill_class
            return skill_class
        except ImportError:
            # Ecosystem skill not available
            return None
    
    return None

def list_available_skills() -> dict:
    """List all available skills by category
    
    Returns:
        Dict with skill categories and their available skills
    """
    return {
        "core": list(CORE_SKILLS.keys()),
        "webagents": list(ROBUTLER_SKILLS.keys()),
        "ecosystem": list(ECOSYSTEM_SKILLS.keys())
    }
