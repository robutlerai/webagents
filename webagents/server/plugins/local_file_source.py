"""
Local File Source

Load agents from local AGENT*.md files.
"""

import fnmatch
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from .interface import AgentSource
from webagents.cli.loader.hierarchy import load_agent
from webagents.cli.daemon.registry import DaemonRegistry

logger = logging.getLogger("webagents.sources.local")


class LocalFileSource(AgentSource):
    """Load agents from local AGENT*.md files"""
    
    def __init__(self, watch_dirs: List[Path], metadata_store, registry: Optional[DaemonRegistry] = None):
        self.watch_dirs = watch_dirs
        self.registry = registry or DaemonRegistry()
        self.metadata_store = metadata_store
    
    async def get_agent(self, name: str) -> Optional[Any]:
        """Get agent by name from local files"""
        logger.debug(f"[LocalFileSource] get_agent({name})")
        
        # Try registry first (fast lookup)
        agent_file = self.registry.get(name)
        if not agent_file:
            # Scan directories
            await self.refresh()
            agent_file = self.registry.get(name)
        
        if not agent_file:
            logger.warning(f"[LocalFileSource] Agent not found: {name}")
            return None
        
        # Load agent with context
        logger.debug(f"[LocalFileSource] Loading agent from {agent_file.source_path}")
        merged = load_agent(Path(agent_file.source_path))
        
        # Load skills - respect the YAML, only use defaults if none specified
        skills_list = merged.metadata.skills
        if not skills_list:
            # Default skills when none specified
            skills_list = ["filesystem", "shell", "web", "todo", "rag", "mcp", "session", "checkpoint"]
        # Note: We no longer force session/checkpoint - respect the agent's YAML
        
        # Instantiate skills
        skills = self._load_skills(skills_list, agent_name=name, agent_path=Path(agent_file.source_path))
        
        
        # Create BaseAgent
        from webagents.agents.core.base_agent import BaseAgent
        
        # Default to google/gemini-2.5-flash if no model specified
        agent_model = merged.metadata.model or "google/gemini-2.5-flash"
        
        agent = BaseAgent(
            name=merged.metadata.name or name,
            instructions=merged.instructions,
            skills=skills,
            scopes=merged.metadata.scopes or ["all"],
            model=agent_model,
        )
        
        # Initialize async skills (like MCP that need to connect to servers)
        logger.info(f"[LocalFileSource] Initializing skills for agent '{name}'")
        await agent._ensure_skills_initialized()
        logger.info(f"[LocalFileSource] Skills initialized for agent '{name}'")
        
        # Store metadata
        self.metadata_store.register_agent(name, {
            "name": name,
            "source": "local",
            "path": str(agent_file.source_path),
            "metadata": merged.metadata.dict(),
        })
        
        return agent
    
    def _load_skills(self, skills_config: List[Union[str, Dict[str, Any]]], agent_name: str, agent_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load and instantiate skills from config"""
        loaded_skills = {}
        
        # Known local skills mapping
        skill_classes = {
            "filesystem": "webagents.agents.skills.local.filesystem.skill.FilesystemSkill",
            "shell": "webagents.agents.skills.local.shell.skill.ShellSkill",
            "rag": "webagents.agents.skills.local.rag.skill.LocalRagSkill",
            "session": "webagents.agents.skills.local.session.skill.SessionManagerSkill",
            "checkpoint": "webagents.agents.skills.local.checkpoint.skill.CheckpointSkill",
            "google": "webagents.agents.skills.core.llm.google.skill.GoogleAISkill",
            "openai": "webagents.agents.skills.core.llm.openai.skill.OpenAISkill",
            "anthropic": "webagents.agents.skills.core.llm.anthropic.skill.AnthropicSkill",
            "xai": "webagents.agents.skills.core.llm.xai.skill.XAISkill",
            "web": "webagents.agents.skills.local.web.skill.WebSkill",
            "todo": "webagents.agents.skills.local.todo.skill.TodoSkill",
            "mcp": "webagents.agents.skills.local.mcp.skill.LocalMcpSkill",
            "sandbox": "webagents.agents.skills.local.sandbox.skill.SandboxSkill",
        }
        
        for item in skills_config:
            skill_name = None
            config = {}
            
            if isinstance(item, str):
                skill_name = item
            elif isinstance(item, dict):
                # item is like {"filesystem": {"whitelist": [...]}}
                # or {"mcp": {"sqlite": {...}}}
                if len(item) == 1:
                    skill_name = list(item.keys())[0]
                    # Specific handling for MCP config structure
                    if skill_name == "mcp":
                        # Allow both {"mcp": {...servers...}} and {"mcp": {"mcpServers": {...}}}
                        raw_config = item[skill_name] or {}
                        if "mcpServers" in raw_config:
                            config = raw_config # Already has mcpServers key
                        else:
                            # Assume top-level keys are servers, wrap them
                            config = {"mcp": raw_config}
                    else:
                        config = item[skill_name] or {}
            
            if not skill_name or skill_name not in skill_classes:
                continue
            
            # Inject agent name into config if needed (e.g. for session skill)
            config["agent_name"] = agent_name
            # Pass agent DIRECTORY, not the file path
            config["agent_path"] = str(agent_path.parent) if agent_path else None
            
            # Inject agent directory into filesystem/shell config
            if agent_path:
                agent_dir = str(agent_path.parent.resolve())
                
                if skill_name == "filesystem":
                    whitelist = config.get("whitelist", [])
                    if agent_dir not in whitelist:
                        whitelist.append(agent_dir)
                    config["whitelist"] = whitelist
                    config["base_dir"] = agent_dir
                elif skill_name == "shell":
                    config["base_dir"] = agent_dir
            
            # Import and instantiate
            try:
                module_path, class_name = skill_classes[skill_name].rsplit(".", 1)
                import importlib
                module = importlib.import_module(module_path)
                skill_class = getattr(module, class_name)
                loaded_skills[skill_name] = skill_class(config)
            except Exception as e:
                # Log but continue - some skills may fail to load
                pass
        
        return loaded_skills
    
    async def list_agents(self) -> List[Dict[str, Any]]:
        """List all local agents"""
        await self.refresh()
        return [
            {
                "name": a.name,
                "source": "local",
                "path": str(a.source_path),
                "description": a.description,
            }
            for a in self.registry.list_agents()
        ]
    
    async def search_agents(self, query: str) -> List[Dict[str, Any]]:
        """Search local agents by name pattern
        
        Supports wildcards:
        - "customer-*" matches "customer-support", "customer-onboarding"
        - "*-test" matches any agent ending in "-test"
        """
        await self.refresh()
        all_agents = self.registry.list_agents()
        
        # Convert wildcard pattern to regex
        matches = [
            {
                "name": a.name,
                "source": "local",
                "path": str(a.source_path),
                "description": a.description,
            }
            for a in all_agents
            if fnmatch.fnmatch(a.name.lower(), query.lower())
        ]
        
        return matches
    
    async def filter_agents(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter local agents by criteria"""
        await self.refresh()
        all_agents = self.registry.list_agents()
        
        filtered = []
        for agent in all_agents:
            match = True
            
            # Check each criterion
            if "source" in criteria and criteria["source"] != "local":
                match = False
            
            if "tags" in criteria:
                 # DaemonAgent doesn't store tags currently, so skip or fail
                 pass
            
            if match:
                filtered.append({
                    "name": agent.name,
                    "source": "local",
                    "path": str(agent.source_path),
                    "description": agent.description,
                })
        
        return filtered
    
    async def refresh(self):
        """Scan directories for agents"""
        for watch_dir in self.watch_dirs:
            await self.registry.scan_directory(watch_dir)
    
    def get_source_type(self) -> str:
        return "local"
