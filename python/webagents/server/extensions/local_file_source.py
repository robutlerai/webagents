"""
Local File Source

Load agents from local AGENT*.md files with caching support.
"""

import asyncio
import fnmatch
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from .interface import AgentSource
from webagents.cli.loader.hierarchy import load_agent
from webagents.cli.daemon.registry import DaemonRegistry

logger = logging.getLogger("webagents.sources.local")


class LocalFileSource(AgentSource):
    """Load agents from local AGENT*.md files with caching.
    
    Agents are cached after first load to avoid expensive re-initialization
    (especially for skills like MCP that establish connections).
    
    Cache invalidation:
    - Call invalidate(name) when agent file changes
    - Call invalidate_all() to clear entire cache
    """
    
    def __init__(self, watch_dirs: List[Path], metadata_store, registry: Optional[DaemonRegistry] = None):
        self.watch_dirs = watch_dirs
        self.registry = registry or DaemonRegistry()
        self.metadata_store = metadata_store
        
        # Agent cache
        self._agent_cache: Dict[str, Any] = {}  # name -> BaseAgent
        self._cache_timestamps: Dict[str, float] = {}  # name -> load timestamp
        self._cache_lock = asyncio.Lock()
    
    async def get_agent(self, name: str, working_dir: Optional[str] = None) -> Optional[Any]:
        """Get agent by name from local files (with caching).
        
        Args:
            name: Agent name
            working_dir: Optional working directory (used for embedded agents like robutler)
        """
        logger.debug(f"[LocalFileSource] get_agent({name}, working_dir={working_dir})")
        
        # Special case: robutler is always created fresh with the provided working_dir
        # This ensures robutler operates in the directory where the command was invoked
        if name.lower() == "robutler":
            return await self._create_robutler_agent(working_dir=working_dir)
        
        # Check cache first (fast path)
        async with self._cache_lock:
            if name in self._agent_cache:
                logger.debug(f"[LocalFileSource] Cache hit for agent '{name}'")
                return self._agent_cache[name]
        
        # Try registry (file lookup)
        agent_file = self.registry.get(name)
        if not agent_file:
            # Scan directories
            await self.refresh()
            agent_file = self.registry.get(name)
        
        if not agent_file:
            logger.warning(f"[LocalFileSource] Agent not found: {name}")
            return None
        
        # Load and cache agent
        agent = await self._create_agent(name, agent_file)
        
        if agent:
            async with self._cache_lock:
                self._agent_cache[name] = agent
                self._cache_timestamps[name] = time.time()
                logger.info(f"[LocalFileSource] Cached agent '{name}'")
        
        return agent
    
    async def _create_agent(self, name: str, agent_file) -> Optional[Any]:
        """Create a new agent instance (internal, not cached)."""
        logger.debug(f"[LocalFileSource] Loading agent from {agent_file.source_path}")
        merged = load_agent(Path(agent_file.source_path))
        
        # Load skills - respect the YAML, only use defaults if none specified
        skills_list = merged.metadata.skills
        if not skills_list:
            # Default skills when none specified
            skills_list = ["filesystem", "shell", "web", "todo", "rag", "mcp", "session", "checkpoint"]
        
        # Add completions transport if no transport skill is explicitly defined
        transport_skills = {"completions", "a2a", "realtime", "acp"}
        has_transport = any(s in transport_skills for s in skills_list if isinstance(s, str))
        if not has_transport:
            skills_list = list(skills_list) + ["completions"]
        
        # Instantiate skills
        skills = self._load_skills(skills_list, agent_name=name, agent_path=Path(agent_file.source_path))
        
        # Always add LLM skill for handoff if not already present
        llm_skills = {"llm", "google", "openai", "anthropic", "xai", "fireworks", "primary_llm"}
        if not any(s in skills for s in llm_skills):
            try:
                from webagents.agents.skills.core.llm.google.skill import GoogleAISkill
                skills["llm"] = GoogleAISkill()
                logger.info(f"[LocalFileSource] Auto-added GoogleAI LLM skill for {name}")
            except Exception as e:
                logger.warning(f"[LocalFileSource] Failed to auto-add LLM skill: {e}")
        
        # Create BaseAgent
        from webagents.agents.core.base_agent import BaseAgent
        
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
    
    async def _create_robutler_agent(self, working_dir: Optional[str] = None) -> Optional[Any]:
        """Create the embedded robutler agent.
        
        This is used when 'robutler' is requested but no local agent file exists.
        The robutler agent uses the provided working directory or falls back to cwd.
        
        Args:
            working_dir: Working directory for the agent (where skills should operate)
        """
        import os
        from webagents.agents.builtin import get_robutler_path
        
        robutler_path = get_robutler_path()
        if not robutler_path.exists():
            logger.error("[LocalFileSource] Embedded ROBUTLER.md not found")
            return None
        
        logger.info(f"[LocalFileSource] Loading embedded robutler agent")
        merged = load_agent(robutler_path)
        
        # Use skills from ROBUTLER.md
        skills_list = merged.metadata.skills or ["filesystem", "shell", "web", "mcp", "session", "todo", "rag", "checkpoint"]
        
        # Add completions transport
        transport_skills = {"completions", "a2a", "realtime", "acp"}
        has_transport = any(s in transport_skills for s in skills_list if isinstance(s, str))
        if not has_transport:
            skills_list = list(skills_list) + ["completions"]
        
        # Use provided working_dir or fallback to cwd
        if working_dir:
            working_dir_path = Path(working_dir)
            logger.info(f"[LocalFileSource] Using provided working_dir: {working_dir_path}")
        else:
            working_dir_path = Path(os.getcwd())
            logger.info(f"[LocalFileSource] Using cwd as working_dir: {working_dir_path}")
        
        # Instantiate skills with working_dir as the agent path
        skills = self._load_skills(skills_list, agent_name="robutler", agent_path=working_dir_path / "AGENT.md")
        
        # Always add LLM skill for handoff if not already present
        llm_skills = {"llm", "google", "openai", "anthropic", "xai", "fireworks", "primary_llm"}
        if not any(s in skills for s in llm_skills):
            try:
                from webagents.agents.skills.core.llm.google.skill import GoogleAISkill
                skills["llm"] = GoogleAISkill()
                logger.info("[LocalFileSource] Auto-added GoogleAI LLM skill for robutler")
            except Exception as e:
                logger.warning(f"[LocalFileSource] Failed to auto-add LLM skill: {e}")
        
        # Create BaseAgent
        from webagents.agents.core.base_agent import BaseAgent
        
        agent = BaseAgent(
            name="robutler",
            instructions=merged.instructions,
            skills=skills,
            scopes=merged.metadata.scopes or ["all"],
            model=merged.metadata.model or "google/gemini-2.5-flash",
        )
        
        # Initialize async skills
        logger.info("[LocalFileSource] Initializing skills for robutler")
        await agent._ensure_skills_initialized()
        logger.info("[LocalFileSource] Skills initialized for robutler")
        
        # Store metadata
        self.metadata_store.register_agent("robutler", {
            "name": "robutler",
            "source": "embedded",
            "path": str(robutler_path),
            "working_dir": str(working_dir),
            "metadata": merged.metadata.dict(),
        })
        
        # Cache the agent
        async with self._cache_lock:
            self._agent_cache["robutler"] = agent
            self._cache_timestamps["robutler"] = time.time()
            logger.info("[LocalFileSource] Cached robutler agent")
        
        return agent
    
    def invalidate(self, name: str) -> bool:
        """Invalidate cached agent (e.g., after file change).
        
        Returns:
            True if agent was in cache and removed, False otherwise
        """
        if name in self._agent_cache:
            del self._agent_cache[name]
            self._cache_timestamps.pop(name, None)
            logger.info(f"[LocalFileSource] Invalidated cache for agent '{name}'")
            return True
        return False
    
    def invalidate_all(self) -> int:
        """Invalidate all cached agents.
        
        Returns:
            Number of agents invalidated
        """
        count = len(self._agent_cache)
        self._agent_cache.clear()
        self._cache_timestamps.clear()
        logger.info(f"[LocalFileSource] Invalidated all {count} cached agents")
        return count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_agents": list(self._agent_cache.keys()),
            "cache_size": len(self._agent_cache),
            "timestamps": self._cache_timestamps.copy(),
        }
    
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
            # LLM skills
            "llm": "webagents.agents.skills.core.llm.google.skill.GoogleAISkill",
            "google": "webagents.agents.skills.core.llm.google.skill.GoogleAISkill",
            "openai": "webagents.agents.skills.core.llm.openai.skill.OpenAISkill",
            "anthropic": "webagents.agents.skills.core.llm.anthropic.skill.AnthropicSkill",
            "xai": "webagents.agents.skills.core.llm.xai.skill.XAISkill",
            "fireworks": "webagents.agents.skills.core.llm.fireworks.skill.FireworksAISkill",
            # Local skills
            "web": "webagents.agents.skills.local.web.skill.WebSkill",
            "todo": "webagents.agents.skills.local.todo.skill.TodoSkill",
            "mcp": "webagents.agents.skills.local.mcp.skill.LocalMcpSkill",
            "sandbox": "webagents.agents.skills.local.sandbox.skill.SandboxSkill",
            # Transport skills - always available
            "completions": "webagents.agents.skills.core.transport.completions.skill.CompletionsTransportSkill",
            "a2a": "webagents.agents.skills.core.transport.a2a.skill.A2ATransportSkill",
            "realtime": "webagents.agents.skills.core.transport.realtime.skill.RealtimeTransportSkill",
            "acp": "webagents.agents.skills.core.transport.acp.skill.ACPTransportSkill",
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
