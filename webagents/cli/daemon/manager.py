"""
Agent Lifecycle Manager

Manage starting, stopping, and monitoring agents.
"""

import asyncio
import logging
from typing import Optional, Dict, List
from pathlib import Path
from datetime import datetime

from .registry import DaemonRegistry, DaemonAgent

logger = logging.getLogger("webagentsd.manager")


class AgentManager:
    """Manage agent lifecycle."""
    
    def __init__(self, registry: DaemonRegistry):
        """Initialize manager.
        
        Args:
            registry: Daemon registry
        """
        self.registry = registry
        self._running_agents: Dict[str, asyncio.Task] = {}
        self._agent_logs: Dict[str, List[str]] = {}
        self._loaded_agents: Dict[str, "BaseAgent"] = {}  # Cache of loaded BaseAgent instances
    
    async def get_or_load_agent(self, name: str) -> Optional["BaseAgent"]:
        """Get or load a BaseAgent instance with skills.
        
        This loads the actual agent with all skills attached, unlike
        the registry which only stores metadata.
        
        Args:
            name: Agent name
            
        Returns:
            BaseAgent instance or None if not found
        """
        logger.debug(f"[Manager] get_or_load_agent({name}), cached={name in self._loaded_agents}")
        
        # Check cache first
        if name in self._loaded_agents:
            cached = self._loaded_agents[name]
            logger.debug(f"[Manager] Returning cached agent, skills={list(cached.skills.keys())}")
            return cached
        
        # Get agent metadata from registry
        daemon_agent = self.registry.get(name)
        if not daemon_agent:
            logger.warning(f"[Manager] Agent '{name}' not found in registry")
            return None
        
        # Load the full agent with skills
        try:
            from ..loader.hierarchy import load_agent
            from webagents.agents.core.base_agent import BaseAgent
            from pathlib import Path
            
            source_path = Path(daemon_agent.source_path)
            merged = load_agent(source_path)
            
            # Load skills - respect the YAML, only use defaults if none specified
            skills_list = merged.metadata.skills or []
            logger.debug(f"[Manager] Agent {name} skills from YAML: {skills_list}")
            
            if not skills_list:
                # Default skills when none specified - includes completions transport
                skills_list = ["filesystem", "shell", "web", "todo", "rag", "session", "checkpoint", "mcp", "completions"]
                logger.debug(f"[Manager] No skills in YAML, using defaults: {skills_list}")
            
            # Add completions transport if no transport skill is explicitly defined
            transport_skills = {"completions", "a2a", "realtime", "acp"}
            has_transport = any(s in transport_skills for s in skills_list if isinstance(s, str))
            if not has_transport:
                skills_list = list(skills_list) + ["completions"]
                logger.debug(f"[Manager] Added completions transport to agent {name}")
            
            # Load skill instances
            skills = self._load_skills(skills_list, name, source_path)
            logger.info(f"[Manager] Loaded skills for {name}: {list(skills.keys())}")
            
            # Always add LLM skill for handoff if not already present
            llm_skills = {"llm", "google", "openai", "anthropic", "xai", "litellm", "primary_llm"}
            if not any(s in skills for s in llm_skills):
                try:
                    from webagents.agents.skills.core.llm.google.skill import GoogleAISkill
                    skills["llm"] = GoogleAISkill()
                    logger.info(f"[Manager] Auto-added GoogleAI LLM skill for {name}")
                except Exception as e:
                    logger.warning(f"[Manager] Failed to auto-add LLM skill: {e}")
            
            # Create BaseAgent
            agent = BaseAgent(
                name=merged.metadata.name or name,
                instructions=merged.instructions,
                skills=skills,
                scopes=merged.metadata.scopes or ["all"],
                model=merged.metadata.model or "google/gemini-2.5-flash",
            )
            
            # Initialize async skills (like MCP that need to connect to servers)
            logger.info(f"[Manager] Calling _ensure_skills_initialized for {name}")
            await agent._ensure_skills_initialized()
            logger.info(f"[Manager] Skills initialized for {name}")
            
            # Cache it
            self._loaded_agents[name] = agent
            return agent
            
        except Exception as e:
            import traceback
            logger.error(f"[Manager] Error loading agent {name}: {e}\n{traceback.format_exc()}")
            self._log(name, f"Error loading agent: {e}\n{traceback.format_exc()}")
            return None
    
    def _load_skills(self, skills_config: List, agent_name: str, agent_path: Path) -> Dict:
        """Load and instantiate skills from config."""
        from webagents.server.plugins.local_file_source import LocalFileSource
        # Reuse the skill loading logic from LocalFileSource
        # Create a minimal loader to avoid circular deps
        loaded_skills = {}
        
        skill_classes = {
            "filesystem": "webagents.agents.skills.local.filesystem.skill.FilesystemSkill",
            "shell": "webagents.agents.skills.local.shell.skill.ShellSkill",
            "rag": "webagents.agents.skills.local.rag.skill.LocalRagSkill",
            "session": "webagents.agents.skills.local.session.skill.SessionManagerSkill",
            "checkpoint": "webagents.agents.skills.local.checkpoint.skill.CheckpointSkill",
            # LLM skills
            "google": "webagents.agents.skills.core.llm.google.skill.GoogleAISkill",
            "openai": "webagents.agents.skills.core.llm.openai.skill.OpenAISkill",
            "anthropic": "webagents.agents.skills.core.llm.anthropic.skill.AnthropicSkill",
            "xai": "webagents.agents.skills.core.llm.xai.skill.XAISkill",
            "litellm": "webagents.agents.skills.core.llm.litellm.skill.LiteLLMSkill",
            # Local skills
            "web": "webagents.agents.skills.local.web.skill.WebSkill",
            "todo": "webagents.agents.skills.local.todo.skill.TodoSkill",
            "mcp": "webagents.agents.skills.local.mcp.skill.LocalMcpSkill",
            "sandbox": "webagents.agents.skills.local.sandbox.skill.SandboxSkill",
            # Transport skills
            "completions": "webagents.agents.skills.core.transport.completions.skill.CompletionsTransportSkill",
            "a2a": "webagents.agents.skills.core.transport.a2a.skill.A2ATransportSkill",
            "realtime": "webagents.agents.skills.core.transport.realtime.skill.RealtimeTransportSkill",
            "acp": "webagents.agents.skills.core.transport.acp.skill.ACPTransportSkill",
            # Testing skills
            "testrunner": "webagents.agents.skills.local.testrunner.skill.TestRunnerSkill",
        }
        
        logger.debug(f"[Manager] _load_skills: processing {len(skills_config)} items")
        
        for item in skills_config:
            skill_name = None
            config = {}
            
            if isinstance(item, str):
                skill_name = item
                logger.debug(f"[Manager] Skill '{skill_name}' (string, no config)")
            elif isinstance(item, dict) and len(item) == 1:
                skill_name = list(item.keys())[0]
                raw_config = item[skill_name] or {}
                logger.debug(f"[Manager] Skill '{skill_name}' with config keys: {list(raw_config.keys()) if isinstance(raw_config, dict) else raw_config}")
                
                # Special handling for MCP config structure
                if skill_name == "mcp":
                    # Allow both {"mcp": {...servers...}} and {"mcp": {"mcpServers": {...}}}
                    if "mcpServers" in raw_config:
                        config = raw_config
                        logger.debug(f"[Manager] MCP config has mcpServers, using as-is")
                    else:
                        # Assume top-level keys are server names, wrap them
                        config = {"mcp": raw_config}
                        logger.debug(f"[Manager] MCP config wrapped: {list(raw_config.keys()) if isinstance(raw_config, dict) else raw_config}")
                else:
                    config = raw_config
            
            if not skill_name or skill_name not in skill_classes:
                logger.debug(f"[Manager] Skipping unknown skill: {skill_name}")
                continue
            
            # Inject agent info
            config["agent_name"] = agent_name
            # Pass agent DIRECTORY, not the file path
            config["agent_path"] = str(agent_path.parent) if agent_path else None
            
            if skill_name == "mcp":
                logger.info(f"[Manager] Final MCP config: {config}")
            
            try:
                import importlib
                module_path, class_name = skill_classes[skill_name].rsplit(".", 1)
                module = importlib.import_module(module_path)
                skill_class = getattr(module, class_name)
                
                # Try to instantiate with config dict (preferred), fallback to kwargs, then no args
                try:
                    loaded_skills[skill_name] = skill_class(config=config)
                except TypeError:
                    try:
                        loaded_skills[skill_name] = skill_class(**config)
                    except TypeError:
                        loaded_skills[skill_name] = skill_class()
                
                logger.debug(f"[Manager] Loaded skill: {skill_name}")
            except Exception as e:
                logger.warning(f"[Manager] Failed to load skill {skill_name}: {e}")
        
        return loaded_skills
    
    def invalidate_agent_cache(self, name: str):
        """Remove an agent from the cache (e.g., after file change)."""
        if name in self._loaded_agents:
            del self._loaded_agents[name]
    
    async def start(self, name: str, prompt: Optional[str] = None) -> bool:
        """Start an agent.
        
        Args:
            name: Agent name
            prompt: Optional initial prompt
            
        Returns:
            Success
        """
        agent = self.registry.get(name)
        if not agent:
            raise ValueError(f"Agent not found: {name}")
        
        if name in self._running_agents:
            raise ValueError(f"Agent already running: {name}")
        
        # Create agent task
        task = asyncio.create_task(self._run_agent(agent, prompt))
        self._running_agents[name] = task
        
        # Update status
        agent.status = "running"
        agent.started_at = datetime.utcnow()
        
        self._log(name, f"Agent started: {name}")
        return True
    
    async def stop(self, name: str) -> bool:
        """Stop a running agent.
        
        Args:
            name: Agent name
            
        Returns:
            Success
        """
        if name not in self._running_agents:
            return False
        
        # Cancel task
        task = self._running_agents[name]
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        del self._running_agents[name]
        
        # Update status
        agent = self.registry.get(name)
        if agent:
            agent.status = "stopped"
        
        self._log(name, f"Agent stopped: {name}")
        return True
    
    async def restart(self, name: str) -> bool:
        """Restart an agent.
        
        Args:
            name: Agent name
            
        Returns:
            Success
        """
        await self.stop(name)
        await asyncio.sleep(0.1)
        return await self.start(name)
    
    async def stop_all(self):
        """Stop all running agents."""
        for name in list(self._running_agents.keys()):
            await self.stop(name)
    
    async def _run_agent(self, agent: DaemonAgent, prompt: Optional[str] = None):
        """Run agent loop.
        
        Args:
            agent: Agent to run
            prompt: Optional initial prompt
        """
        try:
            # Load agent
            from ..loader import AgentLoader
            loader = AgentLoader()
            merged = loader.load(Path(agent.source_path))
            
            self._log(agent.name, f"Loaded agent from: {agent.source_path}")
            
            # TODO: Create actual agent runtime
            # For now, just keep running
            while True:
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            self._log(agent.name, "Agent task cancelled")
            raise
        except Exception as e:
            self._log(agent.name, f"Agent error: {e}")
            agent.status = "error"
            raise
    
    def get_status(self, name: str) -> str:
        """Get agent status.
        
        Args:
            name: Agent name
            
        Returns:
            Status string
        """
        if name in self._running_agents:
            return "running"
        
        agent = self.registry.get(name)
        return agent.status if agent else "unknown"
    
    def get_running_agents(self) -> List[str]:
        """Get list of running agent names."""
        return list(self._running_agents.keys())
    
    def get_logs(self, name: str, lines: int = 100) -> List[str]:
        """Get agent logs.
        
        Args:
            name: Agent name
            lines: Number of lines to return
            
        Returns:
            Log lines
        """
        logs = self._agent_logs.get(name, [])
        return logs[-lines:]
    
    def _log(self, agent_name: str, message: str):
        """Add log entry for agent.
        
        Args:
            agent_name: Agent name
            message: Log message
        """
        if agent_name not in self._agent_logs:
            self._agent_logs[agent_name] = []
        
        timestamp = datetime.utcnow().isoformat()
        self._agent_logs[agent_name].append(f"[{timestamp}] {message}")
        
        # Trim to last 1000 entries
        if len(self._agent_logs[agent_name]) > 1000:
            self._agent_logs[agent_name] = self._agent_logs[agent_name][-1000:]
