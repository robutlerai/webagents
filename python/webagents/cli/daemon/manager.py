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
            
            # The skills the file names, and none when it names none (S-280,
            # 2026-09-26), as the file-source loader (`local_file_source.py`
            # `_create_agent`) and the TypeScript SDK do. This loader added
            # eight defaults here, `filesystem` and `shell` among them, with no
            # `sandbox:`, so an agent file that declared no skills got a shell
            # and filesystem the model could reach unconfined, and the owner's
            # own turns (scheduled ones included) could run commands the file
            # never asked for.
            skills_list = list(merged.metadata.skills or [])
            logger.debug(f"[Manager] Agent {name} skills from YAML: {skills_list}")

            # Add completions transport if no transport skill is explicitly defined.
            # `acp` is not one here: it serves stdio to the editor that spawned
            # `webagents acp`, so an agent naming only `acp` still needs an
            # HTTP transport to be reachable from the daemon (2026-09-26).
            transport_skills = {"completions", "a2a", "realtime"}
            has_transport = any(s in transport_skills for s in skills_list if isinstance(s, str))
            if not has_transport:
                skills_list = list(skills_list) + ["completions"]
                logger.debug(f"[Manager] Added completions transport to agent {name}")
            
            # Load skill instances
            skills = self._load_skills(skills_list, name, source_path)
            logger.info(f"[Manager] Loaded skills for {name}: {list(skills.keys())}")

            # Who may call it, and what each group gets (ADR-0045).
            from webagents.access.install import add_access, finish_access

            access_policy = add_access(skills, merged.metadata.access, Path(source_path) if source_path else None)

            # NO GOOGLE MODEL THE FILE NEVER NAMED (S-280, 2026-09-26). This
            # added `GoogleAISkill` (a tool-bearing model skill) whenever the
            # file named no LLM skill, and defaulted the model to
            # `google/gemini-2.5-flash`, whatever keys the developer had.
            # `choose_model` is the file-source loader's helper: it picks the
            # model the file/skills imply and does NOT force Google. An agent
            # with no model skill simply has none, as it does under `serve`.
            # Never the owner's sign-in for the daemon's callers (S-327).
            from webagents.cli.model_access import choose_model
            model = choose_model(merged.metadata.model, skills, name, for_callers=True)

            # `fallback_models:` (plan item 2.8): the model's skill becomes a chain.
            if merged.metadata.fallback_models:
                from webagents.cli.agent_builder import apply_fallback_models

                model, fallback_failed = apply_fallback_models(
                    skills, merged.metadata.fallback_models, model, name, for_callers=True
                )
                for failed_name, reason in fallback_failed:
                    logger.warning(f'[Manager] Skill "{failed_name}" failed to load: {reason}')

            # Create BaseAgent
            agent = BaseAgent(
                name=merged.metadata.name or name,
                instructions=merged.instructions,
                skills=skills,
                scopes=merged.metadata.scopes or ["all"],
                model=model,
            )
            # The file's `description:`, for the A2A card and the listing (B7, 2026-09-28).
            agent.description = merged.metadata.description or ""
            # `observability: {otel: true}` records the run as OpenTelemetry spans (plan item 2.4).
            agent.observability = merged.metadata.observability if isinstance(merged.metadata.observability, dict) else None
            finish_access(agent, access_policy, skills)
            
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
        """Load and instantiate skills from config.

        This built each skill from its own `skill_classes` table below, so a
        dead `from webagents.server.plugins.local_file_source import
        LocalFileSource` (the module is `webagents.server.extensions.…`, and
        the class was never used) raised `ModuleNotFoundError` on the first
        line and made `get_or_load_agent` always return None (which is why
        S-280 was verified by reading, not exercised). Removed so the loader
        actually loads exactly the declared skills (S-280, 2026-09-26).
        """
        loaded_skills = {}
        
        skill_classes = {
            "filesystem": "webagents.agents.skills.local.filesystem.skill.FilesystemSkill",
            "shell": "webagents.agents.skills.local.shell.skill.ShellSkill",
            "rag": "webagents.agents.skills.local.rag.skill.LocalRagSkill",
            "session": "webagents.agents.skills.local.session.skill.SessionSkill",
            # LLM skills
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
                    # An agent FILE the daemon serves for its local owner is
                    # a place `${env:NAME}` and `${secret:NAME}` resolve
                    # (S-295): the sources are the owner's own environment and
                    # keystore. A skill built without them expands nothing.
                    from webagents.agents.skills.local.mcp.skill import owner_reference_sources

                    # Allow both {"mcp": {...servers...}} and {"mcp": {"mcpServers": {...}}}
                    if "mcpServers" in raw_config:
                        # Under `mcp`, as the agent builder passes it: the
                        # normalizer takes the wrapper there, while a top-level
                        # `mcpServers` was never found (its scan looks for
                        # `command`/`url` one level down).
                        config = {"mcp": raw_config, "references": owner_reference_sources()}
                        logger.debug(f"[Manager] MCP config has mcpServers, using as-is")
                    else:
                        # Assume top-level keys are server names, wrap them
                        config = {"mcp": raw_config, "references": owner_reference_sources()}
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
        """One turn of the agent's real runtime (plan item 1.7, 2026-09-26).

        This was a `while True: sleep(1)` under a TODO, so a "running" agent
        did nothing and its cron job, whose execution was this task, ran
        nothing. Now `start(name, prompt)` loads the agent with its skills
        (`get_or_load_agent`), runs `prompt` as one turn of the agent's OWNER
        (`access.run_as_local_owner`: a scheduled or started prompt is the
        owner's own), records the reply in the agent's log, and finishes.
        Without a prompt it loads the agent and finishes. The `cron:`
        schedules themselves are the schedule runner's
        (`schedule_runner.py`), which runs turns the same way.

        Args:
            agent: Agent to run
            prompt: The turn's message, if any
        """
        try:
            loaded = await self.get_or_load_agent(agent.name)
            if loaded is None:
                raise RuntimeError(f"agent '{agent.name}' could not be loaded from {agent.source_path}")
            self._log(agent.name, f"Loaded agent from: {agent.source_path}")

            if prompt:
                from webagents.access import run_as_local_owner

                from .schedule_runner import reply_text

                run_as_local_owner(loaded)
                response = await loaded.run([{"role": "user", "content": prompt}])
                self._log(agent.name, f"Reply: {reply_text(response)}")
            agent.status = "stopped"

        except asyncio.CancelledError:
            self._log(agent.name, "Agent task cancelled")
            raise
        except Exception as e:
            self._log(agent.name, f"Agent error: {e}")
            agent.status = "error"
            raise
        finally:
            # The turn is over: the name is free to start again.
            self._running_agents.pop(agent.name, None)
    
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
