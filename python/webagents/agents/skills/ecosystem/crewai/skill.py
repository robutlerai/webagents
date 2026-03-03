"""
Simplified CrewAI Skill for WebAgents

This skill runs a pre-configured CrewAI crew with agents and tasks.
The crew configuration is provided during skill initialization.

Main use case: Execute a specific crew workflow on demand.
"""

import os
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, prompt

try:
    from crewai import Agent, Task, Crew, Process
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    Agent, Task, Crew, Process = None, None, None, None  # type: ignore


class CrewAISkill(Skill):
    """Simplified CrewAI skill for running pre-configured crews"""
    
    def __init__(self, crew_or_config: Optional[Union[Dict[str, Any], Any]] = None):
        super().__init__({}, scope="all")
        if not CREWAI_AVAILABLE:
            raise ImportError("CrewAI is not installed. Install with: pip install crewai")
        
        # Handle both Crew object and configuration dictionary
        if crew_or_config is None:
            self.crew_config = {}
            self.crew = None
        elif hasattr(crew_or_config, 'agents') and hasattr(crew_or_config, 'tasks'):
            # It's a Crew object
            self.crew_config = {}
            self.crew = crew_or_config
        else:
            # It's a configuration dictionary
            self.crew_config = crew_or_config or {}
            self.crew = None
            self._setup_crew()
    
    def get_dependencies(self) -> List[str]:
        """Skill dependencies"""
        return []  # No dependencies needed for simple crew execution
    
    def _setup_crew(self):
        """Set up the CrewAI crew from configuration"""
        if not self.crew_config:
            return
        
        try:
            # Get agents configuration
            agents_config = self.crew_config.get('agents', [])
            tasks_config = self.crew_config.get('tasks', [])
            process_type = self.crew_config.get('process', 'sequential')
            
            if not agents_config or not tasks_config:
                return
            
            # Create agents
            agents = []
            for agent_config in agents_config:
                agent = Agent(
                    role=agent_config.get('role', 'Agent'),
                    goal=agent_config.get('goal', 'Complete assigned tasks'),
                    backstory=agent_config.get('backstory', 'An AI agent ready to help'),
                    verbose=agent_config.get('verbose', True),
                    allow_delegation=agent_config.get('allow_delegation', False)
                )
                agents.append(agent)
            
            # Create tasks
            tasks = []
            for i, task_config in enumerate(tasks_config):
                # Assign agent to task (default to first agent if not specified)
                agent_index = task_config.get('agent_index', 0)
                if agent_index >= len(agents):
                    agent_index = 0  # Fallback to first agent
                
                task = Task(
                    description=task_config.get('description', f'Task {i+1}'),
                    agent=agents[agent_index],
                    expected_output=task_config.get('expected_output', 'Task completion')
                )
                tasks.append(task)
            
            # Create crew
            process = Process.sequential
            if process_type.lower() == 'hierarchical':
                process = Process.hierarchical
            
            self.crew = Crew(
                agents=agents,
                tasks=tasks,
                process=process,
                verbose=self.crew_config.get('verbose', True)
            )
            
            # CrewAI crew initialized successfully
            
        except Exception as e:
            self.crew = None

    @prompt(priority=40, scope=["owner", "all"])
    def crewai_prompt(self) -> str:
        """Prompt describing CrewAI capabilities"""
        if not self.crew:
            return """
CrewAI skill is available but no crew is configured. 

To use CrewAI, initialize the skill with either:
1. A CrewAI Crew object: CrewAISkill(crew)
2. A crew configuration dictionary with agents, tasks, and process settings
"""
        
        agents_count = len(self.crew.agents) if self.crew else 0
        tasks_count = len(self.crew.tasks) if self.crew else 0
        
        return f"""
CrewAI multi-agent orchestration is ready. Available tool:

• crewai_run(inputs) - Execute the configured crew with the given inputs

Configured crew:
- {agents_count} agents with specialized roles
- {tasks_count} tasks in the workflow
- Ready to process your requests through collaborative AI agents

Provide inputs as a dictionary to run the crew workflow.
"""

    # Public tool
    @tool(description="Execute the configured CrewAI crew with the given inputs")
    async def crewai_run(self, inputs: Dict[str, Any]) -> str:
        """Execute the configured CrewAI crew with the provided inputs"""
        if not self.crew:
            return "❌ No CrewAI crew configured. Please initialize the skill with a crew configuration."
        
        if not inputs:
            return "❌ Inputs are required to run the crew"
        
        try:
            # Execute the crew with inputs
            result = self.crew.kickoff(inputs=inputs)
            
            agents_count = len(self.crew.agents)
            tasks_count = len(self.crew.tasks)
            
            return f"✅ CrewAI execution completed successfully!\n👥 Agents: {agents_count}\n📝 Tasks: {tasks_count}\n\n📊 Result:\n{result}"
            
        except Exception as e:
            return f"❌ CrewAI execution failed: {str(e)}"
