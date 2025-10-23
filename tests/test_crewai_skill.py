"""
Test suite for simplified CrewAI skill
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

# Check if CrewAI is available
try:
    import crewai
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

from webagents.agents.skills.ecosystem.crewai.skill import CrewAISkill


@pytest.fixture
def crewai_skill():
    """Create CrewAISkill instance for testing without configuration"""
    with patch('webagents.agents.skills.ecosystem.crewai.skill.CREWAI_AVAILABLE', True):
        skill = CrewAISkill()
        
        # Create proper mock agent
        class MockAgent:
            def __init__(self):
                self.name = 'test-agent'
                self.skills = {}
        
        skill.agent = MockAgent()
        skill.logger = MagicMock()
        return skill


@pytest.fixture
def crewai_skill_with_crew_object():
    """Create CrewAISkill instance with a Crew object"""
    with patch('webagents.agents.skills.ecosystem.crewai.skill.CREWAI_AVAILABLE', True):
        # Mock a Crew object
        mock_crew = MagicMock()
        mock_crew.agents = [MagicMock(), MagicMock()]  # 2 agents
        mock_crew.tasks = [MagicMock()]  # 1 task
        mock_crew.kickoff.return_value = "Crew execution completed!"
        
        skill = CrewAISkill(mock_crew)
        
        # Create proper mock agent
        class MockAgent:
            def __init__(self):
                self.name = 'test-agent'
                self.skills = {}
        
        skill.agent = MockAgent()
        return skill


@pytest.fixture
def crewai_skill_with_config():
    """Create CrewAISkill instance with sample crew configuration"""
    config = {
        'agents': [
            {
                'role': 'Senior Researcher',
                'goal': 'Uncover groundbreaking technologies in AI for year 2024',
                'backstory': 'Driven by curiosity, you explore and share the latest innovations.',
                'verbose': True
            }
        ],
        'tasks': [
            {
                'description': 'Identify the next big trend in AI with pros and cons.',
                'expected_output': 'A 3-paragraph report on emerging AI technologies.',
                'agent_index': 0
            }
        ],
        'process': 'sequential',
        'verbose': True
    }
    
    with patch('webagents.agents.skills.ecosystem.crewai.skill.CREWAI_AVAILABLE', True):
        with patch('webagents.agents.skills.ecosystem.crewai.skill.Agent') as mock_agent_class:
            with patch('webagents.agents.skills.ecosystem.crewai.skill.Task') as mock_task_class:
                with patch('webagents.agents.skills.ecosystem.crewai.skill.Crew') as mock_crew_class:
                    with patch('webagents.agents.skills.ecosystem.crewai.skill.Process') as mock_process_class:
                        # Mock the CrewAI classes
                        mock_agent = MagicMock()
                        mock_agent_class.return_value = mock_agent
                        
                        mock_task = MagicMock()
                        mock_task_class.return_value = mock_task
                        
                        mock_crew = MagicMock()
                        mock_crew.agents = [mock_agent]
                        mock_crew.tasks = [mock_task]
                        mock_crew_class.return_value = mock_crew
                        
                        # Mock Process enum
                        mock_process_class.sequential = 'sequential'
                        mock_process_class.hierarchical = 'hierarchical'
                        
                        skill = CrewAISkill(config)
                        
                        # Create proper mock agent
                        class MockAgent:
                            def __init__(self):
                                self.name = 'test-agent'
                                self.skills = {}
                        
                        skill.agent = MockAgent()
                        skill.logger = MagicMock()
                        return skill


class TestCrewAISkill:
    """Test cases for simplified CrewAI skill"""

    def test_skill_initialization(self, crewai_skill):
        """Test skill initializes correctly"""
        assert 'test-agent' == crewai_skill.agent.name
        assert crewai_skill.get_dependencies() == []

    def test_skill_initialization_without_dependencies(self):
        """Test skill fails gracefully without dependencies installed"""
        with patch('webagents.agents.skills.ecosystem.crewai.skill.CREWAI_AVAILABLE', False):
            with pytest.raises(ImportError, match="CrewAI is not installed"):
                CrewAISkill()

    def test_skill_initialization_with_config(self, crewai_skill_with_config):
        """Test skill initializes correctly with configuration"""
        assert crewai_skill_with_config.crew is not None
        assert len(crewai_skill_with_config.crew.agents) == 1
        assert len(crewai_skill_with_config.crew.tasks) == 1

    def test_skill_initialization_with_crew_object(self, crewai_skill_with_crew_object):
        """Test skill initializes correctly with Crew object"""
        assert crewai_skill_with_crew_object.crew is not None
        assert len(crewai_skill_with_crew_object.crew.agents) == 2
        assert len(crewai_skill_with_crew_object.crew.tasks) == 1
        assert crewai_skill_with_crew_object.crew_config == {}

    def test_prompt_without_config(self, crewai_skill):
        """Test prompt when no crew is configured"""
        prompt = crewai_skill.crewai_prompt()
        assert 'CrewAI skill is available but no crew is configured' in prompt
        assert 'A CrewAI Crew object: CrewAISkill(crew)' in prompt
        assert 'crew configuration dictionary' in prompt

    def test_prompt_with_config(self, crewai_skill_with_config):
        """Test prompt when crew is configured"""
        prompt = crewai_skill_with_config.crewai_prompt()
        assert 'CrewAI multi-agent orchestration is ready' in prompt
        assert 'crewai_run(inputs)' in prompt
        assert '1 agents with specialized roles' in prompt
        assert '1 tasks in the workflow' in prompt

    def test_prompt_with_crew_object(self, crewai_skill_with_crew_object):
        """Test prompt when crew object is provided"""
        prompt = crewai_skill_with_crew_object.crewai_prompt()
        assert 'CrewAI multi-agent orchestration is ready' in prompt
        assert 'crewai_run(inputs)' in prompt
        assert '2 agents with specialized roles' in prompt
        assert '1 tasks in the workflow' in prompt

    def test_has_1_tool(self, crewai_skill):
        """Test skill has exactly 1 tool"""
        crewai_tools = [m for m in dir(crewai_skill) if m.startswith('crewai_') and not m.startswith('crewai_prompt')]
        assert len(crewai_tools) == 1
        assert 'crewai_run' in crewai_tools

    @pytest.mark.asyncio
    async def test_crewai_run_no_config(self, crewai_skill):
        """Test running crew without configuration"""
        result = await crewai_skill.crewai_run({'topic': 'AI Agents'})
        
        assert '❌ No CrewAI crew configured' in result

    @pytest.mark.asyncio
    async def test_crewai_run_no_inputs(self, crewai_skill_with_config):
        """Test running crew without inputs"""
        result = await crewai_skill_with_config.crewai_run({})
        
        assert '❌ Inputs are required to run the crew' in result

    @pytest.mark.asyncio
    async def test_crewai_run_with_crew_object(self, crewai_skill_with_crew_object):
        """Test running crew with Crew object"""
        result = await crewai_skill_with_crew_object.crewai_run({'topic': 'AI Agents'})
        
        assert '✅ CrewAI execution completed successfully!' in result
        assert 'Agents: 2' in result
        assert 'Tasks: 1' in result
        assert 'Crew execution completed!' in result

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CREWAI_AVAILABLE, reason="CrewAI not installed")
    async def test_crewai_run_success(self, crewai_skill_with_config):
        """Test successful crew execution"""
        # Mock the kickoff method
        crewai_skill_with_config.crew.kickoff.return_value = "Research completed: AI Agents are the future!"
        
        result = await crewai_skill_with_config.crewai_run({'topic': 'AI Agents'})
        
        assert '✅ CrewAI execution completed successfully!' in result
        assert 'Agents: 1' in result
        assert 'Tasks: 1' in result
        assert 'Research completed: AI Agents are the future!' in result

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CREWAI_AVAILABLE, reason="CrewAI not installed")
    async def test_crewai_run_failure(self, crewai_skill_with_config):
        """Test crew execution failure"""
        # Mock the kickoff method to raise an exception
        crewai_skill_with_config.crew.kickoff.side_effect = Exception("Execution failed")
        
        result = await crewai_skill_with_config.crewai_run({'topic': 'AI Agents'})
        
        assert '❌ CrewAI execution failed: Execution failed' in result

    def test_setup_crew_empty_config(self, crewai_skill):
        """Test crew setup with empty configuration"""
        crewai_skill.crew_config = {}
        crewai_skill._setup_crew()
        
        assert crewai_skill.crew is None

    def test_setup_crew_missing_agents(self, crewai_skill):
        """Test crew setup with missing agents"""
        crewai_skill.crew_config = {'tasks': [{'description': 'test'}]}
        crewai_skill._setup_crew()
        
        assert crewai_skill.crew is None

    def test_setup_crew_missing_tasks(self, crewai_skill):
        """Test crew setup with missing tasks"""
        crewai_skill.crew_config = {'agents': [{'role': 'test'}]}
        crewai_skill._setup_crew()
        
        assert crewai_skill.crew is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])