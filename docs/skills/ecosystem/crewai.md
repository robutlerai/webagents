# CrewAI Skill

Simplified CrewAI integration for multi-agent orchestration. Execute pre-configured crews of specialized AI agents with collaborative workflows.

## Features

- **Pre-configured Crews**: Set up crews during skill initialization
- **Single Tool Execution**: One simple tool to run configured crews
- **Multi-Agent Orchestration**: Specialized AI agents with defined roles
- **Process Management**: Sequential and hierarchical workflow execution
- **Input-driven Execution**: Provide inputs to drive crew workflows
- **No Dependencies**: Simplified architecture with no external skill dependencies

## Quick Setup

### Prerequisites

Install CrewAI:

```bash
pip install crewai
```

### Option 1: Using CrewAI Crew Object (Recommended)

```python
from webagents.agents import BaseAgent
from webagents.agents.skills.ecosystem.crewai import CrewAISkill
from crewai import Agent, Task, Crew, Process

# Create CrewAI agents and tasks using native API
researcher = Agent(
    role='Senior Researcher',
    goal='Uncover groundbreaking technologies in AI for year 2024',
    backstory='Driven by curiosity, you explore and share the latest innovations.',
    verbose=True
)

research_task = Task(
    description='Identify the next big trend in {topic} with pros and cons.',
    expected_output='A 3-paragraph report on emerging {topic} technologies.',
    agent=researcher
)

# Form the crew
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    process=Process.sequential,
    verbose=True
)

# Create agent with CrewAI crew object
agent = BaseAgent(
    name="research-agent",
    model="openai/gpt-4o",
    skills={
        "crewai": CrewAISkill(crew)  # Pass Crew object directly
    }
)
```

### Option 2: Using Configuration Dictionary

```python
from webagents.agents import BaseAgent
from webagents.agents.skills.ecosystem.crewai import CrewAISkill

# Define your crew configuration
crew_config = {
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
            'description': 'Identify the next big trend in {topic} with pros and cons.',
            'expected_output': 'A 3-paragraph report on emerging {topic} technologies.',
            'agent_index': 0
        }
    ],
    'process': 'sequential',
    'verbose': True
}

# Create agent with configured CrewAI skill
agent = BaseAgent(
    name="research-agent",
    model="openai/gpt-4o",
    skills={
        "crewai": CrewAISkill(crew_config)
    }
)
```

## Usage

### Core Tool

**Tool**: `crewai_run(inputs)`

Execute the configured CrewAI crew with the provided inputs.

```python
# Example usage via LLM
messages = [{
    'role': 'user', 
    'content': 'Research AI Agents and their applications'
}]
response = await agent.run(messages=messages)
```

The LLM will automatically call `crewai_run({'topic': 'AI Agents'})` based on the crew configuration and user request.

## Use Case Example: Research Team

This example shows a complete research crew that analyzes topics and provides insights:

### Using CrewAI Objects (Recommended)

```python
from crewai import Agent, Task, Crew, Process

# Create specialized agents
researcher = Agent(
    role='Senior Researcher',
    goal='Conduct comprehensive research on specified topics',
    backstory='You are an experienced researcher with access to latest information and analytical skills.'
)

analyst = Agent(
    role='Data Analyst',
    goal='Analyze research data and extract key insights',
    backstory='You specialize in data analysis and pattern recognition.'
)

# Create sequential tasks
research_task = Task(
    description='Research the current state of {topic} technology and market trends',
    expected_output='Detailed research report with key findings',
    agent=researcher
)

analysis_task = Task(
    description='Analyze the research data and identify the top 3 opportunities in {topic}',
    expected_output='Analysis report with ranked opportunities and recommendations',
    agent=analyst
)

# Form the crew
research_crew = Crew(
    agents=[researcher, analyst],
    tasks=[research_task, analysis_task],
    process=Process.sequential
)

# Initialize skill with crew
agent = BaseAgent(
    name="research-agent",
    model="openai/gpt-4o",
    skills={
        "crewai": CrewAISkill(research_crew)
    }
)
```

### Using Configuration Dictionary

```python
research_crew_config = {
    'agents': [
        {
            'role': 'Senior Researcher',
            'goal': 'Conduct comprehensive research on specified topics',
            'backstory': 'You are an experienced researcher with access to latest information and analytical skills.'
        },
        {
            'role': 'Data Analyst', 
            'goal': 'Analyze research data and extract key insights',
            'backstory': 'You specialize in data analysis and pattern recognition.'
        }
    ],
    'tasks': [
        {
            'description': 'Research the current state of {topic} technology and market trends',
            'expected_output': 'Detailed research report with key findings',
            'agent_index': 0
        },
        {
            'description': 'Analyze the research data and identify the top 3 opportunities in {topic}',
            'expected_output': 'Analysis report with ranked opportunities and recommendations',
            'agent_index': 1
        }
    ],
    'process': 'sequential'
}

# Initialize skill with configuration
agent = BaseAgent(
    name="research-agent", 
    model="openai/gpt-4o",
    skills={
        "crewai": CrewAISkill(research_crew_config)
    }
)
```

### Example Interaction

```python
# User asks for research
messages = [{
    'role': 'user',
    'content': 'Research the latest developments in quantum computing and identify business opportunities'
}]

# The LLM automatically:
# 1. Recognizes this as a research request
# 2. Calls crewai_run({'topic': 'quantum computing'})
# 3. The researcher agent researches quantum computing trends
# 4. The analyst agent identifies top 3 business opportunities
# 5. Returns comprehensive analysis

response = await agent.run(messages=messages)
print(response)
```

## Configuration Reference

```python
crew_config = {
    'agents': [
        {
            'role': 'Agent Role/Title',           # Required
            'goal': 'Agent primary objective',   # Required
            'backstory': 'Agent background',     # Required
            'verbose': True,                     # Optional
            'allow_delegation': False            # Optional
        }
    ],
    'tasks': [
        {
            'description': 'Task with {input_variable} placeholders',  # Required
            'expected_output': 'Expected deliverable format',          # Optional
            'agent_index': 0                                           # Optional, defaults to 0
        }
    ],
    'process': 'sequential',  # 'sequential' or 'hierarchical'
    'verbose': True           # Optional, defaults to True
}
```

## Troubleshooting

**"No CrewAI crew configured"** - Initialize the skill with a valid crew configuration or Crew object

**"Inputs are required to run the crew"** - Ensure your tasks use input variables like `{topic}` that the LLM can populate

**"CrewAI execution failed"** - Check CrewAI installation: `pip install crewai`