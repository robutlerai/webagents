"""
DiscoverySkill - Simplified WebAgents Platform Integration

Agent discovery skill for WebAgents platform.
Provides intent-based agent discovery and intent publishing capabilities.
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, prompt


@dataclass
class DiscoveryResult:
    """Result from intent discovery operation"""
    agent_id: str
    intent: str
    agent_description: str
    similarity: float
    url: str
    rank: int


class DiscoverySkill(Skill):
    """
    Simplified agent discovery skill for WebAgents platform
    
    Key Features:
    - Intent-based agent discovery via Portal API
    - Intent publishing for agent registration
    - Dynamic agent URL configuration for prompts
    
    Configuration hierarchy for robutler_api_key:
    1. config.robutler_api_key (explicit configuration)
    2. agent.api_key (agent's API key)
    3. WEBAGENTS_API_KEY environment variable
    4. SERVICE_TOKEN environment variable
    
    Configuration hierarchy for agent_base_url:
    1. AGENTS_BASE_URL environment variable
    2. config.agent_base_url (explicit configuration)
    3. 'http://localhost:2224' (default for local development)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all")
        
        # Configuration
        self.config = config or {}
        self.enable_discovery = self.config.get('enable_discovery', True)
        
        # WebAgents platform configuration
        self.webagents_api_url = (
            os.getenv('ROBUTLER_INTERNAL_API_URL') or
            os.getenv('ROBUTLER_API_URL') or 
            self.config.get('webagents_api_url') or 
            'https://webagents.ai'
        )
        
        # Agent communication base URL configuration (for dynamic agent URLs in prompts)
        self.agent_base_url = (
            os.getenv('AGENTS_BASE_URL') or
            self.config.get('agent_base_url') or 
            'http://localhost:2224'  # Default for local development (agents server)
        )
        
        # API key: config first (will be resolved in initialize)
        self.robutler_api_key = self.config.get('robutler_api_key')
    
    async def initialize(self, agent) -> None:
        """Initialize DiscoverySkill"""
        from webagents.utils.logging import get_logger, log_skill_event
        
        self.agent = agent
        self.logger = get_logger('skill.webagents.discovery', self.agent.name)
        
        # Resolve WebAgents API key using hierarchy: config -> agent.api_key -> env
        if not self.robutler_api_key:
            if hasattr(self.agent, 'api_key') and self.agent.api_key:
                self.robutler_api_key = self.agent.api_key
                self.logger.debug(f"Using agent's API key for WebAgents platform")
            elif os.getenv('WEBAGENTS_API_KEY'):
                self.robutler_api_key = os.getenv('WEBAGENTS_API_KEY')
                self.logger.debug(f"Using WEBAGENTS_API_KEY from environment")
            elif os.getenv('SERVICE_TOKEN'):
                self.robutler_api_key = os.getenv('SERVICE_TOKEN')
                self.logger.debug(f"Using SERVICE_TOKEN from environment")
            else:
                self.logger.warning("No API key configured for discovery")
        
        log_skill_event(self.agent.name, 'discovery', 'initialized', {
            'enable_discovery': self.enable_discovery,
            'webagents_api_url': self.webagents_api_url,
            'agent_base_url': self.agent_base_url,
            'has_api_key': bool(self.robutler_api_key)
        })
    
    # ===== DISCOVERY TOOLS =====
    
    @prompt(priority=20, scope="all")
    def discovery_general_prompt(self, context: Any = None) -> str:
        base_url = self.agent_base_url.rstrip('/')
        return f"You are part of a network of AI agents working for their owners. Each agent has their own name and address. @agentname corresponds to {base_url}/agents/agentname. Use discovery_tool to discover other agents and their capabilities."

    @tool(description="Discover agents and their capabilities across the WebAgents network by intent. Make the intent as specific as possible. When presenting the results, make it simple and non-technical.", scope="all")
    async def discovery_tool(self, 
                            intent: str,
                            top_k: int = 10,
                            context=None) -> Dict[str, Any]:
        """Discover agents by intent across the WebAgents network"""
        if not self.enable_discovery:
            return {'success': False, 'error': 'Discovery disabled'}
        
        if not self.robutler_api_key:
            return {'success': False, 'error': 'API key not configured for discovery'}
        
        try:
            import aiohttp
            
            # Clamp top_k between 1-50
            limited_top_k = max(1, min(50, top_k))
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.webagents_api_url}/api/intents/search",
                    headers={
                        'Authorization': f'Bearer {self.robutler_api_key}',
                        'Content-Type': 'application/json',
                    },
                    json={
                        'intent': intent.strip(),
                        'top_k': limited_top_k,
                    }
                ) as response:
                    
                    if not response.ok:
                        raise Exception(f"Discovery API error: {response.status}")
                    
                    result = await response.json()
                    results = result.get('data', {}).get('results', [])
                    
                    return {
                        'success': True,
                        'intent': intent,
                        'results_count': len(results),
                        'results': [
                            {
                                'agent_id': r.get('agent_id'),
                                'intent': r.get('intent'),
                                'description': r.get('agent_description'),
                                'similarity': r.get('similarity'),
                                'url': r.get('url'),
                                'rank': r.get('rank'),
                            }
                            for r in results
                        ]
                    }
                    
        except Exception as e:
            self.logger.error(f"Agent discovery failed: {e}")
            return {
                'success': False,
                'intent': intent,
                'results_count': 0,
                'results': [],
                'error': str(e)
            }
    
    @tool(description="Publish agent intents to the platform", scope="owner")
    async def publish_intents_tool(self,
                            intents: List[str],
                            description: str,
                            context=None) -> Dict[str, Any]:
        """Publish agent intents to the WebAgents platform"""
        if not self.enable_discovery:
            return {'success': False, 'error': 'Discovery disabled'}
        
        if not self.robutler_api_key:
            return {'success': False, 'error': 'API key not configured for discovery'}
        
        try:
            import aiohttp
            
            # Get agent information
            agent_id = getattr(self.agent, 'name', 'unknown')
            agent_url = self.config.get('agent_url', f"https://webagents.ai/agents/{agent_id}")
            
            # Prepare intent data
            intents_data = [
                {
                    'intent': intent,
                    'agent_id': agent_id,
                    'description': description,
                    'url': agent_url,
                }
                for intent in intents
            ]
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.webagents_api_url}/api/intents/publish",
                    headers={
                        'Authorization': f'Bearer {self.robutler_api_key}',
                        'Content-Type': 'application/json',
                    },
                    json={'intents': intents_data}
                ) as response:
                    
                    if not response.ok:
                        raise Exception(f"Publish API error: {response.status}")
                    
                    result = await response.json()
                    
                    return {
                        'success': True,
                        'agent_id': agent_id,
                        'published_intents': intents,
                        'results': result
                    }
                    
        except Exception as e:
            self.logger.error(f"Intent publishing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_dependencies(self) -> List[str]:
        """Get skill dependencies"""
        return ['aiohttp']  # Required for HTTP client 