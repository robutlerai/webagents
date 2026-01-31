"""
PublishSkill - Publish Local Agents to WebAgents Platform

Provides commands for publishing local agents to the robutler.ai platform.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import command


@dataclass
class PublishResult:
    """Result from agent publishing operation"""
    success: bool
    agent_id: str
    url: Optional[str] = None
    error: Optional[str] = None


class PublishSkill(Skill):
    """
    Publish local agents to the WebAgents platform.
    
    Commands exposed:
    - /publish - Publish current agent to platform
    - /publish status - Check publication status
    - /publish unpublish - Remove agent from platform
    
    Configuration:
    - robutler_api_key: API key for platform (or env WEBAGENTS_API_KEY)
    - webagents_api_url: Platform URL (default: https://webagents.ai)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="owner")
        
        self.config = config or {}
        
        # Platform configuration
        self.webagents_api_url = (
            os.getenv('ROBUTLER_API_URL') or 
            self.config.get('webagents_api_url') or 
            'https://webagents.ai'
        )
        
        self.robutler_api_key = self.config.get('robutler_api_key')
    
    async def initialize(self, agent) -> None:
        """Initialize PublishSkill"""
        from webagents.utils.logging import get_logger
        
        self.agent = agent
        self.logger = get_logger('skill.webagents.publish', self.agent.name)
        
        # Resolve API key
        if not self.robutler_api_key:
            if hasattr(self.agent, 'api_key') and self.agent.api_key:
                self.robutler_api_key = self.agent.api_key
            elif os.getenv('WEBAGENTS_API_KEY'):
                self.robutler_api_key = os.getenv('WEBAGENTS_API_KEY')
            elif os.getenv('SERVICE_TOKEN'):
                self.robutler_api_key = os.getenv('SERVICE_TOKEN')
    
    # ===== SLASH COMMANDS =====
    
    @command("/publish", description="Publish current agent to the WebAgents platform", scope="owner")
    async def cmd_publish(self, visibility: str = "internal") -> Dict[str, Any]:
        """Publish the current agent to the WebAgents platform.
        
        Usage: /publish [visibility]
        
        Visibility options:
        - internal (default): Only accessible within your organization
        - public: Discoverable by all platform users
        
        Examples:
          /publish
          /publish public
        """
        if not self.robutler_api_key:
            return {"error": "API key not configured. Set WEBAGENTS_API_KEY or configure robutler_api_key."}
        
        try:
            import aiohttp
            
            # Get agent metadata
            agent_name = getattr(self.agent, 'name', 'unknown')
            agent_description = getattr(self.agent, 'description', '')
            agent_intents = getattr(self.agent, 'intents', [])
            
            # Read agent file content
            agent_path = getattr(self.agent, 'agent_path', None)
            agent_content = ""
            if agent_path and Path(agent_path).exists():
                agent_content = Path(agent_path).read_text()
            
            # Prepare payload
            payload = {
                "name": agent_name,
                "description": agent_description,
                "intents": agent_intents,
                "visibility": visibility,
                "content": agent_content,
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.webagents_api_url}/api/agents/publish",
                    headers={
                        'Authorization': f'Bearer {self.robutler_api_key}',
                        'Content-Type': 'application/json',
                    },
                    json=payload
                ) as response:
                    if not response.ok:
                        error_text = await response.text()
                        return {"error": f"Publish failed: {response.status} - {error_text}"}
                    
                    result = await response.json()
                    
                    return {
                        "success": True,
                        "agent": agent_name,
                        "visibility": visibility,
                        "url": result.get("url", f"{self.webagents_api_url}/agents/{agent_name}"),
                        "message": f"Agent '{agent_name}' published successfully!"
                    }
                    
        except ImportError:
            return {"error": "aiohttp not installed. Run: pip install aiohttp"}
        except Exception as e:
            self.logger.error(f"Publish failed: {e}")
            return {"error": str(e)}
    
    @command("/publish/status", description="Check agent publication status", scope="owner")
    async def cmd_publish_status(self) -> Dict[str, Any]:
        """Check if the current agent is published on the platform.
        
        Usage: /publish status
        """
        if not self.robutler_api_key:
            return {"error": "API key not configured"}
        
        try:
            import aiohttp
            
            agent_name = getattr(self.agent, 'name', 'unknown')
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.webagents_api_url}/api/agents/{agent_name}/status",
                    headers={
                        'Authorization': f'Bearer {self.robutler_api_key}',
                    }
                ) as response:
                    if response.status == 404:
                        return {
                            "published": False,
                            "agent": agent_name,
                            "message": "Agent is not published"
                        }
                    
                    if not response.ok:
                        return {"error": f"Status check failed: {response.status}"}
                    
                    result = await response.json()
                    
                    return {
                        "published": True,
                        "agent": agent_name,
                        "visibility": result.get("visibility"),
                        "url": result.get("url"),
                        "updated_at": result.get("updated_at"),
                    }
                    
        except Exception as e:
            return {"error": str(e)}
    
    @command("/publish/unpublish", alias="/unpublish", description="Remove agent from platform", scope="owner")
    async def cmd_unpublish(self) -> Dict[str, Any]:
        """Remove the current agent from the platform.
        
        Usage: /publish unpublish
               /unpublish
        """
        if not self.robutler_api_key:
            return {"error": "API key not configured"}
        
        try:
            import aiohttp
            
            agent_name = getattr(self.agent, 'name', 'unknown')
            
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.webagents_api_url}/api/agents/{agent_name}",
                    headers={
                        'Authorization': f'Bearer {self.robutler_api_key}',
                    }
                ) as response:
                    if not response.ok:
                        return {"error": f"Unpublish failed: {response.status}"}
                    
                    return {
                        "success": True,
                        "agent": agent_name,
                        "message": f"Agent '{agent_name}' removed from platform"
                    }
                    
        except Exception as e:
            return {"error": str(e)}
    
    def get_dependencies(self) -> List[str]:
        """Get skill dependencies"""
        return ['aiohttp']
