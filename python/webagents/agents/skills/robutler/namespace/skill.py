"""
NamespaceSkill - Agent Namespace Management

Provides commands for managing agent namespaces. Agents within a namespace
can only communicate with other agents in the same namespace, providing
isolation for different environments or organizations.
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import command


@dataclass
class Namespace:
    """Namespace information"""
    id: str
    name: str
    description: Optional[str] = None
    owner_id: Optional[str] = None
    member_count: int = 0
    created_at: Optional[str] = None


class NamespaceSkill(Skill):
    """
    Manage agent namespaces on the WebAgents platform.
    
    Namespaces provide isolation - agents within a namespace can only
    communicate with other agents in the same namespace.
    
    Commands exposed:
    - /namespace list - List available namespaces
    - /namespace create - Create a new namespace
    - /namespace join - Join an existing namespace
    - /namespace leave - Leave current namespace
    - /namespace delete - Delete a namespace (owner only)
    - /namespace info - Show current namespace info
    
    Configuration:
    - robutler_api_key: API key for platform
    - webagents_api_url: Platform URL
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all")
        
        self.config = config or {}
        
        # Platform configuration
        self.webagents_api_url = (
            os.getenv('ROBUTLER_API_URL') or 
            self.config.get('webagents_api_url') or 
            'https://webagents.ai'
        )
        
        self.robutler_api_key = self.config.get('robutler_api_key')
        
        # Current namespace (can be set in agent config)
        self.current_namespace = self.config.get('namespace', 'default')
    
    async def initialize(self, agent) -> None:
        """Initialize NamespaceSkill"""
        from webagents.utils.logging import get_logger
        
        self.agent = agent
        self.logger = get_logger('skill.webagents.namespace', self.agent.name)
        
        # Resolve API key
        if not self.robutler_api_key:
            if hasattr(self.agent, 'api_key') and self.agent.api_key:
                self.robutler_api_key = self.agent.api_key
            elif os.getenv('WEBAGENTS_API_KEY'):
                self.robutler_api_key = os.getenv('WEBAGENTS_API_KEY')
            elif os.getenv('SERVICE_TOKEN'):
                self.robutler_api_key = os.getenv('SERVICE_TOKEN')
        
        # Get namespace from agent metadata
        if hasattr(self.agent, 'namespace'):
            self.current_namespace = self.agent.namespace
    
    # ===== SLASH COMMANDS =====
    
    @command("/namespace", alias="/ns", description="Show current namespace info")
    async def cmd_namespace_info(self) -> Dict[str, Any]:
        """Show information about the current namespace.
        
        Usage: /namespace
               /ns
        """
        agent_name = getattr(self.agent, 'name', 'unknown')
        
        return {
            "agent": agent_name,
            "namespace": self.current_namespace,
            "message": f"Agent '{agent_name}' is in namespace '{self.current_namespace}'"
        }
    
    @command("/namespace/list", description="List available namespaces")
    async def cmd_namespace_list(self) -> Dict[str, Any]:
        """List namespaces you have access to.
        
        Usage: /namespace list
        """
        if not self.robutler_api_key:
            return {"error": "API key not configured"}
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.webagents_api_url}/api/namespaces",
                    headers={
                        'Authorization': f'Bearer {self.robutler_api_key}',
                    }
                ) as response:
                    if not response.ok:
                        return {"error": f"List failed: {response.status}"}
                    
                    result = await response.json()
                    namespaces = result.get("namespaces", [])
                    
                    return {
                        "current": self.current_namespace,
                        "namespaces": [
                            {
                                "name": ns.get("name"),
                                "description": ns.get("description"),
                                "members": ns.get("member_count", 0),
                            }
                            for ns in namespaces
                        ],
                        "count": len(namespaces)
                    }
                    
        except ImportError:
            return {"error": "aiohttp not installed"}
        except Exception as e:
            return {"error": str(e)}
    
    @command("/namespace/create", description="Create a new namespace", scope="owner")
    async def cmd_namespace_create(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new namespace.
        
        Usage: /namespace create <name> [description]
        
        Examples:
          /namespace create production
          /namespace create dev "Development environment"
        """
        if not name:
            return {"error": "Namespace name is required", "usage": "/namespace create <name>"}
        
        if not self.robutler_api_key:
            return {"error": "API key not configured"}
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.webagents_api_url}/api/namespaces",
                    headers={
                        'Authorization': f'Bearer {self.robutler_api_key}',
                        'Content-Type': 'application/json',
                    },
                    json={
                        "name": name,
                        "description": description,
                    }
                ) as response:
                    if not response.ok:
                        error_text = await response.text()
                        return {"error": f"Create failed: {response.status} - {error_text}"}
                    
                    result = await response.json()
                    
                    return {
                        "success": True,
                        "namespace": name,
                        "message": f"Namespace '{name}' created successfully"
                    }
                    
        except Exception as e:
            return {"error": str(e)}
    
    @command("/namespace/join", description="Join an existing namespace")
    async def cmd_namespace_join(self, name: str) -> Dict[str, Any]:
        """Join a namespace.
        
        Usage: /namespace join <name>
        
        This will set the agent's namespace for future communications.
        """
        if not name:
            return {"error": "Namespace name is required", "usage": "/namespace join <name>"}
        
        if not self.robutler_api_key:
            return {"error": "API key not configured"}
        
        try:
            import aiohttp
            
            agent_name = getattr(self.agent, 'name', 'unknown')
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.webagents_api_url}/api/namespaces/{name}/join",
                    headers={
                        'Authorization': f'Bearer {self.robutler_api_key}',
                        'Content-Type': 'application/json',
                    },
                    json={"agent_id": agent_name}
                ) as response:
                    if not response.ok:
                        return {"error": f"Join failed: {response.status}"}
                    
                    # Update local namespace
                    self.current_namespace = name
                    if hasattr(self.agent, 'namespace'):
                        self.agent.namespace = name
                    
                    return {
                        "success": True,
                        "namespace": name,
                        "message": f"Joined namespace '{name}'"
                    }
                    
        except Exception as e:
            return {"error": str(e)}
    
    @command("/namespace/leave", description="Leave current namespace")
    async def cmd_namespace_leave(self) -> Dict[str, Any]:
        """Leave the current namespace and return to default.
        
        Usage: /namespace leave
        """
        if self.current_namespace == "default":
            return {"message": "Already in default namespace"}
        
        if not self.robutler_api_key:
            return {"error": "API key not configured"}
        
        try:
            import aiohttp
            
            agent_name = getattr(self.agent, 'name', 'unknown')
            old_namespace = self.current_namespace
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.webagents_api_url}/api/namespaces/{old_namespace}/leave",
                    headers={
                        'Authorization': f'Bearer {self.robutler_api_key}',
                        'Content-Type': 'application/json',
                    },
                    json={"agent_id": agent_name}
                ) as response:
                    if not response.ok:
                        return {"error": f"Leave failed: {response.status}"}
                    
                    # Update local namespace
                    self.current_namespace = "default"
                    if hasattr(self.agent, 'namespace'):
                        self.agent.namespace = "default"
                    
                    return {
                        "success": True,
                        "left": old_namespace,
                        "current": "default",
                        "message": f"Left namespace '{old_namespace}', now in 'default'"
                    }
                    
        except Exception as e:
            return {"error": str(e)}
    
    @command("/namespace/delete", description="Delete a namespace (owner only)", scope="owner")
    async def cmd_namespace_delete(self, name: str) -> Dict[str, Any]:
        """Delete a namespace you own.
        
        Usage: /namespace delete <name>
        
        Warning: This will remove all agents from the namespace.
        """
        if not name:
            return {"error": "Namespace name is required", "usage": "/namespace delete <name>"}
        
        if name == "default":
            return {"error": "Cannot delete the default namespace"}
        
        if not self.robutler_api_key:
            return {"error": "API key not configured"}
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.webagents_api_url}/api/namespaces/{name}",
                    headers={
                        'Authorization': f'Bearer {self.robutler_api_key}',
                    }
                ) as response:
                    if not response.ok:
                        return {"error": f"Delete failed: {response.status}"}
                    
                    # If we were in the deleted namespace, return to default
                    if self.current_namespace == name:
                        self.current_namespace = "default"
                        if hasattr(self.agent, 'namespace'):
                            self.agent.namespace = "default"
                    
                    return {
                        "success": True,
                        "deleted": name,
                        "message": f"Namespace '{name}' deleted"
                    }
                    
        except Exception as e:
            return {"error": str(e)}
    
    def get_dependencies(self) -> List[str]:
        """Get skill dependencies"""
        return ['aiohttp']
