"""
Robutler Platform API Client

API client for robutler.ai platform.
"""

from typing import Optional, Dict, List, Any
import httpx

from ..state.local import get_state


class RobutlerAPI:
    """API client for robutler.ai platform."""
    
    def __init__(self, base_url: str = None):
        """Initialize API client.
        
        Args:
            base_url: Platform API URL
        """
        self.base_url = base_url or "https://api.robutler.ai"
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._get_headers(),
            timeout=30.0,
        )
        return self
    
    async def __aexit__(self, *args):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get authorization headers."""
        state = get_state()
        creds = state.get_credentials()
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "webagents-cli/1.0",
        }
        
        if creds.get("auth_type") == "api_key":
            headers["X-API-Key"] = creds.get("api_key", "")
        elif creds.get("access_token"):
            headers["Authorization"] = f"Bearer {creds['access_token']}"
        
        return headers
    
    # User endpoints
    
    async def get_user(self) -> Dict:
        """Get current user info."""
        response = await self._client.get("/v1/user")
        response.raise_for_status()
        return response.json()
    
    # Agent endpoints
    
    async def list_agents(self, namespace: str = None) -> List[Dict]:
        """List agents.
        
        Args:
            namespace: Filter by namespace
            
        Returns:
            List of agents
        """
        params = {}
        if namespace:
            params["namespace"] = namespace
        
        response = await self._client.get("/v1/agents", params=params)
        response.raise_for_status()
        return response.json().get("agents", [])
    
    async def get_agent(self, name: str) -> Dict:
        """Get agent details.
        
        Args:
            name: Agent name
            
        Returns:
            Agent info
        """
        response = await self._client.get(f"/v1/agents/{name}")
        response.raise_for_status()
        return response.json()
    
    async def register_agent(self, agent_data: Dict) -> Dict:
        """Register agent with platform.
        
        Args:
            agent_data: Agent data
            
        Returns:
            Registered agent
        """
        response = await self._client.post("/v1/agents", json=agent_data)
        response.raise_for_status()
        return response.json()
    
    async def update_agent(self, name: str, agent_data: Dict) -> Dict:
        """Update agent.
        
        Args:
            name: Agent name
            agent_data: Updated data
            
        Returns:
            Updated agent
        """
        response = await self._client.put(f"/v1/agents/{name}", json=agent_data)
        response.raise_for_status()
        return response.json()
    
    async def delete_agent(self, name: str) -> bool:
        """Delete agent.
        
        Args:
            name: Agent name
            
        Returns:
            Success
        """
        response = await self._client.delete(f"/v1/agents/{name}")
        return response.status_code == 204
    
    # Discovery endpoints
    
    async def discover(
        self,
        intent: str,
        namespace: str = None,
        limit: int = 10,
    ) -> List[Dict]:
        """Discover agents by intent.
        
        Args:
            intent: What you want to accomplish
            namespace: Filter by namespace
            limit: Max results
            
        Returns:
            List of matching agents
        """
        params = {
            "intent": intent,
            "limit": limit,
        }
        if namespace:
            params["namespace"] = namespace
        
        response = await self._client.get("/v1/discover", params=params)
        response.raise_for_status()
        return response.json().get("results", [])
    
    async def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Search agents.
        
        Args:
            query: Search query
            limit: Max results
            
        Returns:
            Search results
        """
        response = await self._client.get(
            "/v1/search",
            params={"q": query, "limit": limit}
        )
        response.raise_for_status()
        return response.json().get("results", [])
    
    # Intent endpoints
    
    async def publish_intent(
        self,
        agent_name: str,
        intent: str,
        visibility: str = "namespace",
    ) -> Dict:
        """Publish an intent.
        
        Args:
            agent_name: Agent name
            intent: Intent text
            visibility: local, namespace, or public
            
        Returns:
            Published intent
        """
        response = await self._client.post(
            "/v1/intents",
            json={
                "agent": agent_name,
                "intent": intent,
                "visibility": visibility,
            }
        )
        response.raise_for_status()
        return response.json()
    
    async def list_intents(self, agent_name: str = None) -> List[Dict]:
        """List published intents.
        
        Args:
            agent_name: Filter by agent
            
        Returns:
            List of intents
        """
        params = {}
        if agent_name:
            params["agent"] = agent_name
        
        response = await self._client.get("/v1/intents", params=params)
        response.raise_for_status()
        return response.json().get("intents", [])
    
    async def subscribe_intent(
        self,
        intent: str,
        callback_agent: str = None,
    ) -> Dict:
        """Subscribe to intent notifications.
        
        Args:
            intent: Intent to subscribe to
            callback_agent: Agent to route notifications to
            
        Returns:
            Subscription info
        """
        response = await self._client.post(
            "/v1/subscriptions",
            json={
                "intent": intent,
                "callback_agent": callback_agent,
            }
        )
        response.raise_for_status()
        return response.json()
    
    # Namespace endpoints
    
    async def list_namespaces(self) -> List[Dict]:
        """List accessible namespaces.
        
        Returns:
            List of namespaces
        """
        response = await self._client.get("/v1/namespaces")
        response.raise_for_status()
        return response.json().get("namespaces", [])
    
    async def create_namespace(
        self,
        name: str,
        type_: str = "user",
    ) -> Dict:
        """Create a namespace.
        
        Args:
            name: Namespace name
            type_: user, reversedomain, or global
            
        Returns:
            Created namespace
        """
        response = await self._client.post(
            "/v1/namespaces",
            json={"name": name, "type": type_}
        )
        response.raise_for_status()
        return response.json()
    
    async def get_namespace(self, name: str) -> Dict:
        """Get namespace details.
        
        Args:
            name: Namespace name
            
        Returns:
            Namespace info
        """
        response = await self._client.get(f"/v1/namespaces/{name}")
        response.raise_for_status()
        return response.json()
    
    async def invite_to_namespace(self, name: str) -> str:
        """Generate invite code for namespace.
        
        Args:
            name: Namespace name
            
        Returns:
            Invite code
        """
        response = await self._client.post(f"/v1/namespaces/{name}/invite")
        response.raise_for_status()
        return response.json().get("invite_code")
    
    async def join_namespace(self, invite_code: str) -> Dict:
        """Join namespace via invite code.
        
        Args:
            invite_code: Invite code
            
        Returns:
            Namespace info
        """
        response = await self._client.post(
            "/v1/namespaces/join",
            json={"invite_code": invite_code}
        )
        response.raise_for_status()
        return response.json()


# Convenience functions

def get_namespaces() -> List[str]:
    """Get list of namespace names for autocomplete."""
    state = get_state()
    # TODO: Cache namespaces
    return ["local"]
