"""
The Robutler platform API client (vendored into the SDK 2026-09-25).

This is `robutler/api/client.py` from the `robutlerai/robutler` repository at
`9e6bd4d` (2026-03-05): the client the Python platform skills were written
against. The payment skill verifies, locks, settles and extends locks through
it (`POST /api/payments/verify`, `/lock`, `/settle`, `PATCH
/api/payments/lock/{id}`), the auth skill checks keys with it, and the JSON and
KV storage skills read and write through it. It is the SDK's own code now: a
fix goes here, and nothing keeps it in step with that repository.

WHY IT MOVED. The SDK imported it from the `robutler` package on PyPI, a
declared dependency, and that went wrong three ways:

 1. PyPI only ever received `robutler` 0.2.0, published before `lock`,
    `settle` and `extend_lock` existed. On a pip install the payment skill
    called methods that were not there, so every paid request was refused.
 2. That package is a whole earlier copy of this framework, and it declares a
    `robutler` console script of its own, pointing at a module it does not
    ship. Whichever package an installer wrote last owned `bin/robutler`, so a
    clean install could get a `robutler` command that only printed a
    traceback. Importing it also imported `litellm`, whose import calls
    `load_dotenv()` (the S-216 chain, see `../__init__.py`).
 3. Every install carried that framework, and `litellm` with it, to use this
    one file.

WHAT DIFFERS FROM THE REPOSITORY'S FILE, and nothing else does:

 - With no `base_url`, the client asks the SDK's one platform lookup
   (`../platform_url.py`): `ROBUTLER_API_URL`, `ROBUTLER_INTERNAL_API_URL`,
   the CLI's `platform.url`, then https://robutler.ai. The original tried the
   in-cluster variable first and knew nothing of the CLI.
 - The content methods are replaced by `upload_file` and the agent content
   methods (their section says why): the old ones called routes an API key
   cannot use or that do not exist. Among them, `ContentResource.agent_access()`
   printed the first 20 characters of the API key, all of a shorter one, on
   every call, and `get_content()` would have sent the key to whatever URL a
   listing returned (S-253). The client prints nothing now.
 - `TokensResource.extend_lock` was defined twice and the second definition
   won. The first, which sent `amount` where the platform's `PATCH
   /api/payments/lock/{id}` reads `additionalAmount`, is deleted.
 - `_make_request` sends a POST or PATCH again only when the connection was
   never made. It retried every method after any 5xx or network error, and a
   repeated settle against a lock that is still active is charged again, so a
   lost answer could charge a payer up to four times (S-254).
 - The `settle` docstring says what `release` does on the platform.
 - This header, and the imports.
"""

import os
import re
import asyncio
import aiohttp
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from decimal import Decimal

from .types import (
    User, ApiKey, Integration, CreditTransaction,
    AuthResponse, ApiResponse,
    UserRole, SubscriptionStatus, TransactionType
)
from ..platform_url import resolve_platform_url


#: Methods a repeat cannot change the outcome of (RFC 9110, 9.2.2), so a lost
#: answer is safe to ask for again. A POST or PATCH is not one of them: the
#: platform's settle charges again for a repeat against a lock that is still
#: active (S-254).
_IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE"})

#: The one content route an API key can write to (`POST /api/content/upload`;
#: `POST /api/content` answers 405, F-041).
UPLOAD_PATH = "/api/content/upload"

_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def _uuid(value: Any, what: str) -> str:
    """An agent or content id as the platform issues them, a UUID, so a path
    built from it can only ever name that one thing."""
    text = str(value or "")
    if not _UUID_RE.match(text):
        raise ValueError(f"not a {what}: {value!r}")
    return text


class RobutlerAPIError(Exception):
    """Exception raised when Robutler API returns an error.
    
    This exception provides detailed information about API failures including
    HTTP status codes and response data for debugging.
    
    Attributes:
        message: Human-readable error message
        status_code: HTTP status code from the API response
        response_data: Raw response data from the API
        
    Example:
        ```python
        try:
            user = await client.user.get()
        except RobutlerAPIError as e:
            print(f"API Error: {e} (Status: {e.status_code})")
            print(f"Response: {e.response_data}")
        ```
    """
    def __init__(self, message: str, status_code: int = 400, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}


class Agent:
    """Represents a Robutler AI agent with typed attribute access.
    
    This model provides clean, typed access to agent properties without
    requiring dictionary lookups or .get() calls.
    
    Attributes:
        id: Unique agent identifier
        name: Human-readable agent name
        instructions: System instructions for the agent
        model: AI model used (e.g., 'gpt-4o-mini')
        intents: List of published intent names
        can_use_other_agents: Whether this agent can call other agents
        other_agents_can_talk: Whether other agents can call this agent
        credits_per_token: Cost per token for using this agent
        minimum_balance: Minimum balance required to use this agent
        agent_pricing_percent: Percentage of pricing that goes to agent owner
        is_public: Whether the agent is publicly discoverable
        description: Agent description for discovery
        avatar_url: URL to agent's avatar image
        greeting_message: Welcome message from the agent
        suggested_actions: List of suggested user actions
        greeting_image_mobile: Mobile greeting image URL
        greeting_image_desktop: Desktop greeting image URL
        skills: Agent's configured skills and capabilities
        api_key_encrypted: Encrypted API key for the agent
        
    Example:
        ```python
        agents = await client.agents.list()
        for agent in agents:
            print(f"Agent: {agent.name} using {agent.model}")
            if agent.is_public:
                print(f"Description: {agent.description}")
        ```
    """
    
    def __init__(self, data: Dict[str, Any]):
        # Handle nested agent data structure: {'agent': {...}} or flat {...}
        if 'agent' in data:
            agent_data = data['agent']
            self._raw_data = data  # Keep full response structure
        else:
            agent_data = data
            self._raw_data = data
        
        # Extract all agent properties from the agent_data (robutler uses displayName/username)
        self.id: str = agent_data.get("id", "") or agent_data.get("agentId", "")
        self.name: str = (
            agent_data.get("name")
            or agent_data.get("displayName")
            or agent_data.get("username")
            or ""
        )
        self.instructions: str = agent_data.get("instructions", "")
        self.model: str = agent_data.get("model", "gpt-4o-mini")
        self.intents: List[str] = agent_data.get("intents", [])
        self.can_use_other_agents: bool = agent_data.get("canTalkToOtherAgents", False)
        self.other_agents_can_talk: bool = agent_data.get("otherAgentsCanTalk", False)
        self.credits_per_token: Optional[float] = self._parse_float(agent_data.get("creditsPerToken"))
        self.minimum_balance: Optional[float] = self._parse_float(agent_data.get("minimumBalance"))
        self.agent_pricing_percent: Optional[float] = self._parse_float(agent_data.get("agentPricingPercent"))
        self.is_public: bool = agent_data.get("isPublic", False)
        self.description: str = agent_data.get("description", "")
        self.avatar_url: Optional[str] = agent_data.get("avatarUrl")
        self.greeting_message: Optional[str] = agent_data.get("greetingMessage")
        self.suggested_actions: List[str] = agent_data.get("suggestedActions", [])
        self.greeting_image_mobile: Optional[str] = agent_data.get("greetingImageMobile")
        self.greeting_image_desktop: Optional[str] = agent_data.get("greetingImageDesktop")
        self.skills: Optional[Dict[str, Any]] = agent_data.get("skills")
        self.api_key_encrypted: Optional[str] = agent_data.get("apiKey")  # This is encrypted
    
    def _parse_float(self, value) -> Optional[float]:
        """Parse float value from string or number"""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary format - returns the agent data only"""
        if 'agent' in self._raw_data:
            return self._raw_data['agent']
        return self._raw_data
    
    def get_full_data(self) -> Dict[str, Any]:
        """Get the full raw data structure"""
        return self._raw_data

    @property
    def raw_api_key(self) -> Optional[str]:
        """Raw API key from create response (robutler: only present once at creation)."""
        return self._raw_data.get("rawApiKey") if isinstance(self._raw_data, dict) else None

    def __repr__(self) -> str:
        return f"Agent(id='{self.id}', name='{self.name}', model='{self.model}')"


class UserProfile:
    """User profile model with attributes for clean access"""
    
    def __init__(self, data: Dict[str, Any]):
        self.id: str = data.get("id", "")
        self.name: str = data.get("name", "")
        self.email: str = data.get("email", "")
        self.role: str = data.get("role", "user")
        self.plan_name: str = data.get("planName", "")
        self.total_credits: Decimal = Decimal(str(data.get("totalCredits", "0")))
        self.used_credits: Decimal = Decimal(str(data.get("usedCredits", "0")))
        self.available_credits: Decimal = self.total_credits - self.used_credits
        self.referral_code: str = data.get("referralCode", "")
        self._raw_data = data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary format"""
        return self._raw_data
    
    def __repr__(self) -> str:
        return f"UserProfile(name='{self.name}', email='{self.email}', plan='{self.plan_name}')"


class ApiKeyInfo:
    """API Key info model. Robutler create returns { rawKey, key }; raw_key only present once at creation."""

    def __init__(self, data: Dict[str, Any]):
        key_data = data.get("key", data) if isinstance(data.get("key"), dict) else data
        self.id: str = key_data.get("id", "")
        self.name: str = key_data.get("name", "")
        self.key: str = key_data.get("key", "")
        self.key_prefix: Optional[str] = key_data.get("keyPrefix")
        self.is_active: Optional[bool] = key_data.get("isActive")
        self.expires_at: Optional[str] = key_data.get("expiresAt")
        self.created_at: str = key_data.get("createdAt", key_data.get("created_at", ""))
        self.last_used: Optional[str] = key_data.get("lastUsedAt", key_data.get("lastUsed"))
        self.permissions: Dict[str, Any] = key_data.get("permissions", {})
        self.raw_key: Optional[str] = data.get("rawKey") if data is not key_data else key_data.get("rawKey")
        self._raw_data = data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary format"""
        return self._raw_data
    
    def __repr__(self) -> str:
        return f"ApiKeyInfo(name='{self.name}', id='{self.id}')"


class TransactionInfo:
    """Transaction info model with attributes for clean access"""
    
    def __init__(self, data: Dict[str, Any]):
        self.id: str = data.get("id", "")
        self.type: str = data.get("type", "")
        self.amount: Decimal = Decimal(str(data.get("amount", "0")))
        self.description: str = data.get("description", "")
        self.created_at: str = data.get("createdAt", "")
        self.status: str = data.get("status", "")
        self._raw_data = data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary format"""
        return self._raw_data
    
    def __repr__(self) -> str:
        return f"TransactionInfo(type='{self.type}', amount={self.amount}, status='{self.status}')"


class ChatCompletionResult:
    """Chat completion result model with attributes for clean access"""
    
    def __init__(self, data: Dict[str, Any]):
        self.id: str = data.get("id", "")
        self.choices: List[Dict[str, Any]] = data.get("choices", [])
        self.usage: Dict[str, Any] = data.get("usage", {})
        self.model: str = data.get("model", "")
        self._raw_data = data
    
    @property
    def content(self) -> str:
        """Get the main response content"""
        if self.choices:
            return self.choices[0].get("message", {}).get("content", "")
        return ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary format"""
        return self._raw_data
    
    def __repr__(self) -> str:
        return f"ChatCompletionResult(model='{self.model}', choices={len(self.choices)})"


class AgentsResource:
    """Resource for managing AI agents.
    
    Provides methods for creating, listing, updating, and deleting agents,
    as well as agent discovery and search capabilities.
    
    This resource is accessed via `client.agents` and returns typed Agent
    objects for clean attribute access.
    
    Example:
        ```python
        # List all agents
        agents = await client.agents.list()
        
        # Get specific agent
        agent = await client.agents.get_by_name("my-assistant")
        
        # Create new agent
        new_agent = await client.agents.create({
            "name": "helper",
            "instructions": "You are a helpful assistant",
            "model": "gpt-4o-mini"
        })
        
        # Search for agents
        results = await client.agents.search("data analysis")
        ```
    """
    
    def __init__(self, client):
        self._client = client
    
    async def list(self) -> List[Agent]:
        """List all agents owned by the current user.
        
        Returns:
            List of Agent objects with full agent details
            
        Raises:
            RobutlerAPIError: If the request fails
            
        Example:
            ```python
            agents = await client.agents.list()
            for agent in agents:
                print(f"{agent.name}: {agent.description}")
            ```
        """
        response = await self._client._make_request('GET', '/agents')
        if not response.success:
            raise RobutlerAPIError(f"Failed to list agents: {response.status_code}", response.status_code, response.data)
        
        agents_data = response.data.get("agents", [])
        return [Agent(agent_data) for agent_data in agents_data]
    
    async def get_by_name(self, name: str) -> Agent:
        """Get agent by name (or username) - returns Agent object. Robutler: GET /api/agents/[id] accepts id or username."""
        response = await self._client._make_request('GET', f'/agents/{name}')
        if not response.success:
            raise RobutlerAPIError(f"Failed to get agent by name '{name}': {response.status_code}", response.status_code, response.data)
        return Agent(response.data)
    
    async def get_by_id(self, agent_id: str) -> Agent:
        """Get agent by ID - returns Agent object"""
        response = await self._client._make_request('GET', f'/agents/{agent_id}')
        if not response.success:
            raise RobutlerAPIError(f"Failed to get agent by ID '{agent_id}': {response.status_code}", response.status_code, response.data)
        return Agent(response.data)
    
    async def get(self, agent_id: str) -> 'AgentResource':
        """Get agent resource by ID"""
        return AgentResource(self._client, agent_id)
    
    async def create(self, agent_data: Dict[str, Any]) -> Agent:
        """Create a new AI agent.
        
        Args:
            agent_data: Agent configuration dictionary containing:
                - name (str): Agent name (required)
                - instructions (str): System instructions (required)
                - model (str): AI model to use (default: "gpt-4o-mini")
                - description (str): Agent description for discovery
                - intents (List[str]): Published intent names
                - isPublic (bool): Whether agent is publicly discoverable
                - canTalkToOtherAgents (bool): Can this agent call others
                - otherAgentsCanTalk (bool): Can others call this agent
                
        Returns:
            Created Agent object
            
        Raises:
            RobutlerAPIError: If creation fails
            
        Example:
            ```python
            agent = await client.agents.create({
                "name": "data-analyst",
                "instructions": "You analyze data and create reports",
                "model": "gpt-4o",
                "description": "Specialized in data analysis",
                "isPublic": True
            })
            print(f"Created agent: {agent.id}")
            ```
        """
        response = await self._client._make_request('POST', '/agents', data=agent_data)
        if not response.success:
            raise RobutlerAPIError(f"Failed to create agent: {response.status_code}", response.status_code, response.data)
        return Agent(response.data)
    
    async def update(self, agent_id: str, agent_data: Dict[str, Any]) -> Agent:
        """Update an existing agent - returns updated Agent object. Robutler uses PATCH."""
        response = await self._client._make_request('PATCH', f'/agents/{agent_id}', data=agent_data)
        if not response.success:
            raise RobutlerAPIError(f"Failed to update agent {agent_id}: {response.status_code}", response.status_code, response.data)
        return Agent(response.data)
    
    async def delete(self, agent_id: str) -> bool:
        """Delete an agent - returns True if successful"""
        response = await self._client._make_request('DELETE', f'/agents/{agent_id}')
        if not response.success:
            raise RobutlerAPIError(f"Failed to delete agent {agent_id}: {response.status_code}", response.status_code, response.data)
        return True
    
    async def search(self, query: str, max_results: int = 10, mode: str = 'semantic', min_similarity: float = 0.7) -> List[Dict[str, Any]]:
        """Search for agents using semantic search.
        
        Searches across agent names, descriptions, and intents to find
        relevant agents based on the query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return (default: 10)
            mode: Search mode, currently 'semantic' (default: 'semantic')
            min_similarity: Minimum similarity threshold (default: 0.7)
            
        Returns:
            List of agent search result dictionaries
            
        Raises:
            RobutlerAPIError: If search fails
            
        Example:
            ```python
            # Find agents that can help with data analysis
            results = await client.agents.search("data analysis", max_results=5)
            for result in results:
                agent_data = result.get('agent', {})
                print(f"Found: {agent_data.get('name')} - {agent_data.get('description')}")
            ```
        """
        search_data = {
            'query': query,
            'fields': ['name', 'description', 'intents']
        }
        
        response = await self._client._make_request('POST', '/agents/search', data=search_data)
        if not response.success:
            raise RobutlerAPIError(f"Failed to search agents: {response.message}", response.status_code, response.data)
        
        return response.data.get('agents', [])
    
    async def discover(self, capabilities: List[str], max_results: int = 10) -> List[Dict[str, Any]]:
        """Discover agents by capabilities - returns list of agent results"""
        discovery_params = {
            'capabilities': capabilities,
            'limit': max_results
        }
        
        response = await self._client._make_request('GET', '/agents/discover', params=discovery_params)
        if not response.success:
            raise RobutlerAPIError(f"Failed to discover agents: {response.message}", response.status_code, response.data)
        
        return response.data.get('agents', [])
    
    async def find_similar(self, agent_id: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Find similar agents - returns list of agent results"""
        response = await self._client._make_request('GET', f'/agents/{agent_id}/similar', params={'limit': max_results})
        if not response.success:
            raise RobutlerAPIError(f"Failed to find similar agents: {response.message}", response.status_code, response.data)
        
        return response.data.get('agents', [])


class AgentResource:
    """Individual agent resource"""
    
    def __init__(self, client, agent_id: str):
        self._client = client
        self.agent_id = agent_id
    
    async def get(self) -> Agent:
        """Get agent details - returns Agent object"""
        response = await self._client._make_request('GET', f'/agents/{self.agent_id}')
        if not response.success:
            raise RobutlerAPIError(f"Failed to get agent {self.agent_id}: {response.status_code}", response.status_code, response.data)
        return Agent(response.data)
    
    async def api_key(self) -> Optional[str]:
        """Get API key for this agent. Legacy: returns raw key if present. Robutler GET returns key metadata only (no raw key). Use regenerate_api_key() to obtain a new raw key."""
        response = await self._client._make_request('GET', f'/agents/{self.agent_id}/api-key')
        if not response.success:
            raise RobutlerAPIError(f"Failed to get API key for agent {self.agent_id}: {response.status_code}", response.status_code, response.data)
        api_key = response.data.get("apiKey")
        if api_key:
            return api_key
        # Robutler: GET returns { key: { id, name, keyPrefix, ... } } without raw key
        return None

    async def regenerate_api_key(self, name: Optional[str] = None) -> str:
        """Regenerate API key for this agent (e.g. Robutler POST /api/agents/[id]/api-key). Returns the new raw key (only shown once)."""
        data = {} if name is None else {"name": name}
        response = await self._client._make_request('POST', f'/agents/{self.agent_id}/api-key', data=data)
        if not response.success:
            raise RobutlerAPIError(f"Failed to regenerate API key for agent {self.agent_id}: {response.status_code}", response.status_code, response.data)
        raw_key = response.data.get("rawKey")
        if not raw_key:
            raise RobutlerAPIError(f"No raw key in response for agent {self.agent_id}")
        return raw_key
    
    async def chat_completion(self, data: Dict[str, Any]) -> ChatCompletionResult:
        """Send chat completion request to this agent - returns ChatCompletionResult"""
        response = await self._client._make_request('POST', f'/agents/{self.agent_id}/chat/completions', data=data)
        if not response.success:
            raise RobutlerAPIError(f"Chat completion failed for agent {self.agent_id}: {response.status_code}", response.status_code, response.data)
        return ChatCompletionResult(response.data)
    
    async def get_intents(self) -> List[Dict[str, Any]]:
        """Get published intents for this agent - returns list of intent objects"""
        response = await self._client._make_request('GET', f'/agents/{self.agent_id}/intents')
        if not response.success:
            raise RobutlerAPIError(f"Failed to get intents for agent {self.agent_id}: {response.message}", response.status_code, response.data)
        
        return response.data.get('intents', [])


class IntentsResource:
    """Intents resource for hierarchical API access"""
    
    def __init__(self, client):
        self._client = client
    
    async def publish(self, intents_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Publish intents - returns list of publish results"""
        data = {'intents': intents_data}
        response = await self._client._make_request('POST', '/intents/publish', data=data)
        if not response.success:
            raise RobutlerAPIError(f"Failed to publish intents: {response.message}", response.status_code, response.data)
        
        return response.data.get('results', [])


class UserResource:
    """User resource for hierarchical API access"""
    
    def __init__(self, client):
        self._client = client
    
    async def get(self) -> UserProfile:
        """Get current user profile - returns UserProfile object"""
        response = await self._client._make_request('GET', '/user')
        if not response.success:
            raise RobutlerAPIError(f"Failed to get user: {response.status_code}", response.status_code, response.data)
        # API returns shape { user: { ... } }
        user_data = response.data.get('user', response.data or {})
        return UserProfile(user_data)
    
    async def credits(self) -> Decimal:
        """Get user's available credits - returns Decimal. Uses GET /api/balance (robutler) or GET /api/user/credits (legacy)."""
        # Robutler: GET /api/balance returns availableBalance (nanocents), availableDollars
        response = await self._client._make_request('GET', '/balance')
        if response.success and isinstance(response.data, dict):
            if 'availableDollars' in response.data:
                try:
                    return Decimal(str(response.data['availableDollars']))
                except Exception:
                    pass
            if 'availableBalance' in response.data:
                try:
                    return Decimal(str(response.data['availableBalance'])) / Decimal('1e9')
                except Exception:
                    pass
        # Legacy: GET /api/user/credits
        response = await self._client._make_request('GET', '/user/credits')
        if not response.success:
            raise RobutlerAPIError(f"Failed to get credits: {response.status_code}", response.status_code, response.data)
        if isinstance(response.data, dict) and 'availableCredits' in response.data:
            try:
                return Decimal(str(response.data.get('availableCredits', '0')))
            except Exception:
                pass
        total_credits = Decimal(str(response.data.get('totalCredits', '0'))) if isinstance(response.data, dict) else Decimal('0')
        used_credits = Decimal(str(response.data.get('usedCredits', '0'))) if isinstance(response.data, dict) else Decimal('0')
        return total_credits - used_credits
    
    async def transactions(self, limit: int = 50, offset: int = 0, type_filter: Optional[str] = None) -> List[TransactionInfo]:
        """Get user's transaction history. Uses GET /api/balance/transactions (robutler) or GET /api/user/transactions (legacy)."""
        params = {'limit': limit, 'offset': offset}
        if type_filter:
            params['type'] = type_filter
        response = await self._client._make_request('GET', '/balance/transactions', params=params)
        if not response.success:
            response = await self._client._make_request('GET', f'/user/transactions', params={'limit': limit})
        if not response.success:
            raise RobutlerAPIError(f"Failed to get transactions: {response.status_code}", response.status_code, response.data)
        transactions_data = response.data.get('transactions', [])
        return [TransactionInfo(transaction_data) for transaction_data in transactions_data]


class ApiKeysResource:
    """API Keys resource for hierarchical API access"""
    
    def __init__(self, client):
        self._client = client
    
    async def list(self) -> List[ApiKeyInfo]:
        """List user's API keys - returns list of ApiKeyInfo objects"""
        response = await self._client._make_request('GET', '/api-keys')
        if not response.success:
            raise RobutlerAPIError(f"Failed to list API keys: {response.status_code}", response.status_code, response.data)
        
        keys_data = response.data.get('keys', [])
        return [ApiKeyInfo(key_data) for key_data in keys_data]
    
    async def create(self, name: str, permissions: Optional[Dict[str, Any]] = None) -> ApiKeyInfo:
        """Create new API key. Robutler returns { rawKey, key }; raw_key only in response once."""
        payload = {'name': name}
        if permissions:
            payload['permissions'] = permissions
        response = await self._client._make_request('POST', '/api-keys', data=payload)
        if not response.success:
            raise RobutlerAPIError(f"Failed to create API key: {response.status_code}", response.status_code, response.data)
        return ApiKeyInfo(response.data)
    
    async def delete(self, key_id: str) -> bool:
        """Delete API key - returns True if successful"""
        response = await self._client._make_request('DELETE', f'/api-keys/{key_id}')
        if not response.success:
            raise RobutlerAPIError(f"Failed to delete API key {key_id}: {response.status_code}", response.status_code, response.data)
        return True


class TokensResource:
    """Payment tokens resource. Uses POST /api/payments/verify, /lock, and /settle (robutler).

    Recommended flow:
        1. ``verify()``  – validate token and check balance
        2. ``lock()``    – reserve a budget from the token
        3. ``settle()``  – charge actual usage against the lock
    """

    def __init__(self, client):
        self._client = client

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _verify_request(self, token: str, expected_audience: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {'token': token}
        if expected_audience is not None and (expected_audience if isinstance(expected_audience, list) else [expected_audience]):
            body['expectedAudience'] = expected_audience
        return body

    @staticmethod
    def _parse_dollars(data: Dict[str, Any], dollar_key: str = 'balanceDollars', nano_key: str = 'balance') -> float:
        """Extract a dollar amount from a Robutler response that may contain both nanocents and dollar fields."""
        amount = data.get(dollar_key)
        if amount is not None:
            try:
                return float(amount)
            except (TypeError, ValueError):
                pass
        raw = data.get(nano_key, data.get('availableAmount', 0))
        try:
            v = float(raw)
            return v / 1e9 if v > 1e6 else v
        except (TypeError, ValueError):
            return 0.0

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------

    async def validate(self, token: str, expected_audience: Optional[Union[str, List[str]]] = None) -> bool:
        """Validate payment token (POST /api/payments/verify). Returns True if valid."""
        body = self._verify_request(token, expected_audience)
        response = await self._client._make_request('POST', '/payments/verify', data=body)
        if not response.success:
            return False
        return response.data.get('valid', False)

    async def get_balance(self, token: str, expected_audience: Optional[Union[str, List[str]]] = None) -> float:
        """Get payment token balance in dollars."""
        body = self._verify_request(token, expected_audience)
        response = await self._client._make_request('POST', '/payments/verify', data=body)
        if not response.success:
            raise RobutlerAPIError(
                response.message or "Failed to get token balance",
                response.status_code,
                response.data,
            )
        return self._parse_dollars(response.data)

    async def validate_with_balance(
        self, token: str, expected_audience: Optional[Union[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """Validate token and get balance. Returns dict with valid, balance (dollars), and optional error."""
        body = self._verify_request(token, expected_audience)
        response = await self._client._make_request('POST', '/payments/verify', data=body)
        if not response.success:
            return {'valid': False, 'error': response.message or 'Validation failed', 'balance': 0.0}
        valid = response.data.get('valid', False)
        balance = self._parse_dollars(response.data)
        return {'valid': valid, 'balance': balance}

    # ------------------------------------------------------------------
    # Lock
    # ------------------------------------------------------------------

    async def lock(self, token: str, amount: Union[str, float]) -> Dict[str, Any]:
        """Lock a budget from a payment token (POST /api/payments/lock).

        Args:
            token: Payment token JWT or token ID.
            amount: Amount in dollars to lock.

        Returns:
            dict with ``lockId``, ``lockedAmountDollars``.

        Raises:
            RobutlerAPIError on failure.
        """
        data: Dict[str, Any] = {
            'token': token,
            'amount': float(amount) if isinstance(amount, str) else amount,
        }
        response = await self._client._make_request('POST', '/payments/lock', data=data)
        if not response.success:
            raise RobutlerAPIError(
                response.message or "Failed to lock payment budget",
                response.status_code,
                response.data,
            )
        return {
            'lockId': response.data.get('lockId'),
            'lockedAmountDollars': self._parse_dollars(
                response.data, dollar_key='lockedAmountDollars', nano_key='lockedAmount'
            ),
        }

    # ------------------------------------------------------------------
    # Settle
    # ------------------------------------------------------------------

    async def settle(
        self,
        lock_id: str,
        amount: Optional[Union[str, float]] = None,
        description: Optional[str] = None,
        resource: Optional[str] = None,
        charge_type: Optional[str] = None,
        cid: Optional[str] = None,
        release: bool = False,
        usage: Optional[list] = None,
        provider_key_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Settle (charge) actual usage against a payment lock (POST /api/payments/settle).

        Cost can be specified two ways:
          - ``amount``: pre-computed dollar amount (backward compat, flat-rate tool charges)
          - ``usage``: raw usage records -- Robutler computes cost server-side from MODEL_PRICING

        Args:
            lock_id: Lock ID returned by ``lock()``.
            amount: Actual amount in dollars to charge (0 with release=True to just release).
            description: Human-readable description.
            resource: Optional resource identifier.
            charge_type: One of 'platform_fee', 'platform_llm', 'agent_fee', 'byok_llm', 'byok_mcp_sampling'.
            cid: Optional IPFS content hash for receipt.
            release: Release the lock. The platform honours it only when
                ``amount`` is 0 (a release-only call); with a charge the lock
                stays active while it holds more than the charge.
            usage: Raw usage records for server-side cost computation.
            provider_key_id: UUID of the user's BYOK provider key (required for byok_llm).

        Returns:
            dict with ``success``, ``chargedDollars``, ``remainingDollars``,
            and optionally ``computedFromUsage``.

        Raises:
            RobutlerAPIError on failure.
        """
        data: Dict[str, Any] = {
            'lockId': lock_id,
        }
        if amount is not None:
            data['amount'] = float(amount) if isinstance(amount, str) else amount
        if usage is not None:
            data['usage'] = usage
        if description is not None:
            data['description'] = description
        if resource is not None:
            data['resource'] = resource
        if charge_type is not None:
            data['chargeType'] = charge_type
        if cid is not None:
            data['cid'] = cid
        if release:
            data['release'] = True
        if provider_key_id is not None:
            data['providerKeyId'] = provider_key_id
        response = await self._client._make_request('POST', '/payments/settle', data=data)
        if not response.success:
            raise RobutlerAPIError(
                response.message or "Failed to settle payment",
                response.status_code,
                response.data,
            )
        result = {
            'success': response.data.get('success', True),
            'chargedDollars': self._parse_dollars(
                response.data, dollar_key='chargedDollars', nano_key='charged'
            ),
            'remainingDollars': self._parse_dollars(
                response.data, dollar_key='remainingDollars', nano_key='remaining'
            ),
        }
        if response.data.get('computedFromUsage'):
            result['computedFromUsage'] = True
        return result

    # ------------------------------------------------------------------
    # Extend Lock
    # ------------------------------------------------------------------

    async def extend_lock(
        self,
        lock_id: str,
        additional_amount: Union[str, float],
    ) -> Dict[str, Any]:
        """Extend an existing lock's amount (PATCH /api/payments/lock/:id).

        Args:
            lock_id: Lock ID to extend.
            additional_amount: Additional amount in dollars.

        Returns:
            dict with ``success``, ``newAmountDollars``.
        """
        data: Dict[str, Any] = {
            'additionalAmount': float(additional_amount) if isinstance(additional_amount, str) else additional_amount,
        }
        response = await self._client._make_request('PATCH', f'/payments/lock/{lock_id}', data=data)
        if not response.success:
            raise RobutlerAPIError(
                response.message or "Failed to extend lock",
                response.status_code,
                response.data,
            )
        return {
            'success': response.data.get('success', True),
            'newAmountDollars': self._parse_dollars(
                response.data, dollar_key='newAmountDollars', nano_key='newAmount'
            ),
        }

    # ------------------------------------------------------------------
    # Legacy / convenience
    # ------------------------------------------------------------------

    async def redeem(
        self,
        token: str,
        amount: Union[str, float],
        recipient_id: Optional[str] = None,
        api_key_id: Optional[str] = None,
        description: Optional[str] = None,
        resource: Optional[str] = None,
    ) -> bool:
        """Legacy: settle/charge payment token directly (no lock). Prefer lock() + settle()."""
        data: Dict[str, Any] = {'token': token, 'amount': float(amount) if isinstance(amount, str) else amount}
        if recipient_id:
            data['recipientId'] = recipient_id
        if api_key_id:
            data['apiKeyId'] = api_key_id
        if description is not None:
            data['description'] = description
        if resource is not None:
            data['resource'] = resource
        response = await self._client._make_request('POST', '/payments/settle', data=data)
        if not response.success:
            raise RobutlerAPIError(response.message or "Failed to redeem token", response.status_code, response.data)
        return response.data.get('success', response.success)


class RobutlerClient:
    """Main API client for the Robutler Platform.
    
    Provides hierarchical access to all Robutler Platform services through
    intuitive resource objects. All methods return typed model objects for
    clean, IDE-friendly development.
    
    The client supports automatic retry logic, connection pooling, and
    comprehensive error handling.
    
    Attributes:
        agents: Agent management operations (AgentsResource)
        user: User profile and credit operations (UserResource)
        api_keys: API key management operations (ApiKeysResource)
        tokens: Payment token operations (TokensResource)
        intents: Intent publishing operations (IntentsResource)
        
    Environment Variables:
        WEBAGENTS_API_KEY: Your Robutler API key (required)
        ROBUTLER_API_URL: Base URL for Robutler API (optional)
        ROBUTLER_INTERNAL_API_URL: Internal cluster URL (optional)
        
    Example:
        ```python
        # Basic usage
        async with RobutlerClient() as client:
            # Get user information
            user = await client.user.get()
            print(f"Welcome {user.name}!")
            
            # List agents
            agents = await client.agents.list()
            for agent in agents:
                print(f"Agent: {agent.name}")
        ```
        
    Raises:
        RobutlerAPIError: When API requests fail
        ValueError: When required configuration is missing
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 timeout: int = 30,
                 max_retries: int = 3):
        """Initialize the Robutler API client.
        
        Args:
            api_key: Your Robutler API key. If not provided, will use
                WEBAGENTS_API_KEY environment variable.
            base_url: Base URL for the Robutler API. If not provided, the
                SDK's platform lookup decides (`../platform_url.py`):
                ROBUTLER_API_URL, then ROBUTLER_INTERNAL_API_URL, then the
                CLI's platform.url, then https://robutler.ai
            timeout: Request timeout in seconds (default: 30)
            max_retries: Maximum number of retries for failed requests (default: 3)
            
        Raises:
            ValueError: If no API key is provided via parameter or environment
            
        Example:
            ```python
            # Using environment variables (recommended)
            client = RobutlerClient()
            
            # Explicit configuration
            client = RobutlerClient(
                api_key="your-api-key",
                base_url="https://api.robutler.ai",
                timeout=60
            )
            ```
        """
        self.api_key = api_key or os.getenv('WEBAGENTS_API_KEY')
        # The SDK's one platform lookup when the caller names no URL (header).
        resolved_base = base_url or resolve_platform_url()
        self.base_url = resolved_base.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError("Robutler API key is required. Set WEBAGENTS_API_KEY environment variable or provide api_key parameter.")
        
        # Session for connection pooling
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Initialize hierarchical resources
        self.agents = AgentsResource(self)
        self.user = UserResource(self)
        self.api_keys = ApiKeysResource(self)
        self.tokens = TokensResource(self)
        self.intents = IntentsResource(self)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self):
        """Close the HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    def _get_headers(self, additional_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Get request headers with authentication"""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'X-API-Key': self.api_key,
            'User-Agent': 'Robutler-V2-Client/1.0'
        }
        
        if additional_headers:
            headers.update(additional_headers)
        
        return headers
    
    async def _make_request(self, 
                           method: str, 
                           endpoint: str, 
                           data: Optional[Dict[str, Any]] = None,
                           params: Optional[Dict[str, Any]] = None,
                           headers: Optional[Dict[str, str]] = None) -> ApiResponse:
        """Make authenticated HTTP request to the Robutler API.
        
        Handles authentication, retries with exponential backoff, and
        comprehensive error handling. A POST or PATCH is sent again only when
        the connection was never made, so it cannot have arrived; never after a
        5xx or a dropped connection, which may follow a commit (S-254). The
        TypeScript SDK does not retry at all. Automatically parses JSON responses
        and provides detailed error information.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path (e.g., '/agents')
            data: Request body data (will be JSON-encoded)
            params: URL query parameters
            headers: Additional HTTP headers
            
        Returns:
            ApiResponse object with success status, data, and error details
            
        Note:
            This is an internal method. Use the resource methods instead:
            - client.agents.list() instead of client._make_request('GET', '/agents')
            - client.user.get() instead of client._make_request('GET', '/user')
        """
        url = f"{self.base_url}/api{endpoint}"
        request_headers = self._get_headers(headers)

        session = await self._get_session()
        idempotent = method.upper() in _IDEMPOTENT_METHODS

        for attempt in range(self.max_retries + 1):
            try:
                async with session.request(
                    method=method,
                    url=url,
                    json=data if data else None,
                    params=params,
                    headers=request_headers
                ) as response:
                    
                    # Get response text
                    response_text = await response.text()
                    
                    # Try to parse as JSON
                    try:
                        response_data = json.loads(response_text) if response_text else {}
                    except json.JSONDecodeError:
                        response_data = {'message': response_text}
                    
                    # Handle different status codes
                    if response.status == 200:
                        return ApiResponse(
                            success=True,
                            data=response_data,
                            status_code=response.status
                        )
                    elif response.status == 401:
                        return ApiResponse(
                            success=False,
                            error='Authentication failed',
                            message=response_data.get('message', 'Invalid API key or token'),
                            status_code=response.status
                        )
                    elif response.status == 403:
                        return ApiResponse(
                            success=False,
                            error='Authorization failed',
                            message=response_data.get('message', 'Insufficient permissions'),
                            status_code=response.status
                        )
                    elif response.status == 404:
                        return ApiResponse(
                            success=False,
                            error='Not found',
                            message=response_data.get('message', 'Resource not found'),
                            status_code=response.status
                        )
                    elif response.status >= 500:
                        # Server error: retry what a repeat cannot change (S-254)
                        if idempotent and attempt < self.max_retries:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        
                        return ApiResponse(
                            success=False,
                            error='Server error',
                            message=response_data.get('message') or response_data.get('error') or 'Internal server error',
                            status_code=response.status,
                            data=response_data,
                        )
                    else:
                        msg = (
                            response_data.get('message')
                            or response_data.get('error')
                            or f'HTTP {response.status}'
                        )
                        return ApiResponse(
                            success=False,
                            error='Request failed',
                            message=msg,
                            status_code=response.status,
                            data=response_data,
                        )
                        
            except aiohttp.ClientError as e:
                # A connection that was never made carried nothing, so even a
                # POST is safe to send again; any other failure may follow a
                # commit (S-254).
                never_sent = isinstance(e, aiohttp.ClientConnectorError)
                if attempt < self.max_retries and (idempotent or never_sent):
                    await asyncio.sleep(2 ** attempt)
                    continue
                
                return ApiResponse(
                    success=False,
                    error='Network error',
                    message=str(e),
                    status_code=0
                )
            except Exception as e:
                return ApiResponse(
                    success=False,
                    error='Unexpected error',
                    message=str(e),
                    status_code=0
                )
        
        return ApiResponse(
            success=False,
            error='Max retries exceeded',
            message=f'Failed after {self.max_retries + 1} attempts',
            status_code=0
        )
    
    # ===== USER MANAGEMENT METHODS =====
    
    async def get_user(self) -> AuthResponse:
        """Get current user information"""
        try:
            response = await self._make_request('GET', '/user')
            
            if not response.success:
                return AuthResponse(
                    success=False,
                    error=response.error,
                    message=response.message
                )
            
            user_data = response.data.get('user', {}) if response.data else {}
            
            # Convert to User object
            user = User(
                id=user_data.get('id', ''),
                name=user_data.get('name'),
                email=user_data.get('email', ''),
                role=UserRole(user_data.get('role', 'user')),
                google_id=user_data.get('googleId'),
                avatar_url=user_data.get('avatarUrl'),
                stripe_customer_id=user_data.get('stripeCustomerId'),
                stripe_subscription_id=user_data.get('stripeSubscriptionId'),
                plan_name=user_data.get('planName'),
                total_credits=Decimal(user_data.get('totalCredits', '0')),
                used_credits=Decimal(user_data.get('usedCredits', '0')),
                referral_code=user_data.get('referralCode'),
                referred_by=user_data.get('referredBy'),
                referral_count=user_data.get('referralCount', 0)
            )
            
            return AuthResponse(success=True, user=user)
            
        except Exception as e:
            return AuthResponse(
                success=False,
                error='Failed to get user',
                message=str(e)
            )
    
    async def validate_api_key(self, api_key: str) -> AuthResponse:
        """Validate API key and get associated user"""
        try:
            # Create temporary client with the API key to test
            temp_headers = {
                'Authorization': f'Bearer {api_key}',
                'X-API-Key': api_key
            }
            
            response = await self._make_request('GET', '/user', headers=temp_headers)
            
            if not response.success:
                return AuthResponse(
                    success=False,
                    error='Invalid API key',
                    message=response.message
                )
            
            user_data = response.data.get('user', {}) if response.data else {}
            
            # Convert to User object  
            user = User(
                id=user_data.get('id', ''),
                name=user_data.get('name'),
                email=user_data.get('email', ''),
                role=UserRole(user_data.get('role', 'user'))
            )
            
            return AuthResponse(success=True, user=user)
            
        except Exception as e:
            return AuthResponse(
                success=False,
                error='API key validation failed',
                message=str(e)
            )
    
    async def get_user_credits(self) -> ApiResponse:
        """Get user credit information"""
        return await self._make_request('GET', '/user/credits')
    
    async def get_user_transactions(self, limit: int = 50, offset: int = 0) -> ApiResponse:
        """Get user transaction history"""
        params = {'limit': limit, 'offset': offset}
        return await self._make_request('GET', '/user/transactions', params=params)
    
    # ===== API KEY MANAGEMENT METHODS =====
    
    async def list_api_keys(self) -> ApiResponse:
        """List user's API keys"""
        return await self._make_request('GET', '/api-keys')
    
    async def create_api_key(self, name: str, permissions: Optional[Dict[str, Any]] = None) -> ApiResponse:
        """Create a new API key"""
        data = {
            'name': name,
            'permissions': permissions or {}
        }
        return await self._make_request('POST', '/api-keys', data=data)
    
    async def delete_api_key(self, api_key_id: str) -> ApiResponse:
        """Delete an API key"""
        return await self._make_request('DELETE', f'/api-keys/{api_key_id}')
    
    # ===== INTEGRATION METHODS =====
    
    async def list_integrations(self) -> ApiResponse:
        """List user's integrations"""
        return await self._make_request('GET', '/user/integrations')
    
    async def create_integration(self, 
                               name: str, 
                               integration_type: str = "api",
                               protocol: str = "http") -> ApiResponse:
        """Create a new integration"""
        data = {
            'name': name,
            'type': integration_type,
            'protocol': protocol
        }
        return await self._make_request('POST', '/user/integrations', data=data)
    
    # ===== CREDIT/PAYMENT METHODS =====
    
    async def track_usage(self, 
                         amount: Union[str, Decimal, float],
                         description: str = "API usage",
                         source: str = "api_usage",
                         integration_id: Optional[str] = None) -> ApiResponse:
        """Track credit usage"""
        data = {
            'amount': str(amount),
            'type': 'usage',
            'description': description,
            'source': source
        }
        if integration_id:
            data['integration_id'] = integration_id
            
        return await self._make_request('POST', '/user/transactions', data=data)
    
    # ===== HEALTH/STATUS METHODS =====
    
    async def health_check(self) -> ApiResponse:
        """Check API health status"""
        return await self._make_request('GET', '/health')
    
    async def get_config(self) -> ApiResponse:
        """Get API configuration"""
        return await self._make_request('GET', '/config')
    
    # ===== UTILITY METHODS =====
    
    def _parse_user_data(self, user_data: Dict[str, Any]) -> User:
        """Parse user data from API response into User object"""
        return User(
            id=user_data.get('id', ''),
            name=user_data.get('name'),
            email=user_data.get('email', ''),
            role=UserRole(user_data.get('role', 'user')),
            google_id=user_data.get('googleId'),
            avatar_url=user_data.get('avatarUrl'),
            created_at=self._parse_datetime(user_data.get('createdAt')),
            updated_at=self._parse_datetime(user_data.get('updatedAt')),
            stripe_customer_id=user_data.get('stripeCustomerId'),
            stripe_subscription_id=user_data.get('stripeSubscriptionId'),
            stripe_product_id=user_data.get('stripeProductId'),
            plan_name=user_data.get('planName'),
            total_credits=Decimal(user_data.get('totalCredits', '0')),
            used_credits=Decimal(user_data.get('usedCredits', '0')),
            referral_code=user_data.get('referralCode'),
            referred_by=user_data.get('referredBy'),
            referral_count=user_data.get('referralCount', 0)
        )
    
    def _parse_datetime(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse datetime string from API response"""
        if not date_str:
            return None
        
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except ValueError:
            return None
    
    # ===== AGENT MANAGEMENT METHODS =====
    
    # Legacy methods for backward compatibility - these wrap the new hierarchical methods
    async def list_agents(self) -> List[Agent]:
        """List user's agents (legacy - use client.agents.list())"""
        return await self.agents.list()
    
    async def get_agent_api_key(self, agent_id: str) -> str:
        """Get API key for an agent (legacy - use client.agents.get(id).api_key())"""
        agent_resource = await self.agents.get(agent_id)
        return await agent_resource.api_key()
    
    # ===== AGENT CONTENT (FILES) =====
    #
    # AN API KEY REACHES CONTENT THROUGH THE AGENT ROUTES (2026-09-25). Under
    # `/api/content` only the upload takes a key; listing, reading and deleting
    # there take a browser session, and the `url` a file comes back with is
    # served only to a session, a share token or a link the platform signs.
    # The methods these replace (`list_content`, `upload_content`,
    # `get_content`, `delete_content`, `update_content` and the `content`
    # resource) called those routes or routes that do not exist, and
    # `get_content` would have sent the key to whatever `url` came back
    # (S-253). Every call here is this client's base URL plus a fixed path,
    # with the ids checked, so the key goes nowhere else.

    async def upload_file(
        self,
        filename: str,
        data: bytes,
        content_type: str = 'application/octet-stream',
        visibility: str = 'private',
    ) -> ApiResponse:
        """Upload one file (`POST /api/content/upload`) under its full name.

        With the agent's own key the file is the agent's, also linked into its
        owner's library. Answers `{id, url, displayName, contentType, size,
        mimeType}`. Not retried: it is a POST (S-254).
        """
        form = aiohttp.FormData()
        form.add_field('file', data, filename=filename, content_type=content_type)
        form.add_field('visibility', visibility)
        # The platform drops the extension from the name unless it is given.
        form.add_field('displayName', filename)
        headers = {'Authorization': f'Bearer {self.api_key}', 'User-Agent': 'Robutler-V2-Client/1.0'}
        session = await self._get_session()
        try:
            async with session.post(f"{self.base_url}{UPLOAD_PATH}", data=form, headers=headers) as response:
                text = await response.text()
                try:
                    body = json.loads(text) if text else {}
                except json.JSONDecodeError:
                    body = {'message': text}
                body = body if isinstance(body, dict) else {}
                if response.status == 200:
                    return ApiResponse(success=True, data=body, status_code=200)
                return ApiResponse(
                    success=False,
                    error='Upload failed',
                    message=body.get('error') or body.get('message') or f'HTTP {response.status}',
                    status_code=response.status,
                    data=body,
                )
        except aiohttp.ClientError as e:
            return ApiResponse(success=False, error='Network error', message=str(e), status_code=0)

    async def list_agent_content(
        self,
        agent_id: str,
        query: Optional[str] = None,
        limit: int = 100,
        page: int = 1,
    ) -> ApiResponse:
        """The files an agent holds (`GET /api/agents/{agent_id}/content`):
        `{items: [{id, displayName, contentType, mimeType, size, visibility,
        url, createdAt}], page, limit}`. `query` is the platform's `q`."""
        params: Dict[str, Any] = {'limit': limit, 'page': page}
        if query:
            params['q'] = query
        return await self._make_request('GET', f'/agents/{_uuid(agent_id, "agent id")}/content', params=params)

    async def read_agent_content(self, agent_id: str, content_id: str, format: str = 'text') -> ApiResponse:
        """One of the agent's files (`GET /api/agents/{agent_id}/content/{id}`):
        its metadata, plus `text` with `format='text'` or `base64` with
        `format='base64'`."""
        path = f'/agents/{_uuid(agent_id, "agent id")}/content/{_uuid(content_id, "content id")}'
        return await self._make_request('GET', path, params={'format': format})

    async def delete_agent_content(self, agent_id: str, content_id: str) -> ApiResponse:
        """Take the agent's link off a file (`DELETE
        /api/agents/{agent_id}/content/{id}`). The platform deletes the file
        when no link is left; one the agent uploaded is also linked into its
        owner's library and stays there."""
        path = f'/agents/{_uuid(agent_id, "agent id")}/content/{_uuid(content_id, "content id")}'
        return await self._make_request('DELETE', path)
    
    def __repr__(self) -> str:
        """String representation of the client"""
        return f"RobutlerClient(base_url='{self.base_url}', api_key='***{self.api_key[-4:] if self.api_key else None}')"


# ===== CONVENIENCE FUNCTIONS =====

async def create_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> RobutlerClient:
    """Create and return a Robutler API client"""
    return RobutlerClient(api_key=api_key, base_url=base_url)


async def validate_api_key(api_key: str, base_url: Optional[str] = None) -> AuthResponse:
    """Validate API key using a temporary client"""
    async with RobutlerClient(api_key=api_key, base_url=base_url) as client:
        return await client.get_user() 