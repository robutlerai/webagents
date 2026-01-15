"""
Namespace Management

CRUD operations for namespaces.
"""

from typing import Optional, List, Dict
from dataclasses import dataclass
from enum import Enum


class NamespaceType(str, Enum):
    """Namespace types."""
    GLOBAL = "global"  # ai.global.<name> - requires platform approval
    REVERSEDOMAIN = "reversedomain"  # <domain>.<name> - requires domain verification
    USER = "user"  # @<username>.<name> - user-owned
    LOCAL = "local"  # local.<name> - local only


class AuthMethod(str, Enum):
    """Namespace authentication methods."""
    SECRET = "secret"  # Shared secret
    DOMAINCERT = "domaincert"  # Domain certificate verification
    TOKEN = "token"  # Access token
    INVITE = "invite"  # Invite code


@dataclass
class Namespace:
    """A namespace."""
    name: str
    type: NamespaceType
    owner: str = None
    members: List[str] = None
    agents: List[str] = None
    auth_method: AuthMethod = None
    created_at: str = None
    
    def __post_init__(self):
        self.members = self.members or []
        self.agents = self.agents or []


class NamespaceManager:
    """Manage namespaces."""
    
    def __init__(self, api=None):
        """Initialize manager.
        
        Args:
            api: RobutlerAPI instance
        """
        self.api = api
        self._local_namespaces: Dict[str, Namespace] = {
            "local": Namespace(
                name="local",
                type=NamespaceType.LOCAL,
                owner="local",
            )
        }
    
    async def list(self, include_remote: bool = True) -> List[Namespace]:
        """List accessible namespaces.
        
        Args:
            include_remote: Include remote namespaces
            
        Returns:
            List of namespaces
        """
        namespaces = list(self._local_namespaces.values())
        
        if include_remote and self.api:
            try:
                from .auth import is_authenticated
                if is_authenticated():
                    async with self.api as api:
                        remote = await api.list_namespaces()
                        for ns_data in remote:
                            namespaces.append(Namespace(
                                name=ns_data["name"],
                                type=NamespaceType(ns_data.get("type", "user")),
                                owner=ns_data.get("owner"),
                                members=ns_data.get("members", []),
                                agents=ns_data.get("agents", []),
                            ))
            except Exception:
                pass
        
        return namespaces
    
    async def create(
        self,
        name: str,
        type_: NamespaceType = NamespaceType.USER,
    ) -> Namespace:
        """Create a namespace.
        
        Args:
            name: Namespace name
            type_: Namespace type
            
        Returns:
            Created namespace
        """
        if type_ == NamespaceType.LOCAL:
            # Create locally
            ns = Namespace(name=name, type=type_, owner="local")
            self._local_namespaces[name] = ns
            return ns
        
        # Create on platform
        if not self.api:
            from .api import RobutlerAPI
            self.api = RobutlerAPI()
        
        async with self.api as api:
            ns_data = await api.create_namespace(name, type_.value)
        
        return Namespace(
            name=ns_data["name"],
            type=NamespaceType(ns_data.get("type", "user")),
            owner=ns_data.get("owner"),
        )
    
    async def get(self, name: str) -> Optional[Namespace]:
        """Get namespace by name.
        
        Args:
            name: Namespace name
            
        Returns:
            Namespace or None
        """
        # Check local first
        if name in self._local_namespaces:
            return self._local_namespaces[name]
        
        # Check remote
        if self.api:
            try:
                async with self.api as api:
                    ns_data = await api.get_namespace(name)
                    return Namespace(
                        name=ns_data["name"],
                        type=NamespaceType(ns_data.get("type", "user")),
                        owner=ns_data.get("owner"),
                        members=ns_data.get("members", []),
                        agents=ns_data.get("agents", []),
                    )
            except Exception:
                pass
        
        return None
    
    async def delete(self, name: str) -> bool:
        """Delete namespace.
        
        Args:
            name: Namespace name
            
        Returns:
            True if deleted
        """
        if name in self._local_namespaces:
            del self._local_namespaces[name]
            return True
        
        # TODO: Delete on platform
        return False
    
    async def invite(self, name: str) -> str:
        """Generate invite code for namespace.
        
        Args:
            name: Namespace name
            
        Returns:
            Invite code
        """
        if not self.api:
            from .api import RobutlerAPI
            self.api = RobutlerAPI()
        
        async with self.api as api:
            return await api.invite_to_namespace(name)
    
    async def join(self, invite_code: str) -> Namespace:
        """Join namespace via invite code.
        
        Args:
            invite_code: Invite code
            
        Returns:
            Joined namespace
        """
        if not self.api:
            from .api import RobutlerAPI
            self.api = RobutlerAPI()
        
        async with self.api as api:
            ns_data = await api.join_namespace(invite_code)
        
        return Namespace(
            name=ns_data["name"],
            type=NamespaceType(ns_data.get("type", "user")),
        )
    
    async def set_auth(
        self,
        name: str,
        method: AuthMethod,
        value: str = None,
    ) -> bool:
        """Set namespace authentication.
        
        Args:
            name: Namespace name
            method: Auth method
            value: Auth value (secret, etc.)
            
        Returns:
            Success
        """
        # TODO: Implement auth configuration
        return True
    
    async def verify_domain(self, name: str) -> Dict:
        """Initiate domain verification.
        
        Args:
            name: Namespace name (should be domain format)
            
        Returns:
            Verification instructions
        """
        # TODO: Implement domain verification
        return {
            "record_type": "TXT",
            "record_name": f"_webagents.{name}",
            "record_value": "verify=abc123",
            "instructions": "Add this TXT record to your DNS, then run verify again.",
        }
