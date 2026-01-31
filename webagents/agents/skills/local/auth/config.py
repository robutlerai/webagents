"""
AOAuth Configuration Models

Defines configuration for AOAuth skill including operating modes,
trust settings, and OAuth provider configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class AuthMode(Enum):
    """AOAuth operating mode."""
    PORTAL = "portal"      # Portal signs tokens, assigns namespaces
    SELF_ISSUED = "self"   # Agent signs own tokens


@dataclass
class AOAuthConfig:
    """Configuration for AOAuth skill.
    
    Attributes:
        mode: Operating mode (PORTAL or SELF_ISSUED)
        authority: Portal authority URL (e.g., "https://robutler.ai")
        agent_id: This agent's identifier
        base_url: This agent's base URL
        token_ttl: Token time-to-live in seconds
        allowed_scopes: Scopes this agent accepts
        trusted_issuers: List of explicitly trusted token issuers
        allow: Allow list patterns for agent access
        deny: Deny list patterns for agent access
        google: Google OAuth provider configuration
        robutler: Robutler OAuth provider configuration
        keys_dir: Directory for storing RSA keys
        jwks_cache_ttl: JWKS cache TTL in seconds
    """
    
    # Operating mode
    mode: AuthMode = AuthMode.SELF_ISSUED
    authority: Optional[str] = None  # e.g., "https://robutler.ai"
    
    # Agent identity
    agent_id: Optional[str] = None
    base_url: Optional[str] = None
    
    # Token settings
    token_ttl: int = 300
    
    # Trust configuration
    allowed_scopes: List[str] = field(default_factory=lambda: ["read", "write"])
    trusted_issuers: List[Dict[str, Any]] = field(default_factory=list)
    allow: List[str] = field(default_factory=list)
    deny: List[str] = field(default_factory=list)
    
    # OAuth providers
    google: Optional[Dict[str, Any]] = None
    robutler: Optional[Dict[str, Any]] = None
    
    # Key management
    keys_dir: Optional[str] = None
    jwks_cache_ttl: int = 3600
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "AOAuthConfig":
        """Create AOAuthConfig from dictionary.
        
        Mode is automatically determined:
        - If 'authority' is set -> PORTAL mode
        - Otherwise -> SELF_ISSUED mode
        
        Args:
            config: Configuration dictionary
            
        Returns:
            AOAuthConfig instance
        """
        mode = AuthMode.PORTAL if config.get("authority") else AuthMode.SELF_ISSUED
        
        return cls(
            mode=mode,
            authority=config.get("authority"),
            agent_id=config.get("agent_id"),
            base_url=config.get("base_url"),
            token_ttl=config.get("token_ttl", 300),
            allowed_scopes=config.get("allowed_scopes", ["read", "write"]),
            trusted_issuers=config.get("trusted_issuers", []),
            allow=config.get("allow", []),
            deny=config.get("deny", []),
            google=config.get("google"),
            robutler=config.get("robutler"),
            keys_dir=config.get("keys_dir"),
            jwks_cache_ttl=config.get("jwks_cache_ttl", 3600),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "mode": self.mode.value,
            "authority": self.authority,
            "agent_id": self.agent_id,
            "base_url": self.base_url,
            "token_ttl": self.token_ttl,
            "allowed_scopes": self.allowed_scopes,
            "trusted_issuers": self.trusted_issuers,
            "allow": self.allow,
            "deny": self.deny,
            "google": self.google,
            "robutler": self.robutler,
            "keys_dir": self.keys_dir,
            "jwks_cache_ttl": self.jwks_cache_ttl,
        }
