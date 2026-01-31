"""
AuthSkill - WebAgents V2.0 Platform Integration

Authentication and authorization skill for WebAgents platform.
Integrates with WebAgents Portal APIs for user authentication, API key validation,
and platform service integration.
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, hook
from robutler.api import RobutlerClient
from robutler.api.types import User, ApiKey, AuthResponse
from typing import Any as _Any
try:
    from jose import jwt as jose_jwt  # python-jose
except Exception:
    jose_jwt = None  # type: ignore


class AuthScope(Enum):
    """Authentication scopes for role-based access control"""
    ADMIN = "admin"
    OWNER = "owner" 
    USER = "user"
    ALL = "all"


@dataclass
class AuthContext:
    """Authentication context for requests (harmonized)
    
    - user_id: ID of the caller. Prefer JWT `sub` when present; otherwise the API key owner's user ID.
    - agent_id: Agent ID asserted by JWT, when present and verified.
    - authenticated: True if API key (and/or assertion) verification succeeds.
    - scope: Authorization scope derived from platform user and agent ownership.
    - assertion: Decoded JWT claims when an owner assertion is provided and verified.
    """
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    authenticated: bool = False
    scope: AuthScope = AuthScope.USER
    assertion: Optional[Dict[str, Any]] = None


class AuthSkill(Skill):
    """
    Authentication and authorization skill for WebAgents platform
    
    Features:
    - Platform integration with WebAgents Portal APIs
    - API key authentication and validation
    - User information retrieval
    - Credit tracking and usage management
    - Request authentication hooks
    - Role-based access control
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all")
        
        # Configuration
        self.config = config or {}
        self.require_auth = self.config.get('require_auth', True)
        # Prefer internal portal URL, then public URL, then localhost for dev
        self.platform_api_url = (
            self.config.get('platform_api_url')
            or os.getenv('ROBUTLER_INTERNAL_API_URL')
            or os.getenv('ROBUTLER_API_URL')
            or 'http://localhost:3000'
        )
        self.api_key = self.config.get('api_key')
        
        # Cache configuration
        self._cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes default
        
        # API client for platform integration
        self.client: Optional[RobutlerClient] = None
        
    async def initialize(self, agent) -> None:
        """Initialize AuthSkill with WebAgents Platform client"""
        from webagents.utils.logging import get_logger, log_skill_event
        
        self.agent = agent
        self.logger = get_logger('skill.webagents.auth', agent.name)
        
        # Initialize WebAgents Platform client
        try:
            # Use api_key as priority, fallback to agent's API key
            final_api_key = self.api_key or getattr(agent, 'api_key', None)
            
            self.client = RobutlerClient(
                api_key=final_api_key,
                base_url=self.platform_api_url
            )
            
            # Test connection
            health_response = await self.client.health_check()
            if health_response.success:
                self.logger.info(f"Connected to WebAgents Platform: {self.platform_api_url}")
            else:
                self.logger.warning(f"Platform health check failed: {health_response.message}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize WebAgents Platform client: {e}")
            # Continue without platform integration for testing
            self.client = None
        
        log_skill_event(agent.name, 'auth', 'initialized', {
            'require_auth': self.require_auth,
            'platform_api_url': self.platform_api_url,
            'has_platform_client': bool(self.client),
            'cache_ttl': self._cache_ttl
        })
    
    # ===== AUTHENTICATION HOOKS =====
    
    @hook("on_connection", priority=0, scope="all")
    async def validate_request_auth(self, context) -> Any:
        """Validate authentication for incoming requests using WebAgents Platform"""
        if not self.require_auth:
            return context

        # Extract API key from request (may be absent)
        api_key = self._extract_api_key_from_context(context)

        # 1) Try API key authentication first (preferred when present)
        auth_context = None
        if api_key:
            auth_context = await self._authenticate_api_key(api_key)

        # 2) If API key auth failed or not provided, try owner assertion only
        if not auth_context or not auth_context.authenticated:
            assertion_only_context = await self._authenticate_with_owner_assertion_only(context)
            if assertion_only_context and assertion_only_context.authenticated:
                context.auth = assertion_only_context
                return context

        # 3) If API key auth succeeded, set context
        if auth_context and auth_context.authenticated:
            context.auth = auth_context
            return context

        # Neither worked
        raise AuthenticationError("Authentication failed (API key or owner assertion required)")
    
    
    # ===== INTERNAL METHODS =====
    
    def _extract_api_key_from_context(self, context) -> Optional[str]:
        """Extract API key from request context"""
        # Try to get from headers (Authorization: Bearer <token>)
        headers = getattr(context.request, 'headers', {})
        auth_header = headers.get('authorization', headers.get('Authorization'))
        
        if auth_header and auth_header.startswith('Bearer '):
            return auth_header[7:]  # Remove 'Bearer ' prefix
        
        # Try X-API-Key header
        api_key_header = headers.get('x-api-key', headers.get('X-API-Key'))
        if api_key_header:
            return api_key_header
        
        # Try to get from query parameters
        query_params = getattr(context.request, 'query_params', {})
        if 'api_key' in query_params:
            return query_params['api_key']
        
        # Try to get from context data directly
        return context.get('api_key')

    def _extract_owner_assertion(self, context) -> Optional[str]:
        """Extract X-Owner-Assertion from headers"""
        if not hasattr(context, 'request') or not context.request:
            return None
        headers = getattr(context.request, 'headers', {}) or {}
        return headers.get('X-Owner-Assertion') or headers.get('x-owner-assertion')
    
    def _extract_header(self, context, header_name: str) -> Optional[str]:
        """Extract header value from context.request"""
        if not hasattr(context, 'request') or not context.request:
            return None
            
        headers = getattr(context.request, 'headers', {})
        if not headers:
            return None
        
        # Try exact match first
        if header_name in headers:
            return headers[header_name]
        
        # Try case-insensitive match
        header_name_lower = header_name.lower()
        for key, value in headers.items():
            if key.lower() == header_name_lower:
                return value
        
        return None
    
    def _is_agent_owner(self, user_id: str) -> bool:
        """Check if the user is the owner of the current agent"""
        # Check agent metadata only (context does not carry owner id)
        if hasattr(self.agent, 'owner_user_id'):
            return user_id == self.agent.owner_user_id
        
        return False

    async def _authenticate_with_owner_assertion_only(self, context) -> Optional[AuthContext]:
        """Authenticate using only X-Owner-Assertion (RS256/JWKS), without API key.
        Grants authenticated USER scope; elevates to OWNER if assertion.owner_user_id == agent.owner_user_id.
        """
        try:
            assertion_token = self._extract_owner_assertion(context)
            if not assertion_token or jose_jwt is None:
                return None
            jwks_url = os.getenv('OWNER_ASSERTION_JWKS_URL') or f"{(self.platform_api_url or '').rstrip('/')}/api/auth/jwks"
            if not jwks_url:
                return None
            import requests
            # Fetch JWKS and select key by kid
            hdr = jose_jwt.get_unverified_header(assertion_token)
            kid = hdr.get('kid')
            r = requests.get(jwks_url, timeout=5)
            r.raise_for_status()
            keys = (r.json() or {}).get('keys', [])
            selected_key = None
            for k in keys:
                if not kid or k.get('kid') == kid:
                    selected_key = k
                    break
            if not selected_key and keys:
                selected_key = keys[0]
            if not selected_key:
                raise Exception('No JWKS key available for owner assertion verification')
            # Decode with selected JWK
            claims = jose_jwt.decode(
                assertion_token,
                selected_key,
                algorithms=['RS256'],
                audience=f"webagents-agent:{getattr(self.agent, 'id', '')}",
            )
            if claims.get('agent_id') and getattr(self.agent, 'id', None) and claims['agent_id'] != getattr(self.agent, 'id'):
                raise Exception('Owner assertion agent_id mismatch')

            acting_user_id = claims.get('sub')
            owner_user_id = claims.get('owner_user_id')
            scope = AuthScope.OWNER if (owner_user_id and hasattr(self.agent, 'owner_user_id') and owner_user_id == getattr(self.agent, 'owner_user_id')) else AuthScope.USER

            return AuthContext(
                user_id=acting_user_id,
                agent_id=claims.get('agent_id'),
                authenticated=True,
                scope=scope,
                assertion=claims,
            )
        except Exception as e:
            try:
                self.logger.debug(f"Owner assertion-only authentication failed: {e}")
            except Exception:
                pass
            return None
    
    async def _authenticate_api_key(self, api_key: str) -> Optional[AuthContext]:
        """Authenticate API key with WebAgents Platform and merge optional owner assertion (JWT)."""
        
        if not self.client:
            self.logger.warning("Platform client not available for authentication")
            return None
        
        try:
            auth_response = await self.client.validate_api_key(api_key)
            
            if auth_response.success and auth_response.user:
                # Determine scope based on user role and ownership
                if auth_response.user.is_admin:
                    scope = AuthScope.ADMIN
                elif self._is_agent_owner(auth_response.user.id):
                    scope = AuthScope.OWNER
                    self.logger.info(f"User {auth_response.user.id} is the agent owner - granting OWNER scope")
                else:
                    scope = AuthScope.USER

                auth_context = AuthContext(
                    user_id=getattr(auth_response.user, 'id', None),
                    authenticated=True,
                    scope=scope,
                )

                # Optional: verify owner assertion JWT to attach acting identity and agent binding
                assertion_token = None
                try:
                    from webagents.server.context.context_vars import get_context as _gc
                    ctx_for_assert = _gc()
                    assertion_token = self._extract_owner_assertion(ctx_for_assert) if ctx_for_assert else None
                except Exception:
                    assertion_token = None

                jwks_url = os.getenv('OWNER_ASSERTION_JWKS_URL') or f"{(self.platform_api_url or '').rstrip('/')}/api/auth/jwks"
                if assertion_token and jwks_url and jose_jwt is not None:
                    try:
                        import requests
                        hdr = jose_jwt.get_unverified_header(assertion_token)
                        kid = hdr.get('kid')
                        r = requests.get(jwks_url, timeout=5)
                        r.raise_for_status()
                        keys = (r.json() or {}).get('keys', [])
                        selected_key = None
                        for k in keys:
                            if not kid or k.get('kid') == kid:
                                selected_key = k
                                break
                        if not selected_key and keys:
                            selected_key = keys[0]
                        if not selected_key:
                            raise Exception('No JWKS key available for owner assertion verification')
                        claims = jose_jwt.decode(
                            assertion_token,
                            selected_key,
                            algorithms=['RS256'],
                            audience=f"webagents-agent:{getattr(self.agent, 'id', '')}",
                        )
                        # Enforce agent binding if claim present
                        if claims.get('agent_id') and getattr(self.agent, 'id', None) and claims['agent_id'] != getattr(self.agent, 'id'):
                            raise Exception('Owner assertion agent_id mismatch')
                        # Harmonized fields
                        auth_context.user_id = claims.get('sub') or auth_context.user_id
                        auth_context.agent_id = claims.get('agent_id') or auth_context.agent_id
                        auth_context.assertion = claims
                        # Owner scope remains derived from API key user vs agent ownership
                    except Exception as e:
                        self.logger.debug(f"Owner assertion verification failed or absent: {e}")
                return auth_context
            else:
                self.logger.warning(f"API key validation failed: {auth_response.message}")
                return None
                
        except Exception as e:
            self.logger.error(f"API key authentication error: {e}")
            return None
    

# Custom exceptions for authentication/authorization
class AuthenticationError(Exception):
    """Raised when authentication fails"""
    pass


class AuthorizationError(Exception):
    """Raised when authorization fails"""  
    pass 