"""
AuthSkill - WebAgents V2.0 Platform Integration

Authentication and authorization skill for WebAgents platform.
Integrates with WebAgents Portal APIs for user authentication, API key validation,
and platform service integration.
"""

import os
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, hook
from robutler.api import RobutlerClient
from robutler.api.types import User, ApiKey, AuthResponse
from typing import Any as _Any
# PyJWT is a DECLARED dependency (crypto/jwks.py already requires it). The
# old python-jose import was optional-and-guarded, so on the many installs
# without python-jose the whole verification path silently no-oped.
import jwt as pyjwt


# The audience the platform stamps when the call site has no target to name.
#
# `SERVICE_TOKEN_FALLBACK_AUD` in lib/agents/router.ts — `getServiceToken()`
# with no `target` mints it (the voice relay's outbound leg is the live
# example). It is deliberately NOT the platform issuer, so a token carrying it
# still cannot be replayed at the portal. Accepting it alongside this agent's
# own public URL is what keeps those legs authenticating; treating every
# non-self audience as a refusal would break them the moment this SDK ships.
PLATFORM_FALLBACK_AUDIENCE = "urn:robutler:agent-endpoint"


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
        # The platform's canonical JWT issuer (its PUBLIC base URL). The
        # internal URL above may differ (in-cluster host), so the issuer is
        # pinned separately. Every platform-signed JWT this skill verifies
        # must carry exactly this `iss`.
        self.platform_issuer = (
            self.config.get('platform_issuer')
            or os.getenv('ROBUTLER_PLATFORM_ISSUER')
            or os.getenv('ROBUTLER_API_URL')
            or self.platform_api_url
        ).rstrip('/')
        # This agent's own public URL — the expected `aud` of inbound
        # platform service tokens (the platform binds each service token to
        # the URL it is dialling). See _authenticate_service_token for the
        # no-aud transition window.
        # Trailing slash normalised because the PLATFORM normalises on its
        # side (`serviceTokenAudienceFor`, lib/agents/router.ts, strips a
        # trailing slash and any `/chat/completions` suffix before stamping
        # `aud`). Without this rstrip a WEBAGENTS_PUBLIC_URL ending in '/'
        # refused every service token, silently — and `portal.py` rstrips the
        # same env var, so the two halves disagreed.
        _public_url = (
            self.config.get('agent_url')
            or os.getenv('WEBAGENTS_PUBLIC_URL')
            or os.getenv('WEBAGENTS_AGENT_URL')
            or None
        )
        self.agent_public_url = _public_url.rstrip('/') if _public_url else None
        # Transition flag: '1' = a service token WITHOUT an `aud` claim is
        # refused. Default '0' for exactly one release: the platform only
        # started emitting per-target audiences with the same rollout, and a
        # verifier that requires `aud` before the platform emits it fails
        # closed for every on-prem agent.
        self.require_service_aud = (
            str(self.config.get('require_service_aud', os.getenv('WEBAGENTS_REQUIRE_SERVICE_AUD', '0'))) == '1'
        )
        self.api_key = self.config.get('api_key')

        # Cache configuration
        self._cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes default
        # JWKS cache: fetched once, reused for _cache_ttl seconds, refetched
        # at most once per verification on a kid miss (key rotation).
        self._jwks_keys: Optional[List[Dict[str, Any]]] = None
        self._jwks_fetched_at: float = 0.0
        
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

        # 4) If bearer token looks like a service JWT (RS256), try service auth
        if api_key:
            service_context = await self._authenticate_service_token(api_key)
            if service_context and service_context.authenticated:
                context.auth = service_context
                return context

        # Neither worked
        raise AuthenticationError("Authentication failed (API key, owner assertion, or service token required)")
    
    
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

    # ===== PLATFORM JWKS =====

    def _platform_jwks_url(self) -> str:
        """The JWKS endpoint the portal actually serves.

        The portal publishes its signing keys at `/.well-known/jwks.json`.
        The previous default, `{platform_api_url}/api/auth/jwks`, does not
        exist on the portal (404), so verification always failed over to
        fallback behaviour. `OWNER_ASSERTION_JWKS_URL` still overrides for
        non-standard deployments.
        """
        return (
            os.getenv('OWNER_ASSERTION_JWKS_URL')
            or f"{(self.platform_api_url or '').rstrip('/')}/.well-known/jwks.json"
        )

    async def _load_platform_jwk(self, kid: Optional[str]) -> Optional[Dict[str, Any]]:
        """Load the platform JWK with the EXACT `kid`, from a TTL cache.

        - No `keys[0]` fallback: on a kid mismatch the old code silently
          verified against whatever key happened to be listed first, which
          converts a key-rotation window into a verification bypass.
        - A `kid` is required. A token that names no key gets no key.
        - On a kid miss the JWKS is refetched once (rotation handling).
        """
        if not kid:
            return None

        def _find(keys: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
            for k in keys or []:
                if k.get('kid') == kid:
                    return k
            return None

        now = time.monotonic()
        if self._jwks_keys is not None and (now - self._jwks_fetched_at) < self._cache_ttl:
            hit = _find(self._jwks_keys)
            if hit:
                return hit
            # fall through: kid miss on a fresh cache still refetches once

        import httpx
        jwks_url = self._platform_jwks_url()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(jwks_url)
                r.raise_for_status()
                self._jwks_keys = (r.json() or {}).get('keys', [])
                self._jwks_fetched_at = time.monotonic()
        except Exception as e:
            try:
                self.logger.warning(f"Platform JWKS fetch failed ({jwks_url}): {e}")
            except Exception:
                pass
            return None
        return _find(self._jwks_keys)

    async def _verify_owner_assertion(self, assertion_token: str) -> Optional[Dict[str, Any]]:
        """Verify an X-Owner-Assertion JWT against the platform JWKS.

        Returns the verified claims, or None. Fails CLOSED: no key fallback,
        issuer pinned to the platform, audience required.
        """
        try:
            hdr = pyjwt.get_unverified_header(assertion_token)
            if hdr.get('alg') != 'RS256':
                return None
            selected_key = await self._load_platform_jwk(hdr.get('kid'))
            if not selected_key:
                return None
            claims = pyjwt.decode(
                assertion_token,
                pyjwt.PyJWK(selected_key).key,
                algorithms=['RS256'],
                audience=f"webagents-agent:{getattr(self.agent, 'id', '')}",
            )
            if claims.get('agent_id') and getattr(self.agent, 'id', None) and claims['agent_id'] != getattr(self.agent, 'id'):
                raise Exception('Owner assertion agent_id mismatch')
            return claims
        except Exception as e:
            try:
                self.logger.debug(f"Owner assertion verification failed: {e}")
            except Exception:
                pass
            return None

    async def _authenticate_with_owner_assertion_only(self, context) -> Optional[AuthContext]:
        """Authenticate using only X-Owner-Assertion (RS256/JWKS), without API key.
        Grants authenticated USER scope; elevates to OWNER if assertion.owner_user_id == agent.owner_user_id.
        """
        assertion_token = self._extract_owner_assertion(context)
        if not assertion_token:
            return None
        claims = await self._verify_owner_assertion(assertion_token)
        if not claims:
            return None

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

    async def _authenticate_service_token(self, token: str) -> Optional[AuthContext]:
        """Authenticate an RS256 platform service JWT. Fails CLOSED.

        Service tokens are issued by the Robutler platform router when it
        dials this agent. Verification requirements:

        - alg RS256, `kid` in the header naming an EXACT platform JWKS key
          (no first-key fallback — that converts key rotation into a bypass)
        - `sub` starting with "service:"
        - `iss` equal to the configured platform issuer (`platform_issuer`)
        - `aud` equal to this agent's own public URL (`agent_public_url`) or
          to the platform's targetless fallback (`PLATFORM_FALLBACK_AUDIENCE`)

        AUDIENCE TRANSITION (one release): the platform only recently began
        stamping a per-target `aud` on its service tokens. A token WITHOUT an
        `aud` claim is still accepted while `require_service_aud` is off
        (default), so existing deployments keep working across the platform
        rollout; a token WITH an `aud` that does not match is ALWAYS refused.
        Set WEBAGENTS_REQUIRE_SERVICE_AUD=1 (the next release's default) to
        refuse no-aud tokens too.

        The returned context is NOT admin. The platform is relaying a chat
        turn on behalf of a sender; the request metadata carries that sender
        (`metadata.sender.id`), and the scope derives from it exactly as an
        api-key caller's would.
        """
        try:
            header = pyjwt.get_unverified_header(token)
            if header.get("alg") != "RS256":
                return None

            unverified = pyjwt.decode(token, options={"verify_signature": False})
            sub = unverified.get("sub", "")
            if not isinstance(sub, str) or not sub.startswith("service:"):
                return None

            selected_key = await self._load_platform_jwk(header.get("kid"))
            if not selected_key:
                return None

            decode_kwargs: Dict[str, Any] = {
                "algorithms": ["RS256"],
                "issuer": self.platform_issuer,
            }
            token_aud = unverified.get("aud")
            if token_aud is not None:
                # aud present: it MUST match this agent's public URL.
                if not self.agent_public_url:
                    if self.require_service_aud:
                        self.logger.warning(
                            "Service token carries aud=%r but this agent has no "
                            "configured public URL (set WEBAGENTS_PUBLIC_URL); refusing.",
                            token_aud,
                        )
                        return None
                    self.logger.warning(
                        "Service token carries aud=%r but this agent has no configured "
                        "public URL to check it against. Set WEBAGENTS_PUBLIC_URL — the "
                        "next release refuses this.",
                        token_aud,
                    )
                    decode_kwargs["options"] = {"verify_aud": False}
                else:
                    # Two audiences are legitimate: this agent's own public
                    # URL, and the platform's targetless fallback (see
                    # PLATFORM_FALLBACK_AUDIENCE above).
                    decode_kwargs["audience"] = [
                        self.agent_public_url,
                        PLATFORM_FALLBACK_AUDIENCE,
                    ]
            else:
                # aud absent: allowed for exactly one release (see docstring).
                if self.require_service_aud:
                    self.logger.warning(
                        "Service token without an aud claim refused (WEBAGENTS_REQUIRE_SERVICE_AUD=1)."
                    )
                    return None
                self.logger.info(
                    "Service token without an aud claim accepted (transition window). "
                    "The next release requires the platform's per-target audience."
                )
                decode_kwargs["options"] = {"verify_aud": False}

            claims = pyjwt.decode(token, pyjwt.PyJWK(selected_key).key, **decode_kwargs)

            # Attribute the call to the sender the platform is relaying for,
            # never as a blanket admin. router.ts sends metadata.sender.{id,...}
            # on every completions call.
            sender_id = self._extract_platform_sender_id()
            scope = AuthScope.USER
            if sender_id and self._is_agent_owner(sender_id):
                scope = AuthScope.OWNER

            return AuthContext(
                user_id=sender_id or claims.get("sub"),
                authenticated=True,
                scope=scope,
                assertion=claims,
            )
        except Exception as e:
            try:
                self.logger.debug(f"Service token authentication failed: {e}")
            except Exception:
                pass
            return None

    def _extract_platform_sender_id(self) -> Optional[str]:
        """The sender the platform says this turn is on behalf of
        (`metadata.sender.id` on the completions request), when the server
        put the request metadata on the context. Best-effort."""
        try:
            from webagents.server.context.context_vars import get_context as _gc
            ctx = _gc()
            if ctx is None:
                return None
            metadata = getattr(ctx, "metadata", None)
            if not isinstance(metadata, dict):
                metadata = ctx.get("metadata") if hasattr(ctx, "get") else None
            if not isinstance(metadata, dict):
                metadata = ctx.get("request_metadata") if hasattr(ctx, "get") else None
            if isinstance(metadata, dict):
                sender = metadata.get("sender")
                if isinstance(sender, dict) and sender.get("id"):
                    return str(sender["id"])
            return None
        except Exception:
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

                if assertion_token:
                    claims = await self._verify_owner_assertion(assertion_token)
                    if claims:
                        # Harmonized fields
                        auth_context.user_id = claims.get('sub') or auth_context.user_id
                        auth_context.agent_id = claims.get('agent_id') or auth_context.agent_id
                        auth_context.assertion = claims
                        # Owner scope remains derived from API key user vs agent ownership
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