"""
AOAuth Skill - Agent OAuth Protocol Implementation

OAuth 2.0 extension for agent-to-agent authentication.
Supports both Portal-delegated mode and self-issued mode.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import fnmatch
import uuid
import logging
from datetime import datetime, timedelta

import jwt

from ...base import Skill
from webagents.agents.tools.decorators import hook, command, http
from .config import AOAuthConfig, AuthMode
from .jwks import JWKSManager


# Default Portal URL
PORTAL_URL = "https://robutler.ai"


def normalize_agent_url(agent_ref: str, authority: str = None) -> Optional[str]:
    """Normalize @agentname or URL to full agent URL.
    
    Args:
        agent_ref: Agent reference (URL, @name, or plain name)
        authority: Authority to use for normalization
        
    Returns:
        Full agent URL or None if input is empty
        
    Examples:
        - "https://example.com/agent" -> "https://example.com/agent"
        - "@myagent" -> "https://robutler.ai/agents/myagent"
        - "myagent" -> "https://robutler.ai/agents/myagent"
    """
    if not agent_ref:
        return None
    
    # Already a URL
    if agent_ref.startswith("http://") or agent_ref.startswith("https://"):
        return agent_ref
    
    # Strip @ prefix if present
    if agent_ref.startswith("@"):
        agent_ref = agent_ref[1:]
    
    # Use authority or default Portal
    base = authority or PORTAL_URL
    return f"{base}/agents/{agent_ref}"


@dataclass
class AuthContext:
    """Authentication context for a validated request.
    
    Contains information about the authenticated caller, including
    their identity, scopes, and namespace memberships.
    """
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    source_agent: Optional[str] = None
    authenticated: bool = False
    scopes: List[str] = field(default_factory=list)
    namespaces: List[str] = field(default_factory=list)  # Extracted from namespace:* scopes
    issuer: Optional[str] = None
    issuer_type: str = "unknown"  # "portal", "agent", "user"
    raw_claims: Dict[str, Any] = field(default_factory=dict)
    
    def has_scope(self, scope: str) -> bool:
        """Check if context has a specific scope."""
        return scope in self.scopes
    
    def has_namespace(self, namespace: str) -> bool:
        """Check if context has access to a namespace."""
        return namespace in self.namespaces
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "source_agent": self.source_agent,
            "authenticated": self.authenticated,
            "scopes": self.scopes,
            "namespaces": self.namespaces,
            "issuer": self.issuer,
            "issuer_type": self.issuer_type,
        }


class AuthError(Exception):
    """Authentication/authorization error."""
    
    def __init__(self, message: str, code: str = "auth_error"):
        self.message = message
        self.code = code
        super().__init__(message)


class AuthSkill(Skill):
    """AOAuth: OAuth 2.0 extension for agent authentication.
    
    Supports two operating modes:
    
    1. Portal Mode (authority set):
       - Portal signs tokens with namespace scopes
       - Tokens validated against Portal JWKS
       - Centralized trust management
    
    2. Self-Issued Mode (no authority):
       - Agent generates own RSA keys
       - Signs tokens for outgoing requests
       - Trusts agents via allow/deny lists
    
    Configuration:
        authority: Portal URL (e.g., "https://robutler.ai") for Portal mode
        agent_id: This agent's identifier
        base_url: This agent's base URL (or @name)
        allowed_scopes: Scopes this agent accepts
        allow: Allow list patterns for agent access
        deny: Deny list patterns for agent access
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize AuthSkill.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, scope="all")
        self.aoauth_config = AOAuthConfig.from_dict(self.config)
        self.jwks = JWKSManager(self.config)
        self._providers: Dict[str, Any] = {}
        self._grants: Dict[str, Any] = {}
        self._issuer: Optional[str] = None
        self._kid: Optional[str] = None
        self._agent_id: Optional[str] = None
        self._base_url: Optional[str] = None
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self, agent) -> None:
        """Initialize skill with agent reference.
        
        Sets up:
        - Agent identity
        - Mode-specific configuration
        - OAuth providers
        - Grant handlers
        
        Args:
            agent: The BaseAgent instance
        """
        await super().initialize(agent)
        
        # Determine agent identity
        self._agent_id = (
            self.aoauth_config.agent_id or
            getattr(agent, "id", None) or
            getattr(agent, "name", "default")
        )
        
        self._base_url = normalize_agent_url(
            self.aoauth_config.base_url or self._agent_id,
            self.aoauth_config.authority
        )
        
        # Mode-specific setup
        if self.aoauth_config.mode == AuthMode.PORTAL:
            self._setup_portal_mode()
        else:
            self._setup_self_issued_mode()
        
        # Initialize OAuth providers
        self._init_providers()
        
        # Initialize grant handlers
        self._init_grants()
        
        self.logger.info(
            f"AuthSkill initialized: mode={self.aoauth_config.mode.value}, "
            f"issuer={self._issuer}, agent_id={self._agent_id}"
        )
    
    def _setup_portal_mode(self) -> None:
        """Configure Portal mode.
        
        Portal signs tokens, we validate via Portal JWKS.
        """
        self.logger.info(f"AOAuth Portal mode: authority={self.aoauth_config.authority}")
        
        # Add Portal to trusted issuers
        portal_issuer = {
            "issuer": self.aoauth_config.authority,
            "jwks_uri": f"{self.aoauth_config.authority}/api/auth/jwks",
            "type": "portal"
        }
        
        # Check if already in list
        existing = [ti for ti in self.aoauth_config.trusted_issuers 
                   if ti.get("issuer") == self.aoauth_config.authority]
        if not existing:
            self.aoauth_config.trusted_issuers.append(portal_issuer)
        
        self._issuer = self.aoauth_config.authority
        self._agent_path = "/agents"
        self._kid = None  # Portal manages keys
    
    def _setup_self_issued_mode(self) -> None:
        """Configure self-issued mode.
        
        Generate our own RSA keys for signing tokens.
        The issuer is set to authority-only (scheme + host) for standard OIDC
        discovery. The hosting path prefix is stored as agent_path.
        """
        self._kid = self.jwks.ensure_keys(self._agent_id)

        # Split base_url into authority (issuer) and path prefix (agent_path)
        if self._base_url:
            from urllib.parse import urlparse
            parsed = urlparse(self._base_url)
            self._issuer = f"{parsed.scheme}://{parsed.netloc}"
            # Path prefix is everything before the last segment (the agent name)
            path = parsed.path.rstrip("/")
            segments = path.rsplit("/", 1)
            self._agent_path = segments[0] if len(segments) > 1 and segments[0] else None
        else:
            self._issuer = self._base_url
            self._agent_path = None

        self.logger.info(
            f"AOAuth Self-issued mode: issuer={self._issuer}, "
            f"agent_path={self._agent_path}"
        )
    
    def _init_providers(self) -> None:
        """Initialize OAuth providers."""
        # Import here to avoid circular imports
        from .providers.robutler import RobutlerProvider
        from .providers.google import GoogleProvider
        
        # Google provider
        if self.aoauth_config.google:
            self._providers["google"] = GoogleProvider(self.aoauth_config.google)
        
        # Robutler Portal provider
        robutler_config = self.aoauth_config.robutler or {}
        robutler_config["jwks_manager"] = self.jwks
        robutler_config.setdefault("client_id", self._agent_id)
        
        if self.aoauth_config.authority:
            robutler_config["authority"] = self.aoauth_config.authority
        
        self._providers["robutler"] = RobutlerProvider(robutler_config)
    
    def _init_grants(self) -> None:
        """Initialize OAuth grant handlers."""
        from .grants.client_credentials import ClientCredentialsGrant
        from .grants.authorization_code import AuthorizationCodeGrant
        
        self._grants["client_credentials"] = ClientCredentialsGrant(self)
        self._grants["authorization_code"] = AuthorizationCodeGrant(self, self._providers)
    
    # -------------------------------------------------------------------------
    # Token Generation
    # -------------------------------------------------------------------------
    
    def generate_token(
        self,
        target: str,
        scopes: List[str] = None,
        extra_claims: Dict[str, Any] = None
    ) -> str:
        """Generate token for target agent.
        
        In Portal mode, delegates to Portal for token issuance.
        In self-issued mode, generates and signs own token.
        
        Args:
            target: Target agent (URL or @name)
            scopes: Requested scopes
            extra_claims: Additional JWT claims
            
        Returns:
            JWT token string
        """
        if self.aoauth_config.mode == AuthMode.PORTAL:
            # Delegate to Portal
            return self._providers["robutler"].request_token(
                target=target,
                scopes=scopes or list(self.aoauth_config.allowed_scopes)
            )
        else:
            # Self-issue token
            return self._generate_self_issued_token(target, scopes, extra_claims)
    
    def _generate_self_issued_token(
        self,
        target: str,
        scopes: List[str] = None,
        extra_claims: Dict[str, Any] = None
    ) -> str:
        """Generate self-issued JWT token.
        
        Args:
            target: Target agent
            scopes: Requested scopes
            extra_claims: Additional claims
            
        Returns:
            Signed JWT token
        """
        target_url = normalize_agent_url(target, self.aoauth_config.authority)
        scopes = scopes or list(self.aoauth_config.allowed_scopes)
        
        now = datetime.utcnow()
        
        payload = {
            "iss": self._issuer,
            "sub": self._agent_id,
            "aud": target_url,
            "iat": now,
            "nbf": now,
            "exp": now + timedelta(seconds=self.aoauth_config.token_ttl),
            "jti": str(uuid.uuid4()),
            "scope": " ".join(scopes),
            "client_id": self._agent_id,
            "token_type": "Bearer",
        }
        
        if self._agent_path:
            payload["agent_path"] = self._agent_path
        
        # Add extra claims
        if extra_claims:
            payload.update(extra_claims)
        
        return jwt.encode(
            payload,
            self.jwks.get_signing_key(),
            algorithm="RS256",
            headers={"kid": self._kid}
        )
    
    # -------------------------------------------------------------------------
    # Token Validation
    # -------------------------------------------------------------------------
    
    async def validate_token(self, token: str) -> Optional[AuthContext]:
        """Validate token from any trusted issuer.
        
        Validation flow:
        1. Decode header to get kid
        2. Decode claims to get issuer
        3. Find trusted issuer config or check allow list
        4. Fetch public key from JWKS
        5. Validate signature and claims
        6. Filter scopes based on allowed_scopes
        
        Args:
            token: JWT token to validate
            
        Returns:
            AuthContext if valid, None otherwise
        """
        try:
            # Decode header and claims without verification
            header = jwt.get_unverified_header(token)
            unverified = jwt.decode(token, options={"verify_signature": False})
            
            issuer = unverified.get("iss")
            kid = header.get("kid")
            
            if not issuer:
                self.logger.warning("Token missing issuer claim")
                return None
            
            # Determine issuer type and JWKS URI
            issuer_config = self._find_trusted_issuer(issuer)
            
            if issuer_config:
                jwks_uri = issuer_config.get("jwks_uri")
                issuer_type = issuer_config.get("type", "agent")
            elif self._is_allowed(issuer):
                # AOAuth extension: Unknown but allowed issuer - try their JWKS
                jwks_uri = f"{issuer.rstrip('/')}/.well-known/jwks.json"
                issuer_type = "agent"
            else:
                self.logger.warning(f"Issuer {issuer} not trusted")
                return None
            
            # Fetch key and validate (auto-refreshes on miss)
            public_key = await self.jwks.get_public_key_from_jwks(jwks_uri, kid)
            
            if not public_key:
                self.logger.warning(f"Key {kid} not found at {jwks_uri}")
                return None
            
            # Validate signature and claims
            payload = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                audience=self._base_url,
            )
            
            # Extract and filter scopes
            requested_scopes = payload.get("scope", "").split()
            granted_scopes = [s for s in requested_scopes if self._scope_allowed(s)]
            
            # Extract namespaces from namespace:* scopes
            namespaces = [
                s.split(":")[1] 
                for s in granted_scopes 
                if s.startswith("namespace:")
            ]
            
            return AuthContext(
                agent_id=payload.get("sub"),
                source_agent=payload.get("sub"),
                authenticated=True,
                scopes=granted_scopes,
                namespaces=namespaces,
                issuer=issuer,
                issuer_type=issuer_type,
                raw_claims=payload,
            )
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token expired")
            return None
        except jwt.InvalidAudienceError:
            self.logger.warning("Invalid audience")
            return None
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid token: {e}")
            return None
    
    def _find_trusted_issuer(self, issuer: str) -> Optional[Dict[str, Any]]:
        """Find trusted issuer configuration.
        
        Args:
            issuer: Issuer URL to look up
            
        Returns:
            Issuer config dict or None
        """
        for ti in self.aoauth_config.trusted_issuers:
            if ti.get("issuer") == issuer:
                return ti
        return None
    
    def _is_allowed(self, issuer: str) -> bool:
        """Check if issuer is allowed via allow/deny lists.
        
        Deny list is checked first (blacklist takes precedence).
        If allow list is empty, all non-denied issuers are allowed.
        
        Args:
            issuer: Issuer URL or agent ID to check
            
        Returns:
            True if allowed
        """
        # Extract agent ID from URL
        agent_id = issuer.split("/")[-1] if "/" in issuer else issuer
        
        # Check deny list first
        for pattern in self.aoauth_config.deny:
            pattern = pattern.lstrip("@")
            if fnmatch.fnmatch(agent_id, pattern) or fnmatch.fnmatch(issuer, pattern):
                return False
        
        # If no allow list, allow all non-denied
        if not self.aoauth_config.allow:
            return True
        
        # Check allow list
        for pattern in self.aoauth_config.allow:
            pattern = pattern.lstrip("@")
            if fnmatch.fnmatch(agent_id, pattern) or fnmatch.fnmatch(issuer, pattern):
                return True
        
        return False
    
    def _scope_allowed(self, scope: str) -> bool:
        """Check if a scope is allowed by this agent.
        
        Supports wildcard patterns like "namespace:*" to allow
        all scopes starting with "namespace:".
        
        Args:
            scope: Scope to check
            
        Returns:
            True if scope is allowed
        """
        if scope in self.aoauth_config.allowed_scopes:
            return True
        
        # Check for wildcard patterns
        for allowed in self.aoauth_config.allowed_scopes:
            if allowed.endswith(":*"):
                prefix = allowed[:-1]  # Remove *
                if scope.startswith(prefix):
                    return True
        
        return False
    
    # -------------------------------------------------------------------------
    # Hooks
    # -------------------------------------------------------------------------
    
    @hook("on_request_outgoing", priority=10, scope="all")
    async def inject_token(self, context) -> Any:
        """Inject AOAuth token into outgoing requests.
        
        Automatically adds Bearer token to Authorization header
        for requests to other agents.
        
        Args:
            context: Request context
            
        Returns:
            Modified context
        """
        target = context.get("target_agent") or context.get("target_url")
        if not target:
            return context
        
        scopes = context.get("requested_scopes", ["read"])
        token = self.generate_token(target, scopes)
        
        # Ensure headers dict exists
        if not hasattr(context, "headers") or context.headers is None:
            context.headers = {}
        
        context.headers["Authorization"] = f"Bearer {token}"
        return context
    
    @hook("on_connection", priority=5, scope="all")
    async def validate_request(self, context) -> Any:
        """Validate incoming Bearer token.
        
        Extracts and validates JWT from Authorization header,
        attaching AuthContext to the request context.
        
        Args:
            context: Request context
            
        Returns:
            Modified context with auth information
        """
        # Get authorization header
        auth_header = ""
        if hasattr(context, "request") and context.request:
            auth_header = context.request.headers.get("Authorization", "")
        elif hasattr(context, "headers"):
            auth_header = context.headers.get("Authorization", "")
        
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            auth = await self.validate_token(token)
            if auth:
                context.auth = auth
                self.logger.debug(f"Authenticated: agent={auth.agent_id}, scopes={auth.scopes}")

                # Inbound trust check: if caller is an agent (has client_id),
                # evaluate the target agent's accept_from rules.
                if auth.raw_claims and auth.raw_claims.get("client_id"):
                    accept_from = None
                    if self.agent and hasattr(self.agent, "config"):
                        accept_from = (self.agent.config or {}).get("accept_from")
                    if accept_from is not None:
                        try:
                            from webagents.trust import evaluate_trust_rules, extract_trust_labels
                            caller_labels = extract_trust_labels(
                                auth.raw_claims.get("scope", ""),
                                issuer=auth.issuer or "",
                            )
                            allowed = evaluate_trust_rules(
                                caller=auth.agent_id or "",
                                target=getattr(self.agent, "name", ""),
                                rules=accept_from,
                                caller_trust_labels=caller_labels,
                            )
                            if not allowed:
                                self.logger.warning(
                                    f"Trust denied: {auth.agent_id} not in accept_from rules"
                                )
                                raise AuthError("Caller not in agent's trust scope", "trust_denied")
                        except ImportError:
                            pass
        
        return context
    
    # -------------------------------------------------------------------------
    # HTTP Endpoints
    # -------------------------------------------------------------------------
    
    @http("/.well-known/openid-configuration", method="get", scope="all")
    async def openid_configuration(self) -> Dict[str, Any]:
        """OpenID Connect Discovery endpoint.
        
        Returns:
            OpenID configuration document
        """
        base = self._base_url.rstrip("/")
        
        config = {
            "issuer": self._issuer,
            "authorization_endpoint": f"{base}/auth/authorize",
            "token_endpoint": f"{base}/auth/token",
            "jwks_uri": f"{base}/.well-known/jwks.json",
            "response_types_supported": ["code", "token"],
            "subject_types_supported": ["public"],
            "id_token_signing_alg_values_supported": ["RS256"],
            "scopes_supported": self.aoauth_config.allowed_scopes,
            "token_endpoint_auth_methods_supported": [
                "client_secret_basic",
                "client_secret_post"
            ],
            "claims_supported": [
                "iss", "sub", "aud", "exp", "iat", "scope", "client_id"
            ],
            "grant_types_supported": [
                "authorization_code",
                "client_credentials"
            ],
        }
        
        return config
    
    @http("/.well-known/jwks.json", method="get", scope="all")
    async def jwks_endpoint(self) -> Dict[str, Any]:
        """JWKS endpoint for public key discovery.
        
        Returns:
            JWKS document with public keys
        """
        return self.jwks.get_jwks()
    
    @http("/auth/token", method="post", scope="all")
    async def token_endpoint(
        self,
        grant_type: str,
        client_id: str = None,
        client_secret: str = None,
        scope: str = None,
        code: str = None,
        redirect_uri: str = None,
        target: str = None,
    ) -> Dict[str, Any]:
        """OAuth token endpoint.
        
        Supports grant types:
        - client_credentials: Agent-to-agent authentication
        - authorization_code: User authorization flow
        
        Args:
            grant_type: OAuth grant type
            client_id: Client identifier
            client_secret: Client secret
            scope: Requested scopes (space-separated)
            code: Authorization code (for authorization_code grant)
            redirect_uri: Redirect URI (for authorization_code grant)
            target: Target agent (for client_credentials grant)
            
        Returns:
            Token response
        """
        if grant_type not in self._grants:
            return {
                "error": "unsupported_grant_type",
                "error_description": f"Grant type '{grant_type}' is not supported"
            }
        
        grant = self._grants[grant_type]
        
        return await grant.handle(
            client_id=client_id,
            client_secret=client_secret,
            scope=scope,
            code=code,
            redirect_uri=redirect_uri,
            target=target,
        )
    
    # -------------------------------------------------------------------------
    # Commands
    # -------------------------------------------------------------------------
    
    @command("/auth", description="AOAuth status and configuration")
    async def auth_status(self) -> Dict[str, Any]:
        """Show AOAuth status and configuration.
        
        Returns:
            Current auth configuration
        """
        status = {
            "mode": self.aoauth_config.mode.value,
            "authority": self.aoauth_config.authority,
            "issuer": self._issuer,
            "agent_id": self._agent_id,
            "base_url": self._base_url,
            "kid": self._kid,
            "allowed_scopes": self.aoauth_config.allowed_scopes,
            "trusted_issuers": len(self.aoauth_config.trusted_issuers),
            "providers": list(self._providers.keys()),
            "grants": list(self._grants.keys()),
        }
        
        # Add endpoint URLs
        if self._base_url:
            base = self._base_url.rstrip("/")
            status["endpoints"] = {
                "discovery": f"{base}/.well-known/openid-configuration",
                "jwks": f"{base}/.well-known/jwks.json",
                "token": f"{base}/auth/token",
            }
        
        # Build display
        lines = [
            f"[bold]AOAuth Status[/bold]",
            f"  Mode: [cyan]{status['mode']}[/cyan]",
            f"  Issuer: {status['issuer'] or '[dim]not set[/dim]'}",
            f"  Agent ID: {status['agent_id']}",
            f"  Key ID: {status['kid'] or '[dim]managed by portal[/dim]'}",
            f"  Allowed Scopes: {', '.join(status['allowed_scopes'])}",
        ]
        
        return {
            **status,
            "display": "\n".join(lines),
        }
    
    @command("/auth/token", description="Generate token for target agent")
    async def generate_token_command(
        self,
        target: str,
        scopes: str = "read"
    ) -> Dict[str, Any]:
        """Generate a token for a target agent.
        
        Args:
            target: Target agent (@name or URL)
            scopes: Space-separated scopes
            
        Returns:
            Generated token
        """
        scope_list = scopes.split()
        token = self.generate_token(target, scope_list)
        
        # Decode for display
        claims = jwt.decode(token, options={"verify_signature": False})
        
        return {
            "token": token,
            "claims": claims,
            "display": f"[green]Token generated[/green] for {target}\n[dim]{token[:50]}...[/dim]",
        }
    
    @command("/auth/validate", description="Validate a token")
    async def validate_token_command(self, token: str) -> Dict[str, Any]:
        """Validate a JWT token.
        
        Args:
            token: JWT token to validate
            
        Returns:
            Validation result
        """
        auth = await self.validate_token(token)
        
        if auth:
            return {
                "valid": True,
                "auth": auth.to_dict(),
                "display": f"[green]Valid token[/green]\n  Agent: {auth.agent_id}\n  Scopes: {', '.join(auth.scopes)}",
            }
        else:
            return {
                "valid": False,
                "display": "[red]Invalid token[/red]",
            }
    
    @command("/auth/jwks", description="Show JWKS cache statistics")
    async def jwks_stats(self) -> Dict[str, Any]:
        """Show JWKS cache statistics.
        
        Returns:
            Cache statistics
        """
        stats = self.jwks.get_cache_stats()
        
        lines = [f"[bold]JWKS Cache[/bold]", f"  Entries: {stats['total_entries']}"]
        
        for uri, entry in stats.get("entries", {}).items():
            lines.append(f"  [dim]{uri}[/dim]")
            lines.append(f"    Keys: {entry['keys_count']}, Expires: {entry['expires_in']}s")
        
        return {
            **stats,
            "display": "\n".join(lines),
        }
