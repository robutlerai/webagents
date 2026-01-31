"""
AOAuth Skill Unit Tests

Tests for the AOAuth (Agent OAuth) protocol implementation:
- AOAuthConfig mode detection (portal vs self-issued)
- JWKSManager key generation and caching
- AuthSkill token generation and validation
- Allow/deny list functionality
- Scope filtering
"""

import pytest
import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import asdict

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend


# ===== AOAuthConfig Tests =====

class TestAOAuthConfig:
    """Test AOAuthConfig mode detection and parsing."""
    
    def test_mode_detection_self_issued_when_no_authority(self):
        """Mode should be SELF_ISSUED when authority is not set."""
        from webagents.agents.skills.local.auth import AOAuthConfig, AuthMode
        
        config = AOAuthConfig.from_dict({})
        
        assert config.mode == AuthMode.SELF_ISSUED
        assert config.authority is None
    
    def test_mode_detection_portal_when_authority_set(self):
        """Mode should be PORTAL when authority is set."""
        from webagents.agents.skills.local.auth import AOAuthConfig, AuthMode
        
        config = AOAuthConfig.from_dict({
            "authority": "https://robutler.ai"
        })
        
        assert config.mode == AuthMode.PORTAL
        assert config.authority == "https://robutler.ai"
    
    def test_default_config_values(self):
        """Test default configuration values."""
        from webagents.agents.skills.local.auth import AOAuthConfig
        
        config = AOAuthConfig.from_dict({})
        
        assert config.token_ttl == 300
        assert config.allowed_scopes == ["read", "write"]
        assert config.trusted_issuers == []
        assert config.allow == []
        assert config.deny == []
        assert config.jwks_cache_ttl == 3600
    
    def test_custom_config_values(self):
        """Test custom configuration values are properly parsed."""
        from webagents.agents.skills.local.auth import AOAuthConfig
        
        config = AOAuthConfig.from_dict({
            "authority": "https://portal.example.com",
            "agent_id": "my-agent",
            "base_url": "@my-agent",
            "token_ttl": 600,
            "allowed_scopes": ["read", "write", "admin"],
            "allow": ["@trusted-*"],
            "deny": ["@malicious-*"],
            "jwks_cache_ttl": 7200,
        })
        
        assert config.authority == "https://portal.example.com"
        assert config.agent_id == "my-agent"
        assert config.base_url == "@my-agent"
        assert config.token_ttl == 600
        assert config.allowed_scopes == ["read", "write", "admin"]
        assert config.allow == ["@trusted-*"]
        assert config.deny == ["@malicious-*"]
        assert config.jwks_cache_ttl == 7200
    
    def test_to_dict_roundtrip(self):
        """Test that to_dict produces valid config that can be re-parsed."""
        from webagents.agents.skills.local.auth import AOAuthConfig, AuthMode
        
        original = AOAuthConfig.from_dict({
            "authority": "https://test.com",
            "agent_id": "test-agent",
            "allowed_scopes": ["custom_scope"],
        })
        
        config_dict = original.to_dict()
        restored = AOAuthConfig.from_dict(config_dict)
        
        # Note: mode is derived, not stored directly
        assert restored.authority == original.authority
        assert restored.agent_id == original.agent_id
        assert restored.allowed_scopes == original.allowed_scopes


# ===== JWKSManager Tests =====

class TestJWKSManager:
    """Test JWKSManager key generation and caching."""
    
    def test_ensure_keys_generates_key(self):
        """ensure_keys should generate RSA key pair."""
        from webagents.agents.skills.local.auth import JWKSManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JWKSManager({"keys_dir": tmpdir})
            kid = manager.ensure_keys("test-agent")
            
            assert kid is not None
            assert len(kid) == 16  # SHA256 hash prefix
            assert manager.get_kid() == kid
    
    def test_ensure_keys_loads_existing_key(self):
        """ensure_keys should load existing key from disk."""
        from webagents.agents.skills.local.auth import JWKSManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # First call generates key
            manager1 = JWKSManager({"keys_dir": tmpdir})
            kid1 = manager1.ensure_keys("test-agent")
            
            # Second call should load same key
            manager2 = JWKSManager({"keys_dir": tmpdir})
            kid2 = manager2.ensure_keys("test-agent")
            
            assert kid1 == kid2
    
    def test_get_signing_key_raises_without_init(self):
        """get_signing_key should raise if keys not initialized."""
        from webagents.agents.skills.local.auth import JWKSManager
        
        manager = JWKSManager({})
        
        with pytest.raises(RuntimeError, match="not initialized"):
            manager.get_signing_key()
    
    def test_get_public_jwk_format(self):
        """get_public_jwk should return valid JWK format."""
        from webagents.agents.skills.local.auth import JWKSManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JWKSManager({"keys_dir": tmpdir})
            kid = manager.ensure_keys("test-agent")
            
            jwk = manager.get_public_jwk()
            
            assert jwk["kty"] == "RSA"
            assert jwk["use"] == "sig"
            assert jwk["alg"] == "RS256"
            assert jwk["kid"] == kid
            assert "n" in jwk  # modulus
            assert "e" in jwk  # exponent
    
    def test_get_jwks_returns_keys_array(self):
        """get_jwks should return JWKS format with keys array."""
        from webagents.agents.skills.local.auth import JWKSManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = JWKSManager({"keys_dir": tmpdir})
            manager.ensure_keys("test-agent")
            
            jwks = manager.get_jwks()
            
            assert "keys" in jwks
            assert isinstance(jwks["keys"], list)
            assert len(jwks["keys"]) == 1
    
    def test_cache_entry_structure(self):
        """Test CacheEntry dataclass structure."""
        from webagents.agents.skills.local.auth import CacheEntry
        
        entry = CacheEntry(
            keys=[{"kid": "test"}],
            expires_at=time.time() + 3600,
            etag="abc123",
            last_fetch=time.time()
        )
        
        assert len(entry.keys) == 1
        assert entry.etag == "abc123"
    
    @pytest.mark.asyncio
    async def test_fetch_jwks_caching(self):
        """fetch_jwks should cache responses."""
        from webagents.agents.skills.local.auth import JWKSManager
        
        manager = JWKSManager({"jwks_cache_ttl": 3600})
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"keys": [{"kid": "test-key"}]}
        mock_response.headers = {"Cache-Control": "max-age=3600"}
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            # First fetch
            keys1 = await manager.fetch_jwks("https://example.com/.well-known/jwks.json")
            
            # Second fetch should use cache (no new HTTP call)
            mock_client.get.reset_mock()
            keys2 = await manager.fetch_jwks("https://example.com/.well-known/jwks.json")
            
            assert keys1 == keys2
            assert len(keys1) == 1
            # Second call should not make HTTP request (cached)
            mock_client.get.assert_not_called()
    
    def test_cache_invalidation(self):
        """invalidate_cache should clear cached entries."""
        from webagents.agents.skills.local.auth import JWKSManager, CacheEntry
        
        manager = JWKSManager({})
        
        # Manually populate cache
        manager._jwks_cache["https://test.com/jwks"] = CacheEntry(
            keys=[{"kid": "test"}],
            expires_at=time.time() + 3600,
            last_fetch=time.time()
        )
        
        # Invalidate specific URI
        manager.invalidate_cache("https://test.com/jwks")
        assert "https://test.com/jwks" not in manager._jwks_cache
    
    def test_get_cache_stats(self):
        """get_cache_stats should return cache statistics."""
        from webagents.agents.skills.local.auth import JWKSManager, CacheEntry
        
        manager = JWKSManager({})
        
        # Empty cache
        stats = manager.get_cache_stats()
        assert stats["total_entries"] == 0
        
        # Add entry
        manager._jwks_cache["https://test.com/jwks"] = CacheEntry(
            keys=[{"kid": "test1"}, {"kid": "test2"}],
            expires_at=time.time() + 3600,
            etag="abc",
            last_fetch=time.time()
        )
        
        stats = manager.get_cache_stats()
        assert stats["total_entries"] == 1
        assert "https://test.com/jwks" in stats["entries"]
        assert stats["entries"]["https://test.com/jwks"]["keys_count"] == 2
        assert stats["entries"]["https://test.com/jwks"]["has_etag"] == True


# ===== normalize_agent_url Tests =====

class TestNormalizeAgentUrl:
    """Test normalize_agent_url function."""
    
    def test_full_url_unchanged(self):
        """Full URL should be returned unchanged."""
        from webagents.agents.skills.local.auth import normalize_agent_url
        
        url = "https://example.com/agent"
        result = normalize_agent_url(url)
        
        assert result == url
    
    def test_at_name_normalized_to_portal_url(self):
        """@name should be normalized to portal URL."""
        from webagents.agents.skills.local.auth import normalize_agent_url
        
        result = normalize_agent_url("@myagent")
        
        assert result == "https://robutler.ai/agents/myagent"
    
    def test_plain_name_normalized(self):
        """Plain name should be normalized to portal URL."""
        from webagents.agents.skills.local.auth import normalize_agent_url
        
        result = normalize_agent_url("myagent")
        
        assert result == "https://robutler.ai/agents/myagent"
    
    def test_custom_authority(self):
        """Custom authority should be used in normalization."""
        from webagents.agents.skills.local.auth import normalize_agent_url
        
        result = normalize_agent_url("@myagent", authority="https://custom.portal.com")
        
        assert result == "https://custom.portal.com/agents/myagent"
    
    def test_empty_input_returns_none(self):
        """Empty input should return None."""
        from webagents.agents.skills.local.auth import normalize_agent_url
        
        assert normalize_agent_url("") is None
        assert normalize_agent_url(None) is None


# ===== AuthContext Tests =====

class TestAuthContext:
    """Test AuthContext dataclass."""
    
    def test_auth_context_creation(self):
        """Test AuthContext creation with defaults."""
        from webagents.agents.skills.local.auth.skill import AuthContext
        
        ctx = AuthContext()
        
        assert ctx.authenticated == False
        assert ctx.scopes == []
        assert ctx.namespaces == []
        assert ctx.issuer_type == "unknown"
    
    def test_has_scope(self):
        """Test has_scope method."""
        from webagents.agents.skills.local.auth.skill import AuthContext
        
        ctx = AuthContext(scopes=["read", "write", "namespace:admin"])
        
        assert ctx.has_scope("read") == True
        assert ctx.has_scope("write") == True
        assert ctx.has_scope("delete") == False
    
    def test_has_namespace(self):
        """Test has_namespace method."""
        from webagents.agents.skills.local.auth.skill import AuthContext
        
        ctx = AuthContext(namespaces=["admin", "users"])
        
        assert ctx.has_namespace("admin") == True
        assert ctx.has_namespace("users") == True
        assert ctx.has_namespace("billing") == False
    
    def test_to_dict(self):
        """Test to_dict serialization."""
        from webagents.agents.skills.local.auth.skill import AuthContext
        
        ctx = AuthContext(
            agent_id="test-agent",
            authenticated=True,
            scopes=["read"],
            issuer="https://portal.example.com",
            issuer_type="portal"
        )
        
        d = ctx.to_dict()
        
        assert d["agent_id"] == "test-agent"
        assert d["authenticated"] == True
        assert d["scopes"] == ["read"]
        assert d["issuer"] == "https://portal.example.com"
        assert d["issuer_type"] == "portal"


# ===== AuthSkill Tests =====

class TestAuthSkill:
    """Test AuthSkill token generation and validation."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock agent for testing."""
        agent = Mock()
        agent.name = "test-agent"
        agent.id = "test-agent-id"
        return agent
    
    @pytest.fixture
    def self_issued_skill(self):
        """Create AuthSkill in self-issued mode."""
        from webagents.agents.skills.local.auth import AuthSkill
        
        with tempfile.TemporaryDirectory() as tmpdir:
            skill = AuthSkill({
                "agent_id": "test-agent",
                "base_url": "https://myagent.example.com",
                "keys_dir": tmpdir,
                "allowed_scopes": ["read", "write", "namespace:*"],
            })
            yield skill
    
    @pytest.fixture
    def portal_skill(self):
        """Create AuthSkill in portal mode."""
        from webagents.agents.skills.local.auth import AuthSkill
        
        with tempfile.TemporaryDirectory() as tmpdir:
            skill = AuthSkill({
                "authority": "https://robutler.ai",
                "agent_id": "test-agent",
                "base_url": "@test-agent",
                "keys_dir": tmpdir,
                "allowed_scopes": ["read", "write"],
            })
            yield skill
    
    def test_self_issued_mode_initialization(self, self_issued_skill, mock_agent):
        """Test initialization in self-issued mode."""
        from webagents.agents.skills.local.auth import AuthMode
        
        assert self_issued_skill.aoauth_config.mode == AuthMode.SELF_ISSUED
        assert self_issued_skill.aoauth_config.authority is None
    
    def test_portal_mode_initialization(self, portal_skill, mock_agent):
        """Test initialization in portal mode."""
        from webagents.agents.skills.local.auth import AuthMode
        
        assert portal_skill.aoauth_config.mode == AuthMode.PORTAL
        assert portal_skill.aoauth_config.authority == "https://robutler.ai"
    
    @pytest.mark.asyncio
    async def test_initialize_sets_up_keys_self_issued(self, mock_agent):
        """Initialize should generate keys in self-issued mode."""
        from webagents.agents.skills.local.auth import AuthSkill
        
        with tempfile.TemporaryDirectory() as tmpdir:
            skill = AuthSkill({
                "agent_id": "test-agent",
                "base_url": "https://myagent.example.com",
                "keys_dir": tmpdir,
            })
            
            # Mock the provider imports
            with patch.object(skill, '_init_providers'):
                with patch.object(skill, '_init_grants'):
                    await skill.initialize(mock_agent)
            
            assert skill._kid is not None
            assert skill._issuer == "https://myagent.example.com"
    
    def test_scope_allowed_exact_match(self):
        """Test scope filtering with exact match."""
        from webagents.agents.skills.local.auth import AuthSkill
        
        skill = AuthSkill({
            "allowed_scopes": ["read", "write"]
        })
        
        assert skill._scope_allowed("read") == True
        assert skill._scope_allowed("write") == True
        assert skill._scope_allowed("delete") == False
    
    def test_scope_allowed_wildcard_match(self):
        """Test scope filtering with wildcard patterns."""
        from webagents.agents.skills.local.auth import AuthSkill
        
        skill = AuthSkill({
            "allowed_scopes": ["read", "namespace:*"]
        })
        
        assert skill._scope_allowed("read") == True
        assert skill._scope_allowed("namespace:admin") == True
        assert skill._scope_allowed("namespace:users") == True
        assert skill._scope_allowed("other:scope") == False


# ===== Allow/Deny List Tests =====

class TestAllowDenyLists:
    """Test allow/deny list functionality."""
    
    def test_deny_list_blocks_matching_issuer(self):
        """Deny list should block matching issuers."""
        from webagents.agents.skills.local.auth import AuthSkill
        
        skill = AuthSkill({
            "deny": ["@malicious-*", "evil-agent"]
        })
        
        assert skill._is_allowed("malicious-bot") == False
        assert skill._is_allowed("https://example.com/agents/malicious-hacker") == False
        assert skill._is_allowed("evil-agent") == False
    
    def test_allow_list_permits_matching_issuer(self):
        """Allow list should permit matching issuers."""
        from webagents.agents.skills.local.auth import AuthSkill
        
        skill = AuthSkill({
            "allow": ["@trusted-*", "partner-agent"]
        })
        
        assert skill._is_allowed("trusted-helper") == True
        assert skill._is_allowed("https://example.com/agents/trusted-bot") == True
        assert skill._is_allowed("partner-agent") == True
        assert skill._is_allowed("random-agent") == False
    
    def test_empty_allow_list_allows_all_non_denied(self):
        """Empty allow list should allow all non-denied issuers."""
        from webagents.agents.skills.local.auth import AuthSkill
        
        skill = AuthSkill({
            "allow": [],  # Empty allow list
            "deny": ["blocked-agent"]
        })
        
        assert skill._is_allowed("any-agent") == True
        assert skill._is_allowed("blocked-agent") == False
    
    def test_deny_takes_precedence_over_allow(self):
        """Deny list should take precedence over allow list."""
        from webagents.agents.skills.local.auth import AuthSkill
        
        skill = AuthSkill({
            "allow": ["@trusted-*"],
            "deny": ["trusted-but-blocked"]
        })
        
        # This agent matches allow pattern but is explicitly denied
        assert skill._is_allowed("trusted-but-blocked") == False
        # This agent matches allow pattern and is not denied
        assert skill._is_allowed("trusted-helper") == True
    
    def test_fnmatch_pattern_matching(self):
        """Test fnmatch pattern matching for issuers."""
        from webagents.agents.skills.local.auth import AuthSkill
        
        skill = AuthSkill({
            "allow": ["*-official", "partner-*-v?"]
        })
        
        assert skill._is_allowed("google-official") == True
        assert skill._is_allowed("microsoft-official") == True
        assert skill._is_allowed("partner-api-v1") == True
        assert skill._is_allowed("partner-api-v2") == True
        assert skill._is_allowed("partner-api-v10") == False  # ? matches single char


# ===== Token Generation Tests =====

class TestTokenGeneration:
    """Test self-issued token generation."""
    
    @pytest.mark.asyncio
    async def test_generate_self_issued_token(self):
        """Test generating a self-issued JWT token."""
        from webagents.agents.skills.local.auth import AuthSkill
        
        with tempfile.TemporaryDirectory() as tmpdir:
            skill = AuthSkill({
                "agent_id": "test-agent",
                "base_url": "https://myagent.example.com",
                "keys_dir": tmpdir,
                "allowed_scopes": ["read", "write"],
            })
            
            # Initialize to set up keys
            mock_agent = Mock()
            mock_agent.name = "test-agent"
            mock_agent.id = "test-agent"
            
            with patch.object(skill, '_init_providers'):
                with patch.object(skill, '_init_grants'):
                    await skill.initialize(mock_agent)
            
            # Generate token
            token = skill._generate_self_issued_token(
                target="@target-agent",
                scopes=["read"]
            )
            
            assert token is not None
            
            # Decode without verification to check claims
            claims = jwt.decode(token, options={"verify_signature": False})
            
            assert claims["iss"] == "https://myagent.example.com"
            assert claims["sub"] == "test-agent"
            assert claims["aud"] == "https://robutler.ai/agents/target-agent"
            assert "read" in claims["scope"]
            assert claims["aoauth"]["mode"] == "self"


# ===== AuthError Tests =====

class TestAuthError:
    """Test AuthError exception."""
    
    def test_auth_error_with_message_and_code(self):
        """Test AuthError with custom message and code."""
        from webagents.agents.skills.local.auth.skill import AuthError
        
        error = AuthError("Invalid token", "invalid_token")
        
        assert str(error) == "Invalid token"
        assert error.message == "Invalid token"
        assert error.code == "invalid_token"
    
    def test_auth_error_default_code(self):
        """Test AuthError with default code."""
        from webagents.agents.skills.local.auth.skill import AuthError
        
        error = AuthError("Something went wrong")
        
        assert error.code == "auth_error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
