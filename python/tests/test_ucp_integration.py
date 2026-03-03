"""
UCP Integration Tests

Tests UCP skill with real agents served by WebAgentsServer.
No external server required - tests run self-contained.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.ecosystem.ucp import UCPSkill
from webagents.server.core.app import WebAgentsServer


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def merchant_agent():
    """Create a merchant agent with UCP server mode."""
    ucp_skill = UCPSkill(config={
        "mode": "server",
        "agent_description": "Test merchant selling analysis services",
        "accepted_handlers": ["ai.robutler.token"],
        "services": [
            {"id": "basic", "title": "Basic Analysis", "price": 500},
            {"id": "premium", "title": "Premium Analysis", "price": 2000},
        ]
    })
    
    return BaseAgent(
        name="test-merchant",
        instructions="You sell analysis services.",
        skills={"UCPSkill": ucp_skill},
        scopes=["all"]
    )


@pytest.fixture
def client_agent():
    """Create a client agent with UCP client mode."""
    ucp_skill = UCPSkill(config={
        "mode": "client",
        "enabled_handlers": ["ai.robutler.token"],
    })
    
    return BaseAgent(
        name="test-client",
        instructions="You buy services from merchants.",
        skills={"UCPSkill": ucp_skill},
        scopes=["all"]
    )


# =============================================================================
# Discovery Tests
# =============================================================================

class TestMerchantDiscovery:
    """Test UCP merchant discovery."""
    
    @pytest.mark.asyncio
    async def test_merchant_exposes_ucp_profile(self, merchant_agent):
        """Merchant should expose valid UCP profile."""
        await merchant_agent._ensure_skills_initialized()
        
        ucp_skill = merchant_agent.skills.get("UCPSkill")
        profile = await ucp_skill.serve_ucp_profile()
        
        assert "ucp" in profile
        assert "payment" in profile
        assert profile["ucp"]["version"] == "2026-01-11"
    
    @pytest.mark.asyncio
    async def test_merchant_lists_services(self, merchant_agent):
        """Merchant should list available services."""
        await merchant_agent._ensure_skills_initialized()
        
        ucp_skill = merchant_agent.skills.get("UCPSkill")
        catalog = await ucp_skill.serve_catalog()
        
        assert "services" in catalog
        assert len(catalog["services"]) == 2
        assert any(s["id"] == "basic" for s in catalog["services"])


# =============================================================================
# Checkout Tests
# =============================================================================

class TestCheckoutFlow:
    """Test UCP checkout flow."""
    
    @pytest.mark.asyncio
    async def test_create_checkout_session(self, merchant_agent):
        """Create a checkout session."""
        await merchant_agent._ensure_skills_initialized()
        
        ucp_skill = merchant_agent.skills.get("UCPSkill")
        
        request = {
            "line_items": [{"item": {"id": "basic"}, "quantity": 1}],
            "buyer": {"email": "test@example.com"}
        }
        
        result = await ucp_skill.serve_create_checkout(request)
        
        assert "id" in result
        assert result["status"] == "ready_for_complete"
    
    @pytest.mark.asyncio
    async def test_complete_checkout(self, merchant_agent):
        """Complete a checkout with payment."""
        await merchant_agent._ensure_skills_initialized()
        
        ucp_skill = merchant_agent.skills.get("UCPSkill")
        
        # Create checkout
        checkout = await ucp_skill.serve_create_checkout({
            "line_items": [{"item": {"id": "basic"}, "quantity": 1}],
            "buyer": {"email": "test@example.com"}
        })
        
        # Mock the _verify_payment to always return True for testing
        async def mock_verify_payment(instruments, amount, currency):
            return True
        
        # Complete with payment - use robutler handler which has fallback verification
        with patch.object(ucp_skill, '_verify_payment', mock_verify_payment):
            result = await ucp_skill.serve_complete_checkout(
                checkout["id"],
                {"payment": {"instruments": [{"handler_id": "ai.robutler.token", "type": "token", "data": {"token": "test_token"}}]}}
            )
        
        assert result.get("success") is True or "order_id" in result


# =============================================================================
# Server Integration Tests
# =============================================================================

class TestServerIntegration:
    """Test UCP with WebAgentsServer."""
    
    @pytest.mark.asyncio
    async def test_server_hosts_merchant(self, merchant_agent):
        """Server should host merchant with UCP endpoints."""
        await merchant_agent._ensure_skills_initialized()
        
        server = WebAgentsServer(
            agents=[merchant_agent],
            title="UCP Test Server"
        )
        
        assert server.app is not None
        
        # Check routes are registered
        routes = [str(r.path) for r in server.app.routes]
        assert any("test-merchant" in r for r in routes)
    
    @pytest.mark.asyncio
    async def test_multi_agent_commerce(self, merchant_agent, client_agent):
        """Test commerce between merchant and client agents."""
        await merchant_agent._ensure_skills_initialized()
        await client_agent._ensure_skills_initialized()
        
        # Both agents should be functional
        merchant_skill = merchant_agent.skills.get("UCPSkill")
        client_skill = client_agent.skills.get("UCPSkill")
        
        assert merchant_skill.server_enabled is True
        assert client_skill.client_enabled is True
        
        # Merchant provides services
        services = merchant_skill.server.get_services()
        assert len(services) == 2


# =============================================================================
# Full Purchase Flow
# =============================================================================

class TestFullPurchaseFlow:
    """Test complete purchase flow."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_purchase(self, merchant_agent):
        """Test end-to-end purchase flow."""
        await merchant_agent._ensure_skills_initialized()
        
        ucp_skill = merchant_agent.skills.get("UCPSkill")
        server = ucp_skill.server
        
        # 1. Get services
        services = server.get_services()
        assert len(services) >= 1
        
        # 2. Create checkout
        checkout = server.create_checkout(
            items=[{"id": "basic", "quantity": 1}],
            buyer={"email": "buyer@test.com"}
        )
        assert checkout.id is not None
        
        # 3. Complete with payment - no verify_payment_func means no verification
        result = await server.complete_checkout(
            checkout_id=checkout.id,
            payment_instruments=[{"handler_id": "ai.robutler.token", "type": "token", "data": {"token": "test_token"}}],
            verify_payment_func=None  # Skip payment verification for testing
        )
        
        assert result["success"] is True
        assert "order_id" in result
        
        # 4. Verify order
        order = server.get_order(result["order_id"])
        assert order["status"] == "completed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
