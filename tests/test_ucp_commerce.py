"""
UCP Commerce Integration Tests

Tests agent-to-agent commerce with real merchant and client agents.
No external UCP server required - uses WebAgentsServer internally.
"""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import patch, MagicMock

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.ecosystem.ucp import UCPSkill, UCPClient, UCPServer
from webagents.agents.tools.decorators import tool


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def merchant_agent():
    """Create a merchant agent with UCP server mode."""
    # Create UCP skill first
    ucp_skill = UCPSkill(config={
        "mode": "server",
        "agent_description": "Test merchant agent",
        "accepted_handlers": ["ai.robutler.token"],
        "services": [
            {"id": "svc_basic", "title": "Basic Service", "price": 500},
            {"id": "svc_premium", "title": "Premium Service", "price": 2000},
        ]
    })
    
    # Pass skill via constructor
    agent = BaseAgent(
        name="test-merchant",
        instructions="You are a merchant that sells services.",
        model="openai/gpt-4o-mini",
        skills={"UCPSkill": ucp_skill},
        scopes=["all"]
    )
    
    return agent


@pytest.fixture
def client_agent():
    """Create a client agent with UCP client mode."""
    ucp_skill = UCPSkill(config={
        "mode": "client",
        "enabled_handlers": ["ai.robutler.token"],
    })
    
    agent = BaseAgent(
        name="test-client",
        instructions="You are a client that buys services.",
        model="openai/gpt-4o-mini",
        skills={"UCPSkill": ucp_skill},
        scopes=["all"]
    )
    
    return agent


@pytest.fixture
def both_mode_agent():
    """Create an agent that can both buy and sell."""
    ucp_skill = UCPSkill(config={
        "mode": "both",
        "enabled_handlers": ["ai.robutler.token"],
        "services": [{"id": "my_service", "title": "My Service", "price": 1000}]
    })
    
    agent = BaseAgent(
        name="test-both",
        instructions="You can buy and sell services.",
        model="openai/gpt-4o-mini",
        skills={"UCPSkill": ucp_skill},
        scopes=["all"]
    )
    
    return agent


# =============================================================================
# Agent Initialization Tests
# =============================================================================

class TestAgentInitialization:
    """Test agent initialization with UCP skills."""
    
    @pytest.mark.asyncio
    async def test_merchant_agent_initialization(self, merchant_agent):
        """Test merchant agent initializes with server mode."""
        # Initialize skills (required for server/client to be set up)
        with patch("webagents.utils.logging.get_logger"):
            with patch("webagents.utils.logging.log_skill_event"):
                await merchant_agent._ensure_skills_initialized()
        
        ucp_skill = merchant_agent.skills.get("UCPSkill")
        assert ucp_skill is not None
        assert ucp_skill.server_enabled is True
        assert ucp_skill.client_enabled is False
        assert ucp_skill.server is not None
        
        # Check services registered
        services = ucp_skill.server.get_services()
        assert len(services) == 2
        assert any(s.id == "svc_basic" for s in services)
        assert any(s.id == "svc_premium" for s in services)
    
    @pytest.mark.asyncio
    async def test_client_agent_initialization(self, client_agent):
        """Test client agent initializes with client mode."""
        # Initialize skills (required for server/client to be set up)
        with patch("webagents.utils.logging.get_logger"):
            with patch("webagents.utils.logging.log_skill_event"):
                await client_agent._ensure_skills_initialized()
        
        ucp_skill = client_agent.skills.get("UCPSkill")
        assert ucp_skill is not None
        assert ucp_skill.server_enabled is False
        assert ucp_skill.client_enabled is True
        assert ucp_skill.client is not None
    
    @pytest.mark.asyncio
    async def test_both_mode_agent_initialization(self, both_mode_agent):
        """Test agent initializes with both modes."""
        # Initialize skills (required for server/client to be set up)
        with patch("webagents.utils.logging.get_logger"):
            with patch("webagents.utils.logging.log_skill_event"):
                await both_mode_agent._ensure_skills_initialized()
        
        ucp_skill = both_mode_agent.skills.get("UCPSkill")
        assert ucp_skill is not None
        assert ucp_skill.server_enabled is True
        assert ucp_skill.client_enabled is True


# =============================================================================
# Merchant Server Tests
# =============================================================================

class TestMerchantServer:
    """Test merchant agent's UCP server functionality."""
    
    @pytest.mark.asyncio
    async def test_merchant_ucp_profile(self, merchant_agent):
        """Test merchant exposes valid UCP profile."""
        # Initialize skills (required for server/client to be set up)
        with patch("webagents.utils.logging.get_logger"):
            with patch("webagents.utils.logging.log_skill_event"):
                await merchant_agent._ensure_skills_initialized()
        
        ucp_skill = merchant_agent.skills.get("UCPSkill")
        
        # Call the HTTP endpoint handler directly
        profile = await ucp_skill.serve_ucp_profile()
        
        assert "ucp" in profile
        assert "payment" in profile
        assert profile["ucp"]["version"] == "2026-01-11"
        
        # Check payment handlers
        handlers = profile["payment"]["handlers"]
        assert len(handlers) > 0
        assert any(h["name"] == "ai.robutler.token" for h in handlers)
    
    @pytest.mark.asyncio
    async def test_merchant_list_services(self, merchant_agent):
        """Test merchant can list services via HTTP endpoint."""
        # Initialize skills (required for server/client to be set up)
        with patch("webagents.utils.logging.get_logger"):
            with patch("webagents.utils.logging.log_skill_event"):
                await merchant_agent._ensure_skills_initialized()
        
        ucp_skill = merchant_agent.skills.get("UCPSkill")
        
        # Call catalog endpoint
        result = await ucp_skill.serve_catalog()
        
        assert "services" in result
        assert len(result["services"]) == 2
        assert result["services"][0]["price"] in [500, 2000]
    
    @pytest.mark.asyncio
    async def test_merchant_create_checkout(self, merchant_agent):
        """Test merchant can create checkout session."""
        # Initialize skills (required for server/client to be set up)
        with patch("webagents.utils.logging.get_logger"):
            with patch("webagents.utils.logging.log_skill_event"):
                await merchant_agent._ensure_skills_initialized()
        
        ucp_skill = merchant_agent.skills.get("UCPSkill")
        
        # Create checkout via HTTP endpoint
        request_data = {
            "line_items": [{"item": {"id": "svc_basic"}, "quantity": 1}],
            "buyer": {"email": "test@example.com", "full_name": "Test User"},
            "currency": "USD"
        }
        
        result = await ucp_skill.serve_create_checkout(request_data)
        
        assert "id" in result
        assert result["status"] == "ready_for_complete"
        assert len(result["line_items"]) == 1
    
    @pytest.mark.asyncio
    async def test_merchant_complete_checkout(self, merchant_agent):
        """Test merchant can complete checkout with payment."""
        # Initialize skills (required for server/client to be set up)
        with patch("webagents.utils.logging.get_logger"):
            with patch("webagents.utils.logging.log_skill_event"):
                await merchant_agent._ensure_skills_initialized()
        
        ucp_skill = merchant_agent.skills.get("UCPSkill")
        
        # Create checkout
        create_request = {
            "line_items": [{"item": {"id": "svc_basic"}, "quantity": 1}],
            "buyer": {"email": "test@example.com"}
        }
        checkout = await ucp_skill.serve_create_checkout(create_request)
        checkout_id = checkout["id"]
        
        # Complete checkout
        complete_request = {
            "payment": {
                "instruments": [{
                    "handler_id": "robutler_token",
                    "type": "token",
                    "data": {"token": "test_token:test_secret"}
                }]
            }
        }
        result = await ucp_skill.serve_complete_checkout(checkout_id, complete_request)
        
        assert result.get("success") is True or result.get("checkout", {}).get("status") == "completed"


# =============================================================================
# Client Tests
# =============================================================================

class TestClientAgent:
    """Test client agent's UCP client functionality."""
    
    @pytest.mark.asyncio
    async def test_client_list_handlers(self, client_agent):
        """Test client can list available payment handlers."""
        # Initialize skills (required for server/client to be set up)
        with patch("webagents.utils.logging.get_logger"):
            with patch("webagents.utils.logging.log_skill_event"):
                await client_agent._ensure_skills_initialized()
        
        ucp_skill = client_agent.skills.get("UCPSkill")
        
        result = await ucp_skill.list_payment_handlers()
        
        assert result["success"] is True
        assert len(result["handlers"]) > 0


# =============================================================================
# Full Commerce Flow Tests
# =============================================================================

class TestCommerceFlow:
    """Test complete commerce flow between agents."""
    
    @pytest.mark.asyncio
    async def test_direct_server_checkout_flow(self, merchant_agent):
        """Test complete checkout flow using server directly."""
        # Initialize skills (required for server/client to be set up)
        with patch("webagents.utils.logging.get_logger"):
            with patch("webagents.utils.logging.log_skill_event"):
                await merchant_agent._ensure_skills_initialized()
        
        ucp_skill = merchant_agent.skills.get("UCPSkill")
        server = ucp_skill.server
        
        # 1. Get profile
        profile = server.build_profile()
        assert "payment" in profile
        
        # 2. Get services
        services = server.get_services()
        assert len(services) == 2
        
        # 3. Create checkout
        checkout = server.create_checkout(
            items=[{"id": "svc_basic", "quantity": 2}],
            buyer={"email": "buyer@example.com", "full_name": "Buyer"},
            currency="USD"
        )
        assert checkout.id is not None
        
        # Verify total (500 * 2 = 1000)
        total = next(t.amount for t in checkout.totals if t.type.value == "total")
        assert total == 1000
        
        # 4. Complete with payment
        result = await server.complete_checkout(
            checkout_id=checkout.id,
            payment_instruments=[{
                "handler_id": "robutler_token",
                "type": "token",
                "data": {"token": "valid:token"}
            }]
        )
        
        assert result["success"] is True
        assert "order_id" in result
        
        # 5. Verify order
        order = server.get_order(result["order_id"])
        assert order is not None
        assert order["total"] == 1000
        assert order["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_merchant_orders_tracking(self, merchant_agent):
        """Test merchant can track orders."""
        # Initialize skills (required for server/client to be set up)
        with patch("webagents.utils.logging.get_logger"):
            with patch("webagents.utils.logging.log_skill_event"):
                await merchant_agent._ensure_skills_initialized()
        
        ucp_skill = merchant_agent.skills.get("UCPSkill")
        
        # Complete a purchase
        server = ucp_skill.server
        checkout = server.create_checkout(
            items=[{"id": "svc_premium", "quantity": 1}],
            buyer={"email": "test@example.com"}
        )
        await server.complete_checkout(
            checkout_id=checkout.id,
            payment_instruments=[{"handler_id": "test", "type": "token", "data": {}}]
        )
        
        # List orders via tool
        result = await ucp_skill.list_orders()
        
        assert result["success"] is True
        assert len(result["orders"]) >= 1
        assert result["orders"][-1]["total"] == 2000


# =============================================================================
# Tool Registration Tests
# =============================================================================

class TestToolRegistration:
    """Test that UCP skill registers tools correctly."""
    
    @pytest.mark.asyncio
    async def test_client_tools_registered(self, client_agent):
        """Test client mode registers discovery and checkout tools."""
        # Initialize skills (required for server/client to be set up)
        with patch("webagents.utils.logging.get_logger"):
            with patch("webagents.utils.logging.log_skill_event"):
                await client_agent._ensure_skills_initialized()
        
        # Get registered tools
        tools = client_agent.get_all_tools()
        tool_names = [t["name"] for t in tools]
        
        # Client tools should be registered
        assert "discover_merchant" in tool_names
        assert "create_checkout" in tool_names
        assert "complete_purchase" in tool_names
        assert "get_checkout_status" in tool_names
        assert "list_payment_handlers" in tool_names
    
    @pytest.mark.asyncio
    async def test_server_tools_registered(self, merchant_agent):
        """Test server mode registers service management tools."""
        # Initialize skills (required for server/client to be set up)
        with patch("webagents.utils.logging.get_logger"):
            with patch("webagents.utils.logging.log_skill_event"):
                await merchant_agent._ensure_skills_initialized()
        
        tools = merchant_agent.get_all_tools()
        tool_names = [t["name"] for t in tools]
        
        # Server tools should be registered
        assert "list_services" in tool_names
        assert "list_orders" in tool_names
        assert "register_service" in tool_names


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling in commerce flow."""
    
    @pytest.mark.asyncio
    async def test_complete_nonexistent_checkout(self, merchant_agent):
        """Test completing non-existent checkout returns error."""
        # Initialize skills (required for server/client to be set up)
        with patch("webagents.utils.logging.get_logger"):
            with patch("webagents.utils.logging.log_skill_event"):
                await merchant_agent._ensure_skills_initialized()
        
        ucp_skill = merchant_agent.skills.get("UCPSkill")
        
        result = await ucp_skill.serve_complete_checkout(
            "nonexistent_checkout",
            {"payment": {"instruments": []}}
        )
        
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_server_mode_required_for_services(self, client_agent):
        """Test list_services fails in client-only mode."""
        # Initialize skills (required for server/client to be set up)
        with patch("webagents.utils.logging.get_logger"):
            with patch("webagents.utils.logging.log_skill_event"):
                await client_agent._ensure_skills_initialized()
        
        ucp_skill = client_agent.skills.get("UCPSkill")
        
        result = await ucp_skill.list_services()
        
        assert result["success"] is False
        assert "not enabled" in result.get("error", "").lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
