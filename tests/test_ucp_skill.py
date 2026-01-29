"""
UCP Skill Unit Tests

Tests for the Universal Commerce Protocol skill including:
- Discovery functionality
- Checkout session management
- Payment handlers
- Schema validation
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

# Import UCP components
from webagents.agents.skills.ecosystem.ucp import (
    UCPSkill,
    UCPClient,
    UCPError,
    UCPDiscoveryError,
    UCPCheckoutError,
    UCPPaymentError,
)
from webagents.agents.skills.ecosystem.ucp.discovery import UCPDiscovery
from webagents.agents.skills.ecosystem.ucp.schemas import (
    CheckoutSession,
    CheckoutStatus,
    MerchantProfile,
    UCPProfile,
    PaymentProfile,
    Capability,
    PaymentHandlerConfig,
    LineItem,
    LineItemProduct,
    Total,
    TotalType,
    create_line_item,
    create_checkout_request,
)
from webagents.agents.skills.ecosystem.ucp.handlers import (
    PaymentHandler,
    PaymentInstrument,
    PaymentResult,
    PaymentStatus,
    RobutlerHandler,
    StripeHandler,
    GooglePayHandler,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_merchant_profile_data() -> Dict[str, Any]:
    """Sample merchant profile response"""
    return {
        "ucp": {
            "version": "2026-01-11",
            "services": {
                "dev.ucp.shopping": {
                    "version": "2026-01-11",
                    "spec": "https://ucp.dev/specs/shopping",
                    "rest": {
                        "schema": "https://ucp.dev/services/shopping/openapi.json",
                        "endpoint": "http://localhost:8182/"
                    }
                }
            },
            "capabilities": [
                {
                    "name": "dev.ucp.shopping.checkout",
                    "version": "2026-01-11",
                    "spec": "https://ucp.dev/specs/shopping/checkout"
                },
                {
                    "name": "dev.ucp.shopping.discount",
                    "version": "2026-01-11",
                    "extends": "dev.ucp.shopping.checkout"
                }
            ]
        },
        "payment": {
            "handlers": [
                {
                    "id": "robutler_token",
                    "name": "ai.robutler.token",
                    "version": "2026-01-11",
                    "config": {}
                },
                {
                    "id": "google_pay",
                    "name": "google.pay",
                    "version": "2026-01-11",
                    "config": {
                        "merchant_id": "TEST",
                        "merchant_name": "Test Shop"
                    }
                }
            ]
        }
    }


@pytest.fixture
def sample_checkout_response() -> Dict[str, Any]:
    """Sample checkout session response"""
    return {
        "id": "checkout_123",
        "status": "ready_for_complete",
        "currency": "USD",
        "line_items": [
            {
                "id": "li_1",
                "item": {
                    "id": "product_1",
                    "title": "Test Product",
                    "price": 1000
                },
                "quantity": 2,
                "totals": [
                    {"type": "subtotal", "amount": 2000},
                    {"type": "total", "amount": 2000}
                ]
            }
        ],
        "totals": [
            {"type": "subtotal", "amount": 2000},
            {"type": "total", "amount": 2000}
        ],
        "buyer": {
            "email": "test@example.com",
            "full_name": "Test User"
        }
    }


@pytest.fixture
def mock_agent():
    """Mock agent for skill initialization"""
    agent = MagicMock()
    agent.name = "test-agent"
    return agent


# =============================================================================
# Schema Tests
# =============================================================================

class TestSchemas:
    """Test Pydantic schema models"""
    
    def test_create_line_item(self):
        """Test line item creation helper"""
        item = create_line_item(
            product_id="prod_123",
            title="Widget",
            quantity=3,
            price=500
        )
        
        assert item.item.id == "prod_123"
        assert item.item.title == "Widget"
        assert item.quantity == 3
        assert item.item.price == 500
    
    def test_checkout_status_enum(self):
        """Test checkout status enum values"""
        assert CheckoutStatus.INCOMPLETE.value == "incomplete"
        assert CheckoutStatus.READY_FOR_COMPLETE.value == "ready_for_complete"
        assert CheckoutStatus.COMPLETED.value == "completed"
    
    def test_create_checkout_request(self):
        """Test checkout request creation helper"""
        request = create_checkout_request(
            items=[
                {"id": "p1", "title": "Item 1", "quantity": 1},
                {"id": "p2", "name": "Item 2", "quantity": 2, "price": 1500}
            ],
            buyer_email="test@example.com",
            currency="EUR"
        )
        
        assert len(request.line_items) == 2
        assert request.buyer.email == "test@example.com"
        assert request.currency == "EUR"
    
    def test_total_type_enum(self):
        """Test total type enum"""
        total = Total(type=TotalType.SUBTOTAL, amount=1000)
        assert total.type == TotalType.SUBTOTAL
        assert total.amount == 1000


# =============================================================================
# Discovery Tests
# =============================================================================

class TestDiscovery:
    """Test UCP discovery functionality"""
    
    @pytest.mark.asyncio
    async def test_normalize_url(self):
        """Test URL normalization"""
        discovery = UCPDiscovery()
        
        assert discovery._normalize_url("example.com") == "https://example.com"
        assert discovery._normalize_url("http://example.com/path") == "http://example.com"
        assert discovery._normalize_url("https://api.example.com:8080") == "https://api.example.com:8080"
    
    def test_parse_profile(self, sample_merchant_profile_data):
        """Test profile parsing"""
        discovery = UCPDiscovery()
        profile = discovery._parse_profile(sample_merchant_profile_data)
        
        assert isinstance(profile, MerchantProfile)
        assert len(profile.ucp.capabilities) == 2
        assert profile.payment is not None
        assert len(profile.payment.handlers) == 2
    
    def test_negotiate_capabilities(self, sample_merchant_profile_data):
        """Test capability negotiation"""
        discovery = UCPDiscovery()
        profile = discovery._parse_profile(sample_merchant_profile_data)
        
        result = discovery.negotiate_capabilities(profile)
        
        assert result["can_transact"] == True
        assert "dev.ucp.shopping.checkout" in result["supported_capabilities"]
        assert len(result["supported_handlers"]) > 0
    
    def test_get_checkout_endpoint(self, sample_merchant_profile_data):
        """Test checkout endpoint extraction"""
        discovery = UCPDiscovery()
        profile = discovery._parse_profile(sample_merchant_profile_data)
        
        endpoint = discovery.get_checkout_endpoint(profile)
        assert endpoint == "http://localhost:8182/"


# =============================================================================
# Handler Tests
# =============================================================================

class TestRobutlerHandler:
    """Test Robutler payment handler"""
    
    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test handler initialization"""
        handler = RobutlerHandler()
        await handler.initialize()
        assert handler._initialized == True
    
    @pytest.mark.asyncio
    async def test_create_instrument_with_token(self):
        """Test instrument creation with token"""
        handler = RobutlerHandler()
        await handler.initialize()
        
        instrument = await handler.create_instrument(
            credentials={"token": "tok_123:secret_456"},
            handler_config={"id": "robutler_token"}
        )
        
        assert instrument.type == "token"
        assert instrument.handler_id == "robutler_token"
        assert instrument.data["token"] == "tok_123:secret_456"
    
    @pytest.mark.asyncio
    async def test_create_instrument_missing_token(self):
        """Test instrument creation without token raises error"""
        handler = RobutlerHandler()
        await handler.initialize()
        
        with pytest.raises(UCPPaymentError) as exc_info:
            await handler.create_instrument(
                credentials={},
                handler_config={"id": "robutler_token"}
            )
        
        assert "No Robutler payment token" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_invalid_token_format(self):
        """Test invalid token format raises error"""
        handler = RobutlerHandler()
        await handler.initialize()
        
        with pytest.raises(UCPPaymentError) as exc_info:
            await handler.create_instrument(
                credentials={"token": "invalid_no_colon"},
                handler_config={"id": "robutler_token"}
            )
        
        assert "Invalid token format" in str(exc_info.value)
    
    def test_can_handle(self):
        """Test handler matching"""
        handler = RobutlerHandler()
        
        assert handler.can_handle({"name": "ai.robutler.token"}) == True
        assert handler.can_handle({"name": "google.pay"}) == False


class TestStripeHandler:
    """Test Stripe payment handler"""
    
    @pytest.mark.asyncio
    async def test_initialize_without_key(self):
        """Test initialization without API key"""
        handler = StripeHandler()
        await handler.initialize()
        
        assert handler._initialized == True
        assert handler._stripe_configured == False
    
    @pytest.mark.asyncio
    async def test_create_instrument_from_payment_method(self):
        """Test instrument from payment method ID"""
        handler = StripeHandler()
        await handler.initialize()
        
        instrument = await handler.create_instrument(
            credentials={"payment_method_id": "pm_test_123"},
            handler_config={"id": "stripe_card"}
        )
        
        assert instrument.type == "card"
        assert instrument.data["payment_method_id"] == "pm_test_123"
    
    def test_can_handle(self):
        """Test handler matching"""
        handler = StripeHandler()
        
        assert handler.can_handle({"name": "com.stripe.payments.card"}) == True
        assert handler.can_handle({"name": "stripe"}) == True
        assert handler.can_handle({"name": "ai.robutler.token"}) == False


class TestGooglePayHandler:
    """Test Google Pay payment handler"""
    
    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test handler initialization"""
        handler = GooglePayHandler(config={
            "merchant_id": "TEST123",
            "environment": "TEST"
        })
        await handler.initialize()
        
        assert handler._initialized == True
        assert handler.merchant_id == "TEST123"
    
    @pytest.mark.asyncio
    async def test_create_instrument_from_token(self):
        """Test instrument from payment token"""
        handler = GooglePayHandler()
        await handler.initialize()
        
        instrument = await handler.create_instrument(
            credentials={"payment_token": "gpay_token_123"},
            handler_config={"id": "google_pay"}
        )
        
        assert instrument.type == "wallet"
        assert instrument.data["payment_token"] == "gpay_token_123"
    
    @pytest.mark.asyncio
    async def test_create_instrument_from_payment_data(self):
        """Test instrument from full payment data"""
        handler = GooglePayHandler()
        await handler.initialize()
        
        payment_data = {
            "paymentMethodData": {
                "tokenizationData": {
                    "type": "PAYMENT_GATEWAY",
                    "token": '{"id": "tok_123"}'
                },
                "info": {
                    "cardNetwork": "VISA",
                    "cardDetails": "1234"
                }
            }
        }
        
        instrument = await handler.create_instrument(
            credentials={"payment_data": payment_data},
            handler_config={"id": "google_pay"}
        )
        
        assert instrument.type == "wallet"
        assert instrument.data["card_network"] == "VISA"
        assert instrument.data["card_last4"] == "1234"
    
    def test_can_handle(self):
        """Test handler matching"""
        handler = GooglePayHandler()
        
        assert handler.can_handle({"name": "google.pay"}) == True
        assert handler.can_handle({"name": "googlepay"}) == True
        assert handler.can_handle({"name": "com.stripe.payments"}) == False


# =============================================================================
# Client Tests
# =============================================================================

class TestUCPClient:
    """Test UCP client functionality"""
    
    def test_get_headers(self):
        """Test request headers generation"""
        client = UCPClient()
        headers = client._get_headers()
        
        assert "Content-Type" in headers
        assert headers["Content-Type"] == "application/json"
        assert "UCP-Agent" in headers
        assert "request-id" in headers
        assert "idempotency-key" in headers
    
    def test_parse_checkout_response(self, sample_checkout_response):
        """Test checkout response parsing"""
        client = UCPClient()
        checkout = client._parse_checkout_response(sample_checkout_response)
        
        assert isinstance(checkout, CheckoutSession)
        assert checkout.id == "checkout_123"
        assert checkout.status == CheckoutStatus.READY_FOR_COMPLETE
        assert len(checkout.line_items) == 1
        assert checkout.buyer.email == "test@example.com"


# =============================================================================
# Skill Tests
# =============================================================================

class TestUCPSkill:
    """Test UCPSkill integration"""
    
    @pytest.mark.asyncio
    async def test_skill_initialization(self, mock_agent):
        """Test skill initialization"""
        skill = UCPSkill(config={
            "enabled_handlers": ["ai.robutler.token"],
            "default_currency": "EUR"
        })
        
        # Mock logging
        with patch("webagents.utils.logging.get_logger"):
            with patch("webagents.utils.logging.log_skill_event"):
                await skill.initialize(mock_agent)
        
        assert skill.agent == mock_agent
        assert skill.default_currency == "EUR"
        assert skill.discovery is not None
        assert skill.client is not None
    
    def test_format_checkout_response(self, sample_checkout_response, mock_agent):
        """Test checkout response formatting"""
        skill = UCPSkill()
        skill.agent = mock_agent
        
        # Create a checkout session
        client = UCPClient()
        checkout = client._parse_checkout_response(sample_checkout_response)
        
        result = skill._format_checkout_response(checkout)
        
        assert result["success"] == True
        assert result["checkout_id"] == "checkout_123"
        assert result["status"] == "ready_for_complete"
        assert result["total"] == 2000
        assert result["total_formatted"] == "$20.00"
        assert result["ready_for_payment"] == True
    
    def test_get_handler_display_name(self):
        """Test handler display name mapping"""
        skill = UCPSkill()
        
        assert skill._get_handler_display_name("ai.robutler.token") == "Robutler Credits"
        assert skill._get_handler_display_name("com.stripe.payments.card") == "Credit/Debit Card (Stripe)"
        assert skill._get_handler_display_name("google.pay") == "Google Pay"
        assert skill._get_handler_display_name("unknown.handler") == "unknown.handler"


# =============================================================================
# Exception Tests
# =============================================================================

class TestExceptions:
    """Test UCP exception classes"""
    
    def test_ucp_error_to_dict(self):
        """Test exception serialization"""
        error = UCPError(
            message="Test error",
            code="TEST_ERROR",
            details={"key": "value"},
            status_code=400
        )
        
        result = error.to_dict()
        
        assert result["error"] == "TEST_ERROR"
        assert result["message"] == "Test error"
        assert result["details"]["key"] == "value"
        assert result["status_code"] == 400
    
    def test_discovery_error(self):
        """Test discovery error"""
        error = UCPDiscoveryError(
            message="Discovery failed",
            merchant_url="https://example.com"
        )
        
        assert error.merchant_url == "https://example.com"
        assert error.code == "UCP_DISCOVERY_ERROR"
    
    def test_checkout_error(self):
        """Test checkout error"""
        error = UCPCheckoutError(
            message="Checkout failed",
            checkout_id="checkout_123",
            checkout_status="incomplete"
        )
        
        assert error.checkout_id == "checkout_123"
        assert error.checkout_status == "incomplete"
    
    def test_payment_error(self):
        """Test payment error"""
        error = UCPPaymentError(
            message="Payment failed",
            handler_name="stripe",
            payment_status="declined"
        )
        
        assert error.handler_name == "stripe"
        assert error.payment_status == "declined"
        assert error.status_code == 402


# =============================================================================
# Server Mode Tests
# =============================================================================

from webagents.agents.skills.ecosystem.ucp.server import UCPServer, ServiceOffering


class TestServiceOffering:
    """Test ServiceOffering dataclass"""
    
    def test_create_service(self):
        """Test creating a service offering"""
        service = ServiceOffering(
            id="test_service",
            title="Test Service",
            description="A test service",
            price=1000,
            currency="USD"
        )
        
        assert service.id == "test_service"
        assert service.price == 1000
    
    def test_to_line_item(self):
        """Test converting service to line item"""
        service = ServiceOffering(
            id="svc_1",
            title="Premium Service",
            description="Premium support",
            price=5000
        )
        
        line_item = service.to_line_item(quantity=2)
        
        assert line_item.item.id == "svc_1"
        assert line_item.item.title == "Premium Service"
        assert line_item.quantity == 2
        assert line_item.totals[0].amount == 10000  # 5000 * 2


class TestUCPServer:
    """Test UCPServer functionality"""
    
    @pytest.fixture
    def server(self):
        """Create a test server"""
        return UCPServer(
            agent_id="test_agent",
            agent_name="Test Agent",
            agent_description="A test agent for UCP",
            accepted_handlers=["ai.robutler.token", "google.pay"]
        )
    
    def test_register_service(self, server):
        """Test service registration"""
        service = server.register_service(
            service_id="svc_analysis",
            title="Data Analysis",
            description="Analyze your data",
            price=2500
        )
        
        assert service.id == "svc_analysis"
        assert service.price == 2500
        assert len(server.get_services()) == 1
    
    def test_get_service(self, server):
        """Test getting a specific service"""
        server.register_service(
            service_id="svc_1",
            title="Service 1",
            description="First service",
            price=1000
        )
        
        service = server.get_service("svc_1")
        assert service is not None
        assert service.title == "Service 1"
        
        missing = server.get_service("nonexistent")
        assert missing is None
    
    def test_build_profile(self, server):
        """Test building UCP profile"""
        server.register_service(
            service_id="svc_1",
            title="Service",
            description="Test",
            price=1000
        )
        
        profile = server.build_profile()
        
        assert "ucp" in profile
        assert profile["ucp"]["version"] == "2026-01-11"
        assert "payment" in profile
        assert len(profile["payment"]["handlers"]) == 2  # robutler and google.pay
        assert "merchant" in profile
        assert profile["merchant"]["name"] == "Test Agent"
    
    def test_create_checkout(self, server):
        """Test checkout creation"""
        # Register a service
        server.register_service(
            service_id="svc_1",
            title="Test Service",
            description="Test",
            price=1500
        )
        
        # Create checkout
        session = server.create_checkout(
            items=[{"id": "svc_1", "quantity": 2}],
            buyer={"email": "test@example.com", "full_name": "Test User"},
            currency="USD"
        )
        
        assert session.id is not None
        assert len(session.line_items) == 1
        assert session.line_items[0].quantity == 2
        assert session.buyer.email == "test@example.com"
        
        # Check totals (1500 * 2 = 3000)
        total = next(t for t in session.totals if t.type.value == "total")
        assert total.amount == 3000
    
    def test_checkout_unknown_service(self, server):
        """Test checkout with unknown service"""
        session = server.create_checkout(
            items=[{"id": "unknown", "title": "Unknown Service", "quantity": 1}],
            currency="USD"
        )
        
        assert session.id is not None
        assert len(session.line_items) == 1
        assert session.line_items[0].item.title == "Unknown Service"
    
    def test_get_checkout(self, server):
        """Test getting checkout by ID"""
        session = server.create_checkout(
            items=[{"id": "test", "title": "Test", "quantity": 1}]
        )
        
        retrieved = server.get_checkout(session.id)
        assert retrieved is not None
        assert retrieved.id == session.id
        
        missing = server.get_checkout("nonexistent")
        assert missing is None
    
    def test_update_checkout(self, server):
        """Test updating checkout"""
        session = server.create_checkout(
            items=[{"id": "test", "title": "Test", "quantity": 1}]
        )
        
        updated = server.update_checkout(
            checkout_id=session.id,
            buyer={"email": "updated@example.com", "full_name": "Updated User"}
        )
        
        assert updated.buyer.email == "updated@example.com"
    
    def test_update_nonexistent_checkout(self, server):
        """Test updating nonexistent checkout raises error"""
        with pytest.raises(UCPCheckoutError):
            server.update_checkout(
                checkout_id="nonexistent",
                buyer={"email": "test@example.com"}
            )
    
    @pytest.mark.asyncio
    async def test_complete_checkout(self, server):
        """Test completing checkout"""
        session = server.create_checkout(
            items=[{"id": "test", "title": "Test", "price": 1000, "quantity": 1}],
            buyer={"email": "test@example.com"}
        )
        
        result = await server.complete_checkout(
            checkout_id=session.id,
            payment_instruments=[
                {"handler_id": "robutler_token", "type": "token", "data": {"token": "test:secret"}}
            ]
        )
        
        assert result["success"] is True
        assert "order_id" in result
        assert result["checkout"]["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_complete_checkout_twice_fails(self, server):
        """Test completing checkout twice raises error"""
        session = server.create_checkout(
            items=[{"id": "test", "title": "Test", "price": 1000, "quantity": 1}],
            buyer={"email": "test@example.com"}
        )
        
        # First completion
        await server.complete_checkout(
            checkout_id=session.id,
            payment_instruments=[{"handler_id": "test", "type": "token", "data": {}}]
        )
        
        # Second completion should fail
        with pytest.raises(UCPCheckoutError) as exc_info:
            await server.complete_checkout(
                checkout_id=session.id,
                payment_instruments=[{"handler_id": "test", "type": "token", "data": {}}]
            )
        
        assert "already completed" in str(exc_info.value).lower()
    
    def test_to_response(self, server):
        """Test checkout session to_response"""
        session = server.create_checkout(
            items=[{"id": "svc", "title": "Service", "price": 2000, "quantity": 1}],
            buyer={"email": "test@example.com", "full_name": "Test"}
        )
        
        response = session.to_response()
        
        assert response["id"] == session.id
        assert response["status"] == "ready_for_complete"
        assert len(response["line_items"]) == 1
        assert response["buyer"]["email"] == "test@example.com"
    
    def test_list_orders(self, server):
        """Test listing orders"""
        # Initially empty
        assert len(server.list_orders()) == 0
    
    @pytest.mark.asyncio
    async def test_order_created_after_complete(self, server):
        """Test order is created after checkout completion"""
        session = server.create_checkout(
            items=[{"id": "test", "title": "Test", "price": 1000, "quantity": 1}],
            buyer={"email": "test@example.com"}
        )
        
        result = await server.complete_checkout(
            checkout_id=session.id,
            payment_instruments=[{"handler_id": "test", "type": "token", "data": {}}]
        )
        
        order_id = result["order_id"]
        order = server.get_order(order_id)
        
        assert order is not None
        assert order["status"] == "completed"
        assert order["total"] == 1000


class TestUCPSkillServerMode:
    """Test UCPSkill in server mode"""
    
    @pytest.mark.asyncio
    async def test_skill_server_mode_initialization(self, mock_agent):
        """Test skill initialization in server mode"""
        skill = UCPSkill(config={
            "mode": "server",
            "agent_description": "A merchant agent",
            "services": [
                {"id": "svc_1", "title": "Service One", "description": "First", "price": 1000},
                {"id": "svc_2", "title": "Service Two", "description": "Second", "price": 2000},
            ]
        })
        
        with patch("webagents.utils.logging.get_logger"):
            with patch("webagents.utils.logging.log_skill_event"):
                await skill.initialize(mock_agent)
        
        assert skill.server_enabled is True
        assert skill.client_enabled is False
        assert skill.server is not None
        assert len(skill.server.get_services()) == 2
    
    @pytest.mark.asyncio
    async def test_skill_both_mode_initialization(self, mock_agent):
        """Test skill initialization in both mode"""
        skill = UCPSkill(config={
            "mode": "both",
            "services": [
                {"id": "svc_1", "title": "Service", "price": 1000}
            ]
        })
        
        with patch("webagents.utils.logging.get_logger"):
            with patch("webagents.utils.logging.log_skill_event"):
                await skill.initialize(mock_agent)
        
        assert skill.server_enabled is True
        assert skill.client_enabled is True
        assert skill.server is not None
        assert skill.client is not None
    
    @pytest.mark.asyncio
    async def test_list_services_tool(self, mock_agent):
        """Test list_services tool in server mode"""
        skill = UCPSkill(config={
            "mode": "server",
            "services": [
                {"id": "svc_1", "title": "Test Service", "description": "Test", "price": 1500}
            ]
        })
        
        with patch("webagents.utils.logging.get_logger"):
            with patch("webagents.utils.logging.log_skill_event"):
                await skill.initialize(mock_agent)
        
        result = await skill.list_services()
        
        assert result["success"] is True
        assert result["count"] == 1
        assert result["services"][0]["id"] == "svc_1"
        assert result["services"][0]["price"] == 1500
    
    @pytest.mark.asyncio
    async def test_list_services_client_mode_fails(self, mock_agent):
        """Test list_services fails in client-only mode"""
        skill = UCPSkill(config={"mode": "client"})
        
        with patch("webagents.utils.logging.get_logger"):
            with patch("webagents.utils.logging.log_skill_event"):
                await skill.initialize(mock_agent)
        
        result = await skill.list_services()
        
        assert result["success"] is False
        assert "not enabled" in result["error"]


# =============================================================================
# Elaisium Handler Tests
# =============================================================================

class TestElaisiumHandler:
    """Test Elaisium VIBE payment handler."""
    
    def test_handler_instantiation(self):
        """Test handler can be created."""
        from webagents.agents.skills.ecosystem.ucp.handlers import ElaisiumHandler
        
        handler = ElaisiumHandler(config={
            "elaisium_api_url": "http://localhost:8080",
            "agent_entity_id": "agent_123"
        })
        
        assert handler is not None
        assert handler.name == "world.elaisium.vibe"
        assert handler.display_name == "Elaisium VIBE"
    
    def test_handler_capabilities(self):
        """Test handler capabilities."""
        from webagents.agents.skills.ecosystem.ucp.handlers import ElaisiumHandler
        
        handler = ElaisiumHandler(config={"agent_entity_id": "agent_123"})
        caps = handler.get_capabilities()
        
        assert caps["name"] == "world.elaisium.vibe"
        assert "VIBE" in caps["currencies"]
        assert caps["features"]["escrow"] is True
    
    @pytest.mark.asyncio
    async def test_create_instrument(self):
        """Test creating VIBE payment instrument."""
        from webagents.agents.skills.ecosystem.ucp.handlers import ElaisiumHandler
        
        handler = ElaisiumHandler(config={"agent_entity_id": "merchant_456"})
        
        instrument = await handler.create_instrument(
            credentials={"entity_id": "buyer_789", "amount": 1000, "memo": "Purchase analysis service"},
            handler_config={"namespace": "world.elaisium.vibe"}
        )
        
        assert instrument.handler_id == "world.elaisium.vibe"
        assert instrument.type == "vibe_transfer"
        assert instrument.data["payer_entity_id"] == "buyer_789"
        assert instrument.data["amount"] == 1000
    
    @pytest.mark.asyncio
    async def test_process_payment(self):
        """Test processing VIBE payment."""
        from webagents.agents.skills.ecosystem.ucp.handlers import ElaisiumHandler
        from webagents.agents.skills.ecosystem.ucp.handlers.base import PaymentStatus
        
        handler = ElaisiumHandler(config={"agent_entity_id": "merchant_456"})
        
        instrument = await handler.create_instrument(
            credentials={"entity_id": "buyer_789", "amount": 500},
            handler_config={"namespace": "world.elaisium.vibe"}
        )
        
        result = await handler.process_payment(
            instrument=instrument,
            amount=500,
            currency="VIBE"
        )
        
        assert result.success is True
        assert result.status == PaymentStatus.CAPTURED
        assert result.currency == "VIBE"
        assert "buyer_789" in result.transaction_id
    
    @pytest.mark.asyncio
    async def test_process_payment_wrong_currency(self):
        """Test payment with wrong currency fails."""
        from webagents.agents.skills.ecosystem.ucp.handlers import ElaisiumHandler
        from webagents.agents.skills.ecosystem.ucp.handlers.base import PaymentStatus
        
        handler = ElaisiumHandler(config={"agent_entity_id": "merchant"})
        
        instrument = await handler.create_instrument(
            credentials={"entity_id": "buyer", "amount": 100},
            handler_config={"namespace": "world.elaisium.vibe"}
        )
        
        result = await handler.process_payment(
            instrument=instrument,
            amount=100,
            currency="USD"  # Wrong currency
        )
        
        assert result.success is False
        assert result.status == PaymentStatus.FAILED
        assert "VIBE" in result.error_message
    
    @pytest.mark.asyncio
    async def test_verify_payment(self):
        """Test verifying VIBE payment."""
        from webagents.agents.skills.ecosystem.ucp.handlers import ElaisiumHandler
        from webagents.agents.skills.ecosystem.ucp.handlers.base import PaymentStatus
        
        handler = ElaisiumHandler(config={})
        
        result = await handler.verify_payment("vibe_tx_buyer_seller_1000")
        
        assert result.success is True
        assert result.status == PaymentStatus.CAPTURED
    
    @pytest.mark.asyncio
    async def test_refund_payment(self):
        """Test refunding VIBE payment."""
        from webagents.agents.skills.ecosystem.ucp.handlers import ElaisiumHandler
        from webagents.agents.skills.ecosystem.ucp.handlers.base import PaymentStatus
        
        handler = ElaisiumHandler(config={})
        
        result = await handler.refund_payment(
            transaction_id="vibe_tx_buyer_seller_500",
            amount=500,
            reason="Service not delivered"
        )
        
        assert result.success is True
        assert result.metadata["type"] == "refund"
    
    @pytest.mark.asyncio
    async def test_trade_artifact(self):
        """Test artifact trading."""
        from webagents.agents.skills.ecosystem.ucp.handlers import ElaisiumHandler
        
        handler = ElaisiumHandler(config={})
        
        result = await handler.trade_artifact(
            artifact_id="artifact_magic_sword_001",
            from_entity="seller_entity",
            to_entity="buyer_entity",
            vibe_price=2500
        )
        
        assert result.success is True
        assert result.metadata["type"] == "artifact_trade"
        assert result.metadata["artifact_id"] == "artifact_magic_sword_001"
        assert result.amount == 2500
    
    @pytest.mark.asyncio
    async def test_purchase_service(self):
        """Test service purchase."""
        from webagents.agents.skills.ecosystem.ucp.handlers import ElaisiumHandler
        
        handler = ElaisiumHandler(config={})
        
        result = await handler.purchase_service(
            service_type="analysis",
            provider_entity="oracle_agent",
            consumer_entity="player_123",
            vibe_price=1000,
            service_data={"query": "best path to treasure"}
        )
        
        assert result.success is True
        assert result.metadata["type"] == "service_purchase"
        assert result.metadata["service_type"] == "analysis"


class TestElaisiumInRegistry:
    """Test Elaisium handler is in registry."""
    
    def test_handler_in_registry(self):
        """Elaisium handler should be in handler registry."""
        from webagents.agents.skills.ecosystem.ucp.handlers import HANDLER_REGISTRY
        
        assert "world.elaisium.vibe" in HANDLER_REGISTRY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
