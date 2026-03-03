"""
UCP Schemas - Pydantic Models for Universal Commerce Protocol

These models represent the core UCP data structures for:
- Checkout sessions and line items
- Payment handlers and instruments
- Capability negotiation
- Merchant profiles

Based on UCP Spec v2026-01-11: https://ucp.dev/specification/overview/
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


# ============================================================================
# Enums
# ============================================================================

class CheckoutStatus(str, Enum):
    """UCP Checkout session status"""
    INCOMPLETE = "incomplete"
    REQUIRES_ESCALATION = "requires_escalation"
    READY_FOR_COMPLETE = "ready_for_complete"
    COMPLETE_IN_PROGRESS = "complete_in_progress"
    COMPLETED = "completed"
    CANCELED = "canceled"


class TotalType(str, Enum):
    """Types of totals in checkout"""
    SUBTOTAL = "subtotal"
    DISCOUNT = "discount"
    TAX = "tax"
    SHIPPING = "shipping"
    TOTAL = "total"


# ============================================================================
# Core Models
# ============================================================================

class UCPVersion(BaseModel):
    """UCP version information"""
    version: str = Field(default="2026-01-11", description="UCP spec version")


class Capability(BaseModel):
    """UCP Capability declaration"""
    name: str = Field(..., description="Capability identifier (e.g., dev.ucp.shopping.checkout)")
    version: str = Field(default="2026-01-11")
    spec: Optional[str] = Field(None, description="URL to capability specification")
    schema_url: Optional[str] = Field(None, alias="schema", description="URL to JSON schema")
    extends: Optional[str] = Field(None, description="Parent capability this extends")


class Service(BaseModel):
    """UCP Service declaration"""
    version: str = Field(default="2026-01-11")
    spec: Optional[str] = Field(None, description="URL to service specification")
    rest: Optional[Dict[str, str]] = Field(None, description="REST binding config")
    mcp: Optional[Dict[str, Any]] = Field(None, description="MCP binding config")


class Total(BaseModel):
    """Price total component"""
    type: TotalType
    amount: int = Field(..., description="Amount in smallest currency unit (cents)")


class LineItemProduct(BaseModel):
    """Product information in a line item"""
    id: str
    title: str
    price: Optional[int] = Field(None, description="Unit price in cents")
    description: Optional[str] = None
    image_url: Optional[str] = None
    sku: Optional[str] = None


class LineItem(BaseModel):
    """Checkout line item"""
    id: Optional[str] = None
    item: LineItemProduct
    quantity: int = Field(default=1, ge=1)
    totals: Optional[List[Total]] = None


class Buyer(BaseModel):
    """Buyer information"""
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class Address(BaseModel):
    """Shipping/billing address"""
    line1: Optional[str] = None
    line2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None


class Discount(BaseModel):
    """Applied discount"""
    code: str
    title: Optional[str] = None
    amount: int = Field(..., description="Discount amount in cents")
    automatic: bool = False
    allocations: Optional[List[Dict[str, Any]]] = None


class DiscountInfo(BaseModel):
    """Discount configuration"""
    codes: Optional[List[str]] = None
    applied: Optional[List[Discount]] = None


# ============================================================================
# Payment Models
# ============================================================================

class PaymentHandlerConfig(BaseModel):
    """Payment handler configuration as advertised by merchant"""
    id: str = Field(..., description="Handler identifier (unique per merchant)")
    name: str = Field(..., description="Handler namespace (e.g., google.pay)")
    version: str = Field(default="2026-01-11")
    spec: Optional[str] = Field(None, description="URL to handler specification")
    config_schema: Optional[str] = Field(None, description="URL to config schema")
    instrument_schemas: Optional[List[str]] = Field(None, description="URLs to instrument schemas")
    config: Optional[Dict[str, Any]] = Field(None, description="Handler-specific configuration")


class PaymentInstrumentData(BaseModel):
    """Payment instrument data provided by platform"""
    handler_id: str = Field(..., description="ID of handler to process this instrument")
    type: str = Field(..., description="Instrument type (e.g., 'card', 'token')")
    data: Dict[str, Any] = Field(default_factory=dict, description="Handler-specific instrument data")


class PaymentSection(BaseModel):
    """Payment section of checkout"""
    handlers: List[PaymentHandlerConfig] = Field(default_factory=list)
    instruments: List[PaymentInstrumentData] = Field(default_factory=list)


# ============================================================================
# Checkout Models
# ============================================================================

class CheckoutLink(BaseModel):
    """Checkout action link"""
    rel: str = Field(..., description="Link relation type")
    href: str = Field(..., description="URL")
    method: Optional[str] = Field("GET", description="HTTP method")


class CheckoutSession(BaseModel):
    """UCP Checkout Session"""
    ucp: Optional[UCPVersion] = None
    id: Optional[str] = None
    line_items: List[LineItem] = Field(default_factory=list)
    buyer: Optional[Buyer] = None
    currency: str = Field(default="USD")
    status: CheckoutStatus = Field(default=CheckoutStatus.INCOMPLETE)
    totals: Optional[List[Total]] = None
    links: Optional[List[CheckoutLink]] = None
    payment: Optional[PaymentSection] = None
    discounts: Optional[DiscountInfo] = None
    shipping_address: Optional[Address] = None
    billing_address: Optional[Address] = None
    
    # Escalation info
    continue_url: Optional[str] = Field(None, description="URL for user escalation")
    messages: Optional[List[Dict[str, Any]]] = Field(None, description="Status messages")


class CreateCheckoutRequest(BaseModel):
    """Request to create a checkout session"""
    line_items: List[LineItem]
    buyer: Optional[Buyer] = None
    currency: str = Field(default="USD")
    payment: Optional[PaymentSection] = None
    discounts: Optional[DiscountInfo] = None
    shipping_address: Optional[Address] = None


class UpdateCheckoutRequest(BaseModel):
    """Request to update a checkout session"""
    id: str
    line_items: Optional[List[LineItem]] = None
    buyer: Optional[Buyer] = None
    payment: Optional[PaymentSection] = None
    discounts: Optional[DiscountInfo] = None
    shipping_address: Optional[Address] = None
    billing_address: Optional[Address] = None


class CompleteCheckoutRequest(BaseModel):
    """Request to complete checkout"""
    id: str
    payment: PaymentSection


# ============================================================================
# Merchant Profile Models
# ============================================================================

class UCPProfile(BaseModel):
    """UCP section of merchant profile"""
    version: str = Field(default="2026-01-11")
    services: Dict[str, Service] = Field(default_factory=dict)
    capabilities: List[Capability] = Field(default_factory=list)


class PaymentProfile(BaseModel):
    """Payment section of merchant profile"""
    handlers: List[PaymentHandlerConfig] = Field(default_factory=list)


class MerchantProfile(BaseModel):
    """Complete merchant UCP profile from /.well-known/ucp"""
    ucp: UCPProfile
    payment: Optional[PaymentProfile] = None


# ============================================================================
# Agent Profile Models
# ============================================================================

class AgentProfile(BaseModel):
    """Agent UCP profile for capability negotiation"""
    ucp: UCPVersion = Field(default_factory=UCPVersion)
    supported_capabilities: List[str] = Field(default_factory=list)
    supported_handlers: List[str] = Field(default_factory=list)
    profile_url: Optional[str] = None


# ============================================================================
# Helper Functions
# ============================================================================

def create_line_item(
    product_id: str,
    title: str,
    quantity: int = 1,
    price: Optional[int] = None
) -> LineItem:
    """Create a line item for checkout"""
    return LineItem(
        item=LineItemProduct(id=product_id, title=title, price=price),
        quantity=quantity
    )


def create_checkout_request(
    items: List[Dict[str, Any]],
    buyer_email: Optional[str] = None,
    buyer_name: Optional[str] = None,
    currency: str = "USD"
) -> CreateCheckoutRequest:
    """Create a checkout request from simple item list"""
    line_items = [
        create_line_item(
            product_id=item.get("id", item.get("product_id")),
            title=item.get("title", item.get("name", "Product")),
            quantity=item.get("quantity", 1),
            price=item.get("price")
        )
        for item in items
    ]
    
    buyer = None
    if buyer_email or buyer_name:
        buyer = Buyer(email=buyer_email, full_name=buyer_name)
    
    return CreateCheckoutRequest(
        line_items=line_items,
        buyer=buyer,
        currency=currency
    )
