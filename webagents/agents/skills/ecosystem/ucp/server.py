"""
UCP Server - Merchant Mode for Universal Commerce Protocol

Enables agents to act as UCP merchants/servers:
- Expose /.well-known/ucp profile
- Accept checkout sessions from other agents
- Verify and settle payments
- Fulfill services after payment

Based on UCP Spec v2026-01-11: https://ucp.dev/specification/overview/
"""

import logging
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, field

from .schemas import (
    CheckoutSession,
    CheckoutStatus,
    MerchantProfile,
    UCPProfile,
    PaymentProfile,
    Capability,
    PaymentHandlerConfig,
    Service,
    LineItem,
    LineItemProduct,
    Total,
    TotalType,
    Buyer,
    PaymentSection,
    PaymentInstrumentData,
)
from .exceptions import (
    UCPCheckoutError,
    UCPPaymentError,
    UCPCapabilityNotSupported,
)

logger = logging.getLogger("webagents.skills.ucp.server")


@dataclass
class ServiceOffering:
    """A service that the agent offers for purchase"""
    id: str
    title: str
    description: str
    price: int  # Price in cents
    currency: str = "USD"
    tool_name: Optional[str] = None  # Tool to execute after purchase
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_line_item(self, quantity: int = 1) -> LineItem:
        """Convert to UCP line item"""
        return LineItem(
            id=str(uuid.uuid4()),
            item=LineItemProduct(
                id=self.id,
                title=self.title,
                description=self.description,
                price=self.price
            ),
            quantity=quantity,
            totals=[
                Total(type=TotalType.SUBTOTAL, amount=self.price * quantity),
                Total(type=TotalType.TOTAL, amount=self.price * quantity)
            ]
        )


@dataclass 
class ServerCheckoutSession:
    """Server-side checkout session"""
    id: str
    status: CheckoutStatus
    line_items: List[LineItem]
    buyer: Optional[Buyer]
    currency: str
    created_at: datetime
    updated_at: datetime
    totals: List[Total]
    payment_instruments: List[PaymentInstrumentData] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_response(self) -> Dict[str, Any]:
        """Convert to UCP checkout response"""
        return {
            "ucp": {"version": "2026-01-11", "capabilities": [
                {"name": "dev.ucp.shopping.checkout", "version": "2026-01-11"}
            ]},
            "id": self.id,
            "status": self.status.value,
            "currency": self.currency,
            "line_items": [
                {
                    "id": li.id,
                    "item": {
                        "id": li.item.id,
                        "title": li.item.title,
                        "price": li.item.price,
                        "description": li.item.description
                    },
                    "quantity": li.quantity,
                    "totals": [{"type": t.type.value, "amount": t.amount} for t in (li.totals or [])]
                }
                for li in self.line_items
            ],
            "buyer": {
                "full_name": self.buyer.full_name,
                "email": self.buyer.email
            } if self.buyer else None,
            "totals": [{"type": t.type.value, "amount": t.amount} for t in self.totals],
            "payment": {
                "handlers": [],
                "instruments": [pi.model_dump() for pi in self.payment_instruments]
            }
        }


class UCPServer:
    """
    UCP Server for merchant mode.
    
    Manages:
    - Agent's UCP profile
    - Service catalog (what the agent sells)
    - Checkout sessions
    - Payment verification
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        agent_description: str = "",
        accepted_handlers: Optional[List[str]] = None,
        base_url: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.base_url = base_url or f"https://webagents.ai/agents/{agent_id}"
        
        # Accepted payment handlers
        self.accepted_handlers = accepted_handlers or [
            "ai.robutler.token",
            "com.stripe.payments.card",
            "google.pay"
        ]
        
        # Service catalog
        self._services: Dict[str, ServiceOffering] = {}
        
        # Active checkout sessions
        self._checkouts: Dict[str, ServerCheckoutSession] = {}
        
        # Completed orders
        self._orders: Dict[str, Dict[str, Any]] = {}
    
    def register_service(
        self,
        service_id: str,
        title: str,
        description: str,
        price: int,
        currency: str = "USD",
        tool_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ServiceOffering:
        """
        Register a service that can be purchased.
        
        Args:
            service_id: Unique service identifier
            title: Display title
            description: Service description
            price: Price in cents
            currency: Currency code
            tool_name: Tool to execute after purchase
            metadata: Additional metadata
            
        Returns:
            Created ServiceOffering
        """
        service = ServiceOffering(
            id=service_id,
            title=title,
            description=description,
            price=price,
            currency=currency,
            tool_name=tool_name,
            metadata=metadata or {}
        )
        self._services[service_id] = service
        logger.info(f"Registered service: {service_id} - {title} (${price/100:.2f})")
        return service
    
    def get_services(self) -> List[ServiceOffering]:
        """Get all registered services"""
        return list(self._services.values())
    
    def get_service(self, service_id: str) -> Optional[ServiceOffering]:
        """Get a specific service"""
        return self._services.get(service_id)
    
    def build_profile(self) -> Dict[str, Any]:
        """
        Build the agent's UCP profile for /.well-known/ucp
        
        Returns:
            Complete UCP merchant profile
        """
        # Build payment handlers
        handlers = []
        
        if "ai.robutler.token" in self.accepted_handlers:
            handlers.append({
                "id": "robutler_token",
                "name": "ai.robutler.token",
                "version": "2026-01-11",
                "spec": "https://webagents.ai/specs/robutler-token",
                "config_schema": "https://webagents.ai/schemas/robutler-token/config.json",
                "instrument_schemas": ["https://webagents.ai/schemas/robutler-token/instrument.json"],
                "config": {}
            })
        
        if "com.stripe.payments.card" in self.accepted_handlers:
            handlers.append({
                "id": "stripe_card",
                "name": "com.stripe.payments.card",
                "version": "2026-01-11",
                "spec": "https://stripe.com/docs/ucp",
                "config": {}
            })
        
        if "google.pay" in self.accepted_handlers:
            handlers.append({
                "id": "google_pay",
                "name": "google.pay",
                "version": "2026-01-11",
                "spec": "https://developers.google.com/pay/api",
                "config": {
                    "merchant_name": self.agent_name,
                    "merchant_id": self.agent_id
                }
            })
        
        return {
            "ucp": {
                "version": "2026-01-11",
                "services": {
                    "dev.ucp.shopping": {
                        "version": "2026-01-11",
                        "spec": "https://ucp.dev/specs/shopping",
                        "rest": {
                            "schema": f"{self.base_url}/openapi.json",
                            "endpoint": f"{self.base_url}/"
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
                        "name": "dev.ucp.shopping.catalog",
                        "version": "2026-01-11",
                        "spec": "https://ucp.dev/specs/shopping/catalog"
                    }
                ]
            },
            "payment": {
                "handlers": handlers
            },
            "merchant": {
                "id": self.agent_id,
                "name": self.agent_name,
                "description": self.agent_description
            }
        }
    
    def create_checkout(
        self,
        items: List[Dict[str, Any]],
        buyer: Optional[Dict[str, Any]] = None,
        currency: str = "USD"
    ) -> ServerCheckoutSession:
        """
        Create a new checkout session.
        
        Args:
            items: List of items [{"id": "service_id", "quantity": 1}]
            buyer: Buyer info {"email": "...", "full_name": "..."}
            currency: Currency code
            
        Returns:
            Created checkout session
        """
        checkout_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        # Build line items from services
        line_items = []
        subtotal = 0
        
        for item_req in items:
            service_id = item_req.get("id") or item_req.get("item", {}).get("id")
            quantity = item_req.get("quantity", 1)
            
            service = self._services.get(service_id)
            if not service:
                # Unknown service - create placeholder
                line_item = LineItem(
                    id=str(uuid.uuid4()),
                    item=LineItemProduct(
                        id=service_id,
                        title=item_req.get("title", item_req.get("item", {}).get("title", "Unknown Service")),
                        price=item_req.get("price", 0)
                    ),
                    quantity=quantity
                )
            else:
                line_item = service.to_line_item(quantity)
            
            line_items.append(line_item)
            item_total = (line_item.item.price or 0) * quantity
            subtotal += item_total
        
        # Build totals
        totals = [
            Total(type=TotalType.SUBTOTAL, amount=subtotal),
            Total(type=TotalType.TOTAL, amount=subtotal)
        ]
        
        # Parse buyer
        buyer_obj = None
        if buyer:
            buyer_obj = Buyer(
                full_name=buyer.get("full_name"),
                email=buyer.get("email"),
                phone=buyer.get("phone")
            )
        
        # Determine initial status
        status = CheckoutStatus.READY_FOR_COMPLETE if line_items else CheckoutStatus.INCOMPLETE
        
        session = ServerCheckoutSession(
            id=checkout_id,
            status=status,
            line_items=line_items,
            buyer=buyer_obj,
            currency=currency,
            created_at=now,
            updated_at=now,
            totals=totals
        )
        
        self._checkouts[checkout_id] = session
        logger.info(f"Created checkout {checkout_id}: {len(line_items)} items, total ${subtotal/100:.2f}")
        
        return session
    
    def get_checkout(self, checkout_id: str) -> Optional[ServerCheckoutSession]:
        """Get a checkout session by ID"""
        return self._checkouts.get(checkout_id)
    
    def update_checkout(
        self,
        checkout_id: str,
        buyer: Optional[Dict[str, Any]] = None,
        payment_instruments: Optional[List[Dict[str, Any]]] = None
    ) -> ServerCheckoutSession:
        """
        Update a checkout session.
        
        Args:
            checkout_id: Session ID
            buyer: Updated buyer info
            payment_instruments: Payment instruments to add
            
        Returns:
            Updated session
        """
        session = self._checkouts.get(checkout_id)
        if not session:
            raise UCPCheckoutError(
                message=f"Checkout not found: {checkout_id}",
                checkout_id=checkout_id
            )
        
        if buyer:
            session.buyer = Buyer(
                full_name=buyer.get("full_name"),
                email=buyer.get("email"),
                phone=buyer.get("phone")
            )
        
        if payment_instruments:
            session.payment_instruments = [
                PaymentInstrumentData(**pi) for pi in payment_instruments
            ]
        
        session.updated_at = datetime.utcnow()
        
        # Update status
        if session.buyer and session.line_items:
            session.status = CheckoutStatus.READY_FOR_COMPLETE
        
        return session
    
    async def complete_checkout(
        self,
        checkout_id: str,
        payment_instruments: List[Dict[str, Any]],
        verify_payment_func: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Complete a checkout with payment.
        
        Args:
            checkout_id: Session ID
            payment_instruments: Payment instruments
            verify_payment_func: Optional function to verify payment
            
        Returns:
            Completion result with order info
        """
        session = self._checkouts.get(checkout_id)
        if not session:
            raise UCPCheckoutError(
                message=f"Checkout not found: {checkout_id}",
                checkout_id=checkout_id
            )
        
        if session.status == CheckoutStatus.COMPLETED:
            raise UCPCheckoutError(
                message="Checkout already completed",
                checkout_id=checkout_id,
                checkout_status="completed"
            )
        
        # Get total amount
        total_amount = 0
        for t in session.totals:
            if t.type == TotalType.TOTAL:
                total_amount = t.amount
                break
        
        # Verify payment if function provided
        if verify_payment_func and payment_instruments:
            try:
                verified = await verify_payment_func(
                    payment_instruments,
                    total_amount,
                    session.currency
                )
                if not verified:
                    raise UCPPaymentError(
                        message="Payment verification failed",
                        payment_status="verification_failed"
                    )
            except Exception as e:
                raise UCPPaymentError(
                    message=f"Payment verification error: {str(e)}",
                    payment_status="error"
                )
        
        # Mark as completed
        session.status = CheckoutStatus.COMPLETED
        session.updated_at = datetime.utcnow()
        session.payment_instruments = [
            PaymentInstrumentData(**pi) if isinstance(pi, dict) else pi
            for pi in payment_instruments
        ]
        
        # Create order record
        order_id = f"order_{uuid.uuid4().hex[:12]}"
        order = {
            "id": order_id,
            "checkout_id": checkout_id,
            "status": "completed",
            "total": total_amount,
            "currency": session.currency,
            "items": [
                {"service_id": li.item.id, "quantity": li.quantity}
                for li in session.line_items
            ],
            "buyer": {
                "email": session.buyer.email,
                "name": session.buyer.full_name
            } if session.buyer else None,
            "created_at": session.created_at.isoformat(),
            "completed_at": datetime.utcnow().isoformat()
        }
        
        self._orders[order_id] = order
        logger.info(f"Checkout {checkout_id} completed as order {order_id}")
        
        return {
            "success": True,
            "order_id": order_id,
            "checkout": session.to_response()
        }
    
    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get an order by ID"""
        return self._orders.get(order_id)
    
    def list_orders(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent orders"""
        orders = list(self._orders.values())
        return orders[-limit:]
