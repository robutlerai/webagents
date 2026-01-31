"""
UCP Client - REST Client for Universal Commerce Protocol

Provides a high-level client for interacting with UCP-compliant merchants
including checkout session management and payment processing.

Based on UCP Spec v2026-01-11: https://ucp.dev/specification/overview/
"""

import logging
import uuid
from typing import Optional, Dict, Any, List

import aiohttp

from .schemas import (
    CheckoutSession,
    CheckoutStatus,
    CreateCheckoutRequest,
    UpdateCheckoutRequest,
    CompleteCheckoutRequest,
    PaymentSection,
    PaymentInstrumentData,
    MerchantProfile,
    LineItem,
    Buyer,
)
from .discovery import UCPDiscovery
from .exceptions import (
    UCPCheckoutError,
    UCPPaymentError,
    UCPDiscoveryError,
    UCPEscalationRequired,
)

logger = logging.getLogger("webagents.skills.ucp.client")


class UCPClient:
    """
    UCP REST Client for merchant interactions.
    
    Manages the full checkout lifecycle:
    1. Discover merchant capabilities
    2. Create checkout session
    3. Update with buyer/payment info
    4. Complete checkout
    
    Also handles escalation (browser handoff) when required.
    """
    
    def __init__(
        self,
        discovery: Optional[UCPDiscovery] = None,
        agent_profile_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize UCP client.
        
        Args:
            discovery: UCPDiscovery instance (created if not provided)
            agent_profile_url: URL to agent's UCP profile for requests
            timeout: HTTP request timeout
        """
        self.discovery = discovery or UCPDiscovery()
        self.agent_profile_url = agent_profile_url or "https://webagents.ai/profile"
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        
        # Active checkout sessions
        self._sessions: Dict[str, Dict[str, Any]] = {}
    
    def _get_headers(self) -> Dict[str, str]:
        """Get standard headers for UCP requests"""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "UCP-Agent": f'profile="{self.agent_profile_url}"',
            "request-id": str(uuid.uuid4()),
            "idempotency-key": str(uuid.uuid4()),
        }
    
    async def discover_and_negotiate(
        self,
        merchant_url: str
    ) -> Dict[str, Any]:
        """
        Discover merchant and negotiate capabilities.
        
        Args:
            merchant_url: Merchant base URL
            
        Returns:
            Negotiation result with capabilities and handlers
        """
        profile = await self.discovery.discover_merchant(merchant_url)
        negotiation = self.discovery.negotiate_capabilities(profile)
        
        return {
            "merchant_url": merchant_url,
            "profile": profile,
            "negotiation": negotiation,
            "endpoint": self.discovery.get_checkout_endpoint(profile),
        }
    
    async def create_checkout(
        self,
        merchant_url: str,
        items: List[Dict[str, Any]],
        buyer_email: Optional[str] = None,
        buyer_name: Optional[str] = None,
        currency: str = "USD",
        discount_codes: Optional[List[str]] = None,
    ) -> CheckoutSession:
        """
        Create a new checkout session with a merchant.
        
        Args:
            merchant_url: Merchant base URL
            items: List of items to purchase [{"id": "...", "title": "...", "quantity": 1}]
            buyer_email: Buyer's email address
            buyer_name: Buyer's full name
            currency: Currency code (default USD)
            discount_codes: Optional discount codes to apply
            
        Returns:
            Created CheckoutSession
            
        Raises:
            UCPDiscoveryError: If merchant discovery fails
            UCPCheckoutError: If checkout creation fails
        """
        # Discover merchant
        discovery_result = await self.discover_and_negotiate(merchant_url)
        
        if not discovery_result["negotiation"]["can_transact"]:
            raise UCPCheckoutError(
                message="Cannot transact with merchant - no compatible capabilities/handlers",
                details=discovery_result["negotiation"]
            )
        
        endpoint = discovery_result["endpoint"]
        if not endpoint:
            raise UCPCheckoutError(
                message="Merchant has no checkout endpoint configured",
                details={"merchant_url": merchant_url}
            )
        
        profile: MerchantProfile = discovery_result["profile"]
        
        # Build line items
        from .schemas import create_line_item
        line_items = [
            create_line_item(
                product_id=item.get("id", item.get("product_id", "")),
                title=item.get("title", item.get("name", "Product")),
                quantity=item.get("quantity", 1),
                price=item.get("price")
            )
            for item in items
        ]
        
        # Build request
        request_data = {
            "line_items": [li.model_dump() for li in line_items],
            "currency": currency,
            "payment": {
                "instruments": [],
                "handlers": [h.model_dump() for h in profile.payment.handlers] if profile.payment else []
            }
        }
        
        # Add buyer if provided
        if buyer_email or buyer_name:
            request_data["buyer"] = {
                "email": buyer_email,
                "full_name": buyer_name
            }
        
        # Add discounts if provided
        if discount_codes:
            request_data["discounts"] = {"codes": discount_codes}
        
        # Create session
        checkout_url = f"{endpoint.rstrip('/')}/checkout-sessions"
        logger.info(f"Creating checkout at {checkout_url}")
        
        try:
            async with aiohttp.ClientSession(
                timeout=self.timeout,
                headers=self._get_headers()
            ) as session:
                async with session.post(checkout_url, json=request_data) as response:
                    if not response.ok:
                        error_text = await response.text()
                        raise UCPCheckoutError(
                            message=f"Checkout creation failed: HTTP {response.status}",
                            details={"response": error_text, "status": response.status}
                        )
                    
                    data = await response.json()
                    
        except aiohttp.ClientError as e:
            raise UCPCheckoutError(
                message=f"Network error during checkout creation: {str(e)}",
                details={"error_type": type(e).__name__}
            )
        
        # Parse response
        checkout = self._parse_checkout_response(data)
        
        # Cache session info
        self._sessions[checkout.id] = {
            "merchant_url": merchant_url,
            "endpoint": endpoint,
            "profile": profile,
            "checkout": checkout
        }
        
        logger.info(f"Created checkout {checkout.id} with status {checkout.status}")
        
        return checkout
    
    async def update_checkout(
        self,
        checkout_id: str,
        buyer: Optional[Dict[str, Any]] = None,
        payment_instruments: Optional[List[PaymentInstrumentData]] = None,
        discount_codes: Optional[List[str]] = None,
        shipping_address: Optional[Dict[str, Any]] = None,
    ) -> CheckoutSession:
        """
        Update an existing checkout session.
        
        Args:
            checkout_id: ID of checkout to update
            buyer: Updated buyer information
            payment_instruments: Payment instruments to add
            discount_codes: Discount codes to apply
            shipping_address: Shipping address
            
        Returns:
            Updated CheckoutSession
        """
        session_info = self._sessions.get(checkout_id)
        if not session_info:
            raise UCPCheckoutError(
                message=f"Checkout session not found: {checkout_id}",
                checkout_id=checkout_id
            )
        
        endpoint = session_info["endpoint"]
        checkout_url = f"{endpoint.rstrip('/')}/checkout-sessions/{checkout_id}"
        
        # Build update request
        request_data = {"id": checkout_id}
        
        if buyer:
            request_data["buyer"] = buyer
        
        if payment_instruments:
            request_data["payment"] = {
                "instruments": [pi.model_dump() for pi in payment_instruments],
                "handlers": []
            }
        
        if discount_codes:
            request_data["discounts"] = {"codes": discount_codes}
        
        if shipping_address:
            request_data["shipping_address"] = shipping_address
        
        logger.info(f"Updating checkout {checkout_id}")
        
        try:
            async with aiohttp.ClientSession(
                timeout=self.timeout,
                headers=self._get_headers()
            ) as session:
                async with session.put(checkout_url, json=request_data) as response:
                    if not response.ok:
                        error_text = await response.text()
                        raise UCPCheckoutError(
                            message=f"Checkout update failed: HTTP {response.status}",
                            checkout_id=checkout_id,
                            details={"response": error_text}
                        )
                    
                    data = await response.json()
                    
        except aiohttp.ClientError as e:
            raise UCPCheckoutError(
                message=f"Network error during checkout update: {str(e)}",
                checkout_id=checkout_id
            )
        
        checkout = self._parse_checkout_response(data)
        session_info["checkout"] = checkout
        
        return checkout
    
    async def complete_checkout(
        self,
        checkout_id: str,
        payment_instruments: List[PaymentInstrumentData],
    ) -> CheckoutSession:
        """
        Complete a checkout session with payment.
        
        Args:
            checkout_id: ID of checkout to complete
            payment_instruments: Payment instruments for completion
            
        Returns:
            Completed CheckoutSession
            
        Raises:
            UCPEscalationRequired: If checkout requires browser handoff
            UCPPaymentError: If payment processing fails
        """
        session_info = self._sessions.get(checkout_id)
        if not session_info:
            raise UCPCheckoutError(
                message=f"Checkout session not found: {checkout_id}",
                checkout_id=checkout_id
            )
        
        current_checkout: CheckoutSession = session_info["checkout"]
        
        # Check if escalation required
        if current_checkout.status == CheckoutStatus.REQUIRES_ESCALATION:
            if current_checkout.continue_url:
                raise UCPEscalationRequired(
                    message="Checkout requires user interaction in browser",
                    continue_url=current_checkout.continue_url,
                    checkout_id=checkout_id,
                    reason="Merchant requires additional verification"
                )
        
        # Ensure checkout is ready
        if current_checkout.status not in [
            CheckoutStatus.READY_FOR_COMPLETE,
            CheckoutStatus.INCOMPLETE
        ]:
            raise UCPCheckoutError(
                message=f"Checkout not ready for completion: {current_checkout.status}",
                checkout_id=checkout_id,
                checkout_status=current_checkout.status.value
            )
        
        endpoint = session_info["endpoint"]
        complete_url = f"{endpoint.rstrip('/')}/checkout-sessions/{checkout_id}/complete"
        
        # Build completion request
        request_data = {
            "id": checkout_id,
            "payment": {
                "instruments": [pi.model_dump() for pi in payment_instruments],
                "handlers": []
            }
        }
        
        logger.info(f"Completing checkout {checkout_id}")
        
        try:
            async with aiohttp.ClientSession(
                timeout=self.timeout,
                headers=self._get_headers()
            ) as session:
                async with session.post(complete_url, json=request_data) as response:
                    if not response.ok:
                        error_text = await response.text()
                        
                        # Check for escalation in error response
                        if response.status == 303:
                            data = await response.json()
                            raise UCPEscalationRequired(
                                message="Checkout requires escalation",
                                continue_url=data.get("continue_url", ""),
                                checkout_id=checkout_id
                            )
                        
                        raise UCPPaymentError(
                            message=f"Checkout completion failed: HTTP {response.status}",
                            payment_status="failed",
                            details={"response": error_text}
                        )
                    
                    data = await response.json()
                    
        except aiohttp.ClientError as e:
            raise UCPPaymentError(
                message=f"Network error during checkout completion: {str(e)}",
                payment_status="error"
            )
        
        checkout = self._parse_checkout_response(data)
        session_info["checkout"] = checkout
        
        # Verify completion
        if checkout.status != CheckoutStatus.COMPLETED:
            logger.warning(f"Checkout {checkout_id} not completed: {checkout.status}")
        
        return checkout
    
    async def get_checkout_status(
        self,
        checkout_id: str
    ) -> CheckoutSession:
        """
        Get current status of a checkout session.
        
        Args:
            checkout_id: ID of checkout to query
            
        Returns:
            Current CheckoutSession state
        """
        session_info = self._sessions.get(checkout_id)
        if not session_info:
            raise UCPCheckoutError(
                message=f"Checkout session not found: {checkout_id}",
                checkout_id=checkout_id
            )
        
        endpoint = session_info["endpoint"]
        checkout_url = f"{endpoint.rstrip('/')}/checkout-sessions/{checkout_id}"
        
        try:
            async with aiohttp.ClientSession(
                timeout=self.timeout,
                headers=self._get_headers()
            ) as session:
                async with session.get(checkout_url) as response:
                    if not response.ok:
                        raise UCPCheckoutError(
                            message=f"Failed to get checkout status: HTTP {response.status}",
                            checkout_id=checkout_id
                        )
                    
                    data = await response.json()
                    
        except aiohttp.ClientError as e:
            raise UCPCheckoutError(
                message=f"Network error getting checkout status: {str(e)}",
                checkout_id=checkout_id
            )
        
        checkout = self._parse_checkout_response(data)
        session_info["checkout"] = checkout
        
        return checkout
    
    def _parse_checkout_response(self, data: Dict[str, Any]) -> CheckoutSession:
        """Parse checkout response into CheckoutSession"""
        # Map status string to enum
        status_str = data.get("status", "incomplete")
        try:
            status = CheckoutStatus(status_str)
        except ValueError:
            status = CheckoutStatus.INCOMPLETE
        
        # Parse line items
        line_items = []
        for li_data in data.get("line_items", []):
            from .schemas import LineItem, LineItemProduct, Total, TotalType
            
            item_data = li_data.get("item", {})
            item = LineItemProduct(
                id=item_data.get("id", ""),
                title=item_data.get("title", ""),
                price=item_data.get("price"),
                description=item_data.get("description"),
            )
            
            totals = None
            if "totals" in li_data:
                totals = [
                    Total(type=TotalType(t.get("type", "total")), amount=t.get("amount", 0))
                    for t in li_data.get("totals", [])
                ]
            
            line_items.append(LineItem(
                id=li_data.get("id"),
                item=item,
                quantity=li_data.get("quantity", 1),
                totals=totals
            ))
        
        # Parse totals
        totals = None
        if "totals" in data:
            from .schemas import Total, TotalType
            totals = [
                Total(type=TotalType(t.get("type", "total")), amount=t.get("amount", 0))
                for t in data.get("totals", [])
            ]
        
        return CheckoutSession(
            id=data.get("id"),
            status=status,
            line_items=line_items,
            currency=data.get("currency", "USD"),
            totals=totals,
            buyer=Buyer(**data["buyer"]) if data.get("buyer") else None,
            continue_url=data.get("continue_url"),
            messages=data.get("messages"),
        )
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active checkout session IDs"""
        return list(self._sessions.keys())
    
    def clear_session(self, checkout_id: str) -> None:
        """Clear a checkout session from cache"""
        self._sessions.pop(checkout_id, None)
