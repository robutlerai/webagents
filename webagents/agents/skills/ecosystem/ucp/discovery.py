"""
UCP Discovery - Merchant Discovery and Capability Negotiation

Handles fetching merchant profiles from /.well-known/ucp endpoints
and negotiating capabilities between agent and merchant.

Based on UCP Spec v2026-01-11: https://ucp.dev/specification/overview/
"""

import logging
from typing import Optional, Dict, Any, List, Set
from urllib.parse import urljoin, urlparse

import aiohttp

from .schemas import (
    MerchantProfile,
    UCPProfile,
    PaymentProfile,
    Capability,
    PaymentHandlerConfig,
    AgentProfile,
)
from .exceptions import UCPDiscoveryError

logger = logging.getLogger("webagents.skills.ucp.discovery")


class UCPDiscovery:
    """
    UCP Merchant Discovery and Capability Negotiation.
    
    Fetches and parses merchant profiles, negotiates capabilities,
    and determines compatible payment handlers.
    """
    
    # Well-known UCP endpoint path
    WELL_KNOWN_PATH = "/.well-known/ucp"
    
    # Default agent capabilities
    DEFAULT_AGENT_CAPABILITIES = [
        "dev.ucp.shopping.checkout",
        "dev.ucp.shopping.discount",
        "dev.ucp.shopping.fulfillment",
    ]
    
    # Default agent-supported payment handlers
    DEFAULT_AGENT_HANDLERS = [
        "ai.robutler.token",
        "com.stripe.payments.card",
        "google.pay",
    ]
    
    def __init__(
        self,
        agent_profile: Optional[AgentProfile] = None,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize discovery client.
        
        Args:
            agent_profile: Agent's UCP profile for negotiation
            timeout: HTTP request timeout in seconds
            headers: Additional headers for requests
        """
        self.agent_profile = agent_profile or AgentProfile(
            supported_capabilities=self.DEFAULT_AGENT_CAPABILITIES,
            supported_handlers=self.DEFAULT_AGENT_HANDLERS
        )
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "WebAgents-UCP/1.0",
            **(headers or {})
        }
        
        # Cache of discovered merchant profiles
        self._profile_cache: Dict[str, MerchantProfile] = {}
    
    async def discover_merchant(
        self,
        merchant_url: str,
        force_refresh: bool = False
    ) -> MerchantProfile:
        """
        Discover a merchant's UCP profile.
        
        Fetches the merchant's /.well-known/ucp endpoint and parses
        the profile including services, capabilities, and payment handlers.
        
        Args:
            merchant_url: Base URL of the merchant (e.g., "https://merchant.example.com")
            force_refresh: Force re-fetch even if cached
            
        Returns:
            Parsed MerchantProfile
            
        Raises:
            UCPDiscoveryError: If discovery fails
        """
        # Normalize URL
        base_url = self._normalize_url(merchant_url)
        
        # Check cache
        if not force_refresh and base_url in self._profile_cache:
            logger.debug(f"Returning cached profile for {base_url}")
            return self._profile_cache[base_url]
        
        # Build well-known URL
        well_known_url = urljoin(base_url, self.WELL_KNOWN_PATH)
        logger.info(f"Discovering merchant profile at {well_known_url}")
        
        try:
            async with aiohttp.ClientSession(
                timeout=self.timeout,
                headers=self.headers
            ) as session:
                async with session.get(well_known_url) as response:
                    if response.status == 404:
                        raise UCPDiscoveryError(
                            message="Merchant does not support UCP (no /.well-known/ucp)",
                            merchant_url=merchant_url,
                            details={"status_code": 404}
                        )
                    
                    if not response.ok:
                        raise UCPDiscoveryError(
                            message=f"Failed to fetch merchant profile: HTTP {response.status}",
                            merchant_url=merchant_url,
                            details={"status_code": response.status}
                        )
                    
                    data = await response.json()
                    
        except aiohttp.ClientError as e:
            raise UCPDiscoveryError(
                message=f"Network error during discovery: {str(e)}",
                merchant_url=merchant_url,
                details={"error_type": type(e).__name__}
            )
        
        # Parse profile
        try:
            profile = self._parse_profile(data)
            self._profile_cache[base_url] = profile
            
            logger.info(
                f"Discovered merchant: {len(profile.ucp.capabilities)} capabilities, "
                f"{len(profile.payment.handlers) if profile.payment else 0} payment handlers"
            )
            
            return profile
            
        except Exception as e:
            raise UCPDiscoveryError(
                message=f"Failed to parse merchant profile: {str(e)}",
                merchant_url=merchant_url,
                details={"parse_error": str(e)}
            )
    
    def _normalize_url(self, url: str) -> str:
        """Normalize merchant URL to base form"""
        parsed = urlparse(url)
        
        # Ensure scheme
        if not parsed.scheme:
            url = f"https://{url}"
            parsed = urlparse(url)
        
        # Return base URL without path
        return f"{parsed.scheme}://{parsed.netloc}"
    
    def _parse_profile(self, data: Dict[str, Any]) -> MerchantProfile:
        """Parse raw JSON into MerchantProfile"""
        ucp_data = data.get("ucp", {})
        payment_data = data.get("payment", {})
        
        # Parse capabilities
        capabilities = []
        for cap in ucp_data.get("capabilities", []):
            capabilities.append(Capability(
                name=cap.get("name", ""),
                version=cap.get("version", "2026-01-11"),
                spec=cap.get("spec"),
                schema_url=cap.get("schema"),
                extends=cap.get("extends")
            ))
        
        # Parse services
        services = {}
        for name, svc in ucp_data.get("services", {}).items():
            from .schemas import Service
            services[name] = Service(
                version=svc.get("version", "2026-01-11"),
                spec=svc.get("spec"),
                rest=svc.get("rest"),
                mcp=svc.get("mcp")
            )
        
        # Parse payment handlers
        handlers = []
        for handler in payment_data.get("handlers", []):
            handlers.append(PaymentHandlerConfig(
                id=handler.get("id", ""),
                name=handler.get("name", ""),
                version=handler.get("version", "2026-01-11"),
                spec=handler.get("spec"),
                config_schema=handler.get("config_schema"),
                instrument_schemas=handler.get("instrument_schemas", []),
                config=handler.get("config", {})
            ))
        
        return MerchantProfile(
            ucp=UCPProfile(
                version=ucp_data.get("version", "2026-01-11"),
                services=services,
                capabilities=capabilities
            ),
            payment=PaymentProfile(handlers=handlers) if handlers else None
        )
    
    def negotiate_capabilities(
        self,
        merchant_profile: MerchantProfile
    ) -> Dict[str, Any]:
        """
        Negotiate capabilities between agent and merchant.
        
        Computes the intersection of agent and merchant capabilities
        to determine what operations are supported.
        
        Args:
            merchant_profile: Merchant's UCP profile
            
        Returns:
            Negotiation result with supported capabilities and handlers
        """
        # Get merchant capability names
        merchant_caps: Set[str] = {
            cap.name for cap in merchant_profile.ucp.capabilities
        }
        
        # Get agent capability names
        agent_caps: Set[str] = set(self.agent_profile.supported_capabilities)
        
        # Compute intersection
        supported_caps = merchant_caps & agent_caps
        
        # Negotiate payment handlers
        supported_handlers = []
        if merchant_profile.payment:
            merchant_handler_names = {h.name for h in merchant_profile.payment.handlers}
            agent_handler_names = set(self.agent_profile.supported_handlers)
            
            # Find matching handlers
            for handler in merchant_profile.payment.handlers:
                if handler.name in agent_handler_names:
                    supported_handlers.append(handler)
        
        # Determine if checkout is possible
        can_checkout = "dev.ucp.shopping.checkout" in supported_caps
        has_payment = len(supported_handlers) > 0
        
        return {
            "can_transact": can_checkout and has_payment,
            "supported_capabilities": list(supported_caps),
            "unsupported_capabilities": list(merchant_caps - supported_caps),
            "supported_handlers": [h.model_dump() for h in supported_handlers],
            "merchant_capabilities": list(merchant_caps),
            "agent_capabilities": list(agent_caps),
        }
    
    def get_checkout_endpoint(
        self,
        merchant_profile: MerchantProfile
    ) -> Optional[str]:
        """
        Get the checkout REST endpoint from merchant profile.
        
        Args:
            merchant_profile: Merchant's UCP profile
            
        Returns:
            Checkout endpoint URL or None
        """
        shopping_service = merchant_profile.ucp.services.get("dev.ucp.shopping")
        if shopping_service and shopping_service.rest:
            return shopping_service.rest.get("endpoint")
        return None
    
    def clear_cache(self, merchant_url: Optional[str] = None) -> None:
        """
        Clear profile cache.
        
        Args:
            merchant_url: Specific merchant to clear, or None for all
        """
        if merchant_url:
            base_url = self._normalize_url(merchant_url)
            self._profile_cache.pop(base_url, None)
        else:
            self._profile_cache.clear()
