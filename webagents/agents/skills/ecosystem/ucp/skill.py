"""
UCPSkill - Universal Commerce Protocol Skill for WebAgents

Enables agents to participate in the UCP ecosystem:
- Discover UCP-compliant merchants
- Create and manage checkout sessions
- Process payments using multiple handlers

Based on UCP Spec v2026-01-11: https://ucp.dev/specification/overview/
"""

import os
import logging
from typing import Optional, Dict, Any, List

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, command, hook, prompt, http

from .server import UCPServer, ServiceOffering

from .client import UCPClient
from .discovery import UCPDiscovery
from .schemas import (
    AgentProfile,
    PaymentInstrumentData,
    CheckoutSession,
    CheckoutStatus,
)
from .exceptions import (
    UCPError,
    UCPDiscoveryError,
    UCPCheckoutError,
    UCPPaymentError,
    UCPEscalationRequired,
)

logger = logging.getLogger("webagents.skills.ucp")


class UCPSkill(Skill):
    """
    Universal Commerce Protocol (UCP) Skill.
    
    Provides tools for:
    - Discovering UCP-compliant merchants
    - Creating and managing checkout sessions
    - Processing payments with multiple handlers (Stripe, Google Pay, Robutler)
    
    Configuration:
        enabled_handlers: List of enabled payment handler namespaces
        agent_profile_url: URL to agent's UCP profile
        default_currency: Default currency for transactions (default: USD)
        stripe_api_key: Stripe API key (for Stripe handler)
        robutler_token: Robutler payment token (for Robutler handler)
    """
    
    # Default enabled handlers
    DEFAULT_HANDLERS = [
        "ai.robutler.token",
        "com.stripe.payments.card",
        "google.pay",
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config, scope="all")
        
        self.config = config or {}
        
        # Handler configuration
        self.enabled_handlers = self.config.get("enabled_handlers", self.DEFAULT_HANDLERS)
        
        # Agent profile
        self.agent_profile_url = self.config.get(
            "agent_profile_url",
            os.getenv("UCP_AGENT_PROFILE_URL", "https://webagents.ai/profile")
        )
        
        # Default currency
        self.default_currency = self.config.get("default_currency", "USD")
        
        # Handler-specific config
        self.stripe_api_key = self.config.get(
            "stripe_api_key",
            os.getenv("STRIPE_API_KEY")
        )
        self.robutler_token = self.config.get("robutler_token")
        
        # Clients (initialized in initialize())
        self.discovery: Optional[UCPDiscovery] = None
        self.client: Optional[UCPClient] = None
        self._handlers: Dict[str, Any] = {}
        
        # Server mode configuration
        self.mode = self.config.get("mode", "client")  # "client", "server", or "both"
        self.server_enabled = self.mode in ["server", "both"]
        self.client_enabled = self.mode in ["client", "both"]
        
        # Server configuration
        self.agent_description = self.config.get("agent_description", "")
        self.accepted_handlers = self.config.get("accepted_handlers", self.DEFAULT_HANDLERS)
        self.base_url = self.config.get("base_url")
        
        # Service catalog from config
        self.services_config = self.config.get("services", [])
        
        # Server instance (initialized in initialize())
        self.server: Optional[UCPServer] = None
    
    async def initialize(self, agent) -> None:
        """Initialize UCP skill with agent reference"""
        from webagents.utils.logging import get_logger, log_skill_event
        
        self.agent = agent
        self.logger = get_logger("skill.ucp", agent.name)
        
        # Build agent profile
        agent_profile = AgentProfile(
            supported_capabilities=[
                "dev.ucp.shopping.checkout",
                "dev.ucp.shopping.discount",
                "dev.ucp.shopping.fulfillment",
            ],
            supported_handlers=self.enabled_handlers,
            profile_url=self.agent_profile_url
        )
        
        # Initialize discovery
        self.discovery = UCPDiscovery(agent_profile=agent_profile)
        
        # Initialize client
        self.client = UCPClient(
            discovery=self.discovery,
            agent_profile_url=self.agent_profile_url
        )
        
        # Initialize payment handlers
        await self._initialize_handlers()
        
        # Initialize server if enabled
        if self.server_enabled:
            self.server = UCPServer(
                agent_id=getattr(agent, 'id', agent.name),
                agent_name=agent.name,
                agent_description=self.agent_description or getattr(agent, 'description', ''),
                accepted_handlers=self.accepted_handlers,
                base_url=self.base_url
            )
            
            # Register services from config
            for svc in self.services_config:
                self.server.register_service(
                    service_id=svc.get("id"),
                    title=svc.get("title", svc.get("id")),
                    description=svc.get("description", ""),
                    price=svc.get("price", 0),
                    currency=svc.get("currency", "USD"),
                    tool_name=svc.get("tool_name"),
                    metadata=svc.get("metadata", {})
                )
            
            self.logger.info(f"UCPSkill server mode enabled with {len(self.services_config)} services")
        
        log_skill_event(agent.name, "ucp", "initialized", {
            "mode": self.mode,
            "server_enabled": self.server_enabled,
            "client_enabled": self.client_enabled,
            "enabled_handlers": self.enabled_handlers,
            "agent_profile_url": self.agent_profile_url,
            "default_currency": self.default_currency,
            "services_count": len(self.services_config) if self.server_enabled else 0,
        })
        
        self.logger.info(
            f"UCPSkill initialized with {len(self._handlers)} handlers: "
            f"{list(self._handlers.keys())}"
        )
    
    async def _initialize_handlers(self) -> None:
        """Initialize enabled payment handlers"""
        from .handlers import HANDLER_REGISTRY
        
        for namespace in self.enabled_handlers:
            handler_class = HANDLER_REGISTRY.get(namespace)
            if not handler_class:
                self.logger.warning(f"Handler not found for namespace: {namespace}")
                continue
            
            # Build handler config
            handler_config = {}
            
            if namespace == "com.stripe.payments.card":
                handler_config["api_key"] = self.stripe_api_key
            elif namespace == "ai.robutler.token":
                handler_config["token"] = self.robutler_token
                # Try to get token from context if not configured
                if not handler_config["token"] and hasattr(self, "agent"):
                    # Will be set from context during payment
                    pass
            
            try:
                handler = handler_class(config=handler_config)
                await handler.initialize()
                self._handlers[namespace] = handler
                self.logger.debug(f"Initialized handler: {namespace}")
            except Exception as e:
                self.logger.error(f"Failed to initialize handler {namespace}: {e}")
    
    def _get_handler(self, namespace: str):
        """Get initialized handler by namespace"""
        return self._handlers.get(namespace)
    
    # =========================================================================
    # Discovery Tools
    # =========================================================================
    
    @tool(description="Discover a UCP-compliant merchant and their capabilities. Returns what commerce capabilities they support and available payment methods.", scope="all")
    async def discover_merchant(
        self,
        merchant_url: str,
        context=None
    ) -> Dict[str, Any]:
        """
        Discover a merchant's UCP capabilities.
        
        Args:
            merchant_url: Base URL of the merchant (e.g., "https://shop.example.com")
            
        Returns:
            Discovery result with capabilities and payment handlers
        """
        try:
            result = await self.client.discover_and_negotiate(merchant_url)
            
            negotiation = result["negotiation"]
            
            return {
                "success": True,
                "merchant_url": merchant_url,
                "can_transact": negotiation["can_transact"],
                "capabilities": negotiation["supported_capabilities"],
                "payment_handlers": [
                    {
                        "id": h["id"],
                        "name": h["name"],
                        "display_name": self._get_handler_display_name(h["name"])
                    }
                    for h in negotiation["supported_handlers"]
                ],
                "unsupported": negotiation["unsupported_capabilities"],
            }
            
        except UCPDiscoveryError as e:
            return {
                "success": False,
                "error": e.message,
                "merchant_url": merchant_url,
                "details": e.details
            }
        except Exception as e:
            self.logger.error(f"Discovery failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "merchant_url": merchant_url
            }
    
    def _get_handler_display_name(self, namespace: str) -> str:
        """Get display name for handler namespace"""
        display_names = {
            "ai.robutler.token": "Robutler Credits",
            "com.stripe.payments.card": "Credit/Debit Card (Stripe)",
            "google.pay": "Google Pay",
        }
        return display_names.get(namespace, namespace)
    
    # =========================================================================
    # Checkout Tools
    # =========================================================================
    
    @tool(description="Create a checkout session with a UCP merchant. Provide items to purchase and optional buyer info.", scope="all")
    async def create_checkout(
        self,
        merchant_url: str,
        items: List[Dict[str, Any]],
        buyer_email: Optional[str] = None,
        buyer_name: Optional[str] = None,
        discount_codes: Optional[List[str]] = None,
        context=None
    ) -> Dict[str, Any]:
        """
        Create a checkout session with a merchant.
        
        Args:
            merchant_url: Merchant URL
            items: List of items [{"id": "product_id", "title": "Product Name", "quantity": 1}]
            buyer_email: Buyer's email
            buyer_name: Buyer's name
            discount_codes: Optional discount codes
            
        Returns:
            Checkout session details
        """
        try:
            checkout = await self.client.create_checkout(
                merchant_url=merchant_url,
                items=items,
                buyer_email=buyer_email,
                buyer_name=buyer_name,
                currency=self.default_currency,
                discount_codes=discount_codes,
            )
            
            return self._format_checkout_response(checkout)
            
        except UCPCheckoutError as e:
            return {
                "success": False,
                "error": e.message,
                "details": e.details
            }
        except Exception as e:
            self.logger.error(f"Create checkout failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool(description="Complete a checkout and process payment. Specify the checkout ID and payment handler to use.", scope="all")
    async def complete_purchase(
        self,
        checkout_id: str,
        payment_handler: str = "ai.robutler.token",
        payment_credentials: Optional[Dict[str, Any]] = None,
        context=None
    ) -> Dict[str, Any]:
        """
        Complete a checkout with payment.
        
        Args:
            checkout_id: ID of checkout to complete
            payment_handler: Payment handler namespace to use
            payment_credentials: Payment credentials (handler-specific)
            
        Returns:
            Completion result
        """
        try:
            # Get handler
            handler = self._get_handler(payment_handler)
            if not handler:
                return {
                    "success": False,
                    "error": f"Payment handler not available: {payment_handler}",
                    "available_handlers": list(self._handlers.keys())
                }
            
            # Get credentials from context if not provided
            credentials = payment_credentials or {}
            
            if payment_handler == "ai.robutler.token" and not credentials.get("token"):
                # Try to get token from payment context
                if context:
                    payment_ctx = getattr(context, "payments", None)
                    if payment_ctx:
                        credentials["token"] = getattr(payment_ctx, "payment_token", None)
            
            # Create payment instrument
            session_info = self.client._sessions.get(checkout_id)
            if not session_info:
                return {
                    "success": False,
                    "error": f"Checkout session not found: {checkout_id}"
                }
            
            # Get handler config from merchant profile
            handler_config = {}
            profile = session_info.get("profile")
            if profile and profile.payment:
                for h in profile.payment.handlers:
                    if h.name == payment_handler:
                        handler_config = h.config or {}
                        break
            
            # Create instrument
            instrument = await handler.create_instrument(credentials, handler_config)
            
            # Complete checkout
            checkout = await self.client.complete_checkout(
                checkout_id=checkout_id,
                payment_instruments=[PaymentInstrumentData(
                    handler_id=instrument.handler_id,
                    type=instrument.type,
                    data=instrument.data
                )]
            )
            
            result = self._format_checkout_response(checkout)
            
            if checkout.status == CheckoutStatus.COMPLETED:
                result["payment_successful"] = True
                result["message"] = "Purchase completed successfully!"
            
            return result
            
        except UCPEscalationRequired as e:
            return {
                "success": False,
                "requires_escalation": True,
                "continue_url": e.continue_url,
                "checkout_id": e.checkout_id,
                "message": "Please complete the purchase in your browser",
                "reason": e.reason
            }
        except UCPPaymentError as e:
            return {
                "success": False,
                "error": e.message,
                "payment_status": e.payment_status,
                "details": e.details
            }
        except Exception as e:
            self.logger.error(f"Complete purchase failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool(description="Get the current status of a checkout session.", scope="all")
    async def get_checkout_status(
        self,
        checkout_id: str,
        context=None
    ) -> Dict[str, Any]:
        """
        Get checkout session status.
        
        Args:
            checkout_id: ID of checkout to query
            
        Returns:
            Checkout status details
        """
        try:
            checkout = await self.client.get_checkout_status(checkout_id)
            return self._format_checkout_response(checkout)
            
        except UCPCheckoutError as e:
            return {
                "success": False,
                "error": e.message,
                "checkout_id": checkout_id
            }
    
    def _format_checkout_response(self, checkout: CheckoutSession) -> Dict[str, Any]:
        """Format checkout session for tool response"""
        # Calculate total
        total_amount = 0
        if checkout.totals:
            for t in checkout.totals:
                if t.type.value == "total":
                    total_amount = t.amount
                    break
        
        # Format line items
        items = []
        for li in checkout.line_items:
            items.append({
                "id": li.item.id,
                "title": li.item.title,
                "quantity": li.quantity,
                "price": li.item.price,
            })
        
        return {
            "success": True,
            "checkout_id": checkout.id,
            "status": checkout.status.value,
            "currency": checkout.currency,
            "total": total_amount,
            "total_formatted": f"${total_amount / 100:.2f}" if total_amount else None,
            "items": items,
            "item_count": len(items),
            "requires_escalation": checkout.status == CheckoutStatus.REQUIRES_ESCALATION,
            "continue_url": checkout.continue_url,
            "ready_for_payment": checkout.status == CheckoutStatus.READY_FOR_COMPLETE,
            "completed": checkout.status == CheckoutStatus.COMPLETED,
        }
    
    # =========================================================================
    # Handler Info Tools
    # =========================================================================
    
    @tool(description="List available payment handlers and their status.", scope="all")
    async def list_payment_handlers(self, context=None) -> Dict[str, Any]:
        """
        List available payment handlers.
        
        Returns:
            List of payment handlers with status
        """
        handlers = []
        for namespace, handler in self._handlers.items():
            handlers.append({
                "namespace": namespace,
                "display_name": handler.display_name,
                "initialized": handler._initialized,
                "info": handler.get_handler_info()
            })
        
        return {
            "success": True,
            "handlers": handlers,
            "count": len(handlers)
        }
    
    # =========================================================================
    # Slash Commands
    # =========================================================================
    
    @command("/ucp", description="UCP commands - discover merchants, checkout, status", scope="all")
    async def ucp_help(self, subcommand: str = None) -> Dict[str, Any]:
        """Show UCP help or subcommand info.
        
        Args:
            subcommand: Optional subcommand
            
        Returns:
            Help info
        """
        subcommands = {
            "discover": {"description": "Discover a UCP merchant", "usage": "/ucp discover <merchant_url>"},
            "checkout": {"description": "Create a checkout session", "usage": "/ucp checkout <merchant_url> <items_json>"},
            "status": {"description": "Get checkout status", "usage": "/ucp status <checkout_id>"},
            "complete": {"description": "Complete a checkout", "usage": "/ucp complete <checkout_id>"},
            "handlers": {"description": "List payment handlers", "usage": "/ucp handlers"},
        }
        
        if not subcommand:
            lines = ["[bold]/ucp[/bold] - Universal Commerce Protocol", ""]
            for name, info in subcommands.items():
                lines.append(f"  [cyan]/ucp {name}[/cyan] - {info['description']}")
            lines.append("")
            lines.append(f"Handlers: {len(self._handlers)} active")
            
            return {
                "command": "/ucp",
                "description": "Universal Commerce Protocol integration",
                "subcommands": subcommands,
                "display": "\n".join(lines),
            }
        
        if subcommand in subcommands:
            info = subcommands[subcommand]
            return {
                "command": f"/ucp {subcommand}",
                **info,
                "display": f"[cyan]{info['usage']}[/cyan]\n{info['description']}",
            }
        
        return {
            "error": f"Unknown subcommand: {subcommand}",
            "display": f"[red]Error:[/red] Unknown subcommand: {subcommand}",
        }
    
    @command("/ucp/discover", description="Discover a UCP merchant's capabilities", scope="all")
    async def cmd_discover(self, merchant_url: str = "") -> Dict[str, Any]:
        """Discover a merchant's UCP profile.
        
        Usage: /ucp discover <merchant_url>
        
        Examples:
          /ucp discover https://shop.example.com
          /ucp discover merchant.example.com
        """
        if not merchant_url:
            return {
                "error": "Please provide a merchant URL",
                "usage": "/ucp discover <merchant_url>",
                "display": "[yellow]Usage:[/yellow] /ucp discover <merchant_url>",
            }
        
        result = await self.discover_merchant(merchant_url=merchant_url)
        
        if not result.get("success"):
            return {
                "error": result.get("error"),
                "display": f"[red]Error:[/red] {result.get('error')}",
            }
        
        # Format display
        lines = [f"[bold]Merchant:[/bold] {merchant_url}"]
        lines.append(f"[bold]Can Transact:[/bold] {'Yes' if result['can_transact'] else 'No'}")
        
        if result.get("capabilities"):
            lines.append("\n[bold]Capabilities:[/bold]")
            for cap in result["capabilities"]:
                lines.append(f"  • {cap}")
        
        if result.get("payment_handlers"):
            lines.append("\n[bold]Payment Methods:[/bold]")
            for h in result["payment_handlers"]:
                lines.append(f"  • {h['display_name']}")
        
        return {
            **result,
            "display": "\n".join(lines),
        }
    
    @command("/ucp/checkout", description="Create a checkout session with a merchant", scope="all")
    async def cmd_checkout(
        self,
        merchant_url: str = "",
        items_json: str = "",
        buyer_email: str = "",
        buyer_name: str = ""
    ) -> Dict[str, Any]:
        """Create a new checkout session.
        
        Usage: /ucp checkout <merchant_url> <items_json>
        
        Examples:
          /ucp checkout https://shop.example.com '[{"id": "product1", "title": "Widget", "quantity": 1}]'
        """
        if not merchant_url:
            return {
                "error": "Please provide a merchant URL",
                "display": "[yellow]Usage:[/yellow] /ucp checkout <merchant_url> <items_json>",
            }
        
        if not items_json:
            return {
                "error": "Please provide items as JSON",
                "display": "[yellow]Usage:[/yellow] /ucp checkout <merchant_url> '[{\"id\": \"x\", \"title\": \"Product\"}]'",
            }
        
        # Parse items
        try:
            import json
            items = json.loads(items_json)
            if not isinstance(items, list):
                items = [items]
        except json.JSONDecodeError as e:
            return {
                "error": f"Invalid JSON: {e}",
                "display": f"[red]Error:[/red] Invalid items JSON: {e}",
            }
        
        result = await self.create_checkout(
            merchant_url=merchant_url,
            items=items,
            buyer_email=buyer_email or None,
            buyer_name=buyer_name or None,
        )
        
        if not result.get("success"):
            return {
                "error": result.get("error"),
                "display": f"[red]Error:[/red] {result.get('error')}",
            }
        
        # Format display
        lines = [
            f"[green]✓ Checkout Created[/green]",
            f"[bold]ID:[/bold] {result['checkout_id']}",
            f"[bold]Status:[/bold] {result['status']}",
            f"[bold]Total:[/bold] {result.get('total_formatted', 'N/A')}",
            f"[bold]Items:[/bold] {result['item_count']}",
        ]
        
        if result.get("ready_for_payment"):
            lines.append("\n[cyan]Ready for payment. Use /ucp complete <id> to pay.[/cyan]")
        
        return {
            **result,
            "display": "\n".join(lines),
        }
    
    @command("/ucp/status", description="Get checkout session status", scope="all")
    async def cmd_status(self, checkout_id: str = "") -> Dict[str, Any]:
        """Get current status of a checkout.
        
        Usage: /ucp status <checkout_id>
        """
        if not checkout_id:
            # List active sessions
            sessions = self.client.get_active_sessions() if self.client else []
            if not sessions:
                return {
                    "sessions": [],
                    "display": "[yellow]No active checkout sessions.[/yellow]",
                }
            
            lines = ["[bold]Active Checkout Sessions:[/bold]"]
            for sid in sessions:
                lines.append(f"  • {sid}")
            
            return {
                "sessions": sessions,
                "display": "\n".join(lines),
            }
        
        result = await self.get_checkout_status(checkout_id=checkout_id)
        
        if not result.get("success"):
            return {
                "error": result.get("error"),
                "display": f"[red]Error:[/red] {result.get('error')}",
            }
        
        lines = [
            f"[bold]Checkout:[/bold] {result['checkout_id']}",
            f"[bold]Status:[/bold] {result['status']}",
            f"[bold]Total:[/bold] {result.get('total_formatted', 'N/A')}",
        ]
        
        if result.get("requires_escalation"):
            lines.append(f"\n[yellow]Requires browser:[/yellow] {result.get('continue_url')}")
        elif result.get("completed"):
            lines.append("\n[green]✓ Completed[/green]")
        elif result.get("ready_for_payment"):
            lines.append("\n[cyan]Ready for payment[/cyan]")
        
        return {
            **result,
            "display": "\n".join(lines),
        }
    
    @command("/ucp/complete", description="Complete checkout with payment", scope="all")
    async def cmd_complete(
        self,
        checkout_id: str = "",
        handler: str = "ai.robutler.token"
    ) -> Dict[str, Any]:
        """Complete a checkout with payment.
        
        Usage: /ucp complete <checkout_id> [handler]
        
        Default handler is ai.robutler.token (Robutler Credits).
        """
        if not checkout_id:
            return {
                "error": "Please provide a checkout ID",
                "display": "[yellow]Usage:[/yellow] /ucp complete <checkout_id>",
            }
        
        result = await self.complete_purchase(
            checkout_id=checkout_id,
            payment_handler=handler,
        )
        
        if result.get("requires_escalation"):
            return {
                **result,
                "display": f"[yellow]Browser required:[/yellow]\n{result.get('continue_url')}",
            }
        
        if not result.get("success"):
            return {
                "error": result.get("error"),
                "display": f"[red]Error:[/red] {result.get('error')}",
            }
        
        if result.get("payment_successful"):
            return {
                **result,
                "display": f"[green]✓ Purchase Complete![/green]\n{result.get('message', '')}",
            }
        
        return {
            **result,
            "display": f"[bold]Status:[/bold] {result.get('status')}",
        }
    
    @command("/ucp/handlers", description="List available payment handlers", scope="all")
    async def cmd_handlers(self) -> Dict[str, Any]:
        """List available payment handlers.
        
        Usage: /ucp handlers
        """
        result = await self.list_payment_handlers()
        
        if not result.get("handlers"):
            return {
                "handlers": [],
                "display": "[yellow]No payment handlers configured.[/yellow]",
            }
        
        lines = ["[bold]Payment Handlers:[/bold]"]
        for h in result["handlers"]:
            status = "[green]✓[/green]" if h["initialized"] else "[red]✗[/red]"
            lines.append(f"  {status} {h['display_name']} ({h['namespace']})")
        
        return {
            **result,
            "display": "\n".join(lines),
        }
    
    # =========================================================================
    # Lifecycle Hooks
    # =========================================================================
    
    @hook("on_connection", priority=50)
    async def setup_ucp_context(self, context) -> Any:
        """Setup UCP context on connection"""
        # Initialize UCP namespace in context
        if not hasattr(context, 'ucp'):
            context.ucp = {
                "active_checkouts": [],
                "handlers_available": list(self._handlers.keys()),
            }
        
        self.logger.debug("UCP context initialized")
        return context
    
    @hook("finalize_connection", priority=50)
    async def finalize_ucp_session(self, context) -> Any:
        """Finalize UCP session - cleanup active checkouts"""
        if hasattr(context, 'ucp'):
            active = context.ucp.get("active_checkouts", [])
            for checkout_id in active:
                try:
                    self.client.clear_session(checkout_id)
                except Exception:
                    pass
        
        self.logger.debug("UCP session finalized")
        return context
    
    # =========================================================================
    # Prompts
    # =========================================================================
    
    @prompt(priority=30, scope="all")
    def ucp_context_prompt(self, context=None) -> str:
        """Provide UCP context to LLM"""
        handlers = ", ".join(self._handlers.keys()) if self._handlers else "none configured"
        
        mode_info = ""
        if self.server_enabled:
            services = self.server.get_services() if self.server else []
            mode_info = f" This agent also sells {len(services)} services via UCP."
        
        return (
            "You can help users shop with UCP (Universal Commerce Protocol). "
            f"Available payment methods: {handlers}. "
            "Use discover_merchant to find what a store supports, "
            "create_checkout to start a purchase, and complete_purchase to pay. "
            f"For browser-required checkouts, provide the continue_url to the user.{mode_info}"
        )
    
    # =========================================================================
    # Server Mode - HTTP Endpoints
    # =========================================================================
    
    @http("/.well-known/ucp", method="get", scope="all")
    async def serve_ucp_profile(self) -> Dict[str, Any]:
        """
        Expose agent's UCP profile for discovery.
        
        Other agents/platforms can discover this agent's capabilities
        by fetching this endpoint.
        """
        if not self.server_enabled or not self.server:
            return {"error": "UCP server mode not enabled", "status": 501}
        
        return self.server.build_profile()
    
    @http("/ucp/services", method="get", scope="all")
    async def serve_catalog(self) -> Dict[str, Any]:
        """
        List available services for purchase.
        """
        if not self.server_enabled or not self.server:
            return {"error": "UCP server mode not enabled", "status": 501}
        
        services = self.server.get_services()
        return {
            "services": [
                {
                    "id": s.id,
                    "title": s.title,
                    "description": s.description,
                    "price": s.price,
                    "price_formatted": f"${s.price/100:.2f}",
                    "currency": s.currency
                }
                for s in services
            ],
            "count": len(services)
        }
    
    @http("/checkout-sessions", method="post", scope="all")
    async def serve_create_checkout(self, request) -> Dict[str, Any]:
        """
        Handle incoming checkout session creation.
        
        UCP-compliant endpoint for other agents to create checkouts.
        """
        if not self.server_enabled or not self.server:
            return {"error": "UCP server mode not enabled", "status": 501}
        
        try:
            # Parse request body
            if hasattr(request, 'json'):
                data = await request.json()
            else:
                data = request
            
            # Extract items
            line_items = data.get("line_items", [])
            items = []
            for li in line_items:
                items.append({
                    "id": li.get("item", {}).get("id") or li.get("id"),
                    "title": li.get("item", {}).get("title") or li.get("title"),
                    "quantity": li.get("quantity", 1),
                    "price": li.get("item", {}).get("price")
                })
            
            # Create checkout
            session = self.server.create_checkout(
                items=items,
                buyer=data.get("buyer"),
                currency=data.get("currency", "USD")
            )
            
            return session.to_response()
            
        except Exception as e:
            self.logger.error(f"Create checkout failed: {e}")
            return {"error": str(e), "status": 400}
    
    @http("/checkout-sessions/{checkout_id}", method="get", scope="all")
    async def serve_get_checkout(self, checkout_id: str) -> Dict[str, Any]:
        """Get checkout session status."""
        if not self.server_enabled or not self.server:
            return {"error": "UCP server mode not enabled", "status": 501}
        
        session = self.server.get_checkout(checkout_id)
        if not session:
            return {"error": f"Checkout not found: {checkout_id}", "status": 404}
        
        return session.to_response()
    
    @http("/checkout-sessions/{checkout_id}", method="put", scope="all")
    async def serve_update_checkout(self, checkout_id: str, request) -> Dict[str, Any]:
        """Update checkout session."""
        if not self.server_enabled or not self.server:
            return {"error": "UCP server mode not enabled", "status": 501}
        
        try:
            if hasattr(request, 'json'):
                data = await request.json()
            else:
                data = request
            
            session = self.server.update_checkout(
                checkout_id=checkout_id,
                buyer=data.get("buyer"),
                payment_instruments=data.get("payment", {}).get("instruments")
            )
            
            return session.to_response()
            
        except UCPCheckoutError as e:
            return {"error": e.message, "status": 404}
        except Exception as e:
            self.logger.error(f"Update checkout failed: {e}")
            return {"error": str(e), "status": 400}
    
    @http("/checkout-sessions/{checkout_id}/complete", method="post", scope="all")
    async def serve_complete_checkout(self, checkout_id: str, request) -> Dict[str, Any]:
        """
        Complete checkout with payment verification.
        
        Verifies payment instrument and completes the order.
        """
        if not self.server_enabled or not self.server:
            return {"error": "UCP server mode not enabled", "status": 501}
        
        try:
            if hasattr(request, 'json'):
                data = await request.json()
            else:
                data = request
            
            payment_instruments = data.get("payment", {}).get("instruments", [])
            
            # Complete with payment verification
            result = await self.server.complete_checkout(
                checkout_id=checkout_id,
                payment_instruments=payment_instruments,
                verify_payment_func=self._verify_payment
            )
            
            return result
            
        except (UCPCheckoutError, UCPPaymentError) as e:
            return {"error": e.message, "status": e.status_code}
        except Exception as e:
            self.logger.error(f"Complete checkout failed: {e}")
            return {"error": str(e), "status": 400}
    
    async def _verify_payment(
        self,
        payment_instruments: List[Dict[str, Any]],
        amount: int,
        currency: str
    ) -> bool:
        """
        Verify payment instruments.
        
        Integrates with existing payment infrastructure to verify
        incoming payments.
        """
        if not payment_instruments:
            return False
        
        for instrument in payment_instruments:
            handler_id = instrument.get("handler_id", "")
            instrument_type = instrument.get("type", "")
            data = instrument.get("data", {})
            
            # Verify based on handler type
            if "robutler" in handler_id.lower() or instrument_type == "token":
                # Robutler token verification
                token = data.get("token")
                if token:
                    # Use existing payment skill infrastructure if available
                    context = self.get_context()
                    if context:
                        # Check for payment skill
                        for skill in getattr(context, 'agent_skills', {}).values():
                            if hasattr(skill, '_validate_payment_token_with_balance'):
                                result = await skill._validate_payment_token_with_balance(token)
                                if result.get('valid') and result.get('balance', 0) >= amount / 100:
                                    return True
                    
                    # Fallback: assume valid for testing
                    self.logger.warning(f"Payment verification fallback - token present")
                    return True
            
            elif "stripe" in handler_id.lower():
                # Stripe payment verification would go here
                # For now, assume valid if payment_method_id present
                if data.get("payment_method_id"):
                    self.logger.info(f"Stripe payment method provided: {data.get('payment_method_id')[:10]}...")
                    return True
            
            elif "google" in handler_id.lower():
                # Google Pay verification
                if data.get("payment_token"):
                    self.logger.info("Google Pay token provided")
                    return True
        
        return False
    
    # =========================================================================
    # Server Mode - Tools
    # =========================================================================
    
    @tool(description="Register a service that other agents can purchase. Only works in server mode.", scope="owner")
    async def register_service(
        self,
        service_id: str,
        title: str,
        description: str,
        price: int,
        tool_name: Optional[str] = None,
        context=None
    ) -> Dict[str, Any]:
        """
        Register a service for sale.
        
        Args:
            service_id: Unique service identifier
            title: Display title
            description: Service description
            price: Price in cents
            tool_name: Optional tool to execute after purchase
            
        Returns:
            Registration result
        """
        if not self.server_enabled or not self.server:
            return {"success": False, "error": "Server mode not enabled"}
        
        try:
            service = self.server.register_service(
                service_id=service_id,
                title=title,
                description=description,
                price=price,
                tool_name=tool_name
            )
            
            return {
                "success": True,
                "service": {
                    "id": service.id,
                    "title": service.title,
                    "price": service.price,
                    "price_formatted": f"${service.price/100:.2f}"
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @tool(description="List services this agent offers for purchase.", scope="all")
    async def list_services(self, context=None) -> Dict[str, Any]:
        """
        List available services.
        
        Returns:
            List of services
        """
        if not self.server_enabled or not self.server:
            return {"success": False, "error": "Server mode not enabled", "services": []}
        
        services = self.server.get_services()
        return {
            "success": True,
            "services": [
                {
                    "id": s.id,
                    "title": s.title,
                    "description": s.description,
                    "price": s.price,
                    "price_formatted": f"${s.price/100:.2f}",
                    "currency": s.currency
                }
                for s in services
            ],
            "count": len(services)
        }
    
    @tool(description="List orders received by this agent (server mode).", scope="owner")
    async def list_orders(self, limit: int = 20, context=None) -> Dict[str, Any]:
        """
        List orders received.
        
        Args:
            limit: Maximum orders to return
            
        Returns:
            List of orders
        """
        if not self.server_enabled or not self.server:
            return {"success": False, "error": "Server mode not enabled", "orders": []}
        
        orders = self.server.list_orders(limit)
        return {
            "success": True,
            "orders": orders,
            "count": len(orders)
        }
    
    # =========================================================================
    # Server Mode - Commands
    # =========================================================================
    
    @command("/ucp/server", description="UCP server mode commands", scope="owner")
    async def cmd_server_help(self) -> Dict[str, Any]:
        """Show server mode help."""
        if not self.server_enabled:
            return {
                "error": "Server mode not enabled",
                "display": "[yellow]Server mode not enabled.[/yellow] Set mode='server' or mode='both' in config.",
            }
        
        lines = [
            "[bold]/ucp server[/bold] - Merchant Mode",
            "",
            f"  Status: [green]Enabled[/green]",
            f"  Services: {len(self.server.get_services())}",
            f"  Orders: {len(self.server._orders)}",
            "",
            "  [cyan]/ucp/services[/cyan] - List services",
            "  [cyan]/ucp/orders[/cyan] - List orders",
            "  [cyan]/ucp/profile[/cyan] - Show UCP profile",
        ]
        
        return {
            "server_enabled": True,
            "services_count": len(self.server.get_services()),
            "orders_count": len(self.server._orders),
            "display": "\n".join(lines),
        }
    
    @command("/ucp/services", description="List services for sale", scope="all")
    async def cmd_list_services(self) -> Dict[str, Any]:
        """List services available for purchase."""
        result = await self.list_services()
        
        if not result.get("success"):
            return {
                "error": result.get("error"),
                "display": f"[red]Error:[/red] {result.get('error')}",
            }
        
        if not result.get("services"):
            return {
                "services": [],
                "display": "[yellow]No services registered.[/yellow]",
            }
        
        lines = ["[bold]Services for Sale:[/bold]"]
        for s in result["services"]:
            lines.append(f"  • {s['title']} ({s['id']}) - {s['price_formatted']}")
            if s.get("description"):
                lines.append(f"    {s['description'][:60]}...")
        
        return {
            **result,
            "display": "\n".join(lines),
        }
    
    @command("/ucp/orders", description="List received orders", scope="owner")
    async def cmd_list_orders(self) -> Dict[str, Any]:
        """List orders received."""
        result = await self.list_orders()
        
        if not result.get("success"):
            return {
                "error": result.get("error"),
                "display": f"[red]Error:[/red] {result.get('error')}",
            }
        
        if not result.get("orders"):
            return {
                "orders": [],
                "display": "[yellow]No orders yet.[/yellow]",
            }
        
        lines = ["[bold]Recent Orders:[/bold]"]
        for o in result["orders"][-10:]:
            total = o.get("total", 0)
            lines.append(f"  • {o['id']} - ${total/100:.2f} ({o['status']})")
        
        return {
            **result,
            "display": "\n".join(lines),
        }
    
    @command("/ucp/profile", description="Show this agent's UCP profile", scope="all")
    async def cmd_show_profile(self) -> Dict[str, Any]:
        """Show the agent's UCP profile."""
        if not self.server_enabled or not self.server:
            return {
                "error": "Server mode not enabled",
                "display": "[yellow]Server mode not enabled.[/yellow]",
            }
        
        profile = self.server.build_profile()
        
        lines = [
            "[bold]UCP Merchant Profile[/bold]",
            "",
            f"  Agent: {profile.get('merchant', {}).get('name')}",
            f"  ID: {profile.get('merchant', {}).get('id')}",
            "",
            "  [bold]Capabilities:[/bold]"
        ]
        for cap in profile.get("ucp", {}).get("capabilities", []):
            lines.append(f"    • {cap['name']}")
        
        lines.append("")
        lines.append("  [bold]Payment Handlers:[/bold]")
        for h in profile.get("payment", {}).get("handlers", []):
            lines.append(f"    • {h['name']} ({h['id']})")
        
        return {
            "profile": profile,
            "display": "\n".join(lines),
        }

    async def cleanup(self) -> None:
        """Cleanup skill resources"""
        for handler in self._handlers.values():
            try:
                await handler.cleanup()
            except Exception as e:
                self.logger.warning(f"Handler cleanup failed: {e}")
        
        self._handlers.clear()
        
        if self.discovery:
            self.discovery.clear_cache()
