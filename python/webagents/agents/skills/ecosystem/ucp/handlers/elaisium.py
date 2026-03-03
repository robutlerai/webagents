"""
Elaisium VIBE Payment Handler

Handles payments using Elaisium's VIBE in-game currency for agent-to-agent
commerce within the Elaisium game world.

VIBE Economy:
- VIBE is the universal currency in Elaisium
- Used for trading artifacts, services, information
- Governed by the World Equation: V = m × Φ
- Can be collected, traded, or earned through gameplay
"""

from typing import Dict, Any, Optional, List
from .base import PaymentHandler, PaymentInstrument, PaymentResult, PaymentStatus


class ElaisiumHandler(PaymentHandler):
    """
    Payment handler for Elaisium VIBE currency.
    
    Supports in-game commerce between agents and players:
    - Trading artifacts and items
    - Purchasing services (analysis, guidance, etc.)
    - Information exchange
    - Quest rewards
    
    Config:
        elaisium_api_url: URL of the Elaisium backend API
        agent_entity_id: The entity ID of this agent in Elaisium
        require_proximity: If True, require entities to be nearby for trades
    """
    
    # Handler namespace
    NAMESPACE = "world.elaisium.vibe"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Elaisium VIBE handler.
        
        Args:
            config: Handler configuration
                - elaisium_api_url: Elaisium backend URL
                - agent_entity_id: Agent's entity ID in game
                - require_proximity: Require nearby for trades (default: False)
        """
        super().__init__(config)
        self.api_url = self.config.get("elaisium_api_url", "http://localhost:8080")
        self.entity_id = self.config.get("agent_entity_id")
        self.require_proximity = self.config.get("require_proximity", False)
    
    @property
    def name(self) -> str:
        return self.NAMESPACE
    
    @property
    def display_name(self) -> str:
        return "Elaisium VIBE"
    
    @property
    def supported_currencies(self) -> List[str]:
        return ["VIBE"]
    
    def can_handle(self, handler_config: Dict[str, Any]) -> bool:
        """Check if this handler can process the given merchant handler config.
        
        Args:
            handler_config: Handler configuration from merchant's /.well-known/ucp
            
        Returns:
            True if this handler can process payments for this config
        """
        namespace = handler_config.get("namespace", "")
        return namespace == self.NAMESPACE or namespace.startswith("world.elaisium")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return handler capabilities."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "currencies": self.supported_currencies,
            "features": {
                "refunds": True,
                "escrow": True,  # Can hold VIBE in escrow
                "recurring": False,
                "proximity_trading": self.require_proximity,
            },
            "payment_types": ["instant", "escrow"],
            "entity_id": self.entity_id,
        }
    
    async def create_instrument(
        self,
        credentials: Dict[str, Any],
        handler_config: Dict[str, Any]
    ) -> PaymentInstrument:
        """Create a VIBE payment instrument.
        
        Args:
            credentials: Payment credentials
                - entity_id: Payer's entity ID in Elaisium
                - amount: VIBE amount to pay
                - escrow: If True, hold in escrow until confirmed
            handler_config: Handler configuration from merchant profile
                
        Returns:
            PaymentInstrument for VIBE payment
        """
        entity_id = credentials.get("entity_id")
        amount = credentials.get("amount", 0)
        
        if not entity_id:
            raise ValueError("entity_id is required for VIBE payments")
        
        return PaymentInstrument(
            handler_id=self.name,
            type="vibe_transfer",
            data={
                "payer_entity_id": entity_id,
                "amount": amount,
                "escrow": credentials.get("escrow", False),
                "memo": credentials.get("memo", ""),
                "currency": "VIBE",
                "game": "elaisium",
            }
        )
    
    async def process_payment(
        self,
        instrument: PaymentInstrument,
        amount: int,
        currency: str = "VIBE",
        metadata: Optional[Dict[str, Any]] = None
    ) -> PaymentResult:
        """Process a VIBE payment.
        
        Args:
            instrument: The payment instrument
            amount: Amount in VIBE (smallest unit)
            currency: Must be "VIBE"
            metadata: Additional payment metadata
            
        Returns:
            PaymentResult with transaction details
        """
        if currency != "VIBE":
            return PaymentResult(
                success=False,
                status=PaymentStatus.FAILED,
                error_message="Only VIBE currency is supported"
            )
        
        payer_id = instrument.data.get("payer_entity_id")
        receiver_id = self.entity_id
        use_escrow = instrument.data.get("escrow", False)
        
        # In a real implementation, this would call the Elaisium API
        # For now, we simulate the payment
        try:
            # Simulate API call to transfer VIBE
            # await self._transfer_vibe(payer_id, receiver_id, amount, use_escrow)
            
            transaction_id = f"vibe_tx_{payer_id}_{receiver_id}_{amount}"
            
            return PaymentResult(
                success=True,
                status=PaymentStatus.CAPTURED if not use_escrow else PaymentStatus.PENDING,
                transaction_id=transaction_id,
                amount=amount,
                currency="VIBE",
                metadata={
                    "payer_entity_id": payer_id,
                    "receiver_entity_id": receiver_id,
                    "escrow": use_escrow,
                    "game": "elaisium",
                    **(metadata or {})
                }
            )
            
        except Exception as e:
            return PaymentResult(
                success=False,
                status=PaymentStatus.FAILED,
                error_message=str(e)
            )
    
    async def verify_payment(
        self,
        transaction_id: str
    ) -> PaymentResult:
        """Verify a VIBE payment transaction.
        
        Args:
            transaction_id: The transaction ID to verify
            
        Returns:
            PaymentResult with verification status
        """
        # In real implementation, query Elaisium API
        # For now, simulate verification
        
        if transaction_id.startswith("vibe_tx_"):
            return PaymentResult(
                success=True,
                status=PaymentStatus.CAPTURED,
                transaction_id=transaction_id,
                metadata={"verified": True}
            )
        
        return PaymentResult(
            success=False,
            status=PaymentStatus.FAILED,
            error_message="Transaction not found"
        )
    
    async def refund_payment(
        self,
        transaction_id: str,
        amount: Optional[int] = None,
        reason: Optional[str] = None
    ) -> PaymentResult:
        """Refund VIBE from a transaction.
        
        Args:
            transaction_id: Original transaction ID
            amount: Amount to refund (None = full refund)
            reason: Reason for refund
            
        Returns:
            PaymentResult for the refund
        """
        # Parse original transaction to get details
        if not transaction_id.startswith("vibe_tx_"):
            return PaymentResult(
                success=False,
                status=PaymentStatus.FAILED,
                error_message="Invalid transaction ID"
            )
        
        refund_id = f"vibe_refund_{transaction_id}"
        
        return PaymentResult(
            success=True,
            status=PaymentStatus.REFUNDED,
            transaction_id=refund_id,
            amount=amount,
            currency="VIBE",
            metadata={
                "original_transaction": transaction_id,
                "reason": reason,
                "type": "refund"
            }
        )
    
    async def get_balance(self, entity_id: str) -> Dict[str, Any]:
        """Get VIBE balance for an entity.
        
        Args:
            entity_id: The entity ID to check
            
        Returns:
            Balance information
        """
        # In real implementation, query Elaisium API
        # Simulated response
        return {
            "entity_id": entity_id,
            "currency": "VIBE",
            "balance": 0,  # Would come from API
            "available": 0,
            "in_escrow": 0,
        }
    
    # =========================================================================
    # Elaisium-specific trading methods
    # =========================================================================
    
    async def trade_artifact(
        self,
        artifact_id: str,
        from_entity: str,
        to_entity: str,
        vibe_price: int
    ) -> PaymentResult:
        """Trade an artifact for VIBE.
        
        Args:
            artifact_id: The artifact to trade
            from_entity: Seller entity ID
            to_entity: Buyer entity ID
            vibe_price: Price in VIBE
            
        Returns:
            PaymentResult for the trade
        """
        transaction_id = f"vibe_artifact_{artifact_id}_{from_entity}_{to_entity}"
        
        return PaymentResult(
            success=True,
            status=PaymentStatus.CAPTURED,
            transaction_id=transaction_id,
            amount=vibe_price,
            currency="VIBE",
            metadata={
                "type": "artifact_trade",
                "artifact_id": artifact_id,
                "seller": from_entity,
                "buyer": to_entity,
            }
        )
    
    async def purchase_service(
        self,
        service_type: str,
        provider_entity: str,
        consumer_entity: str,
        vibe_price: int,
        service_data: Optional[Dict[str, Any]] = None
    ) -> PaymentResult:
        """Purchase a service from another entity.
        
        Service types:
        - analysis: Data analysis or insights
        - guidance: Navigation or quest guidance
        - healing: Healing services
        - crafting: Artifact crafting
        - information: Knowledge or secrets
        
        Args:
            service_type: Type of service
            provider_entity: Service provider entity ID
            consumer_entity: Consumer entity ID
            vibe_price: Price in VIBE
            service_data: Additional service parameters
            
        Returns:
            PaymentResult for the service purchase
        """
        transaction_id = f"vibe_service_{service_type}_{provider_entity}_{consumer_entity}"
        
        return PaymentResult(
            success=True,
            status=PaymentStatus.CAPTURED,
            transaction_id=transaction_id,
            amount=vibe_price,
            currency="VIBE",
            metadata={
                "type": "service_purchase",
                "service_type": service_type,
                "provider": provider_entity,
                "consumer": consumer_entity,
                "service_data": service_data or {},
            }
        )
