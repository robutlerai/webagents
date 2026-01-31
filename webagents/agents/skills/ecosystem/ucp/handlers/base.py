"""
Base Payment Handler - Abstract Interface for UCP Payment Handlers

Defines the common interface that all payment handlers must implement.
Each handler processes payments for a specific payment provider/method.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum


class PaymentStatus(str, Enum):
    """Payment processing status"""
    PENDING = "pending"
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    FAILED = "failed"
    CANCELED = "canceled"
    REFUNDED = "refunded"


@dataclass
class PaymentInstrument:
    """
    Payment instrument data for processing.
    
    This represents the payment credentials/data that will be sent
    to the merchant for processing (e.g., card token, wallet data).
    """
    handler_id: str
    type: str  # e.g., "card", "token", "wallet"
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_ucp_format(self) -> Dict[str, Any]:
        """Convert to UCP payment instrument format"""
        return {
            "handler_id": self.handler_id,
            "type": self.type,
            "data": self.data
        }


@dataclass
class PaymentResult:
    """
    Result of payment processing.
    
    Contains the outcome of a payment attempt including
    any transaction identifiers and error information.
    """
    success: bool
    status: PaymentStatus
    transaction_id: Optional[str] = None
    authorization_code: Optional[str] = None
    amount: Optional[int] = None  # Amount in cents
    currency: Optional[str] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "status": self.status.value,
            "transaction_id": self.transaction_id,
            "authorization_code": self.authorization_code,
            "amount": self.amount,
            "currency": self.currency,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "metadata": self.metadata
        }


class PaymentHandler(ABC):
    """
    Abstract base class for UCP payment handlers.
    
    Each payment handler implements support for a specific payment
    provider (Stripe, Google Pay, Robutler, etc.) and handles:
    - Configuration from merchant profile
    - Instrument creation/tokenization
    - Payment processing
    
    Subclasses must implement:
    - namespace: The handler namespace (e.g., "com.stripe.payments.card")
    - create_instrument(): Create payment instrument from credentials
    - can_handle(): Check if handler can process given config
    """
    
    # Class-level handler namespace (override in subclass)
    namespace: str = "base"
    display_name: str = "Base Handler"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize handler with configuration.
        
        Args:
            config: Handler configuration from merchant profile or skill config
        """
        self.config = config or {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """
        Initialize the handler (connect to APIs, validate credentials, etc.)
        
        Override in subclass if async initialization is needed.
        """
        self._initialized = True
    
    @abstractmethod
    async def create_instrument(
        self,
        credentials: Dict[str, Any],
        handler_config: Dict[str, Any]
    ) -> PaymentInstrument:
        """
        Create a payment instrument from raw credentials.
        
        This transforms user-provided payment data (card number, token, etc.)
        into a UCP-compliant payment instrument that can be sent to the merchant.
        
        Args:
            credentials: Raw payment credentials (card data, token, etc.)
            handler_config: Handler configuration from merchant profile
            
        Returns:
            PaymentInstrument ready for checkout
            
        Raises:
            UCPPaymentError: If instrument creation fails
        """
        pass
    
    @abstractmethod
    def can_handle(self, handler_config: Dict[str, Any]) -> bool:
        """
        Check if this handler can process the given merchant handler config.
        
        Args:
            handler_config: Handler configuration from merchant's /.well-known/ucp
            
        Returns:
            True if this handler can process payments for this config
        """
        pass
    
    def get_required_credentials(self) -> List[Dict[str, Any]]:
        """
        Get list of required credential fields for this handler.
        
        Returns:
            List of credential field definitions with name, type, required, etc.
        """
        return []
    
    def validate_credentials(self, credentials: Dict[str, Any]) -> bool:
        """
        Validate that provided credentials meet requirements.
        
        Args:
            credentials: Credentials to validate
            
        Returns:
            True if credentials are valid
        """
        required = self.get_required_credentials()
        for field_def in required:
            if field_def.get("required", False):
                if field_def["name"] not in credentials:
                    return False
        return True
    
    def get_handler_info(self) -> Dict[str, Any]:
        """
        Get handler information for display/negotiation.
        
        Returns:
            Handler metadata
        """
        return {
            "namespace": self.namespace,
            "display_name": self.display_name,
            "required_credentials": self.get_required_credentials(),
            "initialized": self._initialized
        }
    
    async def cleanup(self) -> None:
        """
        Cleanup handler resources.
        
        Override in subclass if cleanup is needed.
        """
        self._initialized = False
