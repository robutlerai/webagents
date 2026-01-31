"""
UCP Exceptions - Universal Commerce Protocol Error Classes

Provides structured error handling for UCP operations including:
- Discovery errors
- Checkout errors  
- Payment processing errors
- Handler-specific errors
"""

from typing import Optional, Dict, Any


class UCPError(Exception):
    """Base exception for all UCP errors"""
    
    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ):
        super().__init__(message)
        self.message = message
        self.code = code or "UCP_ERROR"
        self.details = details or {}
        self.status_code = status_code
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses"""
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
            "status_code": self.status_code
        }


class UCPDiscoveryError(UCPError):
    """Error during merchant/capability discovery"""
    
    def __init__(
        self,
        message: str,
        merchant_url: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code="UCP_DISCOVERY_ERROR",
            details={**(details or {}), "merchant_url": merchant_url},
            status_code=502
        )
        self.merchant_url = merchant_url


class UCPCheckoutError(UCPError):
    """Error during checkout session operations"""
    
    def __init__(
        self,
        message: str,
        checkout_id: Optional[str] = None,
        checkout_status: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code="UCP_CHECKOUT_ERROR",
            details={
                **(details or {}),
                "checkout_id": checkout_id,
                "checkout_status": checkout_status
            },
            status_code=400
        )
        self.checkout_id = checkout_id
        self.checkout_status = checkout_status


class UCPPaymentError(UCPError):
    """Error during payment processing"""
    
    def __init__(
        self,
        message: str,
        handler_name: Optional[str] = None,
        payment_status: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 402
    ):
        super().__init__(
            message=message,
            code="UCP_PAYMENT_ERROR",
            details={
                **(details or {}),
                "handler": handler_name,
                "payment_status": payment_status
            },
            status_code=status_code
        )
        self.handler_name = handler_name
        self.payment_status = payment_status


class UCPHandlerError(UCPError):
    """Error in payment handler operations"""
    
    def __init__(
        self,
        message: str,
        handler_name: str,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code="UCP_HANDLER_ERROR",
            details={
                **(details or {}),
                "handler": handler_name,
                "operation": operation
            },
            status_code=500
        )
        self.handler_name = handler_name
        self.operation = operation


class UCPCapabilityNotSupported(UCPError):
    """Requested capability is not supported by merchant"""
    
    def __init__(
        self,
        capability: str,
        available_capabilities: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"Capability '{capability}' is not supported",
            code="UCP_CAPABILITY_NOT_SUPPORTED",
            details={
                **(details or {}),
                "requested_capability": capability,
                "available_capabilities": available_capabilities or []
            },
            status_code=501
        )
        self.capability = capability
        self.available_capabilities = available_capabilities or []


class UCPEscalationRequired(UCPError):
    """Checkout requires user escalation (browser handoff)"""
    
    def __init__(
        self,
        message: str,
        continue_url: str,
        checkout_id: Optional[str] = None,
        reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code="UCP_ESCALATION_REQUIRED",
            details={
                **(details or {}),
                "continue_url": continue_url,
                "checkout_id": checkout_id,
                "reason": reason
            },
            status_code=303  # See Other - redirect
        )
        self.continue_url = continue_url
        self.checkout_id = checkout_id
        self.reason = reason


class UCPInvalidPaymentInstrument(UCPPaymentError):
    """Payment instrument is invalid or expired"""
    
    def __init__(
        self,
        message: str,
        instrument_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            handler_name=None,
            payment_status="invalid_instrument",
            details={**(details or {}), "instrument_type": instrument_type},
            status_code=400
        )
        self.instrument_type = instrument_type
