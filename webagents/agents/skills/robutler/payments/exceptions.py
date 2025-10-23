"""
Payment Exceptions - WebAgents V2.0 Platform Integration

Comprehensive payment error hierarchy for distinguishing between different 402 payment failure scenarios.
Provides specific error codes, subcodes, and context for better error handling and user experience.
"""

from typing import Dict, Any, Optional
from decimal import Decimal


class PaymentError(Exception):
    """Base payment error with 402 status code and detailed context"""
    
    def __init__(self, 
                 message: str,
                 error_code: str,
                 subcode: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None,
                 user_message: Optional[str] = None):
        super().__init__(message)
        self.status_code = 402  # Payment Required
        self.error_code = error_code
        self.subcode = subcode
        self.context = context or {}
        self.user_message = user_message or message
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses"""
        return {
            'error': self.error_code,
            'subcode': self.subcode,
            'message': str(self),
            'user_message': self.user_message,
            'status_code': self.status_code,
            'context': self.context
        }
    
    def __str__(self) -> str:
        parts = [self.error_code]
        if self.subcode:
            parts.append(f"({self.subcode})")
        parts.append(f": {super().__str__()}")
        return " ".join(parts)

    # FastAPI/Starlette HTTPException expects a `detail` field for structured errors.
    # Our server checks for `hasattr(e, 'status_code') and hasattr(e, 'detail')` to
    # map domain errors to HTTP responses. Expose detail dynamically from to_dict().
    @property
    def detail(self) -> Dict[str, Any]:
        return self.to_dict()


class PaymentTokenRequiredError(PaymentError):
    """Raised when payment token is required but not provided"""
    
    def __init__(self, agent_name: Optional[str] = None):
        context = {}
        if agent_name:
            context['agent_name'] = agent_name
            
        super().__init__(
            message="Payment token required for billing-enabled agent",
            error_code="PAYMENT_TOKEN_REQUIRED",
            context=context,
            user_message="This agent requires payment. Please provide a valid payment token."
        )


class PaymentTokenInvalidError(PaymentError):
    """Raised when payment token is invalid or expired"""
    
    def __init__(self, 
                 token_prefix: Optional[str] = None,
                 reason: Optional[str] = None):
        context = {}
        if token_prefix:
            context['token_prefix'] = token_prefix
        if reason:
            context['validation_error'] = reason
            
        subcode = None
        user_message = "Payment token is invalid or expired. Please check your token and try again."
        
        if reason:
            if "expired" in reason.lower():
                subcode = "TOKEN_EXPIRED"
                user_message = "Payment token has expired. Please obtain a new token."
            elif "not found" in reason.lower():
                subcode = "TOKEN_NOT_FOUND"
                user_message = "Payment token not found. Please check your token and try again."
            elif "malformed" in reason.lower():
                subcode = "TOKEN_MALFORMED"
                user_message = "Payment token format is invalid. Please check your token."
            
        super().__init__(
            message=f"Payment token validation failed: {reason}" if reason else "Payment token is invalid",
            error_code="PAYMENT_TOKEN_INVALID",
            subcode=subcode,
            context=context,
            user_message=user_message
        )


class InsufficientBalanceError(PaymentError):
    """Raised when payment token or account balance is insufficient"""
    
    def __init__(self, 
                 current_balance: float,
                 required_balance: float,
                 token_prefix: Optional[str] = None,
                 is_token_balance: bool = True):
        context = {
            'current_balance': current_balance,
            'required_balance': required_balance,
            'shortfall': required_balance - current_balance
        }
        if token_prefix:
            context['token_prefix'] = token_prefix
        
        # Distinguish between token balance (can retry with new token) 
        # and account balance (user needs to add credits)
        if is_token_balance:
            error_code = "INSUFFICIENT_TOKEN_BALANCE"
            user_message = f"Payment token has insufficient balance (${current_balance:.2f} < ${required_balance:.2f}). Requesting fresh token..."
        else:
            error_code = "INSUFFICIENT_ACCOUNT_BALANCE"
            user_message = f"Your account has insufficient credits. You need ${required_balance:.2f} but only have ${current_balance:.2f}. Please add more credits."
            
        super().__init__(
            message=f"Insufficient balance: ${current_balance:.2f} < ${required_balance:.2f} required",
            error_code=error_code,
            context=context,
            user_message=user_message
        )


class PaymentChargingError(PaymentError):
    """Raised when charging/redeeming payment token fails"""
    
    def __init__(self, 
                 amount: float,
                 token_prefix: Optional[str] = None,
                 reason: Optional[str] = None):
        context = {
            'charge_amount': amount
        }
        if token_prefix:
            context['token_prefix'] = token_prefix
        if reason:
            context['charge_error'] = reason
            
        subcode = None
        user_message = "Payment processing failed. Please try again or contact support."
        
        if reason:
            if "insufficient" in reason.lower():
                subcode = "INSUFFICIENT_FUNDS"
                user_message = "Insufficient funds for this transaction. Please add more credits."
            elif "expired" in reason.lower():
                subcode = "TOKEN_EXPIRED_DURING_CHARGE"
                user_message = "Payment token expired during transaction. Please obtain a new token."
            elif "limit" in reason.lower():
                subcode = "SPENDING_LIMIT_EXCEEDED"
                user_message = "Spending limit exceeded. Please check your account limits."
                
        super().__init__(
            message=f"Payment charging failed: {reason}" if reason else f"Failed to charge ${amount:.2f}",
            error_code="PAYMENT_CHARGING_FAILED",
            subcode=subcode,
            context=context,
            user_message=user_message
        )


class PaymentPlatformUnavailableError(PaymentError):
    """Raised when payment platform is unavailable"""
    
    def __init__(self, operation: Optional[str] = None):
        context = {}
        if operation:
            context['attempted_operation'] = operation
            
        super().__init__(
            message=f"Payment platform unavailable for {operation}" if operation else "Payment platform unavailable",
            error_code="PAYMENT_PLATFORM_UNAVAILABLE",
            context=context,
            user_message="Payment system is temporarily unavailable. Please try again later."
        )


class PaymentConfigurationError(PaymentError):
    """Raised when payment configuration is invalid"""
    
    def __init__(self, 
                 config_issue: str,
                 details: Optional[str] = None):
        context = {
            'config_issue': config_issue
        }
        if details:
            context['details'] = details
            
        super().__init__(
            message=f"Payment configuration error: {config_issue}",
            error_code="PAYMENT_CONFIG_ERROR",
            context=context,
            user_message="Payment system configuration error. Please contact support."
        )


# Convenience functions for common error scenarios
def create_token_required_error(agent_name: Optional[str] = None) -> PaymentTokenRequiredError:
    """Create a payment token required error"""
    return PaymentTokenRequiredError(agent_name=agent_name)


def create_token_invalid_error(token_prefix: Optional[str] = None, 
                              reason: Optional[str] = None) -> PaymentTokenInvalidError:
    """Create a payment token invalid error"""
    return PaymentTokenInvalidError(token_prefix=token_prefix, reason=reason)


def create_insufficient_balance_error(current_balance: float,
                                    required_balance: float,
                                    token_prefix: Optional[str] = None,
                                    is_token_balance: bool = True) -> InsufficientBalanceError:
    """Create an insufficient balance error
    
    Args:
        current_balance: Current balance available
        required_balance: Minimum balance required
        token_prefix: Optional token identifier prefix
        is_token_balance: True if checking token balance (can retry with fresh token),
                         False if checking account balance (user needs to add credits)
    """
    return InsufficientBalanceError(
        current_balance=current_balance,
        required_balance=required_balance,
        token_prefix=token_prefix,
        is_token_balance=is_token_balance
    )


def create_charging_error(amount: float,
                         token_prefix: Optional[str] = None,
                         reason: Optional[str] = None) -> PaymentChargingError:
    """Create a payment charging error"""
    return PaymentChargingError(
        amount=amount,
        token_prefix=token_prefix,
        reason=reason
    )


def create_platform_unavailable_error(operation: Optional[str] = None) -> PaymentPlatformUnavailableError:
    """Create a payment platform unavailable error"""
    return PaymentPlatformUnavailableError(operation=operation)


def create_config_error(config_issue: str,
                       details: Optional[str] = None) -> PaymentConfigurationError:
    """Create a payment configuration error"""
    return PaymentConfigurationError(config_issue=config_issue, details=details)


# Legacy compatibility - keep existing exception names but inherit from new hierarchy
class PaymentValidationError(PaymentTokenInvalidError):
    """Legacy compatibility - use PaymentTokenInvalidError instead"""
    pass


class PaymentRequiredError(PaymentTokenRequiredError):
    """Legacy compatibility - use PaymentTokenRequiredError instead"""
    pass 