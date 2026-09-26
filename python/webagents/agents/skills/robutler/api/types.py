"""
Types for the Robutler platform API client (vendored into the SDK 2026-09-25).

`robutler/api/types.py` from the `robutlerai/robutler` repository at `9e6bd4d`,
unchanged but for this header. Why the client lives in the SDK: `client.py`.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional
from decimal import Decimal
from enum import Enum


class UserRole(Enum):
    """User role enumeration"""
    USER = "user"
    ADMIN = "admin"
    AGENT = "agent"


class SubscriptionStatus(Enum):
    """Subscription status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    CANCELED = "canceled"
    TRIALING = "trialing"
    PAST_DUE = "past_due"


class IntegrationType(Enum):
    """Integration type enumeration"""
    AGENT = "agent"
    MCP = "mcp"
    API = "api"


class IntegrationProtocol(Enum):
    """Integration protocol enumeration"""
    MCP = "mcp"
    A2A = "a2a"
    P2PMCP = "p2pmcp"
    HTTP = "http"


class TransactionType(Enum):
    """Credit transaction type enumeration"""
    ADDITION = "addition"
    USAGE = "usage"
    TRANSFER = "transfer"


@dataclass
class User:
    """Robutler platform user"""
    id: str
    name: Optional[str] = None
    email: str = ""
    role: UserRole = UserRole.USER
    google_id: Optional[str] = None
    avatar_url: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    # Stripe/Payment fields
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    stripe_product_id: Optional[str] = None
    plan_name: Optional[str] = None
    subscription_status: Optional[SubscriptionStatus] = None
    # Credits
    total_credits: Decimal = Decimal('0')
    used_credits: Decimal = Decimal('0')
    # Referrals
    referral_code: Optional[str] = None
    referred_by: Optional[str] = None
    referral_count: int = 0
    
    @property
    def available_credits(self) -> Decimal:
        """Calculate available credits"""
        return self.total_credits - self.used_credits
    
    @property
    def is_admin(self) -> bool:
        """Check if user is admin"""
        return self.role == UserRole.ADMIN
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'role': self.role.value,
            'avatar_url': self.avatar_url,
            'total_credits': str(self.total_credits),
            'used_credits': str(self.used_credits),
            'available_credits': str(self.available_credits),
            'subscription_status': self.subscription_status.value if self.subscription_status else None,
            'plan_name': self.plan_name,
            'referral_code': self.referral_code,
            'is_admin': self.is_admin
        }


@dataclass
class ApiKey:
    """Robutler platform API key"""
    id: str
    user_id: str
    name: str
    key_hash: str
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True
    daily_rate_limit: Optional[int] = None
    daily_credit_limit: Optional[Decimal] = None
    session_credit_limit: Optional[Decimal] = None
    spent_credits: Decimal = Decimal('0')
    earned_credits: Decimal = Decimal('0')
    permissions: Optional[Dict[str, Any]] = None
    integration_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if API key is expired"""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    @property
    def is_valid(self) -> bool:
        """Check if API key is valid"""
        return self.is_active and not self.is_expired
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'is_active': self.is_active,
            'is_expired': self.is_expired,
            'is_valid': self.is_valid,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'daily_credit_limit': str(self.daily_credit_limit) if self.daily_credit_limit else None,
            'session_credit_limit': str(self.session_credit_limit) if self.session_credit_limit else None,
            'spent_credits': str(self.spent_credits),
            'permissions': self.permissions
        }


@dataclass
class Integration:
    """Robutler platform integration"""
    id: str
    user_id: str
    agent_id: Optional[str] = None
    name: Optional[str] = None
    type: IntegrationType = IntegrationType.API
    protocol: IntegrationProtocol = IntegrationProtocol.HTTP
    secret: Optional[str] = None
    api_key_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'agent_id': self.agent_id,
            'name': self.name,
            'type': self.type.value,
            'protocol': self.protocol.value,
            'api_key_id': self.api_key_id,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


@dataclass
class CreditTransaction:
    """Robutler platform credit transaction"""
    id: str
    user_id: str
    integration_id: Optional[str] = None
    api_key_id: Optional[str] = None
    recipient_id: Optional[str] = None
    amount: Decimal = Decimal('0')
    type: TransactionType = TransactionType.USAGE
    source: str = "api_usage"
    description: Optional[str] = None
    receipt: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'integration_id': self.integration_id,
            'api_key_id': self.api_key_id,
            'amount': str(self.amount),
            'type': self.type.value,
            'source': self.source,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


@dataclass
class AuthResponse:
    """Authentication response from Robutler API"""
    success: bool
    user: Optional[User] = None
    api_key: Optional[ApiKey] = None
    error: Optional[str] = None
    message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'success': self.success,
            'error': self.error,
            'message': self.message
        }
        if self.user:
            result['user'] = self.user.to_dict()
        if self.api_key:
            result['api_key'] = self.api_key.to_dict()
        return result


@dataclass
class ApiResponse:
    """Generic API response from Robutler Platform"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    message: Optional[str] = None
    status_code: int = 200
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'message': self.message,
            'status_code': self.status_code
        } 