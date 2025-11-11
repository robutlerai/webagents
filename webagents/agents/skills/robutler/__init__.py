"""
Robutler Platform Skills

Skills that integrate with Robutler platform services.
"""

from .crm import CRMAnalyticsSkill
from .auth import AuthSkill
from .payments import PricingInfo, pricing
from .payments import PaymentSkill as PaymentSkillBase  # Base implementation
from .payments_x402 import PaymentSkillX402 as PaymentSkill  # Default: x402-enabled

# Also export x402 with original name for backward compatibility
PaymentSkillX402 = PaymentSkill

__all__ = [
    'CRMAnalyticsSkill',
    'AuthSkill',
    'PaymentSkill',  # Now points to PaymentSkillX402 (superset)
    'PaymentSkillBase',  # Original PaymentSkill (if needed)
    'PaymentSkillX402',  # Alias for PaymentSkill
    'PricingInfo',
    'pricing',
]