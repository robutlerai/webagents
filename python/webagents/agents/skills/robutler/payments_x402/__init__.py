"""
PaymentSkillX402 - x402 Payment Protocol Support

Full x402 protocol integration for WebAgents, enabling agents to provide
and consume paid APIs using multiple payment schemes.
"""

from .skill import PaymentSkillX402

# The MPP buyer (machine-purchase design section 6.4, 2026-09-18): buys
# platform usage from Robutler on a 402. The TypeScript `skills/payments`
# re-exports its twin the same way.
from .mpp_buyer import (
    CardPaymentSource,
    MppBuyer,
    MppBuyerPersistence,
    MppBuyerPolicy,
    MppBuyerRefusal,
    MppPendingCredential,
    MppPurchaseOutcome,
    MppPurchaseRecord,
    # Raised by `paying_fetch` when the host answers with a redirect, which
    # is never followed (S-180, 2026-09-19); a caller needs the name to catch it.
    MppRedirectError,
    MppTermsRequest,
    SptRequest,
    StablecoinPaymentSource,
    TempoTransferRequest,
)

__all__ = [
    "PaymentSkillX402",
    "CardPaymentSource",
    "MppBuyer",
    "MppBuyerPersistence",
    "MppBuyerPolicy",
    "MppBuyerRefusal",
    "MppPendingCredential",
    "MppPurchaseOutcome",
    "MppPurchaseRecord",
    "MppRedirectError",
    "MppTermsRequest",
    "SptRequest",
    "StablecoinPaymentSource",
    "TempoTransferRequest",
]

