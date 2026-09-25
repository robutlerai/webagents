"""
PaymentSkill Package - WebAgents V2.0

Payment processing and billing skill for WebAgents platform.
Validates payment tokens, calculates costs, and charges on connection finalization.
Based on webagents_v1 implementation patterns.

THE SKILL IS IMPORTED LAZILY (2026-09-24). The transports (a2a, realtime,
acp, uamp) and `BaseAgent` import `payments.exceptions`, which is only
exception classes, but importing it runs THIS file first, and this file used
to import `.skill` eagerly. `.skill` imports `robutler.api`, and that pulls in
`robutler` and `litellm`: about 1.2 s on every agent run, `litellm`'s
import-time `load_dotenv()` (the S-216 chain, which this reopened for a plain
`webagents run`), and `robutler`'s global logger class, which printed INFO
lines into `run -p`'s answer. The exceptions stay eager because they are
cheap; the skill and its pricing helpers resolve on first use through a PEP
562 module `__getattr__`, so every existing spelling still works:
`from webagents.agents.skills.robutler.payments import pricing, PricingInfo`.
"""

from typing import Any

from .exceptions import (
    PaymentError,
    PaymentTokenRequiredError,
    PaymentTokenInvalidError,
    InsufficientBalanceError,
    PaymentChargingError,
    PaymentPlatformUnavailableError,
    PaymentConfigurationError,
    # Legacy compatibility
    PaymentValidationError,
    PaymentRequiredError
)

#: Names defined in `.skill`, resolved on first access.
_LAZY_FROM_SKILL = ("PaymentSkill", "PaymentContext", "PricingInfo", "pricing")


def __getattr__(name: str) -> Any:
    if name in _LAZY_FROM_SKILL:
        from . import skill

        value = getattr(skill, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(set(globals()) | set(_LAZY_FROM_SKILL))


__all__ = [
    # Main classes
    "PaymentSkill",
    "PaymentContext",
    # Pricing decorators
    "PricingInfo",
    "pricing",
    # New comprehensive error hierarchy
    "PaymentError",
    "PaymentTokenRequiredError",
    "PaymentTokenInvalidError",
    "InsufficientBalanceError",
    "PaymentChargingError",
    "PaymentPlatformUnavailableError",
    "PaymentConfigurationError",
    # Legacy compatibility
    "PaymentValidationError",
    "PaymentRequiredError"
]
