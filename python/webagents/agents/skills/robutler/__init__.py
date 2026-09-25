"""
Robutler Platform Skills

Skills that integrate with Robutler platform services.

IMPORTED LAZILY, AND THAT IS THE POINT (2026-09-23, S-216). These imports used
to run eagerly, which made `import webagents` drag in the whole platform stack:

    webagents/agents/skills/__init__.py  -> .robutler.crm
    webagents/agents/skills/robutler/__init__.py -> .auth
    .../robutler/auth/skill.py -> from robutler.api import RobutlerClient
    robutler/__init__.py -> ... -> LiteLLMSkill -> import litellm
    litellm/__init__.py:20 -> load_dotenv()

`litellm` calls `load_dotenv()` at ITS module import, unconditionally. So
importing any part of this SDK rewrote the host process's environment from
whatever `.env` the walk happened to find. Measured before this change:
importing `webagents.cli.main` added 37 variables that the shell did not have,
including `OPENAI_API_KEY`, `STRIPE_SECRET_KEY` and `POSTGRES_URL`, sourced
from the parent repository's `.env` two directories above the SDK.

Removing webagents' own `load_dotenv()` call (in the OpenAI Agent Builder
skill) was necessary and did not change that number, because the dominant
source was two packages away and reached through this file. Deferring these
imports is what actually breaks the chain: nothing here loads until someone
asks for a name, and asking for `AuthSkill` is a deliberate act.

PEP 562 module `__getattr__`, so every existing spelling keeps working:
`from webagents.agents.skills.robutler import AuthSkill` and
`skills.robutler.AuthSkill` both resolve on first use.
"""

from typing import Any

#: Public name -> the submodule that defines it. `__getattr__` consults this,
#: so adding a skill means adding one line here rather than an import that
#: runs for everybody.
_EXPORTS = {
    "CRMAnalyticsSkill": (".crm", "CRMAnalyticsSkill"),
    "AuthSkill": (".auth", "AuthSkill"),
    "ChatsSkill": (".chats", "ChatsSkill"),
    "DiscoverySkill": (".discovery", "DiscoverySkill"),
    "NamespaceSkill": (".namespace", "NamespaceSkill"),
    "PublishSkill": (".publish", "PublishSkill"),
    "PricingInfo": (".payments", "PricingInfo"),
    "pricing": (".payments", "pricing"),
    # The base implementation, kept reachable under its own name.
    "PaymentSkillBase": (".payments", "PaymentSkill"),
    # The default is the x402-enabled superset, under both spellings.
    "PaymentSkill": (".payments_x402", "PaymentSkillX402"),
    "PaymentSkillX402": (".payments_x402", "PaymentSkillX402"),
    "PortalConnectSkill": (".portal_connect", "PortalConnectSkill"),
    "SocialSkill": (".social", "SocialSkill"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve a skill on first use. See the module docstring."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None

    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attribute)
    # Cache on the module so the next lookup skips all of this.
    globals()[name] = value
    return value


def __dir__() -> list:
    """`dir()` still answers, which star-imports and tab-completion rely on."""
    return sorted(set(__all__) | set(globals()))
