"""Secrets Skill - named credentials in the OS keystore."""
from .keychain_ux import KeychainDialogBlocked, forbid_dialogs
from .skill import SecretsSkill
from .store import (
    KeystoreUnavailableError,
    SecretStore,
    legacy_service_key,
    open_secret_store,
    service_key,
)

__all__ = [
    "SecretsSkill",
    "SecretStore",
    "KeystoreUnavailableError",
    "KeychainDialogBlocked",
    "forbid_dialogs",
    "legacy_service_key",
    "open_secret_store",
    "service_key",
]
