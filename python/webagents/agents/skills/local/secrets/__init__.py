"""Secrets Skill - named credentials in the OS keystore."""
from .skill import SecretsSkill
from .store import (
    KeystoreUnavailableError,
    SecretStore,
    open_secret_store,
    service_key,
)

__all__ = [
    "SecretsSkill",
    "SecretStore",
    "KeystoreUnavailableError",
    "open_secret_store",
    "service_key",
]
