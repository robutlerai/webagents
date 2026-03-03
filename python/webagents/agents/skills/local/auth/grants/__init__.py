"""
OAuth Grant Handlers for AOAuth

Grant handlers implement different OAuth 2.0 grant types
for token issuance.
"""

from .client_credentials import ClientCredentialsGrant
from .authorization_code import AuthorizationCodeGrant

__all__ = [
    "ClientCredentialsGrant",
    "AuthorizationCodeGrant",
]
