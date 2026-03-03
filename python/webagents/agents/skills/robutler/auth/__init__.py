"""
AuthSkill Package - WebAgents V2.0

Authentication and authorization skill for WebAgents platform.
Provides user authentication, API key validation, role-based access control,
and integration with WebAgents platform services.
"""

from .skill import AuthSkill, AuthScope, AuthContext, AuthenticationError, AuthorizationError

__all__ = [
    "AuthSkill", 
    "AuthScope", 
    "AuthContext",
    "AuthenticationError", 
    "AuthorizationError"
]
