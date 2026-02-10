"""
x402 payment scheme helpers

Encoding, decoding, and validation utilities for x402 payment headers.
Supports both legacy JSON payload and JWT payment tokens.
"""

import base64
import json
from typing import Dict, Any


def _is_jwt_string(s: str) -> bool:
    """Return True if s looks like a compact JWT (three base64url segments)."""
    parts = s.strip().split(".")
    return len(parts) == 3 and all(
        len(p) > 0 and p.replace("-", "").replace("_", "").isalnum()
        for p in parts
    )


def encode_robutler_payment(token: str, amount: str, network: str = "robutler") -> str:
    """
    Encode robutler token payment as x402 payment header.
    
    Args:
        token: Robutler payment token (JWT string or legacy tok_xxx:secret_yyy)
        amount: Amount to charge
        network: Network identifier (default: "robutler")
        
    Returns:
        Base64 encoded payment header (legacy JSON). If token is already a JWT,
        callers may send the JWT as X-PAYMENT directly per platform docs.
    """
    payment_data = {
        'scheme': 'token',
        'network': network,
        'payload': {
            'token': token,
            'amount': amount
        }
    }
    
    return base64.b64encode(
        json.dumps(payment_data).encode()
    ).decode()


def decode_payment_header(payment_header: str) -> Dict[str, Any]:
    """
    Decode x402 payment header. Supports:
    - Legacy: base64-encoded JSON with scheme, network, payload.
    - JWT: raw JWT string or base64-encoded JWT in X-PAYMENT.
    
    Args:
        payment_header: Base64 encoded payment header or raw JWT
        
    Returns:
        Dict with scheme, network, payload. If JWT: also _is_jwt=True and
        _raw_token set to the token string for verify/settle.
        
    Raises:
        ValueError: If payment header is invalid
    """
    raw = payment_header.strip()
    # Raw JWT (three base64url parts)
    if _is_jwt_string(raw):
        return {
            "scheme": "token",
            "network": "robutler",
            "payload": {"token": raw, "amount": None},
            "_is_jwt": True,
            "_raw_token": raw,
        }
    try:
        decoded = base64.b64decode(raw).decode("utf-8")
    except Exception as e:
        raise ValueError(f"Invalid payment header: {e}") from e
    # Base64-encoded JWT
    if _is_jwt_string(decoded):
        return {
            "scheme": "token",
            "network": "robutler",
            "payload": {"token": decoded, "amount": None},
            "_is_jwt": True,
            "_raw_token": decoded,
        }
    # Legacy JSON
    try:
        data = json.loads(decoded)
        data["_is_jwt"] = False
        data["_raw_token"] = raw
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid payment header: {e}") from e


def validate_payment_header(payment_data: Dict[str, Any]) -> bool:
    """
    Validate decoded payment header structure.
    
    Args:
        payment_data: Decoded payment header dict
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['scheme', 'network', 'payload']
    return all(field in payment_data for field in required_fields)


def extract_token_from_payment(payment_data: Dict[str, Any]) -> str:
    """
    Extract robutler token from decoded payment header.
    Prefers _raw_token (JWT or legacy) for API calls.
    
    Args:
        payment_data: Decoded payment header dict
        
    Returns:
        Token string (JWT or legacy)
        
    Raises:
        ValueError: If token not found in payload
    """
    if payment_data.get("_raw_token"):
        return payment_data["_raw_token"]
    if payment_data.get("scheme") != "token":
        raise ValueError(f"Not a token scheme: {payment_data.get('scheme')}")
    payload = payment_data.get("payload", {})
    token = payload.get("token")
    if not token:
        raise ValueError("Token not found in payment payload")
    return token


def create_x402_requirements(
    scheme: str,
    network: str,
    amount: float,
    resource: str,
    pay_to: str,
    description: str = "",
    mime_type: str = "application/json"
) -> Dict[str, Any]:
    """
    Create x402 PaymentRequirements object.
    
    Args:
        scheme: Payment scheme (e.g., "token", "exact")
        network: Network identifier
        amount: Required amount
        resource: Resource URL/path
        pay_to: Payment recipient identifier
        description: Human-readable description
        mime_type: Response MIME type
        
    Returns:
        x402 PaymentRequirements dict
    """
    return {
        'scheme': scheme,
        'network': network,
        'maxAmountRequired': str(amount),
        'resource': resource,
        'payTo': pay_to,
        'description': description,
        'mimeType': mime_type,
        'maxTimeoutSeconds': 60
    }


def create_x402_response(accepts: list) -> Dict[str, Any]:
    """
    Create x402 402 Payment Required response.
    
    Args:
        accepts: List of accepted payment requirements
        
    Returns:
        x402 response dict
    """
    return {
        'x402Version': 1,
        'accepts': accepts
    }

