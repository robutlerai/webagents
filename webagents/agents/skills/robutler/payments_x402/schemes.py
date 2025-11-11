"""
x402 payment scheme helpers

Encoding, decoding, and validation utilities for x402 payment headers.
"""

import base64
import json
from typing import Dict, Any


def encode_robutler_payment(token: str, amount: str, network: str = "robutler") -> str:
    """
    Encode robutler token payment as x402 payment header.
    
    Args:
        token: Robutler payment token (tok_xxx:secret_yyy)
        amount: Amount to charge
        network: Network identifier (default: "robutler")
        
    Returns:
        Base64 encoded payment header
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
    Decode x402 payment header.
    
    Args:
        payment_header: Base64 encoded payment header
        
    Returns:
        Dict with scheme, network, and payload
        
    Raises:
        ValueError: If payment header is invalid
    """
    try:
        decoded = base64.b64decode(payment_header).decode('utf-8')
        return json.loads(decoded)
    except Exception as e:
        raise ValueError(f"Invalid payment header: {e}")


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
    
    Args:
        payment_data: Decoded payment header dict
        
    Returns:
        Token string
        
    Raises:
        ValueError: If token not found in payload
    """
    if payment_data.get('scheme') != 'token':
        raise ValueError(f"Not a token scheme: {payment_data.get('scheme')}")
    
    payload = payment_data.get('payload', {})
    token = payload.get('token')
    
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

