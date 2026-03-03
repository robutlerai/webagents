"""
NLI Response Signing - RS256 JWT signatures for non-repudiation.

Callee agents sign their NLI responses so that callers can submit
cryptographically verifiable evidence when flagging harmful content.

Signatures are OPTIONAL. Agents without private keys skip signing.
Unsigned responses still work for NLI calls but flags against them
carry reduced weight (UNVERIFIED_AGENT_FLAG_WEIGHT = 0.05 vs 0.3).
"""

import hashlib
import logging
import time
import uuid
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    import jwt as pyjwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    pyjwt = None


def compute_hash(data: str) -> str:
    """SHA-256 hash of a string, hex-encoded."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def sign_nli_response(
    response_text: str,
    request_payload: str,
    callee_agent_id: str,
    caller_agent_id: Optional[str],
    private_key: Any,
    kid: Optional[str] = None,
    ttl_seconds: int = 300,
) -> Optional[str]:
    """Sign an NLI response with RS256.

    Args:
        response_text: Full response text from the callee.
        request_payload: Original request payload string.
        callee_agent_id: The responding agent's identifier (iss).
        caller_agent_id: The calling agent's identifier (aud), if known.
        private_key: RSA private key object for signing.
        kid: Key ID for the JWT header.
        ttl_seconds: Signature validity period.

    Returns:
        Signed JWT string, or None if signing is unavailable.
    """
    if not JWT_AVAILABLE or private_key is None:
        return None

    try:
        now = int(time.time())
        payload: Dict[str, Any] = {
            "iss": callee_agent_id,
            "response_hash": compute_hash(response_text),
            "request_hash": compute_hash(request_payload),
            "iat": now,
            "exp": now + ttl_seconds,
            "jti": str(uuid.uuid4()),
        }
        if caller_agent_id:
            payload["aud"] = caller_agent_id

        headers: Dict[str, Any] = {}
        if kid:
            headers["kid"] = kid

        token = pyjwt.encode(payload, private_key, algorithm="RS256", headers=headers)
        return token
    except Exception as e:
        logger.warning(f"NLI response signing failed: {e}")
        return None


def verify_nli_signature(
    signature_jwt: str,
    expected_response_hash: str,
    expected_request_hash: Optional[str],
    callee_public_key: Any,
    caller_agent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify an NLI response signature.

    Args:
        signature_jwt: The JWT signature to verify.
        expected_response_hash: SHA-256 of the response text.
        expected_request_hash: SHA-256 of the request payload (optional).
        callee_public_key: RSA public key of the callee agent.
        caller_agent_id: Expected audience (optional).

    Returns:
        Dict with 'valid' bool, 'claims' if valid, 'error' if not.
    """
    if not JWT_AVAILABLE:
        return {"valid": False, "error": "jwt library not available"}

    try:
        decode_opts: Dict[str, Any] = {"algorithms": ["RS256"]}
        if caller_agent_id:
            decode_opts["audience"] = caller_agent_id
        else:
            decode_opts["options"] = {"verify_aud": False}

        claims = pyjwt.decode(signature_jwt, callee_public_key, **decode_opts)

        if claims.get("response_hash") != expected_response_hash:
            return {"valid": False, "error": "response_hash mismatch"}

        if expected_request_hash and claims.get("request_hash") != expected_request_hash:
            return {"valid": False, "error": "request_hash mismatch"}

        return {"valid": True, "claims": claims}
    except pyjwt.ExpiredSignatureError:
        # For evidence verification, we accept expired signatures
        try:
            claims = pyjwt.decode(
                signature_jwt,
                callee_public_key,
                algorithms=["RS256"],
                options={"verify_exp": False, "verify_aud": False},
            )
            if claims.get("response_hash") != expected_response_hash:
                return {"valid": False, "error": "response_hash mismatch"}
            return {"valid": True, "claims": claims, "expired": True}
        except Exception as e:
            return {"valid": False, "error": f"expired + invalid: {e}"}
    except Exception as e:
        return {"valid": False, "error": str(e)}
