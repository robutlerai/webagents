"""
Per-hop budgets for the NLI skill (webagents gap-closure plan 2.3,
2026-09-26), the twin of `typescript/src/skills/nli/budget.ts`; the fixture
`tests/fixtures/payments/delegate_budget.json` pins the parameter, the
receipt and the tree for both SDKs.

WHAT CHANGED. The skill delegated "the parent token's full remaining
balance" to every sub-agent, so one hop could spend the whole run's budget,
and nothing recorded what it did spend. A hop is now funded by a child token
for exactly the `budget` the model names (default 0.1 credits, at most 5),
minted through the platform's delegate route, which bounds it by the
parent's depth and balance and the payer's policy. The parent's balance is
never handed over whole; a hop that cannot be funded with a child of its own
is refused instead of forwarding the parent.

EACH HOP RETURNS A RECEIPT: the child token's balance is read back after the
hop and `budget - remaining` is what it spent; the line is appended to the
tool result. The owner reads the run's budget tree at
`GET /api/payments/tokens/{id}/tree` and prints it with `webagents budget`.
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any, Dict, Optional

import httpx

DELEGATE_BUDGET_PARAMETER: Dict[str, Any] = {
    "name": "budget",
    "description": (
        "The most this hop may spend, in credits (default 0.1, at most 5). A child token for exactly this budget "
        "is minted for the agent; the hop is refused when the budget cannot be derived."
    ),
    "default": 0.1,
    "max": 5,
}


def resolve_delegate_budget(value: Any) -> Dict[str, Any]:
    """`{"ok": True, "budget": ...}` for a bounded budget (the default when absent), else the refusal."""
    if value is None:
        return {"ok": True, "budget": DELEGATE_BUDGET_PARAMETER["default"]}
    try:
        budget = float(value)
    except (TypeError, ValueError):
        return {"ok": False, "error": "budget must be a positive number of credits"}
    if not (budget > 0) or budget != budget:
        return {"ok": False, "error": "budget must be a positive number of credits"}
    if budget > DELEGATE_BUDGET_PARAMETER["max"]:
        return {"ok": False, "error": f"budget {format_credits(budget)} exceeds maximum allowed ({DELEGATE_BUDGET_PARAMETER['max']} credits)"}
    return {"ok": True, "budget": budget}


def token_budget_of(jwt: str) -> Optional[float]:
    """The budget a payment token was minted with: its unverified `payment.balance` claim, or None."""
    parts = jwt.split(".")
    if len(parts) < 2:
        return None
    try:
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(padded))
    except Exception:
        return None
    payment = claims.get("payment") if isinstance(claims, dict) else None
    balance = payment.get("balance") if isinstance(payment, dict) else None
    if isinstance(balance, (int, float)) and not isinstance(balance, bool) and balance > 0:
        return float(balance)
    return None


def format_credits(value: float) -> str:
    """A credit amount as the receipts and the tree print it: at most 9 decimals, no trailing zeros."""
    rounded = round(float(value), 9)
    if rounded == int(rounded):
        return str(int(rounded))
    return repr(rounded)


def token_fingerprint(token_id: str) -> str:
    """The first 12 hex characters of the SHA-256 of a token id: names a token
    without being a bearer for it (S-303, 2026-09-26). A receipt used to print
    the child token's id, which the platform's settle auto-locks with no
    audience check, so a signed-in reader of the transcript could settle the
    remainder to themselves."""
    return hashlib.sha256(token_id.encode("utf-8")).hexdigest()[:12]


def format_delegate_receipt(receipt: Dict[str, Any]) -> str:
    return (
        f"[delegate receipt: budget {format_credits(receipt['budget'])} credits, spent {format_credits(receipt['spent'])}, "
        f"remaining {format_credits(receipt['remaining'])}, token {receipt['token']}]"
    )


def token_id_of(jwt: str) -> Optional[str]:
    """The `jti` of a payment token, unverified: the token row's id."""
    parts = jwt.split(".")
    if len(parts) < 2:
        return None
    try:
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(padded))
    except Exception:
        return None
    jti = claims.get("jti") if isinstance(claims, dict) else None
    return jti if isinstance(jti, str) else None


async def mint_child_token(
    platform_url: str,
    api_key: str,
    parent_token: str,
    delegate_to: str,
    budget: float,
    client: Optional[httpx.AsyncClient] = None,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """A child token for one hop from the platform's delegate route: `{"ok": True, "token", "tokenId", "amountCredits"}` or the refusal."""
    url = f"{platform_url.rstrip('/')}/api/payments/delegate"
    body = {"parentToken": parent_token, "delegateTo": delegate_to.lstrip("@"), "amount": budget}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    own = client is None
    client = client or httpx.AsyncClient(timeout=timeout)
    try:
        response = await client.post(url, json=body, headers=headers)
    except Exception as error:
        return {"ok": False, "error": f"the platform could not be reached to derive the budget: {error}"}
    finally:
        if own:
            await client.aclose()
    try:
        data = response.json()
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    token = data.get("token")
    if response.status_code != 200 or not isinstance(token, str) or not token:
        return {"ok": False, "error": data.get("error") if isinstance(data.get("error"), str) and data.get("error") else f"the delegate route answered {response.status_code}"}
    amount = data.get("amountCredits", data.get("amountDollars", budget))
    token_id = data.get("tokenId") if isinstance(data.get("tokenId"), str) else token_id_of(token)
    return {"ok": True, "token": token, "tokenId": token_id, "amountCredits": float(amount) if isinstance(amount, (int, float)) else budget}


async def read_child_receipt(
    platform_url: str,
    child_token: str,
    budget: float,
    api_key: Optional[str] = None,
    token_id: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> Optional[Dict[str, Any]]:
    """The child's balance after the hop, from the platform's verify route, as a receipt; None when unreadable."""
    url = f"{platform_url.rstrip('/')}/api/payments/verify"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    own = client is None
    client = client or httpx.AsyncClient(timeout=8.0)
    try:
        response = await client.post(url, json={"token": child_token}, headers=headers)
        data = response.json()
    except Exception:
        return None
    finally:
        if own:
            await client.aclose()
    if not isinstance(data, dict):
        return None
    remaining = data.get("balanceCredits", data.get("balanceDollars"))
    if not isinstance(remaining, (int, float)) or isinstance(remaining, bool):
        return None
    bounded = min(max(float(remaining), 0.0), float(budget))
    resolved_id = token_id or token_id_of(child_token)
    return {
        "budget": float(budget),
        "spent": round(float(budget) - bounded, 9),
        "remaining": round(bounded, 9),
        "token": token_fingerprint(resolved_id) if resolved_id else "unknown",
    }


__all__ = [
    "DELEGATE_BUDGET_PARAMETER",
    "resolve_delegate_budget",
    "format_credits",
    "token_fingerprint",
    "format_delegate_receipt",
    "token_id_of",
    "token_budget_of",
    "mint_child_token",
    "read_child_receipt",
]
