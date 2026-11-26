"""
Reading `POST /api/payments/settle` (2026-09-18), the Python twin of the
TypeScript `src/skills/payments/settle-result.ts`.

WHY. Since the portal's S-148 fix `settlePaymentToken` charges a short lock
what it holds instead of nothing, and the route answers `success: true`
together with `partial`, and, when partial, `unbilled` and `requested`
(app/api/payments/settle/route.ts; `charged + unbilled = requested`, each in
nanocents as a string beside a `...Dollars` number). The SDK callers read only
`success`, so a partial settle read as charged in full. Every caller now goes
through this reader, which keeps the platform's fields, makes `partial` an
explicit bool (False on an older platform that never sends it, and never True
on a failure), logs a partial at warn, and leaves it to the caller to report
it as partial.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

_log = logging.getLogger(__name__)


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def read_settle_result(raw: Any, where: str, logger: Optional[Any] = None) -> Dict[str, Any]:
    """The settle answer as the SDK reads it; `where` names the caller in the warn line."""
    body = dict(raw) if isinstance(raw, Mapping) else {}
    success = body.get("success") is True
    body["success"] = success
    body["partial"] = success and body.get("partial") is True
    # A repeat under the settle's Idempotency-Key (2026-09-26): the first
    # settle's numbers, nothing charged by this call. Never True on a failure.
    if "replayed" in body:
        body["replayed"] = success and body.get("replayed") is True
    # Credit amounts: the platform answers `<name>Credits`, and older platforms only the same value as
    # `<name>Dollars`. Credits wins; both keys are filled so code reading either keeps working.
    for name in ("charged", "unbilled", "requested", "remaining"):
        value = _number(body.get(f"{name}Credits"))
        if value is None:
            value = _number(body.get(f"{name}Dollars"))
        if value is not None:
            body[f"{name}Credits"] = value
            body[f"{name}Dollars"] = value
    if body["partial"]:
        charged = body.get("chargedCredits", body.get("charged", "?"))
        requested = body.get("requestedCredits", body.get("requested", "?"))
        unbilled = body.get("unbilledCredits", body.get("unbilled", "?"))
        (logger or _log).warning(f"[payments] {where}: PARTIAL settle, charged {charged} of {requested} (unbilled {unbilled})")
    return body


__all__ = ["read_settle_result"]
