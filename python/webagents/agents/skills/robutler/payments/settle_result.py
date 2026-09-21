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


def read_settle_result(raw: Any, where: str, logger: Optional[Any] = None) -> Dict[str, Any]:
    """The settle answer as the SDK reads it; `where` names the caller in the warn line."""
    body = dict(raw) if isinstance(raw, Mapping) else {}
    success = body.get("success") is True
    body["success"] = success
    body["partial"] = success and body.get("partial") is True
    if body["partial"]:
        charged = body.get("chargedDollars", body.get("charged", "?"))
        requested = body.get("requestedDollars", body.get("requested", "?"))
        unbilled = body.get("unbilledDollars", body.get("unbilled", "?"))
        (logger or _log).warning(f"[payments] {where}: PARTIAL settle, charged {charged} of {requested} (unbilled {unbilled})")
    return body


__all__ = ["read_settle_result"]
