"""
Idempotency keys for `POST /api/payments/settle` (2026-09-26), the twin of
`typescript/src/skills/payments/idempotency.ts`. The shared fixture
`tests/fixtures/payments/settle_idempotency.json` pins the names and the
derivation for both SDKs and for the platform.

WHY. The platform charged again for every repeat of a settle against a lock
that still held more than the charge, so a lost answer could charge a payer
several times when this client retried (S-254; the client then stopped
retrying writes, which leaves a settle whose answer was lost as a charge the
agent cannot see). The platform now records each settle under the caller's
`Idempotency-Key` and answers a repeat with the first result, charging
nothing, provided the key is the same. So every settle this SDK sends carries
one, and the value must be STABLE for a retry of the same settle and NEW for
a genuinely new one:

  - `settle_idempotency_key(lock_id, purpose)` names a settle the skill's
    lifecycle can identify: one lock, one purpose (`usage`, `agent_fee`,
    `release`). Deriving it from those two, with no counter, is what makes a
    second finalize on the same context (the client's retry, or the run
    loop's error paths in `base_agent.py`, which finalize twice) a replay
    rather than a second charge. The invariant: a lock is settled for `usage`
    and for `agent_fee` at most once each, at finalize. A transport that must
    settle one lock twice for the same purpose widens the purpose; it never
    reuses one.
  - `fresh_settle_idempotency_key(scope)` is for a settle nothing can name:
    the legacy `redeem` by token (no lock in hand) or a raw client call.
    Minted once per call, so the call's own retry reuses it and two calls are
    two settles.
  - A PER-CALL settle (wave-0 review finding 5, 2026-09-26) names the tool
    call it charges for: `settle:<lock_id>:<purpose>:<call_id>`. Two settles
    for one purpose on one lock, made for two tool calls, derived the SAME
    key before, and the platform answered the second with the first's
    numbers, dropping its charge. With the call id in the purpose they are
    two keys; a retry of either still replays.
"""

from __future__ import annotations

import re
import uuid
from typing import Dict, Optional

#: The request header a settle's key travels in.
IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"
#: The body field carrying the same value, for a client that cannot set headers.
IDEMPOTENCY_KEY_BODY_FIELD = "idempotencyKey"
#: Set by the platform on the answer to a repeated settle: nothing was charged by that call.
IDEMPOTENT_REPLAYED_HEADER = "Idempotent-Replayed"
#: The platform's limit; a derived key is far shorter.
IDEMPOTENCY_KEY_MAX_LENGTH = 255

#: The settles the payment skill's lifecycle names, one of each per lock.
SETTLE_PURPOSES = ("usage", "agent_fee", "release")
#: A purpose is one word: the three above, or a charge type for an amount settle.
_PURPOSE_RE = re.compile(r"^[a-z0-9_]+$")
#: A tool-call id as the providers mint them (`call_…`, `toolu_…`): one token, no separators the key uses.
_CALL_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def settle_idempotency_key(lock_id: str, purpose: str, call_id: Optional[str] = None) -> str:
    """`settle:<lock_id>:<purpose>`: the same settle derives the same key.

    With `call_id`, `settle:<lock_id>:<purpose>:<call_id>`: the same settle
    for the same tool call derives the same key, and another call's never does.
    """
    if not lock_id:
        raise ValueError("settle_idempotency_key: a lock id is required")
    if not isinstance(purpose, str) or not _PURPOSE_RE.match(purpose):
        raise ValueError(f"settle_idempotency_key: invalid purpose {purpose!r}")
    if call_id is None:
        return f"settle:{lock_id}:{purpose}"
    if not isinstance(call_id, str) or not _CALL_ID_RE.match(call_id):
        raise ValueError(f"settle_idempotency_key: invalid call id {call_id!r}")
    return f"settle:{lock_id}:{purpose}:{call_id}"


def fresh_settle_idempotency_key(scope: str) -> str:
    """`settle:<scope>:<uuid4>`: a key for a settle nothing can name, minted once per call."""
    return f"settle:{scope}:{uuid.uuid4()}"


def idempotency_headers(key: str) -> Dict[str, str]:
    """The headers a settle request carries for its key."""
    return {IDEMPOTENCY_KEY_HEADER: key}


__all__ = [
    "IDEMPOTENCY_KEY_HEADER",
    "IDEMPOTENCY_KEY_BODY_FIELD",
    "IDEMPOTENT_REPLAYED_HEADER",
    "IDEMPOTENCY_KEY_MAX_LENGTH",
    "SETTLE_PURPOSES",
    "settle_idempotency_key",
    "fresh_settle_idempotency_key",
    "idempotency_headers",
]
