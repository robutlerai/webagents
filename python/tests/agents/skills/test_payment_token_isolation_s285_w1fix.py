"""
S-285 (2026-09-26): in the SDK's own server, a payment token one caller sent
on the UAMP socket landed on the agent's shared base context, and a later A2A
run by a different caller inherited it. The TypeScript SDK ran `processUAMP`
on a shared base context, which is where the bleed lived; the fix isolates
each run (`core/agent.ts` `_deriveRunContext`) and stops the transports
writing the token onto the base.

Python does not have a shared base context: every request and every websocket
connection builds a fresh `Context` and sets it on the `CONTEXT` contextvar
(`server/core/app.py`), and the UAMP skill writes the caller's token onto
`get_context()` (that per-connection context), never onto the agent. So there
is no cross-caller bleed to fix here. This test pins that isolation, so a
refactor to a shared per-agent context (the shape the TypeScript bug had)
would fail: a token on one request's context is not visible on another's, and
the payment skill reads the token of the context it runs under.

The TypeScript twin, with the review's serve() + UAMP + A2A probe, is
`tests/unit/transport/payment-token-isolation-s285-w1fix.test.ts`.
"""

from __future__ import annotations

from webagents.server.context.context_vars import CONTEXT, create_context, get_context, set_context


def _extract(context) -> str | None:
    """The transport-agnostic read the payment skill does first
    (`robutler/payments/skill.py` `_extract_payment_token`)."""
    if context is not None and getattr(context, "payment_token", None):
        token = context.payment_token
        if token and str(token).strip():
            return str(token).strip()
    return None


def test_each_request_gets_its_own_context_no_shared_base():
    """Two requests, two contexts: caller U's token is not on caller B's."""
    u = create_context(messages=[])
    u.payment_token = "tok-U"
    set_context(u)
    assert _extract(get_context()) == "tok-U"

    # A different request. Nothing carries U's context forward: `create_context`
    # makes a fresh object, and there is no per-agent context to inherit from.
    b = create_context(messages=[])
    set_context(b)
    assert getattr(b, "payment_token", None) is None
    assert _extract(get_context()) is None

    # And B naming its own token stays B's.
    b.payment_token = "tok-B"
    assert _extract(get_context()) == "tok-B"


def test_a_context_carries_no_caller_state_until_this_request_sets_it():
    """A fresh context has no payment token or auth: nothing to inherit."""
    context = create_context(messages=[])
    assert getattr(context, "payment_token", None) is None
    assert context.auth is None
    # custom_data is per-context, not shared across create_context calls.
    context.set("payment_token_key", "x")
    assert create_context(messages=[]).get("payment_token_key") is None


def test_the_contextvar_isolates_concurrent_callers():
    """Setting one caller's context and resetting does not leak to the next,
    which is what keeps two callers of one served agent apart."""
    first = create_context(messages=[])
    first.payment_token = "tok-U"
    token = CONTEXT.set(first)
    try:
        assert _extract(CONTEXT.get()) == "tok-U"
    finally:
        CONTEXT.reset(token)
    second = create_context(messages=[])
    CONTEXT.set(second)
    assert _extract(CONTEXT.get()) is None
