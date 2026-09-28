"""
Where the sandbox tests find srt (`@anthropic-ai/sandbox-runtime`).

The engine ships inside this package (`webagents/sandbox/sandbox_engine/`,
the sandbox-engine lane, 2026-09-27), so the tests run the copy a user gets:
`locate_srt` takes it unless `WEBAGENTS_SRT_CLI` names another. This file
used to point that variable at the TypeScript package's install beside the
checkout; with the engine bundled that would test a copy no Python user
runs, so it no longer does. When srt cannot run here (no node 20.11 or
later, or on Linux no bubblewrap, socat and ripgrep) every enforcement test
skips with the reason from `backend_status()`, never passes: a sandbox test
that passes on a machine that cannot sandbox would be the same class of lie
as S-217. `webagents sandbox setup` says what is missing.
"""

import pytest


@pytest.fixture(autouse=True)
def _no_caller_context_leaks_into_sandbox_tests():
    """Start every sandbox test outside any run: no caller context.

    `webagents.server.context.context_vars.CONTEXT` is a ContextVar, and tests
    elsewhere (`tests/agents/core/test_scopes.py`, `test_handoffs.py`) call
    `set_context()` without resetting it. A caller left there by an earlier
    test reaches the shell's `_caller_is_owner()`, which then treats the
    control test's direct command as a non-owner's and confines it, so the
    test that proves an undeclared agent is NOT confined failed only when run
    after them (found 2026-09-26 in the gap-closure build's full-suite gate).
    """
    from webagents.server.context.context_vars import CONTEXT

    token = CONTEXT.set(None)  # get_context() treats None as "outside any run"
    yield
    CONTEXT.reset(token)
