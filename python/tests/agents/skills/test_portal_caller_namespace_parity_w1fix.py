"""
Memory-scoping parity (gap-closure wave-1 fix pass, lane w1-fix): a Portal
Connect caller with no `access:` block got no memory namespace in Python and
`caller:user:<id>` in TypeScript. The channels lane flagged it.

The cause was that Python's `portal_caller_auth` left `principals` at the
CallerAuth default `[]`, which `namespace_of` and the session skill's
`conversation_owner` read as "an access block verified nobody" (no namespace),
while TypeScript's `portalCallerAuth` sets no `principals` at all, so
`namespaceOf` derives the caller from its platform credential. Python now sets
`principals=None` on that caller, matching TypeScript; the shared fixture
`tests/fixtures/memory_tool/definition.json` gains the case (principals unset,
not empty).
"""

from __future__ import annotations

from webagents.access.caller import CallerAuth
from webagents.agents.skills.local.memory.memory_namespace import namespace_of
from webagents.agents.skills.local.session.skill import conversation_owner
from webagents.agents.skills.robutler.portal_connect.skill import portal_caller_auth


def test_portal_caller_with_no_access_block_scopes_to_its_user():
    caller = portal_caller_auth({"user_id": "p7", "tier": "user", "username": "ada"})
    assert caller is not None
    # Unset, not the empty list, so it is not read as "verified nobody".
    assert caller.principals is None
    # Both the memory namespace and the conversation owner derive the user.
    # namespace_of adds the `caller:` prefix; conversation_owner returns the
    # bare principal it keys a conversation on.
    assert namespace_of(caller) == "caller:user:p7"
    assert conversation_owner(caller) == "user:p7"


def test_a_portal_owner_is_the_owner_namespace():
    caller = portal_caller_auth({"user_id": "u1", "tier": "owner"})
    assert namespace_of(caller) == "owner"
    assert conversation_owner(caller) == "owner"


def test_an_access_block_that_verified_nobody_still_gets_no_namespace():
    # The distinction the fix preserves: an EMPTY principals list (an access
    # block ran and verified nobody) is not the same as an ABSENT one, and
    # still scopes to nothing even with a user_id present.
    verified_nobody = CallerAuth(scope="user", user_id="4c3a", authenticated=True, provider="platform", principals=[])
    assert namespace_of(verified_nobody) is None
    assert conversation_owner(verified_nobody) is None
