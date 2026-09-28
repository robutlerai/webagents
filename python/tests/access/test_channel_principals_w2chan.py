"""
A channel sender's identity on a relayed turn (the channel relay, plan item 2.2,
2026-09-26), from the table both SDKs run (`tests/fixtures/access/channel_caller.json`;
TypeScript tests/unit/access/channel-principals-w2chan.test.ts): what
`portal_caller_auth` makes of the platform's `caller.channel`, the principals the
access skill collects (the channel first, so it is the memory namespace), and the
sentence the model is told. The access block's decisions over channel principals
are in test_policy.py, from policy.json.
"""

import json
from pathlib import Path

import pytest

from webagents.access.caller import CallerAuth, channel_identity_of
from webagents.agents.skills.local.access.skill import _user_principals, who_is_calling
from webagents.agents.skills.local.memory.memory_namespace import namespace_of
from webagents.agents.skills.robutler.portal_connect.skill import portal_caller_auth

TABLE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "access" / "channel_caller.json").read_text())


@pytest.mark.parametrize("case", TABLE["assertions"], ids=lambda c: c["case"])
def test_the_platform_assertion_with_a_channel(case):
    auth = portal_caller_auth(case["caller"])
    assert isinstance(auth, CallerAuth)
    assert auth.scope == case["auth"]["scope"]
    assert auth.user_id == case["auth"]["user_id"]
    assert auth.provider == "portal"
    assert auth.username == case["auth"].get("username")
    assert auth.channel == case["auth"]["channel"]

    principals = _user_principals(auth)
    assert principals == case["principals"]
    # As the access skill leaves the auth after admitting the caller: the
    # principals it collected, and no group. (A fresh `CallerAuth` carries
    # `principals=[]`, which `namespace_of` reads as "an access block that
    # verified nobody"; the TypeScript auth has no `principals` until the
    # skill sets them, so it falls back to the same list. Both SDKs agree once
    # the skill has run, which is the case every relayed turn is in.)
    auth.principals = principals
    auth.groups = []
    assert namespace_of(auth) == case["namespace"]
    assert who_is_calling(auth) == case["who"]


def test_channel_identity_of_lower_cases_the_type_and_keeps_the_sender_id():
    assert channel_identity_of({"type": "Slack", "sender_id": "U9ABC"}) == {"type": "slack", "sender_id": "U9ABC"}


@pytest.mark.parametrize(
    "bad",
    [None, "telegram:8842", [], {"type": "telegram"}, {"sender_id": "8842"}, {"type": 3, "sender_id": "8842"},
     {"type": "telegram", "sender_id": 8842}, {"type": "-telegram", "sender_id": "1"}, {"type": "telegram", "sender_id": "a" * 201}],
)
def test_channel_identity_of_reads_no_channel_from_anything_else(bad):
    assert channel_identity_of(bad) is None


def test_every_type_the_platform_names_is_accepted():
    for kind in TABLE["types"].values():
        assert channel_identity_of({"type": kind, "sender_id": "x"}) == {"type": kind, "sender_id": "x"}
