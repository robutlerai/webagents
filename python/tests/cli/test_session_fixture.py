"""
Sessions, the same in both SDKs (2026-09-25).

The cases are `tests/fixtures/sessions/sessions.json`, which the TypeScript
suite runs too (`typescript/tests/unit/cli/sessions-fixture.test.ts`): where a
served agent keeps a caller's conversation, which session a request names,
what is kept of a turn, what the chat records on Robutler, how `/resume`
merges this machine's list with Robutler's, and the sentences both say.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from webagents.agents.skills.local.session.skill import (
    conversation_owner,
    conversation_to_keep,
    request_session_id,
    session_backend_of,
)
from webagents.cli.robutler_sessions import (
    PlatformConversation,
    failure_sentence,
    merge_conversations,
    unavailable_reason,
    words_since,
)
from webagents.cli.sessions import SessionSummary, caller_key

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "sessions" / "sessions.json").read_text())


def command(rest: str = "") -> str:
    return f"webagents {rest}".strip()


@pytest.mark.parametrize("case", FIXTURE["caller_keys"], ids=[c["principal"] for c in FIXTURE["caller_keys"]])
def test_caller_keys(case):
    assert caller_key(case["principal"]) == case["key"]


def _auth(case):
    """The caller in this SDK's terms: an auth object with no `principals` when no access block ran."""
    fields = {"authenticated": True, "provider": "platform", "scope": case["tier"] or "user", "user_id": case["user_id"]}
    if case["principals"] is not None:
        fields["principals"] = case["principals"]
    if not any((case["tier"], case["principals"], case["user_id"])):
        fields["authenticated"] = False
    return SimpleNamespace(**fields)


@pytest.mark.parametrize("case", FIXTURE["owners"], ids=[c["case"] for c in FIXTURE["owners"]])
def test_whose_conversation(case):
    assert conversation_owner(_auth(case)) == case["whose"]


def test_no_auth_at_all_is_not_kept():
    assert conversation_owner(None) is None


@pytest.mark.parametrize("case", FIXTURE["session_ids"], ids=[repr(c["metadata"])[:40] for c in FIXTURE["session_ids"]])
def test_the_session_a_request_names(case):
    assert request_session_id(case["metadata"]) == case["id"]


@pytest.mark.parametrize("case", FIXTURE["keep"], ids=[c["case"] for c in FIXTURE["keep"]])
def test_what_is_kept_of_a_turn(case):
    assert conversation_to_keep(case["request"], case["reply"]) == case["kept"]


@pytest.mark.parametrize("case", FIXTURE["words_since"], ids=[c["case"] for c in FIXTURE["words_since"]])
def test_what_a_turn_records_on_robutler(case):
    assert words_since(case["messages"], case["from"]) == case["words"]


@pytest.mark.parametrize("case", FIXTURE["merge"], ids=[c["case"] for c in FIXTURE["merge"]])
def test_resume_merges_this_machine_and_robutler(case):
    local = [
        SessionSummary(
            id=entry["id"],
            updated_at=entry["updated_at"],
            message_count=entry["message_count"],
            preview=entry["preview"],
            chat_id=entry["chat_id"],
        )
        for entry in case["local"]
    ]
    platform = [
        PlatformConversation(
            chat_id=c["chatId"],
            session_id=c["sessionId"],
            updated_at=c["updatedAt"],
            message_count=c["messageCount"],
            preview=c["preview"],
        )
        for c in case["platform"]
    ]
    rows = [
        {
            "id": r.id,
            "chat_id": r.chat_id,
            "session_id": r.session_id,
            "updated_at": r.updated_at,
            "local_count": r.local_count,
            "platform_count": r.platform_count,
            "preview": r.preview,
            "only_on_robutler": r.only_on_robutler,
        }
        for r in merge_conversations(local, platform)
    ]
    assert rows == case["rows"]


def test_the_sentences():
    said = FIXTURE["sentences"]
    assert unavailable_reason("signed_out", command) == said["signed_out"]
    assert unavailable_reason("not_published", command) == said["not_published"]
    for status in (401, 404, 0, 502):
        assert failure_sentence(status, command) == said[f"failure_{status}"]
    with pytest.raises(ValueError) as refused:
        session_backend_of("cloud")
    assert str(refused.value) == said["backend"]
    assert session_backend_of(None) == "local" and session_backend_of("robutler") == "robutler"
