"""
The two chats take the same commands, in the same words (2026-09-24, updated
2026-09-26): the shared fixture `tests/fixtures/cli/chat_commands.json` is the
reference both SDKs are held to. This used to parse `chat-commands.ts`; the
command specs now carry `group` and `details`, so the fixture is the contract
and `test_chat_commands_fixture_interactive.py` checks the rest.
"""

import json
from pathlib import Path

from webagents.cli.repl.commands import CHAT_COMMANDS, CHAT_KEYS

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "chat_commands.json").read_text())


def test_the_commands_match_the_shared_fixture_word_for_word():
    table = [
        {"name": c.name, "usage": c.usage, "description": c.description, "group": c.group, "details": list(c.details)}
        for c in CHAT_COMMANDS
    ]
    assert table == FIXTURE["commands"]


def test_the_keys_match_the_shared_fixture():
    assert [list(k) for k in CHAT_KEYS] == FIXTURE["keys"]
