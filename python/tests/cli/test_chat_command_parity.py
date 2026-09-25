"""
The two chats take the same commands, in the same words (2026-09-24).

The TypeScript list (`typescript/src/cli/chat-commands.ts`) is the reference.
This reads that file and compares it with `webagents/cli/repl/commands.py`, so
a command added, renamed or reworded in one chat only fails here rather than
being found by a person who switched SDKs.
"""

import re
from pathlib import Path

from webagents.cli.repl.commands import CHAT_COMMANDS, CHAT_KEYS

TS_FILE = Path(__file__).resolve().parents[3] / "typescript" / "src" / "cli" / "chat-commands.ts"

_STRING = r"""'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)\""""


def _unquote(single: str, double: str) -> str:
    text = single if single is not None and single != "" or double is None else double
    return text.replace("\\\\", "\\").replace("\\'", "'").replace('\\"', '"')


def _ts_commands():
    source = TS_FILE.read_text()
    body = source.split("export const CHAT_COMMANDS", 1)[1].split("];", 1)[0]
    entry = re.compile(
        r"\{\s*name:\s*(?:" + _STRING + r"),\s*usage:\s*(?:" + _STRING + r"),\s*description:\s*(?:" + _STRING + r")\s*\}"
    )
    out = []
    for m in entry.finditer(body):
        g = m.groups()
        out.append((_unquote(g[0], g[1]), _unquote(g[2], g[3]), _unquote(g[4], g[5])))
    return out


def _ts_keys():
    source = TS_FILE.read_text()
    body = source.split("export const CHAT_KEYS", 1)[1].split("];", 1)[0]
    pair = re.compile(r"\[\s*(?:" + _STRING + r"),\s*(?:" + _STRING + r")\s*\]")
    return [(_unquote(m.group(1), m.group(2)), _unquote(m.group(3), m.group(4))) for m in pair.finditer(body)]


def test_the_reference_file_is_where_this_test_looks():
    assert TS_FILE.exists(), TS_FILE
    assert len(_ts_commands()) >= 10


def test_the_commands_match_the_typescript_chat_word_for_word():
    python = [(c.name, c.usage, c.description) for c in CHAT_COMMANDS]
    assert python == _ts_commands()


def test_the_keys_match_the_typescript_chat_word_for_word():
    assert list(CHAT_KEYS) == _ts_keys()
