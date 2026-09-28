"""
The daemon's console lines carry no emoji (2026-09-26, the new-developer e2e
run): `webagents daemon` keeps the agent's INFO log on its console, and every
agent it built printed `BaseAgent created`, `Set default handoff` and `System
prompt` lines behind an emoji, which the TypeScript daemon never does. This
scans the base agent's source for a logger call at INFO or above whose message
starts with a non-ASCII symbol, so the tone cannot drift back.
"""

from __future__ import annotations

import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "webagents" / "agents" / "core" / "base_agent.py"
CALL = re.compile(r'logger\.(info|warning|error|critical)\(\s*f?"([^"]*)"')


def test_no_info_or_louder_log_line_starts_with_an_emoji():
    offenders = []
    for number, line in enumerate(SOURCE.read_text().splitlines(), 1):
        for match in CALL.finditer(line):
            message = match.group(2)
            if message and ord(message[0]) > 0x7F:
                offenders.append(f"{number}: {line.strip()}")
    assert offenders == [], "\n".join(offenders)
