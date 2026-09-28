"""
Startup lines `webagents serve` prints beside the ones it always printed
(2026-09-26, the new-developer e2e run), the same in the TypeScript `serve()`
(`server/startup-lines.ts`), pinned by `tests/fixtures/cli/serve_startup.json`.
"""

from __future__ import annotations

#: An agent with `memory` and nothing that verifies a caller: memory is the
#: owner's alone (the fixture's `memory_without_auth`).
MEMORY_WITHOUT_AUTH = (
    "[webagents] {name} has memory but no AuthSkill: nothing verifies a caller, so served callers read "
    "shared notes and cannot write; only the owner at this terminal can. Add AuthSkill, or an access: block, "
    "to identify callers."
)


def memory_without_auth_line(name: str) -> str:
    return MEMORY_WITHOUT_AUTH.format(name=name)
