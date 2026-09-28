"""
`webagents acp [path]` (gap-closure plan item 1.6, 2026-09-26): the agent at
`path` (an AGENT.md, or the folder holding one) served to a code editor over
the Agent Client Protocol, on this process's stdin and stdout. The TypeScript
CLI's command is `typescript/src/cli/acp-action.ts`; the words are held by
`tests/fixtures/acp/acp_protocol.json` and `tests/cli/test_cli_parity.py`.

THE SAME AGENT AS `serve` (`serve.load_served_agent`): its skills and model,
the stored keys, the folder, the access block. The caller is the person
whose editor spawned this process, the owner, as in the local chat.

STDOUT IS THE WIRE, so it is reserved before the agent file is even read
(`load_served_agent` prints a line when there is no file, and a skill may
print while starting); `sys.stdout` goes to stderr and the transport gets the
real one. Sessions are kept under the profile directory (`~/.webagents`, or
`~/.webagents-<profile>`), in `acp/sessions`, unless the agent file's own
`- acp: {sessions_dir: ...}` says where.
"""

from __future__ import annotations

import asyncio
import sys


def acp_command(path: str) -> None:
    from webagents.agents.skills.core.transport.acp.skill import ACPTransportSkill
    from webagents.server.mcp_server import reserve_stdout

    from .config_store import global_dir
    from .serve import load_served_agent

    real_stdout = reserve_stdout()
    # The owner's own editor over stdio: the owner's turns, so the sign-in
    # may pay, as the `login` auth method promises (S-327 keeps it off
    # `serve` and the daemon only).
    built = load_served_agent(path, for_callers=False)
    agent = built.agent
    skill = next((s for s in (agent.skills or {}).values() if isinstance(s, ACPTransportSkill)), None)
    if skill is None:
        skill = ACPTransportSkill({})
        agent.add_skill("acp", skill)
    if not skill.settings.get("sessions_dir"):
        skill.settings["sessions_dir"] = str(global_dir() / "acp" / "sessions")
    try:
        asyncio.run(skill.serve_stdio(agent, sys.stdin.buffer, real_stdout))
    except KeyboardInterrupt:
        pass
