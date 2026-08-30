"""Platform-connected agent with NO HTTP server at all.

The honest minimal form when nothing will ever dial this process: there is no
port to bind, so there is no server — just the skill's own lifecycle, run on
the event loop and kept alive. Written out rather than hidden behind a
one-word call, because what the process is doing (open a socket, then wait)
is the whole point.

Prefer portal_connect_minimal.py unless you specifically want no HTTP
surface: the server gives you /health and the agent card, and it owns the
same lifecycle for you.

Environment: WEBAGENTS_PORTAL_URL, WEBAGENTS_AGENT_TOKEN, OPENAI_API_KEY.

This file is executed by tests/docs/test_doc_examples.py against a stub portal.
"""

import asyncio

from webagents import BaseAgent
from webagents.agents.skills.robutler.portal_connect import PortalConnectSkill

portal = PortalConnectSkill()
agent = BaseAgent(
    name="mini",
    instructions="You are helpful.",
    model="openai/gpt-4o-mini",
    skills={"portal": portal},
)


async def main() -> None:
    await portal.initialize(agent)  # reads the env, opens the socket
    try:
        await asyncio.Event().wait()  # the bridge lives on the socket, not a port
    finally:
        await portal.stop()


if __name__ == "__main__":
    asyncio.run(main())
