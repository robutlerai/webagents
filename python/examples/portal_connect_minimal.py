"""Minimal platform-connected agent (reverse WebSocket bridge, U2).

Same agent as the own-URL example, plus one skill and no server.
``PortalConnectSkill`` dials the platform, so no public URL and no inbound
port are needed, and nothing listens: the skill's own lifecycle opens the
socket, and the process keeps the event loop alive. (Until 2026-09-26 this
example also ran a server on every interface behind the presence-only
credential floor, S-267. A process that only dials out has no port to bind.)

The skill reads its own configuration — attaching it IS the configuration:

  WEBAGENTS_PORTAL_URL   e.g. https://robutler.ai (``/ws`` appended)
  WEBAGENTS_AGENT_TOKEN  a PER-AGENT key from POST /api/agents/{id}/api-key
  OPENAI_API_KEY         (or your model provider's key)

The token must be per-agent: its JWT carries an ``agent_id`` claim. A generic
owner key connects successfully and then never receives a single turn, so the
skill refuses it at start with the fix in the message.

To ALSO serve the agent over HTTP (``/health``, the agent card, a chat
endpoint), give it to ``create_server(agents=[agent])`` as own_url_minimal.py
does: the server's lifecycle then starts the skill, and it binds loopback
until an AuthSkill verifies callers.

This file is executed by tests/docs/test_doc_examples.py against a stub portal,
and the docs' snippets are generated from it verbatim.
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
