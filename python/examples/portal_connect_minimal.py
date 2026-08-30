"""Minimal platform-connected agent (reverse WebSocket bridge, U2).

Same three lines as the own-URL example, plus one skill. ``PortalConnectSkill``
dials the platform, so no public URL and no inbound port are needed; the server
is still here because it is what owns the lifecycle that opens the socket (and
it gives you ``/health`` and the agent card for free).

The skill reads its own configuration — attaching it IS the configuration:

  WEBAGENTS_PORTAL_URL   e.g. https://robutler.ai (``/ws`` appended)
  WEBAGENTS_AGENT_TOKEN  a PER-AGENT key from POST /api/agents/{id}/api-key
  OPENAI_API_KEY         (or your model provider's key)

The token must be per-agent: its JWT carries an ``agent_id`` claim. A generic
owner key connects successfully and then never receives a single turn, so the
skill refuses it at start with the fix in the message.

For a process with no HTTP surface at all, see portal_connect_socket_only.py.

This file is executed by tests/docs/test_doc_examples.py against a stub portal,
and the docs' snippets are generated from it verbatim.
"""

import uvicorn

from webagents import BaseAgent, create_server
from webagents.agents.skills.robutler.portal_connect import PortalConnectSkill

agent = BaseAgent(
    name="mini",
    instructions="You are helpful.",
    model="openai/gpt-4o-mini",
    skills={"portal": PortalConnectSkill()},
)

server = create_server(agents=[agent])

if __name__ == "__main__":
    uvicorn.run(server.app, host="0.0.0.0", port=8000)
