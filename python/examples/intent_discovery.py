"""Publish what an agent can do, and find an agent by what you need.

``DiscoverySkill`` gives the agent two things. A ``search`` tool its model
calls with a description of a NEED ("translate a contract into German"),
matched by meaning against the intents every agent on the platform has
published, so the caller never has to know a name or a URL. And
``publish_intents``, which lists this agent's own intents so other agents can
find it the same way; with ``intents`` configured, the skill publishes them on
its own a few seconds after the agent starts.

Both calls are SIGNED with the identity ``create_server`` gives the agent
(RFC 9421 HTTP Message Signatures under the Web Bot Auth profile): the key it
persists under ``WEBAGENTS_KEYS_DIR`` and publishes at
``{agent_url}/.well-known/jwks.json``, the same key that signs its
registration. No platform key is involved: the platform takes the signature
first and looks for a bearer only when there is none. Set WEBAGENTS_API_KEY
only for an agent that has no signing identity.

Order matters once, on the very first run. The platform fetches the key set
from the agent URL the first time it sees the signature, so the agent has to
be serving before anything is published. ``register_after_startup`` makes the
first signed call once uvicorn is serving, and the skill's own publish waits a
few seconds after start for the same reason. From the second run on the
platform already holds the key.

Environment:

  OPENAI_API_KEY         your model provider's key
  WEBAGENTS_PUBLIC_URL   the https base URL this agent is reachable at; the
                         agent is served under ``{it}/translator``
  ROBUTLER_API_URL       the platform's base URL (the signature covers its host)
  WEBAGENTS_KEYS_DIR     where the Ed25519 key is stored (default
                         ``~/.webagents/keys``). It MUST survive restarts.

This file is executed by tests/docs/test_intent_discovery_example.py against a
stub platform without binding a port, and the docs' snippets are generated
from it verbatim.
"""

import uvicorn

from webagents import BaseAgent, create_server
from webagents.agents.skills.robutler.discovery import DiscoverySkill
from webagents.server.core.registration import register_after_startup

# What this agent DOES, in the words someone who needs it would use. Each
# intent is one sentence; the platform embeds them and matches a searcher's
# need against them by meaning, not by keyword.
discovery = DiscoverySkill(
    {
        "intents": [
            "translate documents between English and German",
            "proofread German business correspondence",
        ],
        "description": "Translates and proofreads English and German text.",
    }
)

agent = BaseAgent(
    name="translator",
    instructions="You translate and proofread English and German text.",
    model="openai/gpt-4o-mini",
    skills={"discovery": discovery},
)

# `create_server` persists the agent's signing key and publishes it at
# /translator/.well-known/jwks.json, which is the key the skill signs with.
server = create_server(agents=[agent])

# The first signed request registers the agent, once the server is serving.
register_after_startup(server, agent.name)


async def find_agent_for(need: str) -> dict:
    """The other side. The agent's model calls the `search` tool itself when
    it needs another agent; this is the same call made directly, for code
    that wants the answer rather than the model. Each result carries the
    matched intent, the agent that published it and a similarity score."""
    return await discovery.search(query=need, types=["intents"])


if __name__ == "__main__":
    uvicorn.run(server.app, host="0.0.0.0", port=8000)
