"""Minimal agent on its own URL (U1).

Build an agent, build a server, run it. Nothing is hidden behind a helper:
``create_server`` serves ``POST /mini/chat/completions`` (the endpoint the
platform dials) AND the platform registration surface: the key set at
``/mini/.well-known/jwks.json`` carrying the agent's Ed25519 signing key, the
self-naming agent card at ``/mini/.well-known/agent.json``, and a 60s
presence heartbeat.

Environment:

  OPENAI_API_KEY         your model provider's key
  WEBAGENTS_PUBLIC_URL   the https URL this agent is reachable at; the card
                         and the key set are served under ``{it}/mini``
  WEBAGENTS_KEYS_DIR     where the signing key is stored (default
                         ``~/.webagents/keys``). It MUST survive restarts:
                         the platform selects the key by thumbprint from the
                         key set it fetched at registration.
  WEBAGENTS_AGENT_TOKEN  per-agent key; with ROBUTLER_API_URL set, this is
                         what the heartbeat presents

``POST /mini/chat/completions`` requires an Authorization header: it runs the
model on your credit. Add an AuthSkill to have the credential verified rather
than merely required.

Serving that surface is half of joining the platform. The other half is one
request signed with the key the key set publishes, which is what turns a
served key set into an account: see ``own_url_register.py``.

This file is executed by tests/docs/test_doc_examples.py without binding a
port, and the docs' snippets are generated from it verbatim.
"""

import uvicorn

from webagents import BaseAgent, create_server

agent = BaseAgent(
    name="mini",
    instructions="You are helpful.",
    model="openai/gpt-4o-mini",
)

server = create_server(agents=[agent])

if __name__ == "__main__":
    uvicorn.run(server.app, host="0.0.0.0", port=8000)
