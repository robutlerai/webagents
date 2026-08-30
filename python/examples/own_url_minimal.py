"""Minimal agent on its own URL (U1).

Build an agent, build a server, run it. Nothing is hidden behind a helper:
``create_server`` serves ``POST /mini/chat/completions`` (the endpoint the
platform dials) AND the platform registration surface — the agent card at
``/.well-known/agent.json`` (origin *and* agent prefix) carrying
``metadata.publicKey`` as an SPKI PEM, plus ``/.well-known/jwks.json`` and a
60s presence heartbeat.

Environment:

  OPENAI_API_KEY         your model provider's key
  WEBAGENTS_PUBLIC_URL   the URL this agent is reachable at (goes on the card)
  WEBAGENTS_KEYS_DIR     where the signing key is stored (default
                         ``~/.webagents/keys``). It MUST survive restarts:
                         registration pins the public key from the card.
  WEBAGENTS_AGENT_TOKEN  per-agent key; with ROBUTLER_API_URL set, this is
                         what the heartbeat presents

``POST /mini/chat/completions`` requires an Authorization header — it runs the
model on your credit. Add an AuthSkill to have the credential verified rather
than merely required.

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
