"""Self-registering agent on its own URL (U1 plus dynamic registration).

``own_url_minimal.py`` serves everything the platform READS — the agent card
at ``/.well-known/agent.json`` (origin and agent prefix) with the signing key
at the top level and nested under ``metadata``, plus ``/.well-known/jwks.json``
and a presence heartbeat. What it never does is speak: a card nobody has
fetched is not a registration. This file adds the missing half, which is one
authenticated call.

There is no endpoint to post a registration to. ``POST /api/auth/agent/register``
exists and answers 410 on purpose. The platform registers an agent the first
time a request verifies: it reads ``iss`` off the unverified bearer, fetches
the card at that URL, imports the ``publicKey`` it finds there and checks the
signature. ``register_with_platform`` mints that bearer from the key the server
serves the card with, and makes the call.

The token it mints is short lived (five minutes) and carries a ``jti``. The
platform does not record ``jti`` yet, so the expiry is the only thing bounding
replay of a token someone captures; keep it short even though nothing enforces
that.

Environment:

  OPENAI_API_KEY         your model provider's key
  WEBAGENTS_PUBLIC_URL   the URL this agent is reachable at. The PLATFORM
                         fetches the card from here, so it has to resolve to a
                         public address: loopback, RFC 1918, link-local and
                         100.64.0.0/10 (which is where Tailscale addresses
                         live) are all refused.
  ROBUTLER_API_URL       the platform's base URL. It is also the token's
                         ``aud``, which is the single fact this flow most often
                         gets wrong: ``aud`` is the PLATFORM, never the agent's
                         own URL and never the endpoint being called.
  WEBAGENTS_KEYS_DIR     where the signing key is stored (default
                         ``~/.webagents/keys``). It MUST survive restarts:
                         registration pins the public key from the card and
                         verifies every later token against that copy.

The agent registers as an OWNERLESS account named after its own URL reversed,
and the response says which one. Claim it with ``POST /api/agents/{id}/claim``
to attach it to a person.

Registration runs AFTER this server is serving, never awaited inside a startup
handler, which is the one piece of shape this example exists to get right.
``register_after_startup`` handles that; the comment above the call says what
goes wrong without it. This file is executed by tests/docs/test_doc_examples.py
without binding a port, and the docs' snippets are generated from it verbatim.
"""

import uvicorn

from webagents import BaseAgent, create_server
from webagents.server.core.registration import register_after_startup

agent = BaseAgent(
    name="selfreg",
    instructions="You are helpful.",
    model="openai/gpt-4o-mini",
)

server = create_server(agents=[agent])


def _report(result: dict) -> None:
    if result["ok"]:
        print(f"[selfreg] registered as {result['username']} ({result['user_id']})")
    else:
        print(f"[selfreg] not registered: {result['error']}")


# One line, and the ordering trap is handled for you. Registration is a call
# the platform ANSWERS BY CALLING BACK: it fetches the agent card from this
# very server while the registering request is still in flight. uvicorn serves
# nothing until every startup handler has returned, so awaiting the
# registration inside one deadlocks the callback against the handler waiting
# for it, and what you see is a 502 on the card and a bare 401 on the call
# with nothing pointing at the ordering. `register_after_startup` schedules it
# as a task (and keeps a reference, so the task cannot be collected mid-flight)
# and returns, which is the correct order.
register_after_startup(server, agent.name, on_result=_report)


if __name__ == "__main__":
    uvicorn.run(server.app, host="0.0.0.0", port=8000)
