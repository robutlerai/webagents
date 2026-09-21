"""Self-registering agent on its own URL (U1 plus dynamic registration).

``own_url_minimal.py`` serves everything the platform READS: the key set at
``/{name}/.well-known/jwks.json`` with the agent's Ed25519 signing key, the
self-naming agent card at ``/{name}/.well-known/agent.json``, and a presence
heartbeat. What it never does is speak: a key set nobody has fetched is not a
registration. This file adds the missing half, which is one signed call.

There is no endpoint to post a registration to. ``POST /api/auth/agent/register``
exists and answers 410 on purpose. The platform registers an agent the first
time a request verifies: it reads the key-set URL off the request's
``Signature-Agent`` header, fetches the key set, checks the signature over the
request (RFC 9421 HTTP Message Signatures, the Web Bot Auth profile), and
reads the card at the agent URL it derived. ``register_with_platform`` signs
that request with the key the server serves the key set with, and makes the
call.

Each signature is good for 60 seconds and carries a fresh 64 byte nonce. The
platform spends the nonce on first use and refuses a replay, so a captured
request is worthless once presented; the expiry bounds only the window
before that.

Environment:

  OPENAI_API_KEY         your model provider's key
  WEBAGENTS_PUBLIC_URL   the https URL this agent is reachable at. The
                         PLATFORM fetches the key set and the card from
                         under it, so it has to resolve to a public address:
                         loopback, RFC 1918, link-local and 100.64.0.0/10
                         (which is where Tailscale addresses live) are all
                         refused, and so is plain http.
  ROBUTLER_API_URL       the platform's base URL. The signature is bound to
                         its host, which is the single fact this flow most
                         often gets wrong: the call goes to the PLATFORM's
                         own address, never to something that proxies it
                         under another name.
  WEBAGENTS_KEYS_DIR     where the signing key is stored (default
                         ``~/.webagents/keys``). It MUST survive restarts:
                         the platform selects the key by thumbprint from the
                         key set it fetched, and a key regenerated per boot
                         is unknown to it after the first restart.

The agent registers as an OWNERLESS account named after its own URL reversed,
and the response says which one. To attach it to a person, print the claim
link that ``webagents.server.core.registration.claim_url`` builds: it mints
the short-lived claim token and puts it in the URL FRAGMENT, which is what
keeps a bearer out of access logs and ``Referer`` headers. The person opens
the link and their browser posts the token to ``POST /api/agents/{id}/claim``.

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
