"""The agent card's ``url`` — one precedence table, two SDKs.

The TypeScript twin is the ``resolves the card url on the same precedence
table as Python`` case in ``typescript/tests/unit/examples.test.ts``. It drives
``createFetchHandler`` with the same nine rows and expects the same nine
answers, so a divergence in either SDK fails on that side.

Two things this pins that the two SDKs used to disagree about:

* **the last tier.** TypeScript fell back to the REQUEST ORIGIN
  (``http://127.0.0.1:8816/agents/og``) where Python falls back to a relative
  ``/{agent_name}``. The request origin comes from the Host header, so behind
  a proxy, a tunnel or a container it is a wrong address published as fact —
  precisely the failure ``publicUrl`` exists to prevent. A relative reference
  resolves against whatever origin the consumer actually fetched the card
  from, which is by construction one the agent is reachable at. No consumer
  reads ``card.url`` — precisely: portal ``lib/auth/agent-auth.ts:82`` gives
  ``AgentMetadata`` an index signature ``[key: string]: unknown``, so ``url``
  IS carried through the interface, but nothing dereferences it (the only
  ``metadata.`` reads are ``capabilities`` at 272 and 344 and ``publicKey`` at
  485 and 509), and the callable address a registration is keyed on is
  ``composeAgentRegistrationUrl(iss, agent_path, sub)`` at line 186, from the
  agent's own signed token. So the tier cannot break registration either way
  and the honest value wins.

* **whitespace.** TypeScript trimmed a whitespace-only configured value;
  Python published it verbatim. Both trim now — and in both, a whitespace-only
  ``configured`` SUPPRESSES the environment variable rather than falling
  through to it.
"""

import pytest

from webagents.server.core.registration import build_agent_card, resolve_public_base_url


# (configured, env, expected) — kept row-for-row in step with the TS twin.
PRECEDENCE_TABLE = [
    ("https://configured.example", "https://env.example", "https://configured.example"),
    ("https://configured.example/", None, "https://configured.example"),
    ("https://configured.example///", None, "https://configured.example"),
    (None, "https://env.example/", "https://env.example"),
    (None, None, "/card"),
    ("", "https://env.example", "https://env.example"),
    # Whitespace-only: trimmed to nothing, and NOT a fall-through to the env.
    ("   ", "https://env.example", "/card"),
    ("   ", None, "/card"),
    (None, "   ", "/card"),
]


@pytest.mark.parametrize("configured,env,expected", PRECEDENCE_TABLE)
def test_card_url_precedence(monkeypatch, configured, env, expected):
    if env is None:
        monkeypatch.delenv("WEBAGENTS_PUBLIC_URL", raising=False)
    else:
        monkeypatch.setenv("WEBAGENTS_PUBLIC_URL", env)
    assert resolve_public_base_url(configured, "card") == expected


def test_the_relative_fallback_resolves_against_the_fetch_origin(monkeypatch):
    """What a relative last tier BUYS: whatever origin the consumer reached
    the agent at is the origin the card resolves to. A Host-derived absolute
    url would have pinned the container's own address instead."""
    monkeypatch.delenv("WEBAGENTS_PUBLIC_URL", raising=False)
    from urllib.parse import urljoin

    relative = resolve_public_base_url(None, "card")
    assert relative == "/card"
    assert (
        urljoin("https://proxy.example.com/.well-known/agent.json", relative)
        == "https://proxy.example.com/card"
    )


def test_the_fallback_never_carries_a_host(monkeypatch):
    monkeypatch.delenv("WEBAGENTS_PUBLIC_URL", raising=False)

    class _Agent:
        name = "card"
        instructions = "You are helpful."

    card = build_agent_card(_Agent(), resolve_public_base_url(None, "card"))
    assert card["url"] == "/card"
    assert "://" not in card["url"]
