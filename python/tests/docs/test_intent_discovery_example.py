"""``python/examples/intent_discovery.py`` is EXECUTED here against a stub
platform (an ``httpx.MockTransport``, no model call, no platform key), and the
docs' snippet is asserted to be generated from that exact file, so the
documented way to publish and search intents cannot rot (2026-09-23).

What the stub checks is the wire: the publish and the search both arrive
SIGNED (RFC 9421, Web Bot Auth), verifiable under the key set the example's
own server serves at ``/translator/.well-known/jwks.json``, and with no bearer
anywhere. That is the whole point of the page these snippets land on: intent
discovery with a signing identity and no platform key.
"""

import json
import re
from pathlib import Path

import httpx
import pytest

from ..crypto.support import verify_signed_request
from .test_doc_examples import _EXAMPLES_DIR, _REPO_ROOT, _load_example

PLATFORM = "https://platform.example.com"
PUBLIC_URL = "https://agent.example.com"
AGENT_URL = f"{PUBLIC_URL}/translator"
KEY_SET = f"{AGENT_URL}/.well-known/jwks.json"

#: One row as `POST /api/intents/search` answers it; the tool hands rows on as they are.
INTENT_ROW = {
    "id": "i-1",
    "intent": "translate legal documents into German",
    "agentId": "a-jurist",
    "description": "Certified legal translation",
    "url": "https://legal.example.com/agents/jurist",
    "similarity": 0.87,
}


class TestIntentDiscoveryExample:
    async def test_it_publishes_and_searches_signed_with_the_served_key_and_no_platform_key(
        self, monkeypatch, tmp_path
    ):
        from webagents.agents.skills.robutler.discovery.skill import DiscoverySkill

        # The environment the example reads at import: a public https base
        # URL, the platform, a keys directory that is not the developer's
        # home, and no platform key of any kind.
        monkeypatch.setenv("WEBAGENTS_KEYS_DIR", str(tmp_path))
        monkeypatch.setenv("WEBAGENTS_PUBLIC_URL", PUBLIC_URL)
        monkeypatch.setenv("ROBUTLER_API_URL", PLATFORM)
        for name in ("WEBAGENTS_API_KEY", "SERVICE_TOKEN", "ROBUTLER_INTERNAL_API_URL", "ROBUTLER_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(DiscoverySkill, "AUTO_PUBLISH_DELAY_S", 0)

        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if request.url.path == "/api/discovery/announce":
                return httpx.Response(200, json={"ok": True, "intents": {"count": 2}})
            if request.url.path == "/api/intents/search":
                return httpx.Response(200, json={"results": [INTENT_ROW]})
            return httpx.Response(404, json={"error": "not found"})

        real_client = httpx.AsyncClient
        monkeypatch.setattr(
            httpx, "AsyncClient", lambda *a, **kw: real_client(*a, transport=httpx.MockTransport(handler), **kw)
        )

        module = _load_example("intent_discovery.py")
        skill = module.discovery
        assert module.agent.name == "translator"

        # The key set the platform would fetch to verify what follows: served
        # by the example's own server, under the agent's mount.
        transport = httpx.ASGITransport(app=module.server.app)
        async with real_client(transport=transport, base_url="http://test") as client:
            jwks = await client.get("/translator/.well-known/jwks.json")
        assert jwks.status_code == 200
        served = [k for k in jwks.json()["keys"] if k.get("kty") == "OKP"]
        assert served, jwks.json()
        from cryptography.hazmat.primitives.asymmetric import ed25519
        import base64

        public = {
            k["kid"]: ed25519.Ed25519PublicKey.from_public_bytes(base64.urlsafe_b64decode(k["x"] + "=="))
            for k in served
        }

        # Skills initialise when the agent first runs; drive that step
        # directly, which is what schedules the publish. With no identity
        # and no key it would have warned instead; here it signs.
        await skill.initialize(module.agent)
        assert skill.credential().kind == "signature"
        assert skill._auto_publish_task is not None, "the configured intents were not scheduled for publishing"
        await skill._auto_publish_task

        publish = seen[0]
        assert str(publish.url) == f"{PLATFORM}/api/discovery/announce"
        assert "authorization" not in publish.headers
        assert publish.headers["signature-agent"] == f'sig1="{KEY_SET}";type=jwks_uri'
        body = json.loads(publish.content)
        assert body["url"] == AGENT_URL
        assert [i["intent"] for i in body["intents"]] == [
            "translate documents between English and German",
            "proofread German business correspondence",
        ]
        record = verify_signed_request(publish.headers, "POST", str(publish.url), publish.content, public)["sig1"]
        assert record["keyid"] in public
        assert record["base"].split("\n")[1] == '"@authority": platform.example.com'

        # The other side: the same identity, the same verification, and the
        # results as the skill shapes them for a model.
        found = await module.find_agent_for("translate a contract into German")
        search = seen[1]
        assert str(search.url) == f"{PLATFORM}/api/intents/search"
        assert "authorization" not in search.headers
        assert json.loads(search.content)["query"] == "translate a contract into German"
        verify_signed_request(search.headers, "POST", str(search.url), search.content, public)
        assert found == {"intents": [INTENT_ROW]}

        # Nothing this example sent carried a bearer.
        assert all("authorization" not in r.headers for r in seen)

    def test_the_doc_carries_the_example_verbatim(self):
        source = (_EXAMPLES_DIR / "intent_discovery.py").read_text()
        m = re.match(r'\s*""".*?"""\s*\n', source, re.DOTALL)
        code = (source[m.end():] if m else source).strip()
        doc = (_REPO_ROOT / "docs" / "guides" / "intent-discovery.md").read_text()
        assert (
            "<!-- BEGIN GENERATED: typescript/examples/intent-discovery.ts,python/examples/intent_discovery.py -->"
            in doc
        )
        assert code in doc
