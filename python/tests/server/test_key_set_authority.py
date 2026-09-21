"""The server's key set is authoritative at `/{agent}/.well-known/jwks.json`
(2026-09-18, W2 review finding).

`register_with_platform` signs with the Ed25519 key `JWKSManager` loads from
the server's `keys_dir` under the agent's name, and `Signature-Agent` names
`{agent_url}/.well-known/jwks.json` as the set that key is published in. The
platform fetches exactly that URL, selects the key by thumbprint, and answers
`key_set_invalid` (cached for 300 s) when the set carries no usable Ed25519
key. So whatever answers that URL MUST be the set built from the signer's own
directory and name, and the only component that knows both is the server.

Until 2026-09-18 the local `AuthSkill`'s own `@http("/.well-known/jwks.json")`
handler was mounted first and won the route. Its `JWKSManager` starts empty
and is filled only by the skill's lazy `initialize()`, which a chat request
triggers and which portal mode never fills with an Ed25519 key at all, so a
fresh agent served `{"keys": []}` at the registration URL in both modes and
could not register. These cases are written as the post-fix expectation; on
the pre-fix ordering they fail with an empty `keys` list.
"""

from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.local.auth import AuthSkill
from webagents.crypto.jwks import JWKSManager
from webagents.server.core.app import create_server

PUBLIC_URL = "https://agent.example.com"

PORTAL_MODE = {"authority": "https://robutler.ai"}
SELF_ISSUED_MODE: Dict[str, Any] = {}


def signing_thumbprint(keys_dir: str, agent_name: str) -> str:
    """What `register_with_platform(keys_dir=...)` puts in `keyid`."""
    manager = JWKSManager({"keys_dir": keys_dir})
    manager.ensure_ed25519_key(agent_name)
    return manager.held_ed25519_keys()[0].thumbprint


def okp_kids(key_set: Dict[str, Any]) -> List[str]:
    return [k["kid"] for k in key_set.get("keys", []) if k.get("kty") == "OKP"]


def serve(tmp_path, mode: Dict[str, Any], url_prefix: str = ""):
    """A server with one agent carrying the local AuthSkill, whose own
    `keys_dir` is deliberately NOT the server's: the served set must still be
    the signer's."""
    server_keys = str(tmp_path / "server-keys")
    skill = AuthSkill({"keys_dir": str(tmp_path / "skill-keys"), **mode})
    agent = BaseAgent(name="mini", instructions="Say hi.", skills={"auth": skill})
    server = create_server(
        agents=[agent], public_url=PUBLIC_URL, keys_dir=server_keys, url_prefix=url_prefix
    )
    return TestClient(server.app), server_keys


@pytest.mark.parametrize("mode", [PORTAL_MODE, SELF_ISSUED_MODE], ids=["portal", "self-issued"])
def test_the_registration_url_serves_the_signers_key_with_no_prior_request(tmp_path, mode):
    client, server_keys = serve(tmp_path, mode)
    res = client.get("/mini/.well-known/jwks.json")
    assert res.status_code == 200
    assert okp_kids(res.json()) == [signing_thumbprint(server_keys, "mini")]


def test_the_card_names_the_set_that_carries_the_signing_key(tmp_path):
    client, server_keys = serve(tmp_path, PORTAL_MODE, url_prefix="/agents")
    card = client.get("/agents/mini/.well-known/agent.json").json()
    assert card["jwks_uri"] == f"{PUBLIC_URL}/agents/mini/.well-known/jwks.json"
    res = client.get("/agents/mini/.well-known/jwks.json")
    assert okp_kids(res.json()) == [signing_thumbprint(server_keys, "mini")]


def test_the_skill_handler_no_longer_shadows_the_set_after_a_chat_request_either(tmp_path):
    # Self-issued mode fills the skill's own set once it initialises; the
    # served set must be the server's regardless, since the skill's directory
    # is not the one the signer reads.
    client, server_keys = serve(tmp_path, SELF_ISSUED_MODE)
    agent_keys = signing_thumbprint(server_keys, "mini")
    first = okp_kids(client.get("/mini/.well-known/jwks.json").json())
    assert first == [agent_keys]
    # The skill's manager, initialised by hand, publishes a DIFFERENT key
    # (another directory), which must not appear at the registration URL.
    skill_manager = JWKSManager({"keys_dir": str(tmp_path / "skill-keys")})
    skill_manager.ensure_ed25519_key("mini")
    other = skill_manager.held_ed25519_keys()[0].thumbprint
    assert other != agent_keys
    again = okp_kids(client.get("/mini/.well-known/jwks.json").json())
    assert again == [agent_keys]
