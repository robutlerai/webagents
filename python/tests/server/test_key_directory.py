"""The well-known signatures directory (2026-09-19), the Python twin of
`typescript/tests/unit/server/key-directory.test.ts`.

`sign_request(..., form="legacy-string")` sends a bare ORIGIN as
`Signature-Agent`. The platform resolves a bare origin to
`{origin}/.well-known/http-message-signatures-directory` and refuses any
answer not served as `application/http-message-signatures-directory+json`
(lib/auth/web-bot-auth/discovery.ts in the portal). Neither SDK server served
that path, so the form failed discovery against a webagents-hosted agent every
time. These cases resolve the signer's own header the way the platform does,
ask the server for what it names, and apply the platform's directory rules to
the answer.
"""

import base64
import hashlib
import json
import logging
from typing import Any, Dict, List
from urllib.parse import urlsplit

from fastapi.testclient import TestClient

from webagents.agents.core.base_agent import BaseAgent
from webagents.crypto.http_signature import sign_request
from webagents.crypto.jwks import JWKSManager
from webagents.server.core.app import create_server
from webagents.server.core.credential_floor import PUBLIC_SUBPATHS
from webagents.server.core.key_directory import (
    DIRECTORY_MAX_KEYS,
    DIRECTORY_MEDIA_TYPE,
    DIRECTORY_WELL_KNOWN_PATH,
    directory_keys,
    key_directory_response,
)

PUBLIC_URL = "https://agent.example.com"


def thumbprint(jwk: Dict[str, Any]) -> str:
    """RFC 7638 over `{crv, kty, x}`, as the platform computes it."""
    canonical = json.dumps({"crv": jwk["crv"], "kty": jwk["kty"], "x": jwk["x"]}, separators=(",", ":"))
    return base64.urlsafe_b64encode(hashlib.sha256(canonical.encode()).digest()).rstrip(b"=").decode()


def platform_reads(res) -> List[str]:
    """The platform's reading of a directory answer (discovery.ts): the media type, then the entry rules."""
    assert res.status_code == 200
    assert res.headers["content-type"].split(";")[0].strip().lower() == "application/http-message-signatures-directory+json"
    assert "max-age=" in res.headers["cache-control"]
    keys = res.json()["keys"]
    assert 0 < len(keys) <= 16
    for key in keys:
        assert key["kty"] == "OKP" and key["crv"] == "Ed25519" and key["use"] == "sig"
        # At the well-known directory a `kid` MUST equal the thumbprint, or the whole set is refused.
        assert key["kid"] == thumbprint(key)
        assert "d" not in key
    return [k["kid"] for k in keys]


def manager(keys_dir: str, agent_name: str) -> JWKSManager:
    m = JWKSManager({"keys_dir": keys_dir})
    m.ensure_ed25519_key(agent_name)
    return m


def serve(tmp_path, names=("mini",), url_prefix: str = ""):
    keys_dir = str(tmp_path / "server-keys")
    agents = [BaseAgent(name=name, instructions="Say hi.") for name in names]
    server = create_server(agents=agents, public_url=PUBLIC_URL, keys_dir=keys_dir, url_prefix=url_prefix)
    return TestClient(server.app), keys_dir


def test_names_the_exact_path_and_media_type_the_platform_asks_for():
    assert DIRECTORY_WELL_KNOWN_PATH == "/.well-known/http-message-signatures-directory"
    assert DIRECTORY_MEDIA_TYPE == "application/http-message-signatures-directory+json"
    assert DIRECTORY_MAX_KEYS == 16
    # Public by declaration, in the list both SDKs keep identical.
    assert ".well-known/http-message-signatures-directory" in PUBLIC_SUBPATHS


def test_what_a_legacy_string_signer_names_is_served_at_the_origin_and_lists_the_signing_key(tmp_path):
    client, keys_dir = serve(tmp_path, url_prefix="/agents")
    signer = manager(keys_dir, "mini")
    signed = sign_request(
        signer.held_ed25519_keys(), f"{PUBLIC_URL}/agents/mini", "POST", "https://robutler.ai/api/auth/cli/token", b"{}",
        form="legacy-string",
    )
    # The platform's resolution of a bare string: it must be an origin, and the directory hangs off it.
    origin = json.loads(signed.headers["Signature-Agent"])
    parts = urlsplit(origin)
    assert origin == f"{parts.scheme}://{parts.netloc}" == PUBLIC_URL
    res = client.get(urlsplit(f"{origin}{DIRECTORY_WELL_KNOWN_PATH}").path)
    kids = platform_reads(res)
    assert signer.held_ed25519_keys()[0].thumbprint in kids

    # Not under the prefix or the agent: a bare origin cannot name a path.
    assert client.get("/agents/mini/.well-known/http-message-signatures-directory").status_code == 404
    # The key set at its own path keeps its own media type, and its RSA entry stays there.
    own = client.get("/agents/mini/.well-known/jwks.json")
    assert own.headers["content-type"].startswith("application/json")
    assert any(k.get("kty") == "RSA" for k in own.json()["keys"])


def test_one_origin_directory_lists_every_static_agent_in_hosting_order(tmp_path):
    client, keys_dir = serve(tmp_path, names=("echo", "mini"))
    kids = platform_reads(client.get(DIRECTORY_WELL_KNOWN_PATH))
    assert kids == [manager(keys_dir, "echo").held_ed25519_keys()[0].thumbprint, manager(keys_dir, "mini").held_ed25519_keys()[0].thumbprint]


def test_nothing_signing_is_a_404_never_an_empty_directory():
    res = key_directory_response([None])
    assert res.status_code == 404
    assert directory_keys([None, JWKSManager({})]) == []


def test_one_entry_per_thumbprint_and_a_warning_above_the_platform_cap_instead_of_a_dropped_key(tmp_path, caplog):
    managers = [manager(str(tmp_path / f"k{i}"), "a") for i in range(17)]
    # The package's loggers do not propagate to the root, so the capture handler is attached to this one.
    logger = logging.getLogger("webagents.server.core.key_directory")
    logger.addHandler(caplog.handler)
    try:
        res = key_directory_response([*managers, managers[0]])
    finally:
        logger.removeHandler(caplog.handler)
    assert res.status_code == 200
    assert len(json.loads(res.body)["keys"]) == 17
    assert any("17 keys" in r.getMessage() for r in caplog.records)
