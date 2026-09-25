"""
The Ed25519 identity key beside the RSA key (ADR 0038 step 5, W2 design
section 9.2, 2026-09-17): `JWKSManager.ensure_ed25519_key`,
`get_ed25519_public_jwk`, `get_ed25519_thumbprint`, `held_ed25519_keys`,
and `get_jwks` listing the Ed25519 entries first.

The file layout is part of the contract. A NEW key is written as the
TypeScript store writes it, `{stem}.ed25519.jwk.json` (a private JWK,
owner-only, in an owner-only directory), beside the RSA `{safe_id}.pem`; a key
this SDK wrote before 2026-09-25, `{safe_id}.ed25519.pem` (PKCS#8 PEM), is
still read and never rewritten, so one agent keeps one identity whichever SDK
serves it. `.ed25519.previous.jwk.json` or `.ed25519.previous.pem` holds a key
still held while rotating. The key MUST survive restarts, because the platform
selects it by thumbprint from the published key set.
"""

import json

import os
import stat

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa

from webagents.crypto.http_signature import jwk_thumbprint
from webagents.crypto.jwks import JWKSManager

AGENT_ID = "mini"


def _raw(private_key) -> bytes:
    return private_key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())


def _pem(private_key) -> bytes:
    return private_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())


def test_generates_an_owner_only_private_jwk_and_returns_the_thumbprint(tmp_path, capsys):
    keys_dir = tmp_path / "nested" / "keys"
    manager = JWKSManager({"keys_dir": str(keys_dir)})
    thumbprint = manager.ensure_ed25519_key(AGENT_ID)

    key_file = keys_dir / "mini.ed25519.jwk.json"
    assert key_file.exists()
    assert stat.S_IMODE(os.stat(key_file).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(keys_dir).st_mode) == 0o700
    jwk = json.loads(key_file.read_text())
    assert jwk["kty"] == "OKP" and jwk["crv"] == "Ed25519" and set(jwk) == {"kty", "crv", "d", "x"}

    assert len(thumbprint) == 43
    assert manager.get_ed25519_thumbprint() == thumbprint
    # The TypeScript store's line, at the terminal.
    assert f"[webagents] created agent key {key_file}" in capsys.readouterr().out


def test_a_key_the_python_sdk_wrote_as_pem_is_kept_and_never_rewritten(tmp_path):
    private_key = ed25519.Ed25519PrivateKey.generate()
    pem_file = tmp_path / "mini.ed25519.pem"
    pem_file.write_bytes(_pem(private_key))
    os.chmod(pem_file, 0o600)

    manager = JWKSManager({"keys_dir": str(tmp_path)})
    thumbprint = manager.ensure_ed25519_key(AGENT_ID)
    assert _raw(manager.get_ed25519_signing_key()) == _raw(private_key)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["mini.ed25519.pem"]
    assert JWKSManager({"keys_dir": str(tmp_path)}).ensure_ed25519_key(AGENT_ID) == thumbprint


def test_two_files_holding_two_different_keys_are_refused_by_name(tmp_path):
    JWKSManager({"keys_dir": str(tmp_path)}).ensure_ed25519_key(AGENT_ID)
    (tmp_path / "mini.ed25519.pem").write_bytes(_pem(ed25519.Ed25519PrivateKey.generate()))
    with pytest.raises(RuntimeError, match="two different agent keys"):
        JWKSManager({"keys_dir": str(tmp_path)}).ensure_ed25519_key(AGENT_ID)


def test_the_same_key_in_both_formats_is_one_identity(tmp_path):
    manager = JWKSManager({"keys_dir": str(tmp_path)})
    thumbprint = manager.ensure_ed25519_key(AGENT_ID)
    (tmp_path / "mini.ed25519.pem").write_bytes(_pem(manager.get_ed25519_signing_key()))
    assert JWKSManager({"keys_dir": str(tmp_path)}).ensure_ed25519_key(AGENT_ID) == thumbprint


def test_the_published_jwk_shape_and_kid(tmp_path):
    manager = JWKSManager({"keys_dir": str(tmp_path)})
    thumbprint = manager.ensure_ed25519_key(AGENT_ID)
    jwk = manager.get_ed25519_public_jwk()
    assert set(jwk) == {"kty", "crv", "x", "kid", "use"}
    assert jwk["kty"] == "OKP" and jwk["crv"] == "Ed25519" and jwk["use"] == "sig"
    assert jwk["kid"] == thumbprint == jwk_thumbprint(jwk)
    assert "alg" not in jwk


def test_the_key_survives_a_restart(tmp_path):
    first = JWKSManager({"keys_dir": str(tmp_path)}).ensure_ed25519_key(AGENT_ID)
    second = JWKSManager({"keys_dir": str(tmp_path)}).ensure_ed25519_key(AGENT_ID)
    assert first == second
    assert len(list(tmp_path.glob("*.ed25519.jwk.json"))) == 1


def test_it_lives_beside_the_rsa_key_and_neither_replaces_the_other(tmp_path):
    manager = JWKSManager({"keys_dir": str(tmp_path)})
    rsa_kid = manager.ensure_keys(AGENT_ID)
    thumbprint = manager.ensure_ed25519_key(AGENT_ID)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["mini.ed25519.jwk.json", "mini.pem"]
    assert isinstance(manager.get_signing_key(), rsa.RSAPrivateKey)
    assert manager.get_kid() == rsa_kid and len(rsa_kid) == 16
    assert manager.get_ed25519_thumbprint() == thumbprint != rsa_kid


def test_get_jwks_lists_the_ed25519_key_first_and_keeps_the_rsa_entry(tmp_path):
    manager = JWKSManager({"keys_dir": str(tmp_path)})
    manager.ensure_keys(AGENT_ID)
    manager.ensure_ed25519_key(AGENT_ID)
    keys = manager.get_jwks()["keys"]
    assert [k["kty"] for k in keys] == ["OKP", "RSA"]
    assert keys[0] == manager.get_ed25519_public_jwk()
    assert keys[1] == manager.get_public_jwk()
    assert keys[1]["alg"] == "RS256"  # the RS256 consumers' entry, untouched


def test_get_jwks_lists_only_what_was_ensured(tmp_path):
    only_ed = JWKSManager({"keys_dir": str(tmp_path / "a")})
    only_ed.ensure_ed25519_key(AGENT_ID)
    assert [k["kty"] for k in only_ed.get_jwks()["keys"]] == ["OKP"]
    only_rsa = JWKSManager({"keys_dir": str(tmp_path / "b")})
    only_rsa.ensure_keys(AGENT_ID)
    assert [k["kty"] for k in only_rsa.get_jwks()["keys"]] == ["RSA"]


def test_a_previous_key_is_held_and_published_after_the_current_one(tmp_path):
    old = JWKSManager({"keys_dir": str(tmp_path)})
    old_thumbprint = old.ensure_ed25519_key(AGENT_ID)
    # The documented rotation: rename the current file to `.previous.jwk.json`,
    # restart, and a fresh current key is generated beside it.
    (tmp_path / "mini.ed25519.jwk.json").rename(tmp_path / "mini.ed25519.previous.jwk.json")

    rotated = JWKSManager({"keys_dir": str(tmp_path)})
    new_thumbprint = rotated.ensure_ed25519_key(AGENT_ID)
    assert new_thumbprint != old_thumbprint
    held = rotated.held_ed25519_keys()
    assert [k.thumbprint for k in held] == [new_thumbprint, old_thumbprint]
    assert [k["kid"] for k in rotated.get_jwks()["keys"]] == [new_thumbprint, old_thumbprint]
    assert rotated.get_ed25519_thumbprint() == new_thumbprint

    # Deleting the previous file once the platform admitted the new key
    # returns the agent to a single-key set.
    (tmp_path / "mini.ed25519.previous.jwk.json").unlink()
    settled = JWKSManager({"keys_dir": str(tmp_path)})
    assert settled.ensure_ed25519_key(AGENT_ID) == new_thumbprint
    assert [k.thumbprint for k in settled.held_ed25519_keys()] == [new_thumbprint]


def test_a_previous_file_holding_the_current_key_is_not_listed_twice(tmp_path):
    manager = JWKSManager({"keys_dir": str(tmp_path)})
    thumbprint = manager.ensure_ed25519_key(AGENT_ID)
    (tmp_path / "mini.ed25519.previous.jwk.json").write_bytes((tmp_path / "mini.ed25519.jwk.json").read_bytes())
    again = JWKSManager({"keys_dir": str(tmp_path)})
    again.ensure_ed25519_key(AGENT_ID)
    assert [k.thumbprint for k in again.held_ed25519_keys()] == [thumbprint]


def test_a_non_ed25519_key_in_the_ed25519_file_is_refused_by_name(tmp_path):
    manager = JWKSManager({"keys_dir": str(tmp_path)})
    manager.ensure_keys(AGENT_ID)
    (tmp_path / "mini.ed25519.pem").write_bytes((tmp_path / "mini.pem").read_bytes())
    with pytest.raises(RuntimeError, match="mini.ed25519.pem"):
        JWKSManager({"keys_dir": str(tmp_path)}).ensure_ed25519_key(AGENT_ID)


def test_accessors_say_what_to_call_first(tmp_path):
    manager = JWKSManager({"keys_dir": str(tmp_path)})
    for accessor in (
        manager.get_ed25519_signing_key,
        manager.get_ed25519_thumbprint,
        manager.get_ed25519_public_jwk,
        manager.held_ed25519_keys,
    ):
        with pytest.raises(RuntimeError, match="ensure_ed25519_key"):
            accessor()


def test_agent_ids_are_made_filesystem_safe(tmp_path):
    # The identity key takes the TypeScript store's name for the same agent;
    # the RSA key, which only this SDK has, keeps its own.
    manager = JWKSManager({"keys_dir": str(tmp_path)})
    manager.ensure_keys("acme/agents:mini@v2")
    manager.ensure_ed25519_key("acme/agents:mini@v2")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["acme_agents_mini_v2.ed25519.jwk.json", "acme_agents_miniv2.pem"]


def test_a_pem_key_under_the_old_name_rule_is_still_found(tmp_path):
    private_key = ed25519.Ed25519PrivateKey.generate()
    (tmp_path / "acme_agents_miniv2.ed25519.pem").write_bytes(_pem(private_key))
    manager = JWKSManager({"keys_dir": str(tmp_path)})
    manager.ensure_ed25519_key("acme/agents:mini@v2")
    assert _raw(manager.get_ed25519_signing_key()) == _raw(private_key)


# --------------------------------------------------------------------------
# A missing file is the only reason to generate (2026-09-19, the twin of the
# TypeScript store's S-185): an unusable key file raises naming the file, its
# bytes are left alone, a new key never replaces a file, and an existing
# key's permissions are repaired on load.
# --------------------------------------------------------------------------


def test_a_truncated_key_file_raises_naming_the_file_and_is_left_alone(tmp_path):
    manager = JWKSManager({"keys_dir": str(tmp_path)})
    manager.ensure_ed25519_key(AGENT_ID)
    key_file = tmp_path / "mini.ed25519.jwk.json"
    truncated = key_file.read_bytes()[:40]
    key_file.write_bytes(truncated)

    with pytest.raises(RuntimeError, match="mini.ed25519.jwk.json"):
        JWKSManager({"keys_dir": str(tmp_path)}).ensure_ed25519_key(AGENT_ID)
    assert key_file.read_bytes() == truncated
    assert sorted(f.name for f in tmp_path.iterdir()) == ["mini.ed25519.jwk.json"]


def test_a_truncated_pem_key_raises_naming_the_file(tmp_path):
    key_file = tmp_path / "mini.ed25519.pem"
    key_file.write_bytes(_pem(ed25519.Ed25519PrivateKey.generate())[:40])
    with pytest.raises(RuntimeError, match="mini.ed25519.pem"):
        JWKSManager({"keys_dir": str(tmp_path)}).ensure_ed25519_key(AGENT_ID)


def test_an_empty_key_file_is_not_a_missing_one(tmp_path):
    key_file = tmp_path / "mini.ed25519.pem"
    key_file.write_bytes(b"")
    with pytest.raises(RuntimeError, match="mini.ed25519.pem"):
        JWKSManager({"keys_dir": str(tmp_path)}).ensure_ed25519_key(AGENT_ID)
    assert key_file.read_bytes() == b""


def test_a_directory_where_the_key_should_be_raises_instead_of_generating(tmp_path):
    (tmp_path / "mini.ed25519.pem").mkdir()
    with pytest.raises(RuntimeError, match="could not be read"):
        JWKSManager({"keys_dir": str(tmp_path)}).ensure_ed25519_key(AGENT_ID)
    assert (tmp_path / "mini.ed25519.pem").is_dir()


def test_an_unusable_previous_key_raises_before_a_current_one_is_generated(tmp_path):
    (tmp_path / "mini.ed25519.previous.pem").write_bytes(b"-----BEGIN PRIVATE KEY-----\nMC4C")
    with pytest.raises(RuntimeError, match="mini.ed25519.previous.pem"):
        JWKSManager({"keys_dir": str(tmp_path)}).ensure_ed25519_key(AGENT_ID)
    assert sorted(f.name for f in tmp_path.iterdir()) == ["mini.ed25519.previous.pem"]


def test_a_new_key_never_replaces_a_file_and_leaves_no_temporary_file(tmp_path):
    manager = JWKSManager({"keys_dir": str(tmp_path)})
    thumbprint = manager.ensure_ed25519_key(AGENT_ID)
    key_file = tmp_path / "mini.ed25519.jwk.json"
    assert sorted(f.name for f in tmp_path.iterdir()) == ["mini.ed25519.jwk.json"]

    # The persist step itself refuses to replace: a concurrent creator wins.
    other = ed25519.Ed25519PrivateKey.generate()
    before = key_file.read_bytes()
    with pytest.raises(FileExistsError):
        manager._persist_private_key(key_file, manager._ed25519_jwk_bytes(other))
    assert key_file.read_bytes() == before
    assert sorted(f.name for f in tmp_path.iterdir()) == ["mini.ed25519.jwk.json"]
    assert JWKSManager({"keys_dir": str(tmp_path)}).ensure_ed25519_key(AGENT_ID) == thumbprint


def test_a_first_boot_that_loses_the_race_holds_the_winners_key(tmp_path, monkeypatch):
    winner = JWKSManager({"keys_dir": str(tmp_path / "w")})
    winner_thumbprint = winner.ensure_ed25519_key(AGENT_ID)
    winner_bytes = (tmp_path / "w" / "mini.ed25519.jwk.json").read_bytes()

    loser = JWKSManager({"keys_dir": str(tmp_path / "l")})
    real_persist = loser._persist_private_key

    def persist_after_the_winner(key_file, private_key):
        # The other process lands its file between the existence check and the link.
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.write_bytes(winner_bytes)
        return real_persist(key_file, private_key)

    monkeypatch.setattr(loser, "_persist_private_key", persist_after_the_winner)
    assert loser.ensure_ed25519_key(AGENT_ID) == winner_thumbprint
    assert (tmp_path / "l" / "mini.ed25519.jwk.json").read_bytes() == winner_bytes


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
def test_loading_an_existing_key_repairs_its_permissions(tmp_path):
    manager = JWKSManager({"keys_dir": str(tmp_path)})
    thumbprint = manager.ensure_ed25519_key(AGENT_ID)
    key_file = tmp_path / "mini.ed25519.jwk.json"
    os.chmod(key_file, 0o644)
    os.chmod(tmp_path, 0o755)

    assert JWKSManager({"keys_dir": str(tmp_path)}).ensure_ed25519_key(AGENT_ID) == thumbprint
    assert stat.S_IMODE(os.stat(key_file).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(tmp_path).st_mode) == 0o700
