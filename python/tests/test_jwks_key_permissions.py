"""Private key material must never be group/world readable."""
import os, stat, tempfile, pytest
from webagents.crypto.jwks import JWKSManager


def test_private_key_and_dir_are_owner_only():
    with tempfile.TemporaryDirectory() as td:
        keys = os.path.join(td, "nested", "keys")
        JWKSManager({"keys_dir": keys, "agent_id": "permcheck"}).ensure_keys("permcheck") \
            if hasattr(JWKSManager, "ensure_keys") else JWKSManager({"keys_dir": keys}).get_public_jwk("permcheck")
        pem = os.path.join(keys, "permcheck.pem")
        assert os.path.exists(pem), os.listdir(keys)
        assert stat.S_IMODE(os.stat(pem).st_mode) == 0o600, oct(stat.S_IMODE(os.stat(pem).st_mode))
        assert stat.S_IMODE(os.stat(keys).st_mode) == 0o700, oct(stat.S_IMODE(os.stat(keys).st_mode))
