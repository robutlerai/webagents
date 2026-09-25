"""
JWKS Manager: the agent's two signing keys, its published key set, and a
cache of other parties' key sets.

TWO KEYS, ON PURPOSE (ADR 0038 step 5, W2 design section 9.2, 2026-09-17):

  * The Ed25519 key (`{safe_id}.ed25519.pem`) is the agent's identity toward
    the platform. It signs every authenticated request (RFC 9421 HTTP
    Message Signatures under the Web Bot Auth profile, see
    `webagents.crypto.http_signature`) and the claim token
    (`mint_claim_token`). Its `kid` is the RFC 7638 thumbprint, which is
    also the `keyid` on the wire; the platform selects the key by it.
  * The RSA key (`{safe_id}.pem`) stays for the RS256 consumers that are
    not the platform credential: the auth skill's self-issued tokens
    (`agents/skills/local/auth/skill.py`), the authorization-code grant
    (`auth/grants/authorization_code.py`), the x402 payment skill
    (`payments_x402/skill.py`) and `agents/skills/base.py`. Nothing about
    it changed; the bearer assertion it used to mint for registration
    (`mint_aoauth_token`) is gone, with no compatibility window.

`get_jwks()` publishes BOTH in one set, the Ed25519 entries first. The
platform ignores non-OKP entries and never reads a JWK `alg`, so the mixed
set verifies there; a third-party verifier that is strict about a set
carrying one key type is not a consumer of this set in W2, and this
paragraph is the deviation note the design asks for.

ROTATION: a `{safe_id}.ed25519.previous.pem` beside the current key file is
loaded as a held key. The key set lists it after the current key and the
signer co-signs every request with it (`held_ed25519_keys`, W2 design section
2.5), which is what the platform's continuity rule needs. To rotate, rename
the current file to `.previous.pem`, restart, and delete the previous file
once the platform has admitted the new key.

The rest of the file is the key-set CACHE for verifying other parties'
tokens (ETag revalidation, refresh on a `kid` miss, a fetch rate limit, and
the S-135 destination policy below).
"""

from dataclasses import dataclass
from typing import Dict, Optional, List, Any
import os
import secrets
import stat
import time
import base64
import hashlib
import logging
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa
from cryptography.hazmat.backends import default_backend
import httpx
import jwt
import ipaddress
import json
import re
import socket
from urllib.parse import urlsplit

from .http_signature import SigningKey, ed25519_public_jwk


# S-135 twin (2026-09-17): WHERE a key set may be fetched from is decided by
# configuration, never by the token. Two callers built `${iss}/.well-known/
# jwks.json` from a bearer's unverified `iss` and handed it to `fetch_jwks`
# (PaymentSkillX402._verify_payment_token, now pinned to the platform issuer,
# and the AOAuth AuthSkill.validate_token allow-list branch, which dials an
# issuer's own key set by design and is now marked `untrusted_origin`). So
# anyone who could reach such an agent could make it GET link-local, private
# and cloud-metadata addresses, blind, follow redirects from there, and grow
# `_jwks_cache` by one entry per distinct `iss`. The two predicates below are
# the destination policy, `fetch_jwks` enforces them and bounds the cache.

KEY_SET_PATH = "/.well-known/jwks.json"

# S-135 addendum 2 (2026-09-18): the most a key set may weigh on the wire. A
# real set is a few hundred bytes per entry and a handful of entries; the
# platform refuses a set of more than a few entries anyway. `fetch_jwks` reads
# the body as a stream and stops here, because `httpx` buffers a plain GET in
# full and its timeout is per read, so a permitted public origin streaming an
# endless 200 body used to grow the process without bound.
KEY_SET_MAX_BYTES = 64 * 1024

# WHATWG host parsing's "ends in a number" rule: a last label that is decimal
# or `0x` hex makes the whole host an IPv4 candidate, parsed the way the
# socket layer resolves it (`inet_aton`: `127.1`, `2130706433`, `0x7f000001`,
# `0177.0.0.1` are all 127.0.0.1) or refused when it is not an address.
_NUMERIC_LABEL_RE = re.compile(r"^(?:0x[0-9a-f]*|[0-9]+)$")


def is_well_formed_key_set_url(url: str) -> bool:
    """A key-set URL ANY caller may fetch: http or https, a host, no userinfo,
    no query, no fragment. This is the floor applied to every fetch, including
    the operator's own platform URL, which is routinely plain http to an
    in-cluster name (`ROBUTLER_INTERNAL_API_URL=http://portal.<ns>.svc.cluster.local`
    on every deployed overlay) or a loopback address in local development, so
    the address filter of `is_public_key_set_url` is deliberately NOT part of
    this floor. A `?` or `#` anywhere is refused outright: that is how the
    fixed well-known suffix became a query or fragment in the S-135 shape.
    """
    if not isinstance(url, str) or not url or "?" in url or "#" in url:
        return False
    try:
        parts = urlsplit(url)
        hostname = parts.hostname
    except ValueError:
        return False
    if parts.scheme not in ("http", "https"):
        return False
    if not hostname or parts.username is not None or parts.password is not None:
        return False
    return True


def _is_special_address(addr: "ipaddress.IPv4Address | ipaddress.IPv6Address") -> bool:
    """Loopback, private, link-local, CGNAT, metadata, multicast, reserved,
    unspecified, NAT64 and v4-carrying IPv6 forms: nothing a key set should
    live on. The v4-mapped (`::ffff:a.b.c.d`) and 6to4 forms are judged by the
    address they are routed as, which is the embedded IPv4 address."""
    if isinstance(addr, ipaddress.IPv6Address):
        if addr in ipaddress.IPv6Network("64:ff9b::/32"):
            return True  # NAT64 translation prefixes: routes into another network
        embedded = addr.ipv4_mapped or addr.sixtofour
        if embedded is not None:
            return _is_special_address(embedded)
    return bool(
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
        or not addr.is_global
    )


def _numeric_ipv4_host(hostname: str) -> "tuple[str, ipaddress.IPv4Address | None]":
    """`("literal", addr)` for a host the socket layer resolves as an IPv4
    address without DNS, `("invalid", None)` for one that ends in a number but
    is no address (the WHATWG parser, which the TypeScript SDK and the platform
    parse with, rejects the host outright), `("name", None)` otherwise."""
    labels = hostname.lower().rstrip(".").split(".")
    if not labels or not _NUMERIC_LABEL_RE.match(labels[-1]):
        return "name", None
    try:
        packed = socket.inet_aton(hostname)
    except OSError:
        return "invalid", None
    return "literal", ipaddress.IPv4Address(packed)


def is_public_key_set_url(url: str) -> bool:
    """The strict form for a key-set URL derived from anything other than the
    operator's platform URL, which today means ONE thing: an issuer named by a
    bearer the local AuthSkill has not verified yet. Well-formed (above), https
    only, not loopback by name, and no IP literal in a special range
    (`_is_special_address`), the literal read the way the socket layer reads
    it (`_numeric_ipv4_host`). Named hosts are otherwise NOT resolved:
    DNS-level filtering (a public name answering with a private address) is
    out of the SDK's remit, so a name passes here on its spelling.

    S-135 addendum 2 (2026-09-18): until that day this admitted plain http to
    `localhost` and `*.localhost`, copied from the TypeScript
    `keySetUrlFromIssuer`, where it is harmless because that function only
    ever sees a CONFIGURED issuer. Here the only caller is token-derived, so
    an unauthenticated bearer could point the agent at its own loopback
    services on any port. A local peer belongs in `trusted_issuers` with its
    `jwks_uri`, which is configuration and is fetched under the floor alone,
    where loopback stays allowed. The same day closed `https://127.1` and
    friends, which `ipaddress` cannot parse and so passed as names.
    """
    if not is_well_formed_key_set_url(url):
        return False
    parts = urlsplit(url)
    hostname = parts.hostname or ""
    if parts.scheme != "https":
        return False
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return False
    try:
        addr: "ipaddress.IPv4Address | ipaddress.IPv6Address" = ipaddress.ip_address(hostname)
    except ValueError:
        kind, numeric = _numeric_ipv4_host(hostname)
        if kind == "name":
            return True
        if kind == "invalid" or numeric is None:
            return False
        addr = numeric
    return not _is_special_address(addr)


@dataclass
class CacheEntry:
    """JWKS cache entry with TTL and ETag support."""
    keys: List[Dict[str, Any]]
    expires_at: float
    etag: Optional[str] = None
    last_fetch: float = 0


class JWKSManager:
    """JWKS management with smart caching.
    
    Features:
    - RSA key pair generation and persistence
    - JWKS caching with configurable TTL
    - ETag support for efficient cache validation
    - Automatic refresh on key miss (handles key rotation)
    - Rate limiting to prevent cache stampede
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize JWKS manager.
        
        Args:
            config: Configuration dictionary with optional keys:
                - keys_dir: Directory for storing RSA keys
                - jwks_cache_ttl: Cache TTL in seconds (default: 3600)
        """
        self.config = config or {}
        self._private_key = None
        self._public_key = None
        self._kid: Optional[str] = None
        # The Ed25519 identity (module docstring): the current key and any
        # previous key still held for rotation, as the signer consumes them.
        self._ed25519_key: Optional[SigningKey] = None
        self._ed25519_previous: List[SigningKey] = []
        self._jwks_cache: Dict[str, CacheEntry] = {}
        self._cache_ttl = self.config.get("jwks_cache_ttl", 3600)
        self._min_refetch_interval = 60  # Prevent spam
        # S-135: the cache used to grow by one entry per distinct URL forever,
        # and the URL used to be the caller's choice. Every key now comes from
        # configuration, so a handful is the whole population; the bound keeps
        # a future caller from turning it back into a per-request allocation.
        # Least recently fetched is evicted (see _make_room).
        self._max_cache_entries = max(1, int(self.config.get("jwks_cache_max_entries", 32)))
        
        # Determine keys directory. WEBAGENTS_KEYS_DIR is the documented
        # override (the TS SDK reads the same variable); the key MUST persist
        # across restarts because platform registration pins the public key
        # published on the agent card.
        keys_dir = self.config.get("keys_dir") or os.getenv("WEBAGENTS_KEYS_DIR")
        if keys_dir:
            self._keys_dir = Path(keys_dir)
        else:
            self._keys_dir = Path.home() / ".webagents" / "keys"
        
        self.logger = logging.getLogger(__name__)
    
    def ensure_keys(self, agent_id: str) -> str:
        """Generate or load RSA key pair for the agent.
        
        Keys are persisted to disk for consistency across restarts.
        The key ID (kid) is derived from the public key hash.
        
        Args:
            agent_id: Agent identifier for key file naming
            
        Returns:
            Key ID (kid) for the generated/loaded key
        """
        key_file = self._keys_dir / f"{self._safe_id(agent_id)}.pem"

        if key_file.exists():
            # Load existing key
            self._private_key = serialization.load_pem_private_key(
                key_file.read_bytes(),
                password=None,
                backend=default_backend()
            )
            self.logger.debug(f"Loaded existing RSA key for {agent_id}")
        else:
            # Generate new key pair
            self._private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            try:
                self._persist_private_key(key_file, self._private_key)
                self.logger.info(f"Generated new RSA key for {agent_id}")
            except FileExistsError:
                # Another process created it first (`_persist_private_key`
                # never replaces a file): hold ITS key.
                self._private_key = serialization.load_pem_private_key(
                    key_file.read_bytes(),
                    password=None,
                    backend=default_backend()
                )
        
        # Extract public key
        self._public_key = self._private_key.public_key()
        
        # Generate key ID from public key hash
        pub_bytes = self._public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        self._kid = hashlib.sha256(pub_bytes).hexdigest()[:16]
        
        return self._kid
    
    def get_signing_key(self) -> Any:
        """Get the private key for signing JWTs.
        
        Returns:
            RSA private key object
            
        Raises:
            RuntimeError: If keys haven't been initialized
        """
        if not self._private_key:
            raise RuntimeError("Keys not initialized. Call ensure_keys() first.")
        return self._private_key
    
    def get_kid(self) -> Optional[str]:
        """Get the current key ID."""
        return self._kid

    @staticmethod
    def _safe_id(agent_id: str) -> str:
        """Sanitize agent_id for the filesystem."""
        return agent_id.replace("/", "_").replace("@", "").replace(":", "_")

    def _persist_private_key(self, key_file: Path, private_key: Any) -> None:
        """Write an UNENCRYPTED PKCS8 private key owner-only, in an owner-only
        directory, WITHOUT EVER REPLACING A FILE. Both keys are the identity
        the platform pins, so default mkdir (0755) and default write (0644)
        would leave them world-readable on any shared machine. The mode= on
        mkdir only applies when the directory is created, so chmod
        unconditionally to repair a directory an earlier version already made.

        The bytes go to a temporary file in the same directory (created
        exclusively with 0600, never write-then-chmod, which leaves a window
        where the key is world-readable; flushed to disk), which is then
        hard-linked to its final name. A link fails with EEXIST rather than
        replacing, so two processes booting the same agent for the first time
        end up holding the SAME key (`FileExistsError` tells the loser to load
        the winner's file) instead of one of them holding a key that is no
        longer on disk. Where hard links are unsupported the fallback is a
        rename. A crash mid-write leaves a `.tmp` beside an absent key file,
        never half a key under the real name (2026-09-19, the twin of the
        TypeScript store's S-185 fix; until then this was an O_TRUNC write in
        place, and a truncated PEM raised on the next start)."""
        self._keys_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self._keys_dir.chmod(0o700)
        except OSError:  # e.g. a dir we do not own; the file mode still holds
            pass
        data = private_key if isinstance(private_key, bytes) else private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        tmp = key_file.with_name(f".{key_file.name}.{secrets.token_hex(8)}.tmp")
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            try:
                os.link(tmp, key_file)
            except FileExistsError:
                raise
            except OSError:
                # No hard links here (some network and FUSE mounts). The file
                # was absent a moment ago, which is the whole reason a key was
                # generated.
                os.replace(tmp, key_file)
            os.chmod(key_file, 0o600)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _repair_permissions(self, key_file: Path) -> None:
        """Tighten an existing key file to 0600 and its directory to 0700.
        `mode=` applies only at creation, so a key an earlier version (or a
        careless `cp`) left group- or world-readable stayed that way. Best
        effort: a directory this process does not own is left alone and the
        key still loads. POSIX only; Windows reports synthetic mode bits.
        The TypeScript store does the same on load (2026-09-19)."""
        if os.name == "nt":
            return
        for target, mode in ((key_file, 0o600), (key_file.parent, 0o700)):
            try:
                current = stat.S_IMODE(os.stat(target).st_mode)
                if current & 0o077:
                    os.chmod(target, mode)
                    self.logger.warning(
                        f"{target} was accessible to other users (mode {current:o}); tightened to {mode:o}"
                    )
            except OSError:
                pass

    # ---- The Ed25519 identity key, whichever SDK wrote it ----
    #
    # ONE IDENTITY PER AGENT, WHICHEVER SDK SERVES IT (2026-09-25). This SDK
    # kept the key as `{safe_id}.ed25519.pem` and the TypeScript one as
    # `{stem}.ed25519.jwk.json`, so an agent served by one and then the other
    # had two identities, and the platform saw a stranger. Both SDKs now read
    # either file, refuse to guess when both exist and hold different keys, and
    # write a NEW key as the JWK, under the TypeScript file-name rule. A key
    # already on disk is never moved or rewritten.

    @staticmethod
    def _jwk_stem(agent_id: str) -> str:
        """The TypeScript store's file stem (`identity-store.ts`, `keyFileStem`)."""
        return re.sub(r"[^A-Za-z0-9._-]", "_", agent_id)

    def _ed25519_candidates(self, agent_id: str, previous: bool = False) -> List[Path]:
        """Every file this agent's current (or previous) key may be in, JWK first."""
        part = ".ed25519.previous" if previous else ".ed25519"
        names = [
            f"{self._jwk_stem(agent_id)}{part}.jwk.json",
            f"{self._safe_id(agent_id)}{part}.pem",
            f"{self._jwk_stem(agent_id)}{part}.pem",
        ]
        seen: List[Path] = []
        for name in names:
            path = self._keys_dir / name
            if path not in seen:
                seen.append(path)
        return seen

    def _load_ed25519_any(self, candidates: List[Path]) -> Optional[SigningKey]:
        """The key in whichever candidate exists; None when none does. Two
        files holding DIFFERENT keys raise naming both: guessing would pick an
        identity the platform may not have pinned."""
        found = [(path, self._load_ed25519_file(path)) for path in candidates if os.path.lexists(path)]
        if not found:
            return None
        first_path, first = found[0]
        for path, key in found[1:]:
            if key.thumbprint != first.thumbprint:
                raise RuntimeError(
                    f"{first_path} and {path} hold two different agent keys. One agent has one identity, "
                    "so neither is chosen: keep the one the platform knows and move the other aside."
                )
        return first

    def _load_ed25519_file(self, key_file: Path) -> SigningKey:
        """One key file, a private JWK or a PKCS#8 PEM by its name."""
        if key_file.name.endswith(".jwk.json"):
            return self._load_ed25519_jwk(key_file)
        return self._load_ed25519_key(key_file)

    def _load_ed25519_jwk(self, key_file: Path) -> SigningKey:
        """A private Ed25519 JWK as the TypeScript store writes it, or raise naming the file."""
        try:
            jwk = json.loads(key_file.read_text())
        except OSError as e:
            raise RuntimeError(
                f"the agent key file {key_file} could not be read ({e}). It holds the identity the "
                "platform pinned, so it is never replaced automatically: fix the file, or move it "
                "aside yourself to have a NEW key generated (the platform will see a key rotation)."
            ) from e
        except ValueError as e:
            raise RuntimeError(
                f"the agent key file {key_file} is not valid JSON ({e}; a truncated write looks like this). "
                "It holds the identity the platform pinned, so it is never replaced automatically."
            ) from e
        if not (
            isinstance(jwk, dict) and jwk.get("kty") == "OKP" and jwk.get("crv") == "Ed25519"
            and isinstance(jwk.get("d"), str) and isinstance(jwk.get("x"), str)
        ):
            raise RuntimeError(f"{key_file} is not an Ed25519 private JWK (kty \"OKP\", crv \"Ed25519\", with d and x)")
        try:
            raw = base64.urlsafe_b64decode(jwk["d"] + "=" * (-len(jwk["d"]) % 4))
            private_key = ed25519.Ed25519PrivateKey.from_private_bytes(raw)
        except Exception as e:  # noqa: BLE001 - binascii.Error, ValueError
            raise RuntimeError(f"{key_file} could not be read as an Ed25519 key ({e})") from e
        key = SigningKey.from_private_key(private_key)
        public_x = ed25519_public_jwk(private_key.public_key())["x"]
        if public_x != jwk["x"]:
            raise RuntimeError(f"{key_file} is not a consistent key: its x does not match its d")
        self._repair_permissions(key_file)
        return key

    @staticmethod
    def _ed25519_jwk_bytes(private_key: ed25519.Ed25519PrivateKey) -> bytes:
        """A private JWK, as the TypeScript store writes one."""
        raw = private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        d = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
        x = ed25519_public_jwk(private_key.public_key())["x"]
        return json.dumps({"crv": "Ed25519", "d": d, "x": x, "kty": "OKP"}).encode()

    def _load_ed25519_key(self, key_file: Path) -> SigningKey:
        """Load one Ed25519 key file, or raise naming the file. A key file
        that exists and cannot be used is NEVER a reason to generate: it holds
        the identity the platform pinned, and generating would replace it
        (the TypeScript store's S-185, 2026-09-19). Only a missing file is,
        and the caller decides that before calling this."""
        try:
            data = key_file.read_bytes()
        except OSError as e:
            raise RuntimeError(
                f"the agent key file {key_file} could not be read ({e}). It holds the identity the "
                "platform pinned, so it is never replaced automatically: fix the file, or move it "
                "aside yourself to have a NEW key generated (the platform will see a key rotation)."
            ) from e
        try:
            key = serialization.load_pem_private_key(data, password=None)
        except Exception as e:  # noqa: BLE001 - ValueError, TypeError, UnsupportedAlgorithm
            raise RuntimeError(
                f"the agent key file {key_file} is not a readable PKCS#8 PEM private key ({e}); a "
                "truncated write looks like this. It holds the identity the platform pinned, so it is "
                "never replaced automatically: restore it from a backup, or move it aside yourself to "
                "have a NEW key generated (the platform will see a key rotation)."
            ) from e
        if not isinstance(key, ed25519.Ed25519PrivateKey):
            raise RuntimeError(
                f"{key_file} is not an Ed25519 private key. The Ed25519 identity "
                "lives in its own file beside the RSA key; move or delete this "
                "file and restart to generate a fresh one."
            )
        self._repair_permissions(key_file)
        return SigningKey.from_private_key(key)

    def ensure_ed25519_key(self, agent_id: str) -> str:
        """Generate or load the agent's Ed25519 identity key (module docstring).

        The key is read from `{stem}.ed25519.jwk.json` or `{safe_id}.ed25519.pem`
        in the keys directory, whichever exists (see "One identity per agent"
        above); a new one is written as the JWK, the same owner-only way as the
        RSA key. A previous key beside it (`.ed25519.previous.jwk.json` or
        `.ed25519.previous.pem`), when present, is held for rotation (W2
        design section 2.5).

        A MISSING file is the only reason to generate. A key file that exists
        and cannot be read, parsed or used raises `RuntimeError` naming the
        file and nothing is written (`_load_ed25519_key`); a new key never
        replaces a file (`_persist_private_key`); an existing key's
        permissions are repaired on load. The TypeScript store
        (`crypto/identity-store.ts`) follows the same three rules since
        2026-09-19 (S-185).

        Args:
            agent_id: Agent identifier for key file naming

        Returns:
            The RFC 7638 thumbprint: the published `kid` and the `keyid` on
            the wire.
        """
        # Both keys are read, and an unusable file raises, BEFORE anything is
        # generated or written: a previous key the operator put there to
        # co-sign is as much a reason to stop as the current one.
        # `os.path.lexists`, not `exists`: a dangling symlink is a file that
        # cannot be used, not a missing one.
        previous = self._load_ed25519_any(self._ed25519_candidates(agent_id, previous=True))
        current = self._load_ed25519_any(self._ed25519_candidates(agent_id))

        if current is not None:
            self._ed25519_key = current
            self.logger.debug(f"Loaded existing Ed25519 key for {agent_id}")
        else:
            # A new key is written as the TypeScript store writes it (see
            # "One identity per agent" above).
            key_file = self._keys_dir / f"{self._jwk_stem(agent_id)}.ed25519.jwk.json"
            private_key = ed25519.Ed25519PrivateKey.generate()
            try:
                self._persist_private_key(key_file, self._ed25519_jwk_bytes(private_key))
                self._ed25519_key = SigningKey.from_private_key(private_key)
                # The TypeScript store's line (`identity-store.ts`), at the terminal.
                print(f"[webagents] created agent key {key_file}", flush=True)
            except FileExistsError:
                # Another process created the key between the check above and
                # the link: hold ITS key, so both are the same identity.
                self._ed25519_key = self._load_ed25519_jwk(key_file)
                self.logger.info(f"Loaded the Ed25519 key another process just created for {agent_id}")

        self._ed25519_previous = []
        if previous is not None:
            if previous.thumbprint != self._ed25519_key.thumbprint:
                self._ed25519_previous.append(previous)
                self.logger.info(
                    f"Holding previous Ed25519 key {previous.thumbprint} for {agent_id} "
                    "beside the current one; every request is co-signed with it"
                )

        return self._ed25519_key.thumbprint

    @property
    def keys_dir(self) -> Path:
        """Where this manager reads and writes key files: `keys_dir` from the
        config, else `WEBAGENTS_KEYS_DIR`, else `~/.webagents/keys`."""
        return self._keys_dir

    def load_ed25519_key(self, agent_id: str) -> Optional[str]:
        """`ensure_ed25519_key` without the generating half (2026-09-23).

        For a caller that PIGGYBACKS on an identity rather than establishing
        one: `DiscoverySkill` signs its platform calls with the key the server
        publishes for the agent, and a key it minted itself would be one no
        key set serves, so the platform could never verify with it. Absent
        file, answer `None` and write nothing; present file, load it under
        the same rules as `ensure_ed25519_key` (a previous key beside it is
        held for rotation, an unusable file raises naming the path, never a
        silent replacement). Returns the thumbprint, as `ensure_ed25519_key`
        does, so the two are interchangeable once a key exists.
        """
        if not any(os.path.lexists(path) for path in self._ed25519_candidates(agent_id)):
            return None
        return self.ensure_ed25519_key(agent_id)

    def get_ed25519_signing_key(self) -> ed25519.Ed25519PrivateKey:
        """The current Ed25519 private key.

        Raises:
            RuntimeError: If ensure_ed25519_key() has not been called
        """
        if not self._ed25519_key:
            raise RuntimeError("Ed25519 key not initialized. Call ensure_ed25519_key() first.")
        return self._ed25519_key.private_key

    def get_ed25519_thumbprint(self) -> str:
        """RFC 7638 thumbprint of the current Ed25519 key: the `kid` and `keyid`.

        Raises:
            RuntimeError: If ensure_ed25519_key() has not been called
        """
        if not self._ed25519_key:
            raise RuntimeError("Ed25519 key not initialized. Call ensure_ed25519_key() first.")
        return self._ed25519_key.thumbprint

    def get_ed25519_public_jwk(self) -> Dict[str, Any]:
        """The current Ed25519 key as the key set publishes it:
        `{kty: OKP, crv: Ed25519, x, kid: <thumbprint>, use: sig}` and no `alg`.

        Raises:
            RuntimeError: If ensure_ed25519_key() has not been called
        """
        if not self._ed25519_key:
            raise RuntimeError("Ed25519 key not initialized. Call ensure_ed25519_key() first.")
        return self._ed25519_key.public_jwk()

    def held_ed25519_keys(self) -> List[SigningKey]:
        """Every Ed25519 key the agent holds, current first, as the request
        signer consumes them (`WebBotAuth(keys=...)`). One entry unless a
        previous key is held for rotation.

        Raises:
            RuntimeError: If ensure_ed25519_key() has not been called
        """
        if not self._ed25519_key:
            raise RuntimeError("Ed25519 key not initialized. Call ensure_ed25519_key() first.")
        return [self._ed25519_key, *self._ed25519_previous]

    def get_public_jwk(self) -> Dict[str, Any]:
        """Get public key in JWK format for JWKS endpoint.
        
        Returns:
            JWK dictionary with RSA public key
            
        Raises:
            RuntimeError: If keys haven't been initialized
        """
        if not self._public_key:
            raise RuntimeError("Keys not initialized. Call ensure_keys() first.")
        
        numbers = self._public_key.public_numbers()
        
        def int_to_base64(n: int, length: int) -> str:
            """Convert integer to base64url encoding without padding."""
            return base64.urlsafe_b64encode(
                n.to_bytes(length, 'big')
            ).decode().rstrip('=')
        
        return {
            "kty": "RSA",
            "use": "sig",
            "alg": "RS256",
            "kid": self._kid,
            "n": int_to_base64(numbers.n, 256),
            "e": int_to_base64(numbers.e, 3),
        }
    
    def get_jwks(self) -> Dict[str, Any]:
        """The key set served at `{agent_url}/.well-known/jwks.json`.

        The Ed25519 entries come first (current key, then any held previous
        key), then the RSA entry for the RS256 consumers, each present only
        when its `ensure_*` was called. See the module docstring for why the
        set is mixed.

        Returns:
            JWKS dictionary with keys array
        """
        keys: List[Dict[str, Any]] = []
        if self._ed25519_key:
            keys.extend(key.public_jwk() for key in self.held_ed25519_keys())
        if self._public_key:
            keys.append(self.get_public_jwk())

        return {"keys": keys}

    def get_public_key_spki_pem(self) -> str:
        """The RSA public key as an SPKI PEM string.

        Kept for RS256 consumers that want the PEM form. Nothing publishes it
        on the agent card any more: the card carries `jwks_uri` and the
        platform reads the Ed25519 key from the key set (W2 design section
        3.3, 2026-09-17).

        Raises:
            RuntimeError: If keys haven't been initialized
        """
        if not self._public_key:
            raise RuntimeError("Keys not initialized. Call ensure_keys() first.")
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("ascii")

    def mint_claim_token(
        self,
        agent_id: str,
        platform_url: str,
        ttl_seconds: int = 600,
        *,
        issuer: Optional[str] = None,
    ) -> str:
        """Mint the claim token: the one JWT the SDK still issues (W2 design
        sections 7.2 and 9.2, 2026-09-17).

        A person claims an ownerless agent by POSTing this token to the
        platform's claim route for that agent. It is not an HTTP request from
        the agent, which is why it stays a JWT while every request the agent
        makes is signed instead. EdDSA with the Ed25519 identity key; the
        header `kid` is the thumbprint, which is how the platform selects the
        key it registered. The audience is `{platform_url}/claim`, the scope
        is `agent:claim`, and the UUID `jti` is spent once by the platform.

        Args:
            agent_id: `sub` claim.
            platform_url: the platform's base URL; the audience is derived
                from it, never from the agent's own URL.
            ttl_seconds: token TTL, 10 minutes by default. The token is
                handed to a person to paste, so it is longer-lived than a
                request signature and still bounded.
            issuer: the agent URL, written as `iss` when given. The claim
                route does not read it; it is there for a human reading the
                token.
        """
        if not self._ed25519_key:
            raise RuntimeError("Ed25519 key not initialized. Call ensure_ed25519_key() first.")
        import uuid
        now = int(time.time())
        payload: Dict[str, Any] = {
            "sub": agent_id,
            "aud": f"{platform_url.rstrip('/')}/claim",
            "iat": now,
            "nbf": now,
            "exp": now + ttl_seconds,
            "jti": str(uuid.uuid4()),
            "scope": "agent:claim",
        }
        if issuer:
            payload["iss"] = issuer.rstrip("/")
        return jwt.encode(
            payload,
            self._ed25519_key.private_key,
            algorithm="EdDSA",
            headers={"kid": self._ed25519_key.thumbprint},
        )

    def _make_room(self, jwks_uri: str) -> None:
        """Evict the least recently fetched entries until `jwks_uri` fits (S-135)."""
        if jwks_uri in self._jwks_cache:
            return
        while len(self._jwks_cache) >= self._max_cache_entries:
            oldest = min(self._jwks_cache, key=lambda uri: self._jwks_cache[uri].last_fetch)
            del self._jwks_cache[oldest]

    async def fetch_jwks(
        self,
        jwks_uri: str,
        force_refresh: bool = False,
        *,
        untrusted_origin: bool = False,
    ) -> List[Dict[str, Any]]:
        """Fetch JWKS from remote endpoint with smart caching.

        Features:
        - Respects Cache-Control max-age
        - Uses ETag for conditional requests (304 Not Modified)
        - Rate limits refetches to prevent cache stampede

        Args:
            jwks_uri: URL to fetch JWKS from
            force_refresh: Force fetch even if cache is valid
            untrusted_origin: the URL was derived from something other than
                the operator's own platform configuration (S-135: an issuer
                named by the token). It must then satisfy
                `is_public_key_set_url`, and redirects are not followed, so a
                permitted origin cannot bounce the fetch to an address the
                filter refused. Every URL, trusted or not, must satisfy
                `is_well_formed_key_set_url`.

        Returns:
            List of JWK dictionaries
        """
        keys, _dialled = await self._fetch_jwks(
            jwks_uri, force_refresh=force_refresh, untrusted_origin=untrusted_origin
        )
        return keys

    async def _fetch_jwks(
        self,
        jwks_uri: str,
        *,
        force_refresh: bool,
        untrusted_origin: bool,
    ) -> "tuple[List[Dict[str, Any]], bool]":
        """`fetch_jwks` plus whether this call actually went to the network.

        `get_public_key_from_jwks` needs that bit (S-135 addendum 2,
        2026-09-18): a `kid` miss on keys that were just fetched is final, and
        refetching on it is what made one unauthenticated bearer cost two
        outbound requests against a dead origin. The same change remembers a
        failed fetch as an empty entry for the refetch interval, so the next
        bearer naming the same dead origin costs nothing, and reads the body as
        a stream capped at `KEY_SET_MAX_BYTES` rather than buffering whatever a
        permitted origin chooses to send.
        """
        if not is_well_formed_key_set_url(jwks_uri) or (
            untrusted_origin and not is_public_key_set_url(jwks_uri)
        ):
            self.logger.warning(f"Refusing JWKS fetch: destination not allowed: {jwks_uri!r}")
            return [], False

        now = time.time()
        cached = self._jwks_cache.get(jwks_uri)

        # Return cached if valid and not forcing refresh
        if cached and not force_refresh and cached.expires_at > now:
            return cached.keys, False

        # Rate limit refetches to prevent spam
        if cached and (now - cached.last_fetch) < self._min_refetch_interval:
            self.logger.debug(f"Rate limiting JWKS fetch for {jwks_uri}")
            return cached.keys, False

        try:
            headers = {}
            if cached and cached.etag:
                headers["If-None-Match"] = cached.etag

            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "GET",
                    jwks_uri,
                    timeout=10,
                    headers=headers,
                    follow_redirects=not untrusted_origin,
                ) as resp:
                    # Handle 304 Not Modified
                    if resp.status_code == 304 and cached:
                        self._jwks_cache[jwks_uri] = CacheEntry(
                            keys=cached.keys,
                            expires_at=now + self._cache_ttl,
                            etag=cached.etag,
                            last_fetch=now
                        )
                        self.logger.debug(f"JWKS cache validated (304) for {jwks_uri}")
                        return cached.keys, True

                    resp.raise_for_status()
                    body = bytearray()
                    async for chunk in resp.aiter_bytes():
                        body += chunk
                        if len(body) > KEY_SET_MAX_BYTES:
                            raise ValueError(
                                f"key set larger than {KEY_SET_MAX_BYTES} bytes"
                            )
                    jwks = json.loads(bytes(body))
                    if not isinstance(jwks, dict) or not isinstance(jwks.get("keys", []), list):
                        raise ValueError("key set is not a JSON object with a keys array")

                    # Parse Cache-Control for TTL
                    ttl = self._cache_ttl
                    cache_control = resp.headers.get("Cache-Control", "")
                    if "max-age=" in cache_control:
                        try:
                            max_age_str = cache_control.split("max-age=")[1].split(",")[0].strip()
                            ttl = int(max_age_str)
                        except (ValueError, IndexError):
                            pass

                    # Update cache
                    self._make_room(jwks_uri)
                    self._jwks_cache[jwks_uri] = CacheEntry(
                        keys=jwks.get("keys", []),
                        expires_at=now + ttl,
                        etag=resp.headers.get("ETag"),
                        last_fetch=now
                    )

                    self.logger.debug(f"JWKS fetched from {jwks_uri}, TTL={ttl}s")
                    return jwks.get("keys", []), True

        except httpx.HTTPStatusError as e:
            self.logger.warning(f"JWKS fetch failed for {jwks_uri}: HTTP {e.response.status_code}")
            return self._remember_failure(jwks_uri, cached, now), True
        except Exception as e:
            self.logger.warning(f"JWKS fetch failed for {jwks_uri}: {e}")
            return self._remember_failure(jwks_uri, cached, now), True

    def _remember_failure(
        self, jwks_uri: str, cached: Optional[CacheEntry], now: float
    ) -> List[Dict[str, Any]]:
        """A failed dial keeps serving the stale keys when there are any, and
        otherwise leaves an EMPTY entry, so either way the refetch interval
        holds and a dead origin costs one dial per interval, not one per
        bearer (S-135 addendum 2). The entry is bounded like every other."""
        if cached:
            cached.last_fetch = now
            return cached.keys
        self._make_room(jwks_uri)
        self._jwks_cache[jwks_uri] = CacheEntry(
            keys=[],
            expires_at=now + self._min_refetch_interval,
            etag=None,
            last_fetch=now,
        )
        return []

    async def get_public_key_from_jwks(
        self,
        jwks_uri: str,
        kid: str,
        *,
        untrusted_origin: bool = False,
    ) -> Optional[Any]:
        """Get public key by kid from JWKS, with auto-refresh on miss.

        This handles key rotation gracefully:
        1. First tries to find key in cache
        2. If not found, refreshes JWKS and tries again

        Args:
            jwks_uri: URL to fetch JWKS from
            kid: Key ID to look for
            untrusted_origin: see `fetch_jwks`; pass True whenever `jwks_uri`
                was derived from a token rather than from configuration.

        Returns:
            Public key object or None if not found
        """
        # First try with cached JWKS
        keys, dialled = await self._fetch_jwks(
            jwks_uri, force_refresh=False, untrusted_origin=untrusted_origin
        )

        for key in keys:
            if key.get("kid") == kid:
                try:
                    return jwt.algorithms.RSAAlgorithm.from_jwk(key)
                except Exception as e:
                    self.logger.warning(f"Failed to parse JWK {kid}: {e}")
                    return None

        # A miss on keys this very call fetched is final: the origin has just
        # said what it publishes, and dialling it again is what cost two
        # requests per bearer (S-135 addendum 2, 2026-09-18).
        if dialled:
            self.logger.warning(f"Key {kid} not found at {jwks_uri}")
            return None

        # Key not found in the cached set - try refreshing JWKS (handles key rotation)
        self.logger.info(f"Key {kid} not found in cache, refreshing JWKS from {jwks_uri}")
        keys = await self.fetch_jwks(jwks_uri, force_refresh=True, untrusted_origin=untrusted_origin)
        
        for key in keys:
            if key.get("kid") == kid:
                try:
                    return jwt.algorithms.RSAAlgorithm.from_jwk(key)
                except Exception as e:
                    self.logger.warning(f"Failed to parse JWK {kid} after refresh: {e}")
                    return None
        
        self.logger.warning(f"Key {kid} not found at {jwks_uri} even after refresh")
        return None
    
    def invalidate_cache(self, jwks_uri: Optional[str] = None) -> None:
        """Invalidate JWKS cache.
        
        Args:
            jwks_uri: Specific URI to invalidate, or None to clear all
        """
        if jwks_uri:
            self._jwks_cache.pop(jwks_uri, None)
            self.logger.debug(f"Invalidated JWKS cache for {jwks_uri}")
        else:
            self._jwks_cache.clear()
            self.logger.debug("Cleared all JWKS cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for debugging.
        
        Returns:
            Dictionary with cache statistics
        """
        now = time.time()
        stats = {
            "total_entries": len(self._jwks_cache),
            "entries": {}
        }
        
        for uri, entry in self._jwks_cache.items():
            stats["entries"][uri] = {
                "keys_count": len(entry.keys),
                "expires_in": max(0, int(entry.expires_at - now)),
                "has_etag": entry.etag is not None,
                "last_fetch_ago": int(now - entry.last_fetch),
            }
        
        return stats
