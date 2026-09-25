"""
The CLI's platform credential, in the OS keystore.

WHY (2026-09-23, logged as S-211). The token was written with a bare
`path.write_text` and no mode, so it landed at 0644: any other local user could
read a 7-day platform JWT scoped `agents:own` out of
`~/.webagents/credentials.json`. The TypeScript CLI did the same into
`auth.json`, and the two names meant logging in with one SDK left the other
logged out.

This does NOT implement a credential store. Both SDKs already ship a good one
for `SecretsSkill` (`agents/skills/local/secrets/store.py` and
`typescript/src/skills/secrets/store.ts`): keystore first via `keyring`, an
owner-only 0600 file in a 0700 directory as the documented fallback, and,
crucially, a check of the BACKEND rather than the import, because
`import keyring` succeeds on a headless box and then hands back a `fail`
backend that only raises on use. Writing a third store would mean getting that
subtlety right a third time. This opens theirs under a `cli` namespace.

A KEYSTORE CAN BLOCK. On macOS the first access by a given binary may raise an
authorization dialog, and a process with nobody to click it waits. Observed
2026-09-23: the TypeScript CLI hung on exactly this in a headless shell while
Python did not, because the two use different keyring bindings and macOS
authorises per binary. This is inherent to OS keystores, and it is one more
reason the file fallback and `WEBAGENTS_TOKEN` exist. Anything that must not
block (CI, a container, a script) should pass `--token` or set the env var,
both of which are checked BEFORE the keystore is opened.

PRECEDENCE for reading a token, highest first:

    1. an explicit `--token` flag
    2. `WEBAGENTS_TOKEN` in the environment
    3. the keystore (or its 0600 file fallback)

The env var is the CI answer. A container has no keystore, which is exactly why
the file fallback exists, but a token in an env var is better than a token in a
file on a build agent either way.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

#: The keystore namespace. Distinct from any agent's own secrets, which are
#: namespaced per agent, so `reset --scope credentials` cannot take them out.
CLI_NAMESPACE = "cli"

#: The single entry inside that namespace.
TOKEN_KEY = "platform_token"

#: Read for a token before the keystore. Named for the CLI, not the platform,
#: because it is this CLI's credential.
TOKEN_ENV_VAR = "WEBAGENTS_TOKEN"

#: The `--token` flag for this process, handed over by the root callback.
#:
#: THE FLAG USED TO BE DROPPED (S-222, 2026-09-24). The callback read it into
#: `ctx.obj`, and nothing below the callback looks there: `get_token()` is
#: reached with no `explicit` argument from every command. So `webagents
#: --token "$T" deploy` ran as whoever was logged in on the machine, which on a
#: shared CI runner is the wrong account. Deliberately NOT fixed by exporting
#: `WEBAGENTS_TOKEN`: that would also hand the bearer to every child process
#: (the daemon auto-start among them), which the flag never asked for.
_flag_token: Optional[str] = None


def set_flag_token(token: Optional[str]) -> None:
    """Record this invocation's `--token` (or clear it with None)."""
    global _flag_token
    _flag_token = token or None


def _store(profile: Optional[str] = None, quiet: bool = True):
    """Open the shared secret store for the CLI namespace."""
    from ..agents.skills.local.secrets.store import open_secret_store
    from .config_store import global_dir, profile_name, scoped_namespace

    # BOTH have to carry the profile. Scoping only `secrets_dir` moved the
    # fallback FILE and left the keychain entry shared, so `--profile` isolated
    # the token on machines with no keystore and not on machines with one
    # (S-219, measured 2026-09-23).
    resolved = profile_name(profile)
    secrets_dir = str(global_dir(resolved) / "secrets")
    return open_secret_store(
        namespace=scoped_namespace(CLI_NAMESPACE, resolved),
        secrets_dir=secrets_dir,
        quiet=quiet,
    )


def get_token(explicit: Optional[str] = None, profile: Optional[str] = None) -> Optional[str]:
    """The platform token, by the precedence in the module docstring."""
    if explicit:
        return explicit
    if _flag_token:
        return _flag_token
    from_env = os.environ.get(TOKEN_ENV_VAR)
    if from_env:
        return from_env
    try:
        return _store(profile).get(TOKEN_KEY)
    except Exception:
        # A missing keystore must not make the CLI unusable; the caller treats
        # None as "not logged in" and says so.
        return None


def set_token(token: str, profile: Optional[str] = None, quiet: bool = False) -> str:
    """Store the token. Returns the backend it landed in: 'keystore' or 'file'.

    `quiet` for a caller that says where it went in its own words (the sign-in
    flows: "token stored in an owner-only file"), so the person does not read
    the same fact twice, once as a log warning."""
    store = _store(profile, quiet=quiet)
    store.set(TOKEN_KEY, token)
    return "keystore" if store.keystore else "file"


def clear_token(profile: Optional[str] = None) -> bool:
    """Remove the stored token. True if there was one."""
    try:
        return _store(profile).delete(TOKEN_KEY)
    except Exception:
        return False


def backend_status(profile: Optional[str] = None) -> Dict[str, Any]:
    """Which backend is in use and why.

    This is what `doctor` prints. "I thought my token was in the Keychain" is
    the exact failure the underlying store was written to prevent, and it can
    only be prevented by SAYING which backend answered.
    """
    try:
        store = _store(profile)
        status = dict(store.status())
    except Exception as e:
        status = {"backend": "unavailable", "reason": str(e)}
    status["env_var_set"] = bool(os.environ.get(TOKEN_ENV_VAR))
    if status["env_var_set"]:
        # Worth saying plainly: an env var silently outranks whatever is stored,
        # so a stale export explains "I logged in and it still uses the old
        # account" better than anything else will.
        status["note"] = (
            f"{TOKEN_ENV_VAR} is set and takes precedence over the stored token."
        )
    return status
