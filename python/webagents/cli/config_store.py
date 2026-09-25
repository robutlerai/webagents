"""
The CLI's configuration and credential store.

ONE on-disk contract, shared with the TypeScript SDK. Before this
(2026-09-23) there were two, and they were wrong in opposite directions:

  * The two CLIs COLLIDED on config. Both wrote `~/.webagents/config.json` with
    incompatible schemas: Python wrote nested objects, TypeScript wrote flat
    strings, so `webagents config set daemon 8080` from the TypeScript CLI
    replaced Python's `{"daemon": {"port": 8765}}` with the string `"8080"`.
  * The two CLIs DIVERGED on credentials. Python wrote
    `~/.webagents/credentials.json`, TypeScript wrote `~/.webagents/auth.json`,
    so logging in with one left the other logged out.

Neither had a defaults layer, an env layer or a flag layer, and the TypeScript
CLI read no environment variables at all.

PRECEDENCE, highest first. Documented because an undocumented precedence is a
bug report waiting to happen:

    1. an explicit flag            (--token, --profile)
    2. the process environment     (WEBAGENTS_TOKEN, ${VAR} in config)
    3. ./.env                      (the project a developer is standing in)
    4. ~/.webagents/.env           (their machine-wide fallback)
    5. ./.webagents/config.json    (project config, commit it)
    6. ~/.webagents/config.json    (global config)
    7. the defaults below

`${VAR}` REFERENCES, NOT STORED SECRETS. A config value of the form `${NAME}`
resolves from the env chain at read time, so a provider key lives in the
environment or a `.env` and the config file that names it can be committed.
Borrowed from OpenClaw, which gets this right.

PROFILES. `--profile x` or `WEBAGENTS_PROFILE=x` moves EVERY path under
`~/.webagents-x`, so test and production credentials cannot clobber each other.

THE PLATFORM TOKEN IS NOT IN HERE. It goes to the OS keystore via the
`SecretStore` both SDKs already ship, with an owner-only file as the documented
fallback for containers and CI. See `credentials.py`.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

#: Built-in defaults, the lowest layer. Also the schema: a key absent from here
#: is not a key this CLI knows, and `validate()` says so.
DEFAULTS: Dict[str, Any] = {
    "platform.url": "https://robutler.ai",
    "daemon.port": 8765,
    "daemon.host": "127.0.0.1",
    "model": None,
    "profile": None,
    "telemetry.enabled": False,
    # The platform agent this directory deploys to, written by `deploy`/`link`
    # (and by the TypeScript `publish`). Known keys since 2026-09-24: they were
    # missing here, so `config validate` and `doctor` reported the very keys
    # `deploy` had just written as unknown.
    "link.agentId": None,
    "link.agentName": None,
}

#: `${VAR}` or `${VAR:-fallback}`.
_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def profile_name(explicit: Optional[str] = None) -> Optional[str]:
    """The active profile: an explicit flag beats `WEBAGENTS_PROFILE`."""
    return explicit or os.environ.get("WEBAGENTS_PROFILE") or None


def cli_command(rest: str = "", profile: Optional[str] = None) -> str:
    """A `webagents` command as the user should type it here: with
    `--profile <name>` while a profile is active (2026-09-25).

    Every hint that names a command goes through this. They said
    `webagents login` whatever the profile, so under
    `webagents --profile local` following the hint signed the DEFAULT profile
    in and left the local one signed out. The twin of the TypeScript
    `cliCommand` (`config-store.ts`); a name that is not one shell word is
    quoted by `shlex.quote`.
    """
    import shlex

    active = profile_name(profile)
    base = "webagents"
    if active:
        word = active if re.fullmatch(r"[A-Za-z0-9._-]+", active) else shlex.quote(active)
        base = f"webagents --profile {word}"
    return f"{base} {rest}" if rest else base


def scoped_namespace(base: str, profile: Optional[str] = None) -> str:
    """A keystore namespace that carries the profile (S-219, 2026-09-23).

    THE OS KEYSTORE IS KEYED BY NAMESPACE ALONE. `service_key()` is
    `f"webagents:{namespace}"` with no profile in it, so scoping only the
    FALLBACK FILE's directory (which is what the credential store used to do)
    isolates two profiles on a machine with no keystore and lets them share one
    entry on a machine with one. That is backwards: the desktop is where people
    actually keep a test and a production login side by side.

    Measured before the fix: two stores with different `secrets_dir` values,
    keystore backend active, read each other's values.

    The default profile keeps the bare `base` so a token stored before this
    change stays readable from the profile that wrote it.
    """
    resolved = profile_name(profile)
    return base if not resolved else f"{base}-{resolved}"


def global_dir(profile: Optional[str] = None) -> Path:
    """`~/.webagents`, or `~/.webagents-<profile>` under a profile.

    RESOLVES the profile when not given one, rather than treating `None` as
    "no profile". An earlier version took the argument literally, so
    `global_dir()` returned the real directory even with `WEBAGENTS_PROFILE`
    set: the token went to the profile-scoped keystore while the metadata
    beside it landed in the shared directory. Half-isolation is worse than
    none, because it looks like it worked.
    """
    resolved = profile_name(profile)
    name = ".webagents" if not resolved else f".webagents-{resolved}"
    return Path.home() / name


def project_dir(start: Optional[Path] = None) -> Path:
    """`./.webagents`, NOT created as a side effect of reading it.

    `LocalState.__init__` used to scaffold eight directories plus a `.gitignore`
    in whatever directory you happened to be standing in, reached from
    `is_authenticated()`, so `webagents auth whoami` littered any repo you ran
    it in. Reading config creates nothing.
    """
    return (start or Path.cwd()) / ".webagents"


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        # A corrupt config must not make every command unrunnable. `validate()`
        # is where a user asks about correctness; everything else degrades.
        return {}


def _parse_dotenv(path: Path) -> Dict[str, str]:
    """A deliberately small `.env` reader: `KEY=value`, `#` comments, quotes."""
    out: Dict[str, str] = {}
    try:
        text = path.read_text()
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def env_chain(profile: Optional[str] = None, cwd: Optional[Path] = None) -> Dict[str, str]:
    """The resolved environment: process env wins, then ./.env, then ~/.webagents/.env.

    Files never override a variable the process already has. A developer who
    exports something in their shell means it.
    """
    resolved: Dict[str, str] = {}
    resolved.update(_parse_dotenv(global_dir(profile) / ".env"))
    resolved.update(_parse_dotenv((cwd or Path.cwd()) / ".env"))
    resolved.update({k: v for k, v in os.environ.items()})
    return resolved


def expand(value: Any, env: Mapping[str, str]) -> Any:
    """Resolve `${VAR}` and `${VAR:-fallback}` inside a string value.

    An unresolvable `${VAR}` with no fallback is left AS WRITTEN rather than
    replaced with an empty string, so the failure is visible in the value
    instead of silently becoming "".
    """
    if not isinstance(value, str):
        return value

    def sub(match: re.Match) -> str:
        name, fallback = match.group(1), match.group(2)
        if name in env:
            return env[name]
        if fallback is not None:
            return fallback
        return match.group(0)

    return _VAR_RE.sub(sub, value)


class ConfigStore:
    """Layered config with a documented precedence. See the module docstring."""

    def __init__(
        self,
        profile: Optional[str] = None,
        cwd: Optional[Path] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.profile = profile_name(profile)
        self.cwd = cwd or Path.cwd()
        #: Layer 1: explicit flags, highest precedence.
        self.overrides: Dict[str, Any] = dict(overrides or {})
        self.global_path = global_dir(self.profile) / "config.json"
        self.project_path = project_dir(self.cwd) / "config.json"

    # -- reading ----------------------------------------------------------

    @property
    def env(self) -> Dict[str, str]:
        return env_chain(self.profile, self.cwd)

    def layers(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Every layer, highest precedence first, for `config show` and doctor."""
        return [
            ("flag", dict(self.overrides)),
            ("project", _read_json(self.project_path)),
            ("global", _read_json(self.global_path)),
            ("default", dict(DEFAULTS)),
        ]

    def get(self, key: str, default: Any = None) -> Any:
        """The effective value of `key`, with `${VAR}` resolved."""
        env = self.env
        for _, layer in self.layers():
            if key in layer:
                value = layer[key]
                return expand(value, env) if value is not None else default
        return default

    def source_of(self, key: str) -> Optional[str]:
        """Which layer supplied `key`. What `config get --why` prints."""
        for name, layer in self.layers():
            if key in layer:
                return name
        return None

    def effective(self) -> Dict[str, Any]:
        """Every known key with its resolved value."""
        keys = set(DEFAULTS)
        for _, layer in self.layers():
            keys.update(layer)
        return {k: self.get(k) for k in sorted(keys)}

    # -- writing ----------------------------------------------------------

    def set(self, key: str, value: Any, scope: str = "global") -> Path:
        """Write `key` to the project or global layer. Returns the file written."""
        path = self.project_path if scope == "project" else self.global_path
        data = _read_json(path)
        data[key] = value
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        return path

    def unset(self, key: str, scope: str = "global") -> bool:
        path = self.project_path if scope == "project" else self.global_path
        data = _read_json(path)
        if key not in data:
            return False
        del data[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        return True

    # -- validation -------------------------------------------------------

    def validate(self) -> List[str]:
        """Problems with the config as written. Empty means clean.

        Reports UNKNOWN KEYS, because the alternative is what the agent schema
        does today: `extra = "allow"`, which accepts a typo silently and leaves
        the user wondering why their setting does nothing.
        """
        problems: List[str] = []
        env = self.env
        for name, path in (("project", self.project_path), ("global", self.global_path)):
            if not path.exists():
                continue
            try:
                raw = path.read_text()
            except OSError as e:
                problems.append(f"{path}: cannot read ({e})")
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                problems.append(f"{path}: not valid JSON ({e})")
                continue
            if not isinstance(data, dict):
                problems.append(f"{path}: top level must be an object")
                continue
            for key, value in data.items():
                if key not in DEFAULTS:
                    suggestion = _closest(key, DEFAULTS)
                    hint = f"; did you mean {suggestion}?" if suggestion else ""
                    problems.append(f"{path}: unknown key {key!r}{hint}")
                if isinstance(value, str):
                    for match in _VAR_RE.finditer(value):
                        if match.group(1) not in env and match.group(2) is None:
                            problems.append(
                                f"{path}: {key} references ${{{match.group(1)}}}, which is not set"
                            )
        return problems


def _closest(key: str, known: Mapping[str, Any]) -> Optional[str]:
    """A cheap suggestion for a misspelled key. No dependency for this."""
    import difflib

    matches = difflib.get_close_matches(key, list(known), n=1, cutoff=0.6)
    return matches[0] if matches else None


#: Points every platform command at another portal. It is the variable
#: `platform/auth.py` has always read, so an existing export keeps working, and
#: it now means the same thing to `login` as to `deploy`.
PLATFORM_URL_ENV_VAR = "ROBUTLER_API_URL"


def resolve_platform_url(
    profile: Optional[str] = None, cwd: Optional[Path] = None
) -> Tuple[str, str]:
    """The portal the platform commands talk to, and which layer said so.

    ONE ANSWER FOR LOGIN AND DEPLOY (2026-09-24). `platform/auth.py` read
    `ROBUTLER_API_URL`, frozen at import, and ignored `platform.url`, while
    `RobutlerAPI` read `platform.url` and ignored the variable. Setting either
    one alone signed you in to one portal and deployed to another: point
    `platform.url` at a local cluster and `login` still opened production.

    Precedence is the CLI's usual one, environment over config:
    `ROBUTLER_API_URL`, then `platform.url` (project, then the profile's global
    file), then the default. The source is returned so `whoami` and `deploy`
    can say which portal they mean and why.

    `ROBUTLER_INTERNAL_API_URL` is deliberately NOT consulted here any more,
    although `auth.py` used to. It names the in-cluster address that agent
    skills use from inside a pod; a browser on a developer's machine cannot
    open it, so the login flow could never have worked against it.
    """
    from_env = os.environ.get(PLATFORM_URL_ENV_VAR, "").strip()
    if from_env:
        return from_env.rstrip("/"), PLATFORM_URL_ENV_VAR
    store = ConfigStore(profile=profile, cwd=cwd)
    value = store.get("platform.url") or DEFAULTS["platform.url"]
    return str(value).rstrip("/"), store.source_of("platform.url") or "default"


def platform_url(profile: Optional[str] = None, cwd: Optional[Path] = None) -> str:
    """Just the URL from `resolve_platform_url`."""
    return resolve_platform_url(profile, cwd)[0]
