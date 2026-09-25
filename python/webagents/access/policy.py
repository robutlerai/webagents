"""
The `access:` block of an agent file, and the decision it makes (ADR-0045).

    access:
      deny:    [agent:https://spam.example/**]
      groups:
        admins:  [user:@alice]
        friends: [agent:https://*.acme.com/**, key:<thumbprint>, domain:partner.example]
      default: public           # the group of a verified caller in no group; `none` refuses them
      instructions:
        friends: FRIENDS.md     # added to the instructions for callers in `friends`
      tools:
        friends: [rest]         # skills or tools only these groups (and the owner) may use

THE DECISION, in this order, over principals that came from verified
credentials only (the access skill collects them; nothing here reads a request):
  1. a principal matching `deny` refuses;
  2. the owner, or an admin, is let in and placed in no group (they pass every
     group scope anyway);
  3. the caller joins EVERY group one of its principals matches;
  4. none matched: the `default` group, or refused when `default: none`.

PATTERNS. `user:<id>` and `user:@<handle>` (handles compared without case),
`key:<RFC 7638 thumbprint>`, `domain:<host>` (that host or any subdomain of it,
never a suffix: `domain:partner.example` does not match `evilpartner.example`),
and `agent:<URL>` where `*` is one host label or one path segment and `**` any
number of path segments. The TypeScript twin is `typescript/src/access/policy.ts`;
both run `tests/fixtures/access/policy.json` and refuse a malformed block with the
same sentence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

KNOWN_KEYS = ("deny", "groups", "default", "instructions", "tools")
RESERVED_GROUPS = frozenset({"owner", "admin", "user", "all", "none"})
DEFAULT_GROUP = "everyone"
_GROUP_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_THUMBPRINT = re.compile(r"^[A-Za-z0-9_-]{43}$")
_HOST = re.compile(r"^(?=.{1,253}$)[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?)*$")
_AGENT = re.compile(r"^(https?)://([^/?#\s]+)(/[^?#\s]*)?$", re.I)
_HOSTPORT_LABEL = re.compile(r"^(\*|[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?)$")
_NOT_AN_IDENTITY = (
    "is not an identity. Use user:<id>, user:@<handle>, agent:<https URL>, key:<thumbprint> or domain:<host>."
)


class AccessConfigError(ValueError):
    """A malformed `access:` block; the message names where and what to write."""


@dataclass(frozen=True)
class Pattern:
    kind: str
    value: str
    regex: Optional["re.Pattern[str]"] = None

    def matches(self, principal: str) -> bool:
        kind, _, value = principal.partition(":")
        if self.kind == "user":
            if kind != "user":
                return False
            if self.value.startswith("@"):
                return value.startswith("@") and value.lower() == self.value.lower()
            return value == self.value
        if self.kind == "key":
            return kind == "key" and value == self.value
        if kind != "agent":
            return False
        if self.kind == "domain":
            host = _principal_host(value)
            return host is not None and (host == self.value or host.endswith("." + self.value))
        return bool(self.regex and self.regex.match(_lower_origin(value)))


def _principal_host(url: str) -> Optional[str]:
    match = _AGENT.match(url)
    if not match:
        return None
    return match.group(2).lower().rsplit(":", 1)[0] if not match.group(2).startswith("[") else None


def _lower_origin(url: str) -> str:
    match = _AGENT.match(url)
    if not match:
        return url
    return f"{match.group(1).lower()}://{match.group(2).lower()}{(match.group(3) or '').rstrip('/')}"


def _agent_regex(url: str) -> Optional["re.Pattern[str]"]:
    match = _AGENT.match(url)
    if not match:
        return None
    scheme, hostport, path = match.group(1).lower(), match.group(2).lower(), (match.group(3) or "").rstrip("/")
    host, _, port = hostport.partition(":")
    labels = host.split(".")
    if not labels or any(not _HOSTPORT_LABEL.match(label) for label in labels):
        return None
    if port and not port.isdigit():
        return None
    host_re = r"\.".join("[a-z0-9-]+" if label == "*" else re.escape(label) for label in labels)
    port_re = f":{port}" if port else ""
    path_re = ""
    for segment in [s for s in path.split("/")[1:]]:
        if segment == "**":
            path_re += "(?:/[^/]+)*"
        elif segment == "*":
            path_re += "/[^/]+"
        elif not segment or "*" in segment:
            return None
        else:
            path_re += "/" + re.escape(segment)
    return re.compile(f"^{scheme}://{host_re}{port_re}{path_re}$")


def parse_pattern(entry: Any, where: str) -> Pattern:
    bad = AccessConfigError(f'{where}: "{entry}" {_NOT_AN_IDENTITY}')
    if not isinstance(entry, str):
        raise bad
    kind, sep, value = entry.partition(":")
    if not sep or not value:
        raise bad
    if kind == "user":
        name = value[1:] if value.startswith("@") else value
        if name and not any(c.isspace() for c in name):
            return Pattern("user", value)
        raise bad
    if kind == "key":
        if _THUMBPRINT.match(value):
            return Pattern("key", value)
        raise bad
    if kind == "domain":
        lowered = value.lower()
        if _HOST.match(lowered):
            return Pattern("domain", lowered)
        raise bad
    if kind == "agent":
        regex = _agent_regex(value)
        if regex is not None:
            return Pattern("agent", value, regex)
    raise bad


def _patterns(value: Any, where: str, list_message: str) -> Tuple[Pattern, ...]:
    if not isinstance(value, list):
        raise AccessConfigError(list_message)
    return tuple(parse_pattern(entry, where) for entry in value)


@dataclass
class AccessPolicy:
    deny: Tuple[Pattern, ...] = ()
    groups: Dict[str, Tuple[Pattern, ...]] = field(default_factory=dict)
    #: The group of a caller in no group; None is `default: none`, refuse them.
    default: Optional[str] = DEFAULT_GROUP
    #: Group name to the Markdown file named for it, relative to the agent file.
    instructions: Dict[str, str] = field(default_factory=dict)
    #: Group name to the skill or tool names only its members (and the owner) may use.
    tools: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    def known_groups(self) -> List[str]:
        names = list(self.groups)
        if self.default and self.default not in names:
            names.append(self.default)
        return names


def parse_access(raw: Any) -> AccessPolicy:
    """The block as written in the agent file, or an `AccessConfigError`."""
    if not isinstance(raw, dict):
        raise AccessConfigError("access must be a mapping of deny, groups, default, instructions and tools.")
    for key in raw:
        if key not in KNOWN_KEYS:
            raise AccessConfigError(f'access: unknown key "{key}". It takes deny, groups, default, instructions and tools.')
    policy = AccessPolicy()
    if "deny" in raw:
        policy.deny = _patterns(raw["deny"], "access.deny", "access.deny must be a list of identities, for example agent:https://spam.example/**.")
    if "groups" in raw:
        groups = raw["groups"]
        if not isinstance(groups, dict):
            raise AccessConfigError("access.groups must map a group name to a list of identities.")
        for name, members in groups.items():
            if not isinstance(name, str) or not _GROUP_NAME.match(name):
                raise AccessConfigError(
                    f'access.groups: "{name}" is not a group name (lower-case letters, digits, - and _, starting with a letter).'
                )
            if name in RESERVED_GROUPS:
                raise AccessConfigError(f'access.groups: "{name}" is reserved.')
            policy.groups[name] = _patterns(members, f"access.groups.{name}", f"access.groups.{name} must be a list of identities.")
    if "default" in raw:
        default = raw["default"]
        if default == "none":
            policy.default = None
        elif isinstance(default, str) and _GROUP_NAME.match(default) and default not in RESERVED_GROUPS:
            policy.default = default
        else:
            raise AccessConfigError("access.default must be a group name or none.")
    known = set(policy.known_groups())
    if "instructions" in raw:
        instructions = raw["instructions"]
        message = "access.instructions must map a group name to a Markdown file next to the agent file."
        if not isinstance(instructions, dict):
            raise AccessConfigError(message)
        for name, path in instructions.items():
            if name not in known:
                raise AccessConfigError(f'access.instructions: "{name}" is not a group this block defines.')
            if not isinstance(path, str) or not path.strip():
                raise AccessConfigError(message)
            policy.instructions[name] = path.strip()
    if "tools" in raw:
        tools = raw["tools"]
        message = "access.tools must map a group name to a list of skill or tool names."
        if not isinstance(tools, dict):
            raise AccessConfigError(message)
        for name, names in tools.items():
            if name not in known:
                raise AccessConfigError(f'access.tools: "{name}" is not a group this block defines.')
            if not isinstance(names, list) or not all(isinstance(n, str) and n for n in names):
                raise AccessConfigError(message)
            policy.tools[name] = tuple(names)
    return policy


@dataclass(frozen=True)
class Decision:
    allow: bool
    groups: Tuple[str, ...] = ()


def decide(policy: AccessPolicy, principals: Sequence[str], tier: Optional[str] = None) -> Decision:
    """Deny, then the owner, then every matching group, then the default."""
    if any(pattern.matches(p) for pattern in policy.deny for p in principals):
        return Decision(False)
    if tier in ("owner", "admin"):
        return Decision(True, ())
    groups = tuple(name for name, patterns in policy.groups.items() if any(pt.matches(p) for pt in patterns for p in principals))
    if groups:
        return Decision(True, groups)
    if policy.default is None:
        return Decision(False)
    return Decision(True, (policy.default,))
