"""
The `access:` block of an agent file, and the decision it makes (ADR-0045).

    access:
      deny:    [agent:https://spam.example/**]
      groups:
        admins:  [user:@alice]
        friends: [agent:https://*.acme.com/**, key:<thumbprint>, domain:partner.example]
        billing:                 # a trust-gated group (plan item 2.5): any verified agent
          trust: {min: 0.6, topic: billing}   # whose TrustFlow on the topic is at least 0.6
        partners:                # members AND trust, both required
          members: [domain:partner.example]
          trust: {min: 0.4}
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
  3. the caller joins EVERY group one of its principals matches, and, for a group
     with a `trust` requirement, only when the VERIFIED calling agent (its
     `agent:` principal, from a Web Bot Auth signature) has the TrustFlow the
     group asks for;
  4. none matched: the `default` group, or refused when `default: none`.

TRUST IS EVIDENCE, NOT A PROMISE. TrustFlow is a platform service: the access
skill asks the platform for the calling agent's score (`trustflow.trust_lookup`,
authenticated as THIS agent) and hands what it learned to `decide` as
`TrustEvidence`: the agent it is about and a score per topic key (`""` for the
overall score). Nothing here dials anything, so the table both SDKs run can
stub the platform. FAIL CLOSED: no evidence (the platform unreachable, no
credential, no agent principal), evidence about another agent, or a topic the
platform could not score all mean the trust-gated group is NOT joined; the
caller still gets whatever else it matched, or the default.

PATTERNS. `user:<id>` and `user:@<handle>` (handles compared without case),
`key:<RFC 7638 thumbprint>`, `domain:<host>` (that host or any subdomain of it,
never a suffix: `domain:partner.example` does not match `evilpartner.example`),
`agent:<URL>` where `*` is one host label or one path segment and `**` any
number of path segments, and `channel:<type>:<sender id>` for a sender the
platform relayed from a connected channel (plan item 2.2: `channel:telegram:8842`,
`channel:email:alice@example.com`), where the sender id is compared without case
and `*` is every sender of that type (`channel:slack:*`). A channel principal
comes only from the platform's caller assertion on a relayed turn
(`portal_connect/skill.py`), never from a request. The TypeScript twin is
`typescript/src/access/policy.ts`; both run `tests/fixtures/access/policy.json`
and refuse a malformed block with the same sentence.
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
    "is not an identity. Use user:<id>, user:@<handle>, agent:<https URL>, key:<thumbprint>, domain:<host> "
    "or channel:<type>:<sender id>."
)
#: A channel type is a slug, the same rule as a group name; the sender id has no whitespace.
_CHANNEL_TYPE = re.compile(r"^[a-z][a-z0-9-]{0,31}$")


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
        if self.kind == "channel":
            if kind != "channel":
                return False
            got_type, sep, sender = value.partition(":")
            want_type, _, want_sender = self.value.partition(":")
            if not sep or not sender or got_type.lower() != want_type:
                return False
            return want_sender == "*" or sender.lower() == want_sender.lower()
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
    if kind == "channel":
        # `<type>:<sender id>`: the type named (never `*`), the sender id one
        # token without whitespace, or `*` for every sender of that type.
        channel_type, sep, sender = value.partition(":")
        channel_type = channel_type.lower()
        if sep and _CHANNEL_TYPE.match(channel_type) and sender and len(sender) <= 200 and not any(c.isspace() for c in sender):
            return Pattern("channel", f"{channel_type}:{sender}")
    raise bad


def _patterns(value: Any, where: str, list_message: str) -> Tuple[Pattern, ...]:
    if not isinstance(value, list):
        raise AccessConfigError(list_message)
    return tuple(parse_pattern(entry, where) for entry in value)


@dataclass(frozen=True)
class TrustRequirement:
    """What a trust-gated group asks of the calling agent's TrustFlow."""

    #: The least TrustFlow score, 0 to 1.
    min: float
    #: The topic the score is on; None is the overall score.
    topic: Optional[str] = None


@dataclass(frozen=True)
class GroupRule:
    """One group: who may join by identity, and what trust they need. Either
    may be absent, never both."""

    #: Identity patterns; None when the group is open to any verified agent with the trust.
    members: Optional[Tuple[Pattern, ...]] = None
    trust: Optional[TrustRequirement] = None


@dataclass
class TrustEvidence:
    """What the platform said about the calling agent (module docstring, "TRUST IS EVIDENCE")."""

    #: The agent URL the scores are about: must be the caller's `agent:` principal.
    agent: str
    #: Score per topic key (`trust_key`); a topic missing here was not scored.
    scores: Dict[str, float] = field(default_factory=dict)


def trust_key(topic: Optional[str]) -> str:
    """The key a requirement's score is filed under in `TrustEvidence.scores`."""
    return topic if topic is not None else ""


_GROUP_RULE_KEYS = ("members", "trust")


def _parse_trust(raw: Any, where: str) -> TrustRequirement:
    message = f"{where}.trust must be a mapping of min (a number from 0 to 1) and an optional topic."
    if not isinstance(raw, dict):
        raise AccessConfigError(message)
    for key in raw:
        if key not in ("min", "topic"):
            raise AccessConfigError(message)
    minimum = raw.get("min")
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)) or not (0 <= minimum <= 1):
        raise AccessConfigError(message)
    topic: Optional[str] = None
    if "topic" in raw:
        if not isinstance(raw["topic"], str) or not raw["topic"].strip():
            raise AccessConfigError(message)
        topic = raw["topic"].strip()
    return TrustRequirement(min=float(minimum), topic=topic)


def _parse_group(raw: Any, where: str) -> GroupRule:
    """A group as written: a list of identities, or `{members, trust}`."""
    if isinstance(raw, list):
        return GroupRule(members=_patterns(raw, where, f"{where} must be a list of identities."))
    if not isinstance(raw, dict):
        raise AccessConfigError(f"{where} must be a list of identities, or a mapping of members and trust.")
    for key in raw:
        if key not in _GROUP_RULE_KEYS:
            raise AccessConfigError(f'{where}: unknown key "{key}". It takes members and trust.')
    members = _patterns(raw["members"], where, f"{where}.members must be a list of identities.") if "members" in raw else None
    trust = _parse_trust(raw["trust"], where) if "trust" in raw else None
    if members is None and trust is None:
        raise AccessConfigError(f"{where} must name members or trust.")
    return GroupRule(members=members, trust=trust)


@dataclass
class AccessPolicy:
    deny: Tuple[Pattern, ...] = ()
    groups: Dict[str, GroupRule] = field(default_factory=dict)
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
            policy.groups[name] = _parse_group(members, f"access.groups.{name}")
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


def trust_requirements(policy: AccessPolicy) -> List[TrustRequirement]:
    """Every distinct trust requirement the block makes, one per topic key: what the access skill must look up."""
    by_key: Dict[str, TrustRequirement] = {}
    for rule in policy.groups.values():
        if rule.trust is not None and trust_key(rule.trust.topic) not in by_key:
            by_key[trust_key(rule.trust.topic)] = rule.trust
    return list(by_key.values())


def verified_agent_of(principals: Sequence[str]) -> Optional[str]:
    """The verified calling agent: the URL of the first `agent:` principal, or None."""
    for principal in principals:
        if principal.startswith("agent:"):
            return principal[len("agent:"):]
    return None


def meets_trust(requirement: TrustRequirement, principals: Sequence[str], evidence: Optional[TrustEvidence]) -> bool:
    """Whether `evidence` shows the verified calling agent meets `requirement` (module docstring: fail closed)."""
    agent = verified_agent_of(principals)
    if agent is None or evidence is None or evidence.agent != agent:
        return False
    score = evidence.scores.get(trust_key(requirement.topic))
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return False
    return score >= requirement.min


def decide(
    policy: AccessPolicy,
    principals: Sequence[str],
    tier: Optional[str] = None,
    trust: Optional[TrustEvidence] = None,
) -> Decision:
    """Deny, then the owner, then every matching group, then the default. A
    group with `members` needs one principal to match; a group with `trust`
    needs `trust` (the evidence) to show it (both, when it has both)."""
    if any(pattern.matches(p) for pattern in policy.deny for p in principals):
        return Decision(False)
    if tier in ("owner", "admin"):
        return Decision(True, ())
    groups = []
    for name, rule in policy.groups.items():
        by_identity = rule.members is None or any(pt.matches(p) for pt in rule.members for p in principals)
        by_trust = rule.trust is None or meets_trust(rule.trust, principals, trust)
        if by_identity and by_trust:
            groups.append(name)
    if groups:
        return Decision(True, tuple(groups))
    if policy.default is None:
        return Decision(False)
    return Decision(True, (policy.default,))
