"""
Trust matching utility for dot-namespace agent names.

Evaluates trust rules (acceptFrom / talkTo) against agent identities.
Supports presets, glob patterns on dot-namespace names, and trust labels.

Trust rules are either:
  - A list of patterns (any match -> allow):  ["family", "#verified", "@alice.*"]
  - An object with allow/deny (deny takes precedence):
    { "allow": ["everyone"], "deny": ["@spammer"] }
"""

from __future__ import annotations

import fnmatch
import re
from typing import List, Optional, Set, Union

# Well-known IANA TLDs (subset for fast checks; the full set can be loaded externally)
_COMMON_TLDS: Set[str] = {
    "com", "org", "net", "edu", "gov", "mil", "int",
    "io", "ai", "co", "me", "tv", "cc", "info", "biz",
    "dev", "app", "xyz", "tech", "online", "site", "store",
    "cloud", "design", "agency", "digital", "media", "studio",
    "blog", "shop", "world", "global", "pro", "live",
}

_extended_tlds: Optional[Set[str]] = None


def load_tlds(tlds: Set[str]) -> None:
    """Load the full IANA TLD set (call once at startup)."""
    global _extended_tlds
    _extended_tlds = {t.lower() for t in tlds}


def _get_tlds() -> Set[str]:
    return _extended_tlds if _extended_tlds is not None else _COMMON_TLDS


def is_reserved_tld(name: str) -> bool:
    return name.lower() in _get_tlds()


def derive_namespace(dot_name: str) -> str:
    """
    Derive namespace from a dot-namespace name.
    - TLD first segment (com.example.bot) -> SLD (com.example)
    - Non-TLD first segment (alice.my-bot) -> root (alice)
    """
    segments = dot_name.split(".")
    if not segments:
        return dot_name
    if is_reserved_tld(segments[0]) and len(segments) >= 2:
        return f"{segments[0]}.{segments[1]}"
    return segments[0]


def is_same_namespace(name_a: str, name_b: str) -> bool:
    return derive_namespace(name_a) == derive_namespace(name_b)


def url_to_reversed_domain_name(agent_url: str) -> str:
    """Convert an external agent URL to a reversed-domain dot-namespace name.

    https://agents.example.com/my-bot  ->  com.example.agents.my-bot
    https://cool-bot.io/               ->  io.cool-bot
    """
    from urllib.parse import urlparse

    parsed = urlparse(agent_url)
    domain_parts = list(reversed(parsed.hostname.split(".")))
    path_parts = [
        re.sub(r"^-+|-+$", "", re.sub(r"[^a-z0-9_-]", "", s.lower()))
        for s in parsed.path.strip("/").split("/")
        if s
    ]
    path_parts = [p for p in path_parts if p]
    return ".".join(domain_parts + path_parts)


TrustRules = Union[List[str], dict]

# Default platform issuer
DEFAULT_TRUSTED_ISSUERS = ["https://robutler.ai"]


def extract_trust_labels(
    scope: str,
    issuer: str = "",
    trusted_issuers: Optional[List[str]] = None,
) -> Set[str]:
    """
    Extract trust:* labels from a JWT scope string.
    Only returns labels if the issuer is trusted (default: robutler.ai).
    """
    if trusted_issuers is None:
        trusted_issuers = DEFAULT_TRUSTED_ISSUERS

    if issuer and issuer not in trusted_issuers:
        return set()

    labels = set()
    for part in scope.split():
        if part.startswith("trust:"):
            labels.add(part)
    return labels


def _match_preset(
    preset: str,
    caller: str,
    target: str,
    caller_labels: Set[str],
) -> bool:
    if preset == "everyone":
        return True
    if preset == "nobody":
        return False
    if preset == "family":
        return is_same_namespace(caller, target)
    if preset == "platform":
        first_seg = caller.split(".")[0] if caller else ""
        return bool(first_seg) and not is_reserved_tld(first_seg)
    return False


def _match_pattern(pattern: str, caller: str) -> bool:
    """Match a glob pattern (prefixed with @) against a dot-namespace name."""
    pat = pattern.lstrip("@")

    if "**" in pat:
        regex_pat = pat.replace(".", r"\.").replace("**", r".+")
        return bool(re.fullmatch(regex_pat, caller))

    if "*" in pat:
        parts = pat.split(".")
        caller_parts = caller.split(".")
        if len(parts) != len(caller_parts):
            return False
        for p, c in zip(parts, caller_parts):
            if not fnmatch.fnmatch(c, p):
                return False
        return True

    return pat == caller


def _match_label(label_rule: str, caller_labels: Set[str]) -> bool:
    """
    Match a trust label rule (prefixed with #) against JWT trust labels.
    Special: #reputation:N does >= comparison on trust:reputation-{M}.
    """
    label = label_rule.lstrip("#")

    rep_match = re.match(r"^reputation:(\d+)$", label)
    if rep_match:
        threshold = int(rep_match.group(1))
        for l in caller_labels:
            m = re.match(r"^trust:reputation-(\d+)$", l)
            if m and int(m.group(1)) >= threshold:
                return True
        return False

    return f"trust:{label}" in caller_labels


def _match_element(
    element: str,
    caller: str,
    target: str,
    caller_labels: Set[str],
) -> bool:
    """Match a single trust rule element against a caller."""
    if element.startswith("#"):
        return _match_label(element, caller_labels)
    if element.startswith("@"):
        return _match_pattern(element, caller)
    return _match_preset(element, caller, target, caller_labels)


def evaluate_trust_rules(
    caller: str,
    target: str,
    rules: TrustRules,
    caller_trust_labels: Optional[Set[str]] = None,
) -> bool:
    """
    Evaluate trust rules for an agent interaction.

    Args:
        caller: Dot-namespace name of the calling agent.
        target: Dot-namespace name of the receiving agent.
        rules: Trust rules (list of patterns, or {allow, deny} object).
        caller_trust_labels: Set of trust:* labels from the caller's JWT.

    Returns:
        True if the interaction is allowed.
    """
    labels = caller_trust_labels or set()

    if isinstance(rules, list):
        return any(
            _match_element(elem, caller, target, labels)
            for elem in rules
        )

    if isinstance(rules, dict):
        deny_list = rules.get("deny", [])
        allow_list = rules.get("allow", [])

        for elem in deny_list:
            if _match_element(elem, caller, target, labels):
                return False

        return any(
            _match_element(elem, caller, target, labels)
            for elem in allow_list
        )

    return False
