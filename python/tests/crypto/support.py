"""
A test-only verifier for the headers `sign_request` produces (W2 design
section 10.2, 2026-09-17).

The SDK deliberately ships no structured-field parser (verifying inbound
signed requests is not W2 work, design section 13), so the tests carry the
little reading they need here. It is not a general parser: it splits the
SDK's own Dictionaries on `, ` and reads the fixed parameter shape of
design section 2.3, which is all a test of this signer has to read.

`verify_signed_request` rebuilds the signature base FROM THE HEADERS (the
`signature-agent` value from `Signature-Agent`, the digest from
`Content-Digest`, the derived components from the method and URL), checks
that the `Signature-Input` member re-serialises to exactly what was sent,
and verifies each label's signature with the key its `keyid` names. That is
the platform's checklist in miniature, and it is what makes a test of
`WebBotAuth` over a real request more than a tautology.
"""

from __future__ import annotations

import base64
import re
from typing import Dict, List, Mapping, Tuple

from cryptography.hazmat.primitives.asymmetric import ed25519

from webagents.crypto.http_signature import (
    InnerList,
    Item,
    SignatureParams,
    build_signature_base,
    content_digest,
    request_target,
    serialize_inner_list,
)

_MEMBER_SPLIT_RE = re.compile(r", (?=[a-z*][a-z0-9_\-.*]*=)")
_COMPONENT_RE = re.compile(r'"([^"]+)"(?:;key="([^"]+)")?')
_PARAM_RE = re.compile(r';([a-z]+)=("(?:[^"\\]|\\.)*"|\d+)')


def split_dictionary(value: str) -> Dict[str, str]:
    """`{key: member}` for a serialised Dictionary whose members carry no `, `."""
    members: Dict[str, str] = {}
    for part in _MEMBER_SPLIT_RE.split(value):
        key, sep, member = part.partition("=")
        assert sep, f"not a key=value member: {part!r}"
        assert key not in members, f"duplicate member {key!r}"
        members[key] = member
    return members


def parse_signature_input_member(member: str) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
    """`([(component, key-or-''), ...], {param: value})` for one member."""
    assert member.startswith("("), member
    close = member.index(")")
    components = _COMPONENT_RE.findall(member[1:close])
    params: Dict[str, str] = {}
    for name, raw in _PARAM_RE.findall(member[close + 1 :]):
        params[name] = raw[1:-1] if raw.startswith('"') else raw
    return components, params


def verify_signed_request(
    headers: Mapping[str, str],
    method: str,
    url: str,
    body: bytes,
    public_keys: Mapping[str, ed25519.Ed25519PublicKey],
) -> Dict[str, Dict[str, object]]:
    """Verify every label; return `{label: {keyid, created, expires, nonce, components, base}}`.

    Raises `AssertionError` for an incoherent header set and
    `cryptography.exceptions.InvalidSignature` for a bad signature.
    """
    lower = {name.lower(): value for name, value in headers.items()}
    target = request_target(method, url)
    inputs = split_dictionary(lower["signature-input"])
    signatures = split_dictionary(lower["signature"])
    assert list(inputs) == list(signatures), (list(inputs), list(signatures))

    if body:
        assert lower["content-digest"] == content_digest(body)
    else:
        assert "content-digest" not in lower

    agent_field = lower["signature-agent"]
    verified: Dict[str, Dict[str, object]] = {}
    for label, member in inputs.items():
        components, params = parse_signature_input_member(member)
        items: List[Item] = []
        values: List[Tuple[Item, str]] = []
        for name, key in components:
            item = Item(name, (("key", key),)) if key else Item(name)
            items.append(item)
            if name == "@method":
                value = target.method
            elif name == "@authority":
                value = target.authority
            elif name == "@path":
                value = target.path
            elif name == "@query":
                value = target.query
            elif name == "content-digest":
                value = lower["content-digest"]
            elif name == "signature-agent":
                if key:
                    value = split_dictionary(agent_field)[key]
                else:
                    assert agent_field.startswith('"'), agent_field
                    value = agent_field
            else:
                raise AssertionError(f"unexpected covered component {name!r}")
            values.append((item, value))

        signature_params = SignatureParams(
            created=int(params["created"]),
            expires=int(params["expires"]),
            keyid=params["keyid"],
            nonce=params["nonce"],
            alg=params["alg"],
            tag=params["tag"],
        )
        inner = InnerList(tuple(items), signature_params.as_parameters())
        # Strict re-serialisation must give back the member as sent (RFC 9421
        # section 3.2 step 7): the verifier signs what it re-serialises.
        assert serialize_inner_list(inner) == member
        base = build_signature_base(values, inner)

        encoded = signatures[label]
        assert encoded.startswith(":") and encoded.endswith(":"), encoded
        raw = base64.b64decode(encoded[1:-1], validate=True)
        assert len(raw) == 64, len(raw)
        public_keys[params["keyid"]].verify(raw, base.encode("ascii"))

        verified[label] = {
            "keyid": params["keyid"],
            "created": int(params["created"]),
            "expires": int(params["expires"]),
            "nonce": params["nonce"],
            "alg": params["alg"],
            "tag": params["tag"],
            "components": components,
            "base": base,
        }
    return verified
