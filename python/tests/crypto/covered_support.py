"""
A test-only verifier for signatures that cover plain header fields
(machine-purchase design section 6.4, pass P9b, 2026-09-18).

`support.verify_signed_request` reads the fixed W2 component set and refuses
anything else, which is right for the suites that pin that set. A paid retry
also covers `payment-authorization` and `robutler-terms-accepted` (and the
buyer's hints), so this one reads any other bare component as a header field
off the message, trimmed, exactly as the platform's verifier does
(`Headers.get(name).trim()`, RFC 9421 section 2.1). It rebuilds the base from
the HEADERS as sent, checks the `Signature-Input` member re-serialises to
what was sent, and verifies each label with the key its `keyid` names.
"""

from __future__ import annotations

import base64
from typing import Dict, List, Mapping, Tuple, Union

import httpx
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

from .support import parse_signature_input_member, split_dictionary


def covered_of(signature_input: str, label: str = "sig1") -> List[str]:
    """The component identifiers of one `Signature-Input` member, in wire order, as serialised."""
    member = split_dictionary(signature_input)[label]
    return member[1 : member.index(")")].split(" ")


def verify_with_covered_headers(
    headers: Union[Mapping[str, str], httpx.Headers],
    method: str,
    url: Union[str, httpx.URL],
    body: bytes,
    public_keys: Mapping[str, ed25519.Ed25519PublicKey],
) -> Dict[str, Dict[str, object]]:
    """Verify every label; return `{label: {components, base}}`. Raises
    `AssertionError` for an incoherent header set and `InvalidSignature` for
    a bad signature."""
    fields = httpx.Headers(headers)
    target = request_target(method, url)
    inputs = split_dictionary(fields["signature-input"])
    signatures = split_dictionary(fields["signature"])
    assert list(inputs) == list(signatures), (list(inputs), list(signatures))
    if body:
        assert fields["content-digest"] == content_digest(body)
    else:
        assert "content-digest" not in fields

    agent_field = fields["signature-agent"]
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
            elif name == "signature-agent":
                value = split_dictionary(agent_field)[key] if key else agent_field
            else:
                assert not name.startswith("@"), f"unexpected derived component {name!r}"
                raw = fields.get(name)
                assert raw is not None, f"covered header {name!r} is not on the message"
                value = raw.strip()
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
        assert serialize_inner_list(inner) == member
        base = build_signature_base(values, inner)
        encoded = signatures[label]
        raw_sig = base64.b64decode(encoded[1:-1], validate=True)
        public_keys[params["keyid"]].verify(raw_sig, base.encode("ascii"))
        verified[label] = {"components": components, "base": base}
    return verified
