"""RFC 9110 single-range parsing and the RFC 9530 ``Repr-Digest`` field,
shared by the producer endpoint and the client."""

from __future__ import annotations

import base64
import re
from typing import Optional, Tuple, Union

_RANGE = re.compile(r"^\s*bytes\s*=\s*([^,]+?)\s*$", re.IGNORECASE)
_SPEC = re.compile(r"^(\d*)\s*-\s*(\d*)$")
_CONTENT_RANGE = re.compile(r"^\s*bytes\s+(\d+)-(\d+)/(\d+|\*)\s*$", re.IGNORECASE)
_REPR_DIGEST = re.compile(r"sha-256=:([A-Za-z0-9+/=]+):")

ParsedRange = Union[Tuple[str], Tuple[str, int, int]]  # ("none",) | ("unsatisfiable",) | ("single", start, end)


def parse_range_header(header: Optional[str], size: int) -> ParsedRange:
    """One ``bytes=`` range against a representation of ``size`` bytes.
    Several ranges or another unit read as the whole representation."""
    if not header:
        return ("none",)
    m = _RANGE.match(header)
    if not m:
        return ("none",)
    spec = _SPEC.match(m.group(1))
    if not spec or (spec.group(1) == "" and spec.group(2) == ""):
        return ("none",)
    if size <= 0:
        return ("unsatisfiable",)
    if spec.group(1) == "":
        n = int(spec.group(2))
        if n <= 0:
            return ("unsatisfiable",)
        return ("single", max(0, size - n), size - 1)
    start = int(spec.group(1))
    if start >= size:
        return ("unsatisfiable",)
    end = size - 1 if spec.group(2) == "" else int(spec.group(2))
    if end < start:
        return ("none",)
    return ("single", start, min(end, size - 1))


def repr_digest(sha256_hex: str) -> str:
    return f"sha-256=:{base64.b64encode(bytes.fromhex(sha256_hex)).decode('ascii')}:"


def parse_repr_digest(header: Optional[str]) -> Optional[str]:
    if not header:
        return None
    m = _REPR_DIGEST.search(header)
    if not m:
        return None
    try:
        raw = base64.b64decode(m.group(1), validate=True)
    except ValueError:
        return None
    return raw.hex() if len(raw) == 32 else None


def parse_content_range(header: Optional[str]) -> Optional[Tuple[int, int, int]]:
    """``(start, end, size)``; size is -1 for ``*``."""
    if not header:
        return None
    m = _CONTENT_RANGE.match(header)
    if not m:
        return None
    start, end = int(m.group(1)), int(m.group(2))
    if end < start:
        return None
    size = -1 if m.group(3) == "*" else int(m.group(3))
    return start, end, size


__all__ = ["parse_range_header", "repr_digest", "parse_repr_digest", "parse_content_range", "ParsedRange"]
