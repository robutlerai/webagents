"""
RFC 8785 JSON Canonicalization Scheme (JCS), the form an A2A agent card is
signed over (A2A v1.0 section 8.4) and the form every verifier rebuilds.

Standard library only (plan item 1.3, 2026-09-26), and the same bytes as the
TypeScript twin (`a2a/jcs.ts`) for the same value, which is what makes a card
signed by one SDK verify in the other. The rules JCS adds to `json.dumps`:

  * object keys sorted by UTF-16 code units, not code points: a supplementary
    character (an emoji, two surrogates) sorts BEFORE U+FF5E under JCS and
    after it under Python's default string order, so the sort key is the
    UTF-16BE encoding;
  * numbers as ES6 `Number.prototype.toString` prints them: `1.0` is `1`,
    `1e-6` is `0.000001`, `1e21` is `1e+21`, `-0` is `0`, and an integer
    beyond 2**53 is the double it rounds to;
  * strings with the short escapes, other control characters as lowercase
    `\\u00xx`, everything else literal UTF-8, which `json.dumps` already does
    with `ensure_ascii=False`;
  * no whitespace; NaN and infinities refused rather than written as the
    non-JSON tokens `json.dumps` would emit.
"""

from __future__ import annotations

import json
import math
from typing import Any

__all__ = ["canonicalize", "canonical_bytes"]

_TWO_53 = 2**53


def canonicalize(value: Any) -> str:
    """The canonical (RFC 8785) serialisation of `value`."""
    return _serialize(value, "")


def canonical_bytes(value: Any) -> bytes:
    """The canonical bytes, what a signature covers."""
    return canonicalize(value).encode("utf-8")


def _serialize(value: Any, path: str) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int):
        if abs(value) >= _TWO_53:
            return _es6_number(float(value), path)
        return str(value)
    if isinstance(value, float):
        return _es6_number(value, path)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_serialize(item, f"{path}[{index}]") for index, item in enumerate(value)) + "]"
    if isinstance(value, dict):
        members = []
        for key in sorted(value.keys(), key=lambda k: str(k).encode("utf-16-be")):
            if not isinstance(key, str):
                raise TypeError(f"JCS: {path or 'the value'} has a non-string key {key!r}")
            members.append(json.dumps(key, ensure_ascii=False) + ":" + _serialize(value[key], f"{path}.{key}" if path else key))
        return "{" + ",".join(members) + "}"
    raise TypeError(f"JCS: {path or 'the value'} is a {type(value).__name__}, which JSON cannot carry")


def _es6_number(value: float, path: str) -> str:
    """`Number.prototype.toString` (ECMA-262 section 6.1.6.1.20) for a double."""
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"JCS: {path or 'the value'} is {value!r}, which JSON cannot carry")
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    # The shortest round-trip digits, as ES6 requires: `repr` gives them as
    # `d.ddd` or `d.ddde[+-]xx`; take the digit string and the exponent.
    text = repr(abs(value))
    if "e" in text or "E" in text:
        mantissa, exponent_text = text.lower().split("e")
        exponent = int(exponent_text)
    else:
        mantissa, exponent = text, 0
    if "." in mantissa:
        int_part, frac_part = mantissa.split(".")
    else:
        int_part, frac_part = mantissa, ""
    digits = (int_part + frac_part).lstrip("0")
    # n such that value = 0.d1d2...dk * 10^n, with k digits
    n = len(int_part.lstrip("0")) + exponent if int_part.strip("0") else exponent - (len(frac_part) - len(frac_part.lstrip("0")))
    digits = digits.rstrip("0")
    k = len(digits)
    if k == 0:
        return "0"
    if k <= n <= 21:
        return sign + digits + "0" * (n - k)
    if 0 < n <= 21:
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * (-n) + digits
    e = n - 1
    exp = f"e{'+' if e >= 0 else '-'}{abs(e)}"
    if k == 1:
        return sign + digits + exp
    return sign + digits[0] + "." + digits[1:] + exp
