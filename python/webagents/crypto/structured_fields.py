"""
RFC 9651 Structured Field Values, the PARSING subset the Web Bot Auth verifier
reads (ADR-0045, 2026-09-25): `Signature-Input`, `Signature`, `Signature-Agent`
and `Content-Digest`, into the value types the signer already serialises
(`Item`, `InnerList`, `Token` in `http_signature.py`), so a parsed
`Signature-Input` member re-serialises with the SAME code that wrote it.

The TypeScript twin is `typescript/src/crypto/structured-fields.ts`; both are
held to the same cases (`tests/fixtures/web_bot_auth/structured-fields.json`).

STRICTER THAN THE RFC WHERE LAXITY COULD ONLY HELP A FORGER, and every such
point fails closed:
  - a field longer than 8 KiB is refused unparsed, a Dictionary with more than
    16 members after;
  - a repeated Dictionary key or Parameter key is refused (the RFC keeps the
    last, which lets two readers see two different fields);
  - Decimals, Dates and Display Strings are refused: nothing in the profile
    uses them, and a base must re-serialise exactly what was signed;
  - a Byte Sequence must be canonical padded base64.
"""

from __future__ import annotations

import base64
import binascii
import re
from typing import List, Optional, Tuple, Union

from .http_signature import BareItem, InnerList, Item, Token

MAX_FIELD_CHARS = 8 * 1024
MAX_DICTIONARY_MEMBERS = 16

Member = Union[Item, InnerList]

_LCALPHA = set("abcdefghijklmnopqrstuvwxyz")
_DIGIT = set("0123456789")
_ALPHA = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
_KEY_CHARS = _LCALPHA | _DIGIT | set("_-.*")
_TOKEN_CHARS = _ALPHA | _DIGIT | set(":/!#$%&'*+-.^_`|~")
_BASE64 = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")


class StructuredFieldParseError(ValueError):
    """The one error; `reason` is a short phrase for a log and never echoes the field."""

    def __init__(self, reason: str):
        super().__init__(f"structured field: {reason}")
        self.reason = reason


class _Parser:
    def __init__(self, text: str):
        if not isinstance(text, str):
            raise StructuredFieldParseError("not a string")
        if len(text) > MAX_FIELD_CHARS:
            raise StructuredFieldParseError("field too long")
        self.s = text
        self.i = 0

    def peek(self) -> str:
        return self.s[self.i] if self.i < len(self.s) else ""

    def fail(self, reason: str):
        raise StructuredFieldParseError(reason)

    def skip_sp(self) -> None:
        while self.peek() == " ":
            self.i += 1

    def skip_ows(self) -> None:
        while self.peek() in (" ", "\t") and self.peek() != "":
            self.i += 1

    def finish(self, value):
        self.skip_sp()
        if self.i != len(self.s):
            self.fail("trailing characters")
        return value

    def dictionary(self) -> List[Tuple[str, Member]]:
        out: List[Tuple[str, Member]] = []
        seen = set()
        if self.i >= len(self.s):
            return out
        while True:
            key = self.key()
            if key in seen:
                self.fail("repeated dictionary key")
            seen.add(key)
            if self.peek() == "=":
                self.i += 1
                member = self.item_or_inner_list()
            else:
                member = Item(True, self.parameters())
            out.append((key, member))
            if len(out) > MAX_DICTIONARY_MEMBERS:
                self.fail("too many dictionary members")
            self.skip_ows()
            if self.i >= len(self.s):
                return out
            if self.peek() != ",":
                self.fail("expected a comma")
            self.i += 1
            self.skip_ows()
            if self.i >= len(self.s):
                self.fail("trailing comma")

    def item_or_inner_list(self) -> Member:
        return self.inner_list() if self.peek() == "(" else self.item()

    def inner_list(self) -> InnerList:
        self.i += 1
        items: List[Item] = []
        while True:
            self.skip_sp()
            if self.peek() == ")":
                self.i += 1
                return InnerList(tuple(items), self.parameters())
            if self.i >= len(self.s):
                self.fail("unterminated inner list")
            items.append(self.item())
            if self.peek() not in (" ", ")") or self.peek() == "":
                self.fail("bad inner list separator")

    def item(self) -> Item:
        value = self.bare_item()
        return Item(value, self.parameters())

    def parameters(self) -> Tuple[Tuple[str, BareItem], ...]:
        out: List[Tuple[str, BareItem]] = []
        seen = set()
        while self.peek() == ";":
            self.i += 1
            self.skip_sp()
            key = self.key()
            if key in seen:
                self.fail("repeated parameter key")
            seen.add(key)
            value: BareItem = True
            if self.peek() == "=":
                self.i += 1
                value = self.bare_item()
            out.append((key, value))
        return tuple(out)

    def key(self) -> str:
        first = self.peek()
        if not (first in _LCALPHA or first == "*") or first == "":
            self.fail("bad key")
        start = self.i
        while self.i < len(self.s) and self.s[self.i] in _KEY_CHARS:
            self.i += 1
        return self.s[start : self.i]

    def bare_item(self) -> BareItem:
        c = self.peek()
        if c == "":
            self.fail("bad item")
        if c == "-" or c in _DIGIT:
            return self.integer()
        if c == '"':
            return self.string()
        if c == ":":
            return self.byte_sequence()
        if c == "?":
            return self.boolean()
        if c in _ALPHA or c == "*":
            return self.token()
        if c == "@":
            self.fail("dates are not accepted")
        if c == "%":
            self.fail("display strings are not accepted")
        self.fail("bad item")

    def integer(self) -> int:
        sign = 1
        if self.peek() == "-":
            sign = -1
            self.i += 1
        if self.peek() == "" or self.peek() not in _DIGIT:
            self.fail("bad number")
        start = self.i
        while self.peek() != "" and self.peek() in _DIGIT:
            self.i += 1
            if self.i - start > 15:
                self.fail("integer too long")
        if self.peek() == ".":
            self.fail("decimals are not accepted")
        return sign * int(self.s[start : self.i])

    def string(self) -> str:
        self.i += 1
        out = []
        while True:
            if self.i >= len(self.s):
                self.fail("unterminated string")
            c = self.s[self.i]
            self.i += 1
            if c == "\\":
                nxt = self.s[self.i] if self.i < len(self.s) else ""
                if nxt not in ('"', "\\") or nxt == "":
                    self.fail("bad escape")
                out.append(nxt)
                self.i += 1
                continue
            if c == '"':
                return "".join(out)
            if not 0x20 <= ord(c) <= 0x7E:
                self.fail("string is not printable ASCII")
            out.append(c)

    def token(self) -> Token:
        start = self.i
        while self.i < len(self.s) and self.s[self.i] in _TOKEN_CHARS:
            self.i += 1
        return Token(self.s[start : self.i])

    def byte_sequence(self) -> bytes:
        self.i += 1
        end = self.s.find(":", self.i)
        if end < 0:
            self.fail("unterminated byte sequence")
        text = self.s[self.i : end]
        self.i = end + 1
        if not _BASE64.match(text) or len(text) % 4 != 0:
            self.fail("byte sequence is not padded base64")
        try:
            decoded = base64.b64decode(text, validate=True)
        except (binascii.Error, ValueError):
            self.fail("byte sequence is not base64")
        if base64.b64encode(decoded).decode("ascii") != text:
            self.fail("byte sequence is not canonical base64")
        return decoded

    def boolean(self) -> bool:
        self.i += 1
        c = self.peek()
        if c not in ("0", "1") or c == "":
            self.fail("bad boolean")
        self.i += 1
        return c == "1"


def parse_dictionary(text: str) -> List[Tuple[str, Member]]:
    """RFC 9651 section 4.2 Dictionary, members in wire order."""
    p = _Parser(text)
    p.skip_sp()
    return p.finish(p.dictionary())


def parse_item(text: str) -> Item:
    """RFC 9651 section 4.2 Item."""
    p = _Parser(text)
    p.skip_sp()
    return p.finish(p.item())


def param_of(params, key: str) -> Optional[BareItem]:
    """A parameter's value by key, or None."""
    for name, value in params or ():
        if name == key:
            return value
    return None
