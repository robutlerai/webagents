"""
Which network addresses an agent may connect to on a caller's or a model's behalf.

WHY THIS EXISTS (ADR-0045, 2026-09-25). The REST tool calls URLs a model chose,
and the Web Bot Auth verifier fetches key directories a caller named. Either one,
left alone, is a way to make the agent's host fetch its own loopback services,
the private network it sits in, or the cloud metadata endpoint that hands out the
machine's credentials (S-245 is the `web` skill doing exactly that). So both ask
this module first, about every address a name resolves to, and connect only to
an address it passed (`guarded_http` pins the connection to it).

THE TABLE IS EXPLICIT, not `ipaddress.is_global`, because the TypeScript SDK
must answer the same question the same way and has no such property; both SDKs
run the same cases (`tests/fixtures/net/addresses.json`). It is the IANA
special-purpose registries, plus the embedded-IPv4 forms judged by the address
they carry (`::ffff:a.b.c.d` and 6to4 `2002::/16`).

ALLOW LISTS. An agent file may name private addresses the REST tool may call
anyway (`allow_private`, a list of IPs or CIDRs, each optionally with a port).
Link-local and cloud metadata addresses stay refused whatever the list says:
there is no legitimate reason for an agent to call them, and every reason for
an attacker to try.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

IPAddress = Union[ipaddress.IPv4Address, ipaddress.IPv6Address]

_NOT_PUBLIC_V4 = tuple(
    ipaddress.IPv4Network(cidr)
    for cidr in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.0.0.0/24",
        "192.0.2.0/24",
        "192.88.99.0/24",
        "192.168.0.0/16",
        "198.18.0.0/15",
        "198.51.100.0/24",
        "203.0.113.0/24",
        "224.0.0.0/4",
        "240.0.0.0/4",
    )
)

_NOT_PUBLIC_V6 = tuple(
    ipaddress.IPv6Network(cidr)
    for cidr in (
        "::/96",
        "64:ff9b::/32",
        "100::/64",
        "2001::/23",
        "2001:db8::/32",
        "3fff::/20",
        "5f00::/16",
        "fc00::/7",
        "fe80::/10",
        "fec0::/10",
        "ff00::/8",
    )
)

_ALWAYS_BLOCKED = (
    ipaddress.IPv4Network("169.254.0.0/16"),
    ipaddress.IPv4Network("100.100.100.200/32"),
    ipaddress.IPv6Network("fe80::/10"),
    ipaddress.IPv6Network("fd00:ec2::254/128"),
)

_MAPPED_V4 = ipaddress.IPv6Network("::ffff:0:0/96")
_SIX_TO_FOUR = ipaddress.IPv6Network("2002::/16")


def _effective(addr: IPAddress) -> IPAddress:
    """The address a packet to `addr` is routed as: the IPv4 address inside a
    v4-mapped or 6to4 IPv6 address, else `addr` itself."""
    if isinstance(addr, ipaddress.IPv6Address):
        if addr in _MAPPED_V4:
            return ipaddress.IPv4Address(addr.packed[12:])
        if addr in _SIX_TO_FOUR:
            return ipaddress.IPv4Address(addr.packed[2:6])
    return addr


def is_public_address(addr: IPAddress) -> bool:
    """Whether `addr` is an ordinary public internet address."""
    addr = _effective(addr)
    table = _NOT_PUBLIC_V4 if isinstance(addr, ipaddress.IPv4Address) else _NOT_PUBLIC_V6
    return not any(addr in network for network in table)


def is_always_blocked(addr: IPAddress) -> bool:
    """Link-local and cloud metadata: refused even when an allow list covers them."""
    addr = _effective(addr)
    return any(addr.version == network.version and addr in network for network in _ALWAYS_BLOCKED)


def ip_text(addr: IPAddress) -> str:
    """The address as text: dotted IPv4, and IPv6 in hex groups with the first
    longest run of two or more zero groups written `::`. That is the WHATWG URL
    serialisation the TypeScript SDK gets from `new URL()`; `str()` differs for
    a v4-mapped address (`::ffff:127.0.0.1` from Python 3.13 on, where the URL
    parser writes `::ffff:7f00:1`)."""
    if addr.version == 4:
        return str(addr)
    packed = addr.packed
    groups = [(packed[i] << 8) | packed[i + 1] for i in range(0, 16, 2)]
    best_start, best_length = -1, 1
    i = 0
    while i < 8:
        if groups[i] != 0:
            i += 1
            continue
        j = i
        while j < 8 and groups[j] == 0:
            j += 1
        if j - i > best_length:
            best_start, best_length = i, j - i
        i = j
    hexes = [format(g, "x") for g in groups]
    if best_start < 0:
        return ":".join(hexes)
    return ":".join(hexes[:best_start]) + "::" + ":".join(hexes[best_start + best_length:])


def parse_ip(text: str) -> Optional[IPAddress]:
    """An IP literal, including the shorthand IPv4 forms the socket layer
    accepts (`127.1`, `0x7f.0.0.1`, `2130706433`), which the TypeScript URL
    parser normalises; None for a name."""
    host = text.strip("[]")
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    labels = host.lower().rstrip(".").split(".")
    if labels and re.fullmatch(r"(0x[0-9a-f]*|[0-9]+)", labels[-1] or "x"):
        try:
            return ipaddress.IPv4Address(socket.inet_aton(host))
        except OSError:
            return None
    return None


class AllowListError(ValueError):
    """An `allow_private` entry that is not an IP, a CIDR, or either with a port."""


@dataclass(frozen=True)
class AllowEntry:
    network: Union[ipaddress.IPv4Network, ipaddress.IPv6Network]
    port: Optional[int]

    def covers(self, addr: IPAddress, port: int) -> bool:
        addr = _effective(addr)
        if addr.version != self.network.version or addr not in self.network:
            return False
        return self.port is None or self.port == port


_BRACKETED = re.compile(r"^\[([^\]]+)\]:(\d+)$")
_V4_PORT = re.compile(r"^([0-9.]+):(\d+)$")


def _port(text: str, entry: str) -> int:
    port = int(text)
    if not 1 <= port <= 65535:
        raise AllowListError(f"allow_private entry {entry!r} has a port outside 1-65535")
    return port


def parse_allow_entry(entry: object) -> AllowEntry:
    """`10.0.0.0/8`, `127.0.0.1`, `127.0.0.1:8080`, `::1` or `[::1]:8080`."""
    if not isinstance(entry, str) or not entry.strip():
        raise AllowListError("allow_private entries are IP addresses or CIDR ranges, for example 127.0.0.1:8080 or 10.0.0.0/8")
    text = entry.strip()
    port: Optional[int] = None
    match = _BRACKETED.match(text) or _V4_PORT.match(text)
    if match:
        text, port = match.group(1), _port(match.group(2), entry)
    try:
        network = ipaddress.ip_network(text, strict=True)
    except ValueError:
        raise AllowListError(
            f"allow_private entry {entry!r} is not an IP address or CIDR range "
            "(for example 127.0.0.1:8080, 10.0.0.0/8 or [::1]:8080)"
        ) from None
    return AllowEntry(network=network, port=port)


def parse_allow_list(entries: Optional[Sequence[object]]) -> Tuple[AllowEntry, ...]:
    if entries is None:
        return ()
    if isinstance(entries, (str, bytes)) or not isinstance(entries, (list, tuple)):
        raise AllowListError("allow_private is a list of IP addresses or CIDR ranges")
    return tuple(parse_allow_entry(entry) for entry in entries)


def address_allowed(addr: IPAddress, port: int, allow: Sequence[AllowEntry] = ()) -> bool:
    """Whether a connection to `addr:port` may be made."""
    if is_always_blocked(addr):
        return False
    if is_public_address(addr):
        return True
    return any(entry.covers(addr, port) for entry in allow)


def sort_addresses(addresses: Sequence[IPAddress]) -> List[IPAddress]:
    """IPv4 first, then by value, without duplicates: both SDKs check and pin
    the same address for a name, whatever order the resolver answered in."""
    unique = {(a.version, a.packed): a for a in addresses}
    return [unique[k] for k in sorted(unique)]
