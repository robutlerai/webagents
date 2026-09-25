"""
One HTTP exchange that connects only to an address `addresses` allowed.

CHECK, THEN CONNECT TO WHAT WAS CHECKED (ADR-0045, 2026-09-25). Resolving a
name, checking the answer and then handing the NAME to an HTTP client is not a
guard: the client resolves again, and a name can answer differently the second
time (DNS rebinding). So the name is resolved here, every address it answers
with is checked (one private answer refuses the request, rather than hoping the
client picks a public one), and the connection is made to the first checked
address through a network backend that ignores the hostname httpcore asks for.
TLS still verifies the certificate against the HOSTNAME: httpcore passes the
origin's host as `server_hostname` whatever address the socket went to.

httpcore rather than httpx, on purpose: no redirects followed behind the
caller's back (each hop must be checked and, for the REST tool, signed again),
no proxy taken from the environment (`HTTPS_PROXY` would move the connection
somewhere this module never checked), no cookie jar, and no content decoding
(the caller asks for `Accept-Encoding: identity`).

The TypeScript twin is `typescript/src/net/guarded-request.ts`; the two answer
with the same codes and messages.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
import ssl
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import httpcore

from .addresses import AllowEntry, IPAddress, address_allowed, ip_text, parse_ip, sort_addresses


class GuardError(Exception):
    """A refusal or failure with a stable `code` and a sentence for a person."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def refused_address_message(host: str, address: IPAddress, literal: bool) -> str:
    if literal:
        return f"{ip_text(address)} is not a public address, so it is not called."
    return f"{host} resolves to {ip_text(address)}, which is not a public address, so it is not called."


async def resolve_allowed(host: str, port: int, allow: Sequence[AllowEntry] = ()) -> IPAddress:
    """The address to connect to for `host:port`, or a `GuardError`.

    Every address the name resolves to must be allowed; the first in
    `sort_addresses` order is the one to connect to."""
    literal = parse_ip(host)
    if literal is not None:
        if not address_allowed(literal, port, allow):
            raise GuardError("blocked_address", refused_address_message(host, literal, True))
        return literal
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError, OSError):
        raise GuardError("network_error", f"{host} did not resolve.") from None
    found: List[IPAddress] = []
    for info in infos:
        try:
            found.append(ipaddress.ip_address(str(info[4][0]).split("%", 1)[0]))
        except ValueError:
            continue
    addresses = sort_addresses(found)
    if not addresses:
        raise GuardError("network_error", f"{host} did not resolve.")
    for address in addresses:
        if not address_allowed(address, port, allow):
            raise GuardError("blocked_address", refused_address_message(host, address, False))
    return addresses[0]


class _PinnedBackend(httpcore.AsyncNetworkBackend):
    """Connects to one checked address whatever host httpcore asks for."""

    def __init__(self, address: IPAddress):
        self._inner = httpcore.AnyIOBackend()
        self._address = str(address)

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        return await self._inner.connect_tcp(
            self._address, port, timeout=timeout, local_address=local_address, socket_options=socket_options
        )

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):  # pragma: no cover
        raise httpcore.ConnectError("unix sockets are not used here")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


@dataclass
class Exchange:
    """What came back. `body` holds at most `max_bytes`; `truncated` says more existed."""

    status: int
    headers: List[Tuple[str, str]]
    body: bytes
    truncated: bool
    address: str = ""
    extra: dict = field(default_factory=dict)

    def header(self, name: str) -> Optional[str]:
        lowered = name.lower()
        for key, value in self.headers:
            if key == lowered:
                return value
        return None


def _network_message(host: str, error: BaseException) -> str:
    text = str(error).lower()
    if isinstance(error, ssl.SSLError) or "certificate" in text or "ssl" in text or "tls" in text:
        return f"Could not connect to {host}: its TLS certificate was not accepted."
    if "refused" in text:
        return f"Could not connect to {host}: the connection was refused."
    if "reset" in text:
        return f"Could not connect to {host}: the connection was reset."
    return f"Could not connect to {host}: the connection failed."


async def exchange(
    *,
    method: str,
    scheme: str,
    host: str,
    port: int,
    target: str,
    headers: Sequence[Tuple[str, str]],
    body: bytes,
    address: IPAddress,
    deadline: float,
    max_bytes: int,
) -> Exchange:
    """Send one request to `address` (already checked) and read at most
    `max_bytes` of the answer before `deadline` (a `time.monotonic()` value).
    `target` is the request line's path and query, sent exactly as given."""

    def remaining() -> float:
        left = deadline - time.monotonic()
        if left <= 0:
            raise GuardError("timeout", "")
        return left

    # An IP literal is ASCII already, and the IDNA codec refuses `::1`.
    host_bytes = host.encode("ascii") if parse_ip(host) is not None else host.encode("idna")
    url = httpcore.URL(scheme=scheme.encode("ascii"), host=host_bytes, port=port, target=target.encode("ascii"))
    ssl_context = ssl.create_default_context() if scheme == "https" else None
    request_headers = [(k.encode("latin-1"), v.encode("latin-1")) for k, v in headers]

    async def run() -> Exchange:
        left = remaining()
        timeouts = {"connect": left, "read": left, "write": left, "pool": left}
        async with httpcore.AsyncConnectionPool(
            ssl_context=ssl_context,
            network_backend=_PinnedBackend(address),
            http1=True,
            http2=False,
            retries=0,
            max_connections=1,
        ) as pool:
            async with pool.stream(
                method.encode("ascii"),
                url,
                headers=request_headers,
                content=body if body else None,
                extensions={"timeout": timeouts},
            ) as response:
                chunks: List[bytes] = []
                size = 0
                truncated = False
                async for chunk in response.aiter_stream():
                    room = max_bytes - size
                    if len(chunk) > room:
                        chunks.append(chunk[:room])
                        size = max_bytes
                        truncated = True
                        break
                    chunks.append(chunk)
                    size += len(chunk)
                return Exchange(
                    status=response.status,
                    headers=[(k.decode("latin-1").lower(), v.decode("latin-1")) for k, v in response.headers],
                    body=b"".join(chunks),
                    truncated=truncated,
                    address=str(address),
                )

    try:
        return await asyncio.wait_for(run(), timeout=remaining())
    except GuardError:
        raise
    except (asyncio.TimeoutError, httpcore.TimeoutException):
        raise GuardError("timeout", "") from None
    except (httpcore.ConnectError, httpcore.NetworkError, httpcore.ProtocolError, ssl.SSLError, OSError) as error:
        raise GuardError("network_error", _network_message(host, error)) from None
