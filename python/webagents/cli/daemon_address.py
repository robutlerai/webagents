"""
Where the local daemon is: one answer for every command (2026-09-24).

THE CONFIGURED ADDRESS WAS IGNORED ALMOST EVERYWHERE. `daemon.port` and
`daemon.host` have been known keys since the shared config contract
(`config_store.DEFAULTS`), and `webagents config set daemon.port 38770` stored
the value and echoed it back. Then the chat built `DaemonClient()` on its
hard-coded `http://localhost:8765` and auto-started a daemon on 8765, every
`daemon start|stop|restart|status|endpoints|logs` defaulted `--port` to a
literal 8765, `dev` did the same, `doctor` read the port but not the host, and
`session`/`checkpoint` read both but auto-started on the default host anyway.
Setting the port changed nothing but `config get`. Found by driving the chat;
the TypeScript CLI's `daemon` command was changed to read the keys the same day.

PRECEDENCE, highest first, for each key on its own:

    1. an explicit flag (`--port`, `--host`)
    2. the config store: `./.webagents/config.json`, then the profile's
       `~/.webagents/config.json`, with `${VAR}` references resolved
    3. the default, 127.0.0.1:8765

A CONFIGURED HOST MUST BE ON THIS MACHINE (S-218). The daemon's management
routes (the agent inventory with file paths, cron) carry no credential, so it
binds loopback by default and only an explicit `--host` binds anything else. A
config value must not become a quieter way to do the same thing: the project
file is the one people commit, so a repository could otherwise ship
`"daemon.host": "0.0.0.0"` and expose the daemon of whoever runs the chat in a
clone of it. The TypeScript CLI does exactly that today (S-232). So a host
that comes from CONFIG must be a loopback address, for binding and for
connecting alike, and anything else is refused with a message naming the file
and the flag. The chat has no `--host`, so it only ever talks to a daemon on
this machine, which is all it ever did.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

#: What `host_source`/`port_source` say for the lowest layer.
DEFAULT_SOURCE = "the default"

# RFC 1123 host names. Anything else in a host (`@`, `/`, `#`, spaces) would
# change what the URL built from it means: `127.0.0.1@elsewhere` is a URL for
# `elsewhere` with a user name in front.
_HOSTNAME = re.compile(
    r"^(?=.{1,253}\.?$)[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*\.?$"
)


class DaemonAddressError(ValueError):
    """An address that cannot be used, with the step that fixes it."""

    def __init__(self, message: str, fix: str = "") -> None:
        super().__init__(message)
        self.fix = fix


def is_loopback_host(host: str) -> bool:
    """`localhost`, or a literal loopback address (127.0.0.0/8, ::1).

    Other names are NOT resolved: a name that points at loopback today can
    point elsewhere tomorrow, and the answer would depend on DNS at the moment
    of asking. `0.0.0.0` and `::` are not loopback; they are every interface.
    """
    name = host.strip().strip("[]").lower()
    if name in ("localhost", "localhost."):
        return True
    try:
        address = ipaddress.ip_address(name)
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return bool((mapped or address).is_loopback)


def _url_host(host: str) -> str:
    """A host as it goes in a URL: IPv6 literals need their brackets."""
    return f"[{host}]" if ":" in host else host


@dataclass(frozen=True)
class DaemonAddress:
    host: str
    port: int
    #: Where each half came from: `--host`/`--port`, a config file, or the default.
    host_source: str = DEFAULT_SOURCE
    port_source: str = DEFAULT_SOURCE

    @property
    def base_url(self) -> str:
        return f"http://{_url_host(self.host)}:{self.port}"

    @property
    def is_loopback(self) -> bool:
        return is_loopback_host(self.host)

    def __str__(self) -> str:
        return f"{_url_host(self.host)}:{self.port}"

    def describe(self) -> str:
        """`127.0.0.1:38770 (port from ~/.webagents/config.json)`: the address and why."""
        why = [
            f"{name} from {source}"
            for name, source in (("port", self.port_source), ("host", self.host_source))
            if source != DEFAULT_SOURCE
        ]
        return f"{self} ({', '.join(why)})" if why else str(self)


def address_of(base_url: str) -> DaemonAddress:
    """The address a client's base URL points at."""
    parts = urlsplit(base_url)
    return DaemonAddress(parts.hostname or DEFAULT_HOST, parts.port or DEFAULT_PORT)


def _display(path: Path, cwd: Path) -> str:
    """`./.webagents/config.json` or `~/.webagents/config.json`, as people type them."""
    try:
        return "./" + str(path.relative_to(cwd))
    except ValueError:
        pass
    home = Path.home()
    try:
        return "~/" + str(path.relative_to(home))
    except ValueError:
        return str(path)


def _set_command(key: str, value: object, layer: str) -> str:
    """The `config set` that fixes `key` in the layer the bad value came from.

    `--project` for the project file: a plain `config set` writes the global
    file, which the project file outranks, so it would change nothing.
    """
    from .config_store import cli_command

    return f"`{cli_command(f'config set {key} {value}')}{' --project' if layer == 'project' else ''}`"


def _port(value: Any, where: str, fix: str) -> int:
    if isinstance(value, bool):
        raise DaemonAddressError(f"{where} is {value!r}, which is not a port number.", fix)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    try:
        port = value if isinstance(value, int) else int(str(value).strip())
    except (TypeError, ValueError):
        raise DaemonAddressError(f"{where} is {value!r}, which is not a port number.", fix) from None
    if not 1 <= port <= 65535:
        raise DaemonAddressError(f"{where} is {port}, which is outside 1-65535.", fix)
    return port


def _host(value: Any, where: str, fix: str) -> Optional[str]:
    """The host as given, without IPv6 brackets. None when unset."""
    if value is None:
        return None
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return None
    try:
        ipaddress.ip_address(text)
        return text
    except ValueError:
        pass
    if _HOSTNAME.match(text):
        return text
    raise DaemonAddressError(f"{where} is {value!r}, which is not a host name or an IP address.", fix)


def resolve_daemon_address(
    port: Optional[int] = None,
    host: Optional[str] = None,
    *,
    profile: Optional[str] = None,
    cwd: Optional[Path] = None,
) -> DaemonAddress:
    """The daemon's address: flag, then config, then 127.0.0.1:8765.

    `port` and `host` are the command's own flags, None when not given.

    Raises:
        DaemonAddressError: a value that is not a port or a host, or a
            configured host that is not a loopback address (see the module
            docstring). The message names where the value came from.
    """
    from .config_store import ConfigStore

    store = ConfigStore(profile=profile, cwd=cwd)
    files = {
        "project": _display(store.project_path, store.cwd),
        "global": _display(store.global_path, store.cwd),
    }

    if port is not None:
        port_value, port_source = _port(port, "--port", "pass a port between 1 and 65535"), "--port"
    else:
        layer = store.source_of("daemon.port") or "default"
        port_source = files.get(layer, DEFAULT_SOURCE)
        raw = store.get("daemon.port")
        port_value = _port(
            DEFAULT_PORT if raw is None else raw,
            f"daemon.port in {port_source}",
            _set_command("daemon.port", DEFAULT_PORT, layer),
        )

    flag_host = _host(host, "--host", "pass a host name or an IP address")
    if flag_host is not None:
        return DaemonAddress(flag_host, port_value, "--host", port_source)

    layer = store.source_of("daemon.host") or "default"
    host_source = files.get(layer, DEFAULT_SOURCE)
    host_value = _host(
        store.get("daemon.host"),
        f"daemon.host in {host_source}",
        _set_command("daemon.host", DEFAULT_HOST, layer),
    )
    if host_value is None:
        return DaemonAddress(DEFAULT_HOST, port_value, DEFAULT_SOURCE, port_source)
    if not is_loopback_host(host_value):
        raise DaemonAddressError(
            f"daemon.host in {host_source} is {host_value!r}, which is not an address on this "
            "machine. The daemon's management routes take no credential, so a configured host "
            "must be loopback (127.0.0.1, ::1 or localhost); only an explicit --host binds "
            "anywhere else.",
            f"{_set_command('daemon.host', DEFAULT_HOST, layer)}, or bind elsewhere on purpose "
            f"with `webagents daemon start --host {host_value}`",
        )
    return DaemonAddress(host_value, port_value, host_source, port_source)
