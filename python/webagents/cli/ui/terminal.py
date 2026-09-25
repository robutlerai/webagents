"""
Asking the terminal what it looks like, and listening to it while a reply
streams (2026-09-24).

BACKGROUND. The shaded bands behind a sent message and a code block are mixed
from the terminal's own background, which only the terminal knows: a lift
that reads as gentle on one dark theme is a grey slab on another, and wrong on
every light one. The query (OSC 11) goes out together with Primary Device
Attributes (DA1), which every terminal answers, in order: DA1 arriving first
means the terminal ignores OSC 11, so the wait ends there instead of at a
timeout, and no late reply lands in the input box as stray characters. The
TypeScript chat asks the same way (`typescript/src/cli/ui/terminal.ts`).

KEYS WHILE A REPLY STREAMS. Between prompts nothing reads the keyboard, so
the terminal echoed whatever was typed straight into the middle of the answer,
and Esc did nothing. `stream_keys` turns echo off, reads keys itself, and
calls back on Esc; Ctrl+C still arrives as SIGINT, which the chat already
handles. POSIX terminals only; elsewhere both are a no-op.
"""

from __future__ import annotations

import os
import re
import select
import sys
import time
from contextlib import contextmanager
from typing import Callable, Iterator, Optional, Tuple

_OSC11 = re.compile(rb"\x1b\]11;rgb:([0-9a-fA-F]{1,4})/([0-9a-fA-F]{1,4})/([0-9a-fA-F]{1,4})")
_DA1 = re.compile(rb"\x1b\[\?[0-9;]*c")


def _channel(value: bytes) -> str:
    scale = int(value, 16) / (16 ** len(value) - 1)
    return f"{round(scale * 255):02x}"


def parse_background_reply(reply: bytes) -> Tuple[Optional[str], bool]:
    """`(#rrggbb or None, complete)` for what the terminal has said so far.

    Complete only once the DA1 reply has come: terminals answer in order, so
    it follows any OSC 11 answer. Stopping at the colour (2026-09-24) left the
    DA1 reply to arrive after the terminal was put back in cooked mode, and
    the person saw it echoed as `^[[?62;22c` above the chat.
    """
    osc = _OSC11.search(reply)
    background = f"#{_channel(osc.group(1))}{_channel(osc.group(2))}{_channel(osc.group(3))}" if osc else None
    return background, bool(_DA1.search(reply))


def _posix_tty(stream) -> bool:
    try:
        return os.name == "posix" and stream.isatty()
    except (AttributeError, ValueError):
        return False


def query_background(timeout: float = 0.4) -> Optional[str]:
    """The terminal's background colour as `#rrggbb`, or None when it will not say."""
    if not (_posix_tty(sys.stdin) and _posix_tty(sys.stdout)):
        return None
    import termios
    import tty

    fd = sys.stdin.fileno()
    try:
        saved = termios.tcgetattr(fd)
    except termios.error:
        return None
    reply = b""
    try:
        tty.setraw(fd)
        os.write(sys.stdout.fileno(), b"\x1b]11;?\x1b\\\x1b[c")
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                break
            reply += os.read(fd, 1024)
            background, complete = parse_background_reply(reply)
            if complete:
                return background
    except OSError:
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
    return parse_background_reply(reply)[0]


@contextmanager
def stream_keys(on_escape: Callable[[], None], loop=None) -> Iterator[None]:
    """While a reply streams: no echo, and Esc calls `on_escape`.

    The terminal stays in cbreak mode, so Ctrl+C still raises SIGINT. Other
    keys are read and dropped, so typing ahead does not scribble over the
    answer. An arrow key's escape sequence arrives whole and is not Esc.
    """
    if not _posix_tty(sys.stdin):
        yield
        return
    import asyncio
    import termios
    import tty

    fd = sys.stdin.fileno()
    try:
        saved = termios.tcgetattr(fd)
    except termios.error:
        yield
        return
    loop = loop or asyncio.get_event_loop()

    def _readable() -> None:
        try:
            data = os.read(fd, 1024)
        except OSError:
            return
        if data == b"\x1b":
            on_escape()

    tty.setcbreak(fd)
    loop.add_reader(fd, _readable)
    try:
        yield
    finally:
        loop.remove_reader(fd)
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
