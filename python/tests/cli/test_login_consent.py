"""
The portal asks before it signs a command line in (2026-09-24).

`/cli/auth` used to mint a token the moment a signed-in browser opened it. It
now shows a consent screen, and Deny sends the command line
`error=access_denied` on the same callback. Before this, the handler knew only
"token" or "Missing token", so a Deny would have surfaced as an unexplained
failure. These pin that it is reported as what it is.
"""

import asyncio
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    for var in ("WEBAGENTS_PROFILE", "WEBAGENTS_TOKEN", "ROBUTLER_API_URL"):
        monkeypatch.delenv(var, raising=False)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _hit(port: int, query: str) -> bytes:
    """GET the callback, retrying until the one-shot server is listening."""
    for _ in range(100):
        try:
            with urllib.request.urlopen(
                f"http://localhost:{port}/callback?{query}", timeout=2
            ) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            return error.read()
        except urllib.error.URLError:
            time.sleep(0.05)
    raise AssertionError("callback server never started")


def _serve_once(state: str, port: int) -> dict:
    from webagents.cli.platform.auth import _wait_for_callback

    out: dict = {}
    thread = threading.Thread(
        target=lambda: out.update(result=_wait_for_callback(state=state, port=port, timeout=10))
    )
    thread.start()
    return {"thread": thread, "out": out}


class TestTheCallbackHandler:
    def test_deny_is_reported_as_cancelled(self):
        port = _free_port()
        running = _serve_once("s1", port)
        page = _hit(port, "error=access_denied&state=s1")
        running["thread"].join(5)

        assert running["out"]["result"] == {"error": "access_denied"}
        assert b"cancelled" in page.lower()

    def test_a_token_still_signs_in(self):
        port = _free_port()
        running = _serve_once("s2", port)
        _hit(port, "token=dummy-token&username=ada&state=s2")
        running["thread"].join(5)

        assert running["out"]["result"] == {"token": "dummy-token", "username": "ada"}

    def test_a_foreign_state_is_still_refused_first(self):
        """The state check precedes everything, a Deny included."""
        port = _free_port()
        running = _serve_once("expected", port)
        _hit(port, "error=access_denied&state=someone-else")
        running["thread"].join(5)

        assert running["out"]["result"] is None


class TestLoginSaysWhatHappened:
    def test_deny_becomes_a_clear_message(self, monkeypatch):
        import webagents.cli.platform.auth as auth

        monkeypatch.setattr(auth, "_wait_for_callback", lambda **_: {"error": "access_denied"})
        monkeypatch.setattr(auth.webbrowser, "open", lambda url: True)
        with pytest.raises(ValueError, match="cancelled in the browser"):
            asyncio.run(auth._login_with_browser())

    def test_another_error_is_named(self, monkeypatch):
        import webagents.cli.platform.auth as auth

        monkeypatch.setattr(auth, "_wait_for_callback", lambda **_: {"error": "server_error"})
        monkeypatch.setattr(auth.webbrowser, "open", lambda url: True)
        with pytest.raises(ValueError, match="server_error"):
            asyncio.run(auth._login_with_browser())


class TestTheCallbackPage:
    """What the browser shows after the portal hands back (2026-09-24)."""

    def test_success_names_the_account(self):
        port = _free_port()
        running = _serve_once("s3", port)
        page = _hit(port, "token=dummy-token&username=ada&state=s3").decode()
        running["thread"].join(5)
        assert "Signed in as @ada" in page
        # Replaces the bare serif "Authentication successful!" page.
        assert "<style>" in page and "Authentication successful" not in page

    def test_the_token_is_taken_out_of_the_address_bar(self):
        port = _free_port()
        running = _serve_once("s4", port)
        page = _hit(port, "token=dummy-token&username=ada&state=s4").decode()
        running["thread"].join(5)
        assert 'history.replaceState(null,"","/callback")' in page
        # And never echoed into the page itself.
        assert "dummy-token" not in page

    def test_the_username_is_escaped(self):
        """It arrives in a URL; it is rendered as text, never as markup."""
        from webagents.cli.platform.auth import _callback_page

        page = _callback_page('Signed in as @<img src=x onerror=alert(1)>', "m").decode()
        assert "<img" not in page
        assert "&lt;img" in page

    def test_a_foreign_state_gets_a_styled_refusal_not_the_stock_error_page(self):
        port = _free_port()
        running = _serve_once("expected", port)
        page = _hit(port, "token=dummy-token&state=someone-else").decode()
        running["thread"].join(5)
        assert "Sign-in failed" in page
        assert "Error response" not in page  # http.server's stock send_error page
        assert "dummy-token" not in page
