"""
Platform Authentication

Browser-based login and API key fallback for Robutler/Robutler.
"""

import webbrowser
from typing import Callable, Dict, Optional
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler

from ..state.local import get_state
from ..config_store import cli_command


CALLBACK_PORT = 8789
CLI_AUTH_PATH = "/cli/auth"
TOKEN_PATH = "/api/auth/cli/token"
USER_ME_PATH = "/api/users/me"
EXPIRES_IN_DAYS = 7


def _base() -> str:
    """The portal to authenticate against, resolved the same way `deploy` does.

    THIS USED TO BE A MODULE CONSTANT, and it split the CLI in two
    (2026-09-24). It read `ROBUTLER_API_URL` at IMPORT time and knew nothing of
    the `platform.url` config key, while `RobutlerAPI` read `platform.url` and
    knew nothing of the variable. Setting either one alone signed you in to one
    portal and deployed to another, with no sign that they disagreed: point
    `platform.url` at a local cluster and `login` still opened production, then
    stored that production token for `deploy` to use against the cluster.

    `resolve_platform_url` is now the single answer for both, environment over
    config. Resolved per call rather than at import so a `config set` in the
    same process is visible, and so the test suite can point it somewhere.

    `ROBUTLER_INTERNAL_API_URL` is deliberately no longer consulted. It names
    the in-cluster address an agent skill uses from inside a pod; a browser on
    a developer's machine cannot open it, so the login flow could never have
    worked against it, and silently preferring it over the configured portal is
    how a login lands somewhere nobody chose.
    """
    from ..config_store import platform_url

    return platform_url()


async def login(api_key: Optional[str] = None, say: Callable[[str], None] = print, base: Optional[str] = None) -> Dict:
    """Login to the platform (Robutler/Robutler).

    Args:
        api_key: Optional API key (rok_*) for headless/CI; skip for browser flow.
        say: Where the browser flow's two lines go; the chat draws them its own way.

    Returns:
        User info dict with username, user_id, etc.
    """
    if api_key:
        return await _login_with_api_key(api_key, base)
    return await _login_with_browser(say, base)


async def _login_with_browser(say: Callable[[str], None] = print, base: Optional[str] = None) -> Dict:
    """Open browser to platform /cli/auth, receive JWT via localhost callback.

    Its two lines are the TypeScript sign-in's, word for word
    (`typescript/src/cli/browser-login.ts`).
    """
    import secrets

    state = secrets.token_urlsafe(16)
    port = str(CALLBACK_PORT)
    auth_url = f"{base or _base()}{CLI_AUTH_PATH}?port={port}&state={state}"
    say("Opening your browser to confirm the sign-in...")
    say(f"If the browser does not open, visit: {auth_url}")
    webbrowser.open(auth_url)

    token_data = _wait_for_callback(state=state, port=int(port))
    if not token_data:
        raise ValueError("Authentication failed - no token received")
    if token_data.get("error") == "access_denied":
        raise ValueError("Sign-in was cancelled in the browser.")
    if token_data.get("error"):
        raise ValueError(f"Sign-in failed: {token_data['error']}")

    token = token_data.get("token")
    username = token_data.get("username", "")
    if not token:
        raise ValueError("Authentication failed - invalid callback")

    expires_at = (datetime.utcnow() + timedelta(days=EXPIRES_IN_DAYS)).isoformat()
    state_obj = get_state()
    state_obj.set_credentials(
        access_token=token,
        auth_type="jwt",
        username=username,
        expires_at=expires_at,
        authenticated_at=datetime.utcnow().isoformat(),
    )
    return {"username": username, "auth_type": "jwt"}


#: Lucide icons, inlined: this page is served from localhost by the CLI and
#: loads nothing from anywhere, so there is no icon font or stylesheet to fetch.
_ICONS = {
    "ok": '<path d="M20 6 9 17l-5-5"/>',  # lucide: check
    "cancel": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',  # lucide: x
    "error": (  # lucide: triangle-alert
        '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/>'
        '<path d="M12 9v4"/><path d="M12 17h.01"/>'
    ),
}


def _callback_page(title: str, message: str, tone: str = "ok") -> bytes:
    """The page the browser lands on after the portal sends it back here.

    It was a bare `<h1>Authentication successful!</h1>` in the browser's
    default serif (2026-09-24), right after the portal's own consent screen,
    and the error cases were Python's stock `send_error` page. This matches the
    consent screen's look (gradient tile, lucide icon, one card), follows the
    system light or dark setting, and escapes everything interpolated.

    It also takes the token OUT OF THE ADDRESS BAR. The portal delivers it in
    this URL's query string (S-213, open), so until this page loads the bearer
    is on screen in full. `history.replaceState` swaps the visible URL and this
    tab's history entry for a bare `/callback`. It does not reach wherever the
    browser recorded the navigation itself; the real fix for that is S-213's
    code-for-token exchange.
    """
    from html import escape

    tile = "linear-gradient(135deg,#6366f1,#a855f7)" if tone == "ok" else (
        "linear-gradient(135deg,#71717a,#3f3f46)" if tone == "cancel"
        else "linear-gradient(135deg,#f59e0b,#ef4444)"
    )
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<meta name=\"referrer\" content=\"no-referrer\">"
        f"<title>{escape(title)}</title><style>"
        ":root{color-scheme:light dark;--bg:#fafafa;--fg:#0a0a0a;--muted:#6b7280;"
        "--card:#ffffff;--border:#e5e7eb}"
        "@media (prefers-color-scheme:dark){:root{--bg:#0a0a0a;--fg:#fafafa;"
        "--muted:#a1a1aa;--card:#171717;--border:#27272a}}"
        "*{box-sizing:border-box}body{margin:0;min-height:100vh;display:flex;"
        "align-items:center;justify-content:center;padding:16px;background:var(--bg);"
        "color:var(--fg);font:15px/1.5 -apple-system,BlinkMacSystemFont,\"Segoe UI\","
        "Roboto,Helvetica,Arial,sans-serif}"
        ".card{width:100%;max-width:380px;text-align:center;background:var(--card);"
        "border:1px solid var(--border);border-radius:16px;padding:32px 24px}"
        f".tile{{width:56px;height:56px;border-radius:16px;display:inline-flex;"
        f"align-items:center;justify-content:center;margin-bottom:16px;background:{tile};"
        "color:#fff}"
        "h1{font-size:20px;font-weight:600;margin:0 0 8px}"
        "p{margin:0;color:var(--muted);font-size:14px;text-wrap:balance}"
        "</style></head><body><main class=\"card\"><div class=\"tile\">"
        "<svg width=\"28\" height=\"28\" viewBox=\"0 0 24 24\" fill=\"none\" "
        "stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" "
        f"stroke-linejoin=\"round\" aria-hidden=\"true\">{_ICONS.get(tone, _ICONS['error'])}</svg>"
        f"</div><h1>{escape(title)}</h1><p>{escape(message)}</p></main>"
        "<script>history.replaceState(null,\"\",\"/callback\")</script>"
        "</body></html>"
    ).encode("utf-8")


def _wait_for_callback(state: str, port: int = CALLBACK_PORT, timeout: int = 300) -> Optional[Dict]:
    """Run a local HTTP server to receive the redirect with token and state."""
    result: Optional[Dict] = None

    class CallbackHandler(BaseHTTPRequestHandler):
        def _page(self, status: int, title: str, message: str, tone: str) -> None:
            body = _callback_page(title, message, tone)
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            nonlocal result
            query = parse_qs(urlparse(self.path).query)
            got_state = query.get("state", [None])[0]
            if got_state != state:
                self._page(
                    400,
                    "Sign-in failed",
                    "This does not match the sign-in started in your terminal. "
                    f"Run {cli_command('login')} again.",
                    "error",
                )
                return
            # The portal asks before it signs anyone in (2026-09-24), and Deny
            # comes back here as `error=access_denied` rather than silence, so
            # the terminal can say what happened instead of reporting a
            # missing token after a five-minute wait.
            error = query.get("error", [None])[0]
            if error:
                result = {"error": error}
                self._page(
                    200,
                    "Sign-in cancelled",
                    "Nothing was shared. You can close this tab.",
                    "cancel",
                )
                return
            token = query.get("token", [None])[0]
            username = query.get("username", [""])[0] or ""
            if token:
                result = {"token": token, "username": username}
                self._page(
                    200,
                    f"Signed in as @{username}" if username else "Signed in",
                    "You can close this tab and return to your terminal.",
                    "ok",
                )
            else:
                self._page(
                    400,
                    "Sign-in failed",
                    f"No sign-in details arrived. Run {cli_command('login')} again.",
                    "error",
                )

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("localhost", port), CallbackHandler)
    server.timeout = timeout
    try:
        server.handle_request()
    finally:
        server.server_close()
    return result


async def _login_with_api_key(api_key: str, base: Optional[str] = None) -> Dict:
    """Exchange API key (rok_*) for JWT via POST /api/auth/cli/token."""
    import httpx

    url = f"{base or _base()}{TOKEN_PATH}"
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={},
        )
    if resp.status_code != 200:
        raise ValueError(f"Token exchange failed: {resp.status_code} {resp.text}")
    data = resp.json()
    token = data.get("access_token")
    username = data.get("username", "")
    user_id = data.get("user_id", "")
    expires_in = data.get("expires_in", EXPIRES_IN_DAYS * 24 * 3600)
    if not token:
        raise ValueError("No access_token in response")

    expires_at = (datetime.utcnow() + timedelta(seconds=expires_in)).isoformat()
    state = get_state()
    state.set_credentials(
        access_token=token,
        auth_type="jwt",
        username=username,
        user_id=user_id,
        expires_at=expires_at,
        authenticated_at=datetime.utcnow().isoformat(),
    )
    return {"username": username, "user_id": user_id, "auth_type": "jwt"}


def logout() -> None:
    """Clear stored credentials."""
    get_state().clear_credentials()


def is_authenticated() -> bool:
    """True if we have a valid stored token (JWT or legacy api_key)."""
    creds = get_state().get_credentials()
    if not creds:
        return False
    if creds.get("auth_type") == "api_key":
        return bool(creds.get("api_key"))
    token = creds.get("access_token")
    if not token:
        return False
    expires_at = creds.get("expires_at")
    if expires_at:
        try:
            exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            # Compare with utcnow() for naive exp, or exp for aware
            now = datetime.utcnow()
            if exp.tzinfo:
                from datetime import timezone
                now = now.replace(tzinfo=timezone.utc)
            if now >= exp:
                return False
        except Exception:
            pass
    return True


async def get_current_user() -> Optional[Dict]:
    """Fetch current user from platform (GET /api/users/me) using stored token."""
    if not is_authenticated():
        return None
    creds = get_state().get_credentials()
    token = creds.get("access_token")
    if not token:
        return None
    import httpx
    url = f"{_base()}{USER_ME_PATH}"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers={"Authorization": f"Bearer {token}"})
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data.get("user") or data
    except Exception:
        return None


async def refresh_token() -> bool:
    """Not used for JWT (no refresh); return False."""
    return False
