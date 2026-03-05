"""
Platform Authentication

Browser-based login and API key fallback for Robutler/Robutler.
"""

import os
import webbrowser
from typing import Optional, Dict
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler

from ..state.local import get_state


CALLBACK_PORT = 8789
PLATFORM_URL = os.getenv("ROBUTLER_API_URL") or os.getenv("ROBUTLER_INTERNAL_API_URL") or "https://robutler.ai"
CLI_AUTH_PATH = "/cli/auth"
TOKEN_PATH = "/api/auth/cli/token"
USER_ME_PATH = "/api/users/me"
EXPIRES_IN_DAYS = 7


def _base() -> str:
    return PLATFORM_URL.rstrip("/")


async def login(api_key: Optional[str] = None) -> Dict:
    """Login to the platform (Robutler/Robutler).

    Args:
        api_key: Optional API key (rok_*) for headless/CI; skip for browser flow.

    Returns:
        User info dict with username, user_id, etc.
    """
    if api_key:
        return await _login_with_api_key(api_key)
    return await _login_with_browser()


async def _login_with_browser() -> Dict:
    """Open browser to platform /cli/auth, receive JWT via localhost callback."""
    import secrets

    state = secrets.token_urlsafe(16)
    port = str(CALLBACK_PORT)
    auth_url = f"{_base()}{CLI_AUTH_PATH}?port={port}&state={state}"
    print("Opening browser for authentication...")
    print(f"If the browser doesn't open, visit: {auth_url}")
    webbrowser.open(auth_url)

    token_data = _wait_for_callback(state=state, port=int(port))
    if not token_data:
        raise ValueError("Authentication failed - no token received")

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


def _wait_for_callback(state: str, port: int = CALLBACK_PORT, timeout: int = 300) -> Optional[Dict]:
    """Run a local HTTP server to receive the redirect with token and state."""
    result: Optional[Dict] = None

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal result
            query = parse_qs(urlparse(self.path).query)
            got_state = query.get("state", [None])[0]
            if got_state != state:
                self.send_error(400, "Invalid state")
                return
            token = query.get("token", [None])[0]
            username = query.get("username", [""])[0] or ""
            if token:
                result = {"token": token, "username": username}
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"<html><body><h1>Authentication successful!</h1>"
                    b"<p>You can close this window and return to the terminal.</p></body></html>"
                )
            else:
                self.send_error(400, "Missing token")

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("localhost", port), CallbackHandler)
    server.timeout = timeout
    try:
        server.handle_request()
    finally:
        server.server_close()
    return result


async def _login_with_api_key(api_key: str) -> Dict:
    """Exchange API key (rok_*) for JWT via POST /api/auth/cli/token."""
    import httpx

    url = f"{_base()}{TOKEN_PATH}"
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
