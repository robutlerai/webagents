"""
Platform Authentication

OAuth login with robutler.ai.
"""

import webbrowser
import asyncio
from typing import Optional, Dict
from datetime import datetime, timedelta
import secrets
import hashlib
import base64

from ..state.local import get_state


# OAuth configuration
PLATFORM_URL = "https://robutler.ai"
AUTH_URL = f"{PLATFORM_URL}/oauth/authorize"
TOKEN_URL = f"{PLATFORM_URL}/oauth/token"
CLIENT_ID = "webagents-cli"
REDIRECT_URI = "http://localhost:8789/callback"


async def login(api_key: Optional[str] = None) -> Dict:
    """Login to robutler.ai.
    
    Args:
        api_key: Optional API key (skip OAuth)
        
    Returns:
        User info dict
    """
    state = get_state()
    
    if api_key:
        # API key authentication
        return await _login_with_api_key(api_key)
    else:
        # OAuth flow
        return await _login_with_oauth()


async def _login_with_api_key(api_key: str) -> Dict:
    """Login with API key.
    
    Args:
        api_key: API key
        
    Returns:
        User info
    """
    state = get_state()
    
    # TODO: Validate API key with platform
    # For now, just store it
    state.set_credentials(
        api_key=api_key,
        auth_type="api_key",
        authenticated_at=datetime.utcnow().isoformat(),
    )
    
    return {"auth_type": "api_key"}


async def _login_with_oauth() -> Dict:
    """Login with OAuth flow.
    
    Returns:
        User info
    """
    import httpx
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import urllib.parse
    
    # Generate PKCE code verifier and challenge
    code_verifier = secrets.token_urlsafe(32)
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).decode().rstrip("=")
    
    # Generate state
    oauth_state = secrets.token_urlsafe(16)
    
    # Build authorization URL
    auth_params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "agents:read agents:write discovery:read discovery:write",
        "state": oauth_state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    auth_url = f"{AUTH_URL}?{urllib.parse.urlencode(auth_params)}"
    
    # Open browser
    print(f"Opening browser for authentication...")
    print(f"If browser doesn't open, visit: {auth_url}")
    webbrowser.open(auth_url)
    
    # Wait for callback
    auth_code = await _wait_for_callback(oauth_state)
    
    if not auth_code:
        raise ValueError("Authentication failed - no code received")
    
    # Exchange code for token
    async with httpx.AsyncClient() as client:
        token_response = await client.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": auth_code,
                "redirect_uri": REDIRECT_URI,
                "code_verifier": code_verifier,
            }
        )
        
        if token_response.status_code != 200:
            raise ValueError(f"Token exchange failed: {token_response.text}")
        
        tokens = token_response.json()
    
    # Store credentials
    state = get_state()
    state.set_credentials(
        access_token=tokens.get("access_token"),
        refresh_token=tokens.get("refresh_token"),
        expires_at=(datetime.utcnow() + timedelta(seconds=tokens.get("expires_in", 3600))).isoformat(),
        auth_type="oauth",
        authenticated_at=datetime.utcnow().isoformat(),
    )
    
    # Get user info
    return await get_current_user()


async def _wait_for_callback(expected_state: str, timeout: int = 300) -> Optional[str]:
    """Wait for OAuth callback.
    
    Args:
        expected_state: Expected state parameter
        timeout: Timeout in seconds
        
    Returns:
        Authorization code or None
    """
    auth_code = None
    
    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal auth_code
            
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            
            # Check state
            state = query.get("state", [None])[0]
            if state != expected_state:
                self.send_error(400, "Invalid state")
                return
            
            # Get code
            code = query.get("code", [None])[0]
            if code:
                auth_code = code
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(b"""
                    <html><body>
                    <h1>Authentication successful!</h1>
                    <p>You can close this window and return to the terminal.</p>
                    </body></html>
                """)
            else:
                error = query.get("error", ["unknown"])[0]
                self.send_error(400, f"Authentication failed: {error}")
        
        def log_message(self, format, *args):
            pass  # Suppress logging
    
    # Start callback server
    from http.server import HTTPServer
    server = HTTPServer(("localhost", 8789), CallbackHandler)
    server.timeout = timeout
    
    try:
        # Handle one request
        server.handle_request()
    finally:
        server.server_close()
    
    return auth_code


def logout():
    """Logout and clear credentials."""
    state = get_state()
    state.clear_credentials()


def is_authenticated() -> bool:
    """Check if user is authenticated.
    
    Returns:
        True if authenticated
    """
    state = get_state()
    creds = state.get_credentials()
    
    if not creds:
        return False
    
    if creds.get("auth_type") == "api_key":
        return bool(creds.get("api_key"))
    
    # Check OAuth token
    access_token = creds.get("access_token")
    if not access_token:
        return False
    
    # Check expiration
    expires_at = creds.get("expires_at")
    if expires_at:
        try:
            expires = datetime.fromisoformat(expires_at)
            if datetime.utcnow() > expires:
                return False
        except Exception:
            pass
    
    return True


async def get_current_user() -> Optional[Dict]:
    """Get current authenticated user.
    
    Returns:
        User info or None
    """
    if not is_authenticated():
        return None
    
    state = get_state()
    creds = state.get_credentials()
    
    # TODO: Fetch user info from platform API
    # For now, return placeholder
    return {
        "authenticated": True,
        "auth_type": creds.get("auth_type"),
    }


async def refresh_token() -> bool:
    """Refresh OAuth token.
    
    Returns:
        True if refreshed
    """
    state = get_state()
    creds = state.get_credentials()
    
    refresh_token = creds.get("refresh_token")
    if not refresh_token:
        return False
    
    # TODO: Implement token refresh
    return False
