import os
import json
import urllib.parse
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone

import httpx

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, prompt, http, hook
from webagents.server.context.context_vars import get_context
from webagents.utils.logging import get_logger, log_skill_event, log_tool_execution


class GoogleCalendarSkill(Skill):
    """Google Calendar integration for the dedicated agent r-google-calendar.

    Flow:
    - Other agents call tools on r-google-calendar and present an ownership assertion (X-Owner-Assertion).
    - On first use per user, this skill returns an AuthSub-style consent URL for the user to authorize access.
    - After consent, Google redirects to our callback endpoint; we exchange code for a token and persist it.
    - Tokens are stored via the JSON storage skill as simple key-value entries for now.
    - Subsequent calendar operations use the stored token.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config or {}, scope="all")
        self.logger = None
        self.oauth_client_id = os.getenv("GOOGLE_CLIENT_ID", "")
        self.oauth_client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "")
        self.oauth_redirect_path = "/oauth/google/calendar/callback"
        self.oauth_scopes = [
            "https://www.googleapis.com/auth/calendar.readonly",
            "https://www.googleapis.com/auth/calendar.events.readonly",
        ]
        # Base URL where this agent is exposed (needed to form redirect URI)
        # Single source of truth: AGENTS_BASE_URL. It may be provided with or without trailing /agents.
        # We normalize to always include exactly one /agents segment.
        env_agents = os.getenv("AGENTS_BASE_URL")
        base_root = (env_agents or "http://localhost:2224").rstrip('/')
        if base_root.endswith("/agents"):
            self.agent_base_url = base_root
        else:
            self.agent_base_url = base_root + "/agents"

    async def initialize(self, agent) -> None:
        self.agent = agent
        self.logger = get_logger('skill.google.calendar', agent.name)
        log_skill_event(agent.name, 'google_calendar', 'initialized', {})

    # ---------------- Helpers ----------------
    def _redirect_uri(self) -> str:
        # e.g., http://localhost:8000/agents/{agent-name}/oauth/google/calendar/callback
        base = self.agent_base_url.rstrip('/')
        return f"{base}/{self.agent.name}{self.oauth_redirect_path}"

    async def _get_kv_skill(self):
        # Prefer dedicated KV skill if present; fallback to json_storage
        return self.agent.skills.get("kv") or self.agent.skills.get("json_storage")

    def _token_filename(self, user_id: str) -> str:
        return f"gcal_tokens_{user_id}.json"

    async def _save_user_tokens(self, user_id: str, tokens: Dict[str, Any]) -> None:
        kv_skill = await self._get_kv_skill()
        if kv_skill and hasattr(kv_skill, 'kv_set'):
            try:
                await kv_skill.kv_set(key=self._token_filename(user_id), value=json.dumps(tokens), namespace="auth")
                return
            except Exception:
                pass
        # Fallback: in-memory
        setattr(self.agent, '_gcal_tokens', getattr(self.agent, '_gcal_tokens', {}))
        self.agent._gcal_tokens[user_id] = tokens

    async def _load_user_tokens(self, user_id: str) -> Optional[Dict[str, Any]]:
        kv_skill = await self._get_kv_skill()
        if kv_skill and hasattr(kv_skill, 'kv_get'):
            try:
                stored = await kv_skill.kv_get(key=self._token_filename(user_id), namespace="auth")
                if isinstance(stored, str) and stored.startswith('{'):
                    return json.loads(stored)
            except Exception:
                pass
        mem = getattr(self.agent, '_gcal_tokens', {})
        return mem.get(user_id)

    def _build_auth_url(self, user_id: str) -> str:
        q = {
            "client_id": self.oauth_client_id,
            "response_type": "code",
            "redirect_uri": self._redirect_uri(),
            "access_type": "offline",
            "prompt": "consent",
            "scope": " ".join(self.oauth_scopes),
            # State can carry user id for correlation
            "state": user_id,
        }
        return f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(q)}"

    async def _exchange_code_for_tokens(self, code: str) -> Dict[str, Any]:
        token_url = "https://oauth2.googleapis.com/token"
        payload = {
            "client_id": self.oauth_client_id,
            "client_secret": self.oauth_client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": self._redirect_uri(),
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        async with httpx.AsyncClient(timeout=20) as client:
            try:
                resp = await client.post(token_url, data=payload, headers=headers)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                # Surface detailed Google error payload when available
                detail = None
                try:
                    detail = e.response.json()
                except Exception:
                    try:
                        detail = {"text": e.response.text}
                    except Exception:
                        detail = {"error": str(e)}
                raise Exception(f"Google token exchange failed: status={e.response.status_code}, redirect_uri={payload['redirect_uri']}, detail={detail}")

    async def _refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        token_url = "https://oauth2.googleapis.com/token"
        payload = {
            "client_id": self.oauth_client_id,
            "client_secret": self.oauth_client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(token_url, data=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()

    async def _get_user_id_from_context(self) -> Optional[str]:
        try:
            ctx = get_context()
            if not ctx:
                return None
            auth = getattr(ctx, 'auth', None) or (ctx and ctx.get('auth'))
            return getattr(auth, 'user_id', None)
        except Exception:
            return None

    # ---------------- Prompts ----------------
    @prompt(priority=40, scope=["owner", "all"])  # Visible to all for discoverability
    def calendar_prompt(self) -> str:
        return (
            "Google Calendar skill is available. If calendar access is needed, call init_calendar_auth() to obtain an authorization link. "
            "After the user authorizes, use list_events() to read calendar events."
        )

    # ---------------- HTTP callback handler ----------------
    # Public callback: rely on state param and context-derived user id; no owner scope required
    @http(subpath="/oauth/google/calendar/callback", method="get", scope=["all"])  # Accept from any caller; identity via state/context
    async def oauth_callback(self, code: str = None, state: str = None) -> Dict[str, Any]:
        from fastapi.responses import HTMLResponse
        # Minimal HTML confirmation UI with brand-consistent styling (use Template to avoid brace issues)
        def html(success: bool, message: str) -> str:
            from string import Template
            color_ok = "#16a34a"  # green-600
            color_err = "#dc2626"  # red-600
            accent = color_ok if success else color_err
            title = "Calendar connected" if success else "Calendar connection failed"
            # Basic HTML escape and escape $ for Template
            safe_msg = (message or '')
            safe_msg = safe_msg.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('$','$$')
            template = Template(
                """<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>WebAgents – Google Calendar</title>
    <style>
      :root { color-scheme: light dark; }
      html, body { height: 100%; margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji"; }
      body { background: var(--bg, #0b0b0c); color: var(--fg, #e5e7eb); display: grid; place-items: center; }
      @media (prefers-color-scheme: light) { body { --bg: #f7f7f8; --card: #ffffff; --border: #e5e7eb; --fg: #0f172a; } }
      @media (prefers-color-scheme: dark) { body { --bg: #0b0b0c; --card: #111214; --border: #23252a; --fg: #e5e7eb; } }
      .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 24px 20px; width: min(92vw, 420px); box-shadow: 0 6px 24px rgba(0,0,0,.12); text-align: center; }
      .icon { color: $accent; display:inline-flex; align-items:center; justify-content:center; margin-bottom: 12px; }
      h1 { margin: 0 0 6px; font-size: 18px; font-weight: 600; color: var(--fg); }
      p { margin: 0 0 16px; font-size: 13px; opacity: .78; line-height: 1.4; }
      button { appearance: none; border: 1px solid var(--border); background: transparent; color: var(--fg); border-radius: 8px; padding: 8px 14px; font-size: 13px; cursor: pointer; }
      button:hover { background: rgba(127,127,127,.12); }
      .calendar { width: 44px; height: 44px; }
    </style>
  </head>
  <body>
    <div class=\"card\">
      <div class=\"icon\">
        <svg class=\"calendar\" viewBox=\"0 0 24 24\" fill=\"none\" xmlns=\"http://www.w3.org/2000/svg\">
          <rect x=\"3\" y=\"4\" width=\"18\" height=\"17\" rx=\"3\" ry=\"3\" stroke=\"currentColor\" stroke-width=\"1.5\"/>
          <path d=\"M3 8H21\" stroke=\"currentColor\" stroke-width=\"1.5\"/>
          <rect x=\"7\" y=\"11\" width=\"4\" height=\"4\" fill=\"currentColor\" />
        </svg>
      </div>
      <h1>$title</h1>
      <p>$safe_msg</p>
      <button id=\"ok\">OK</button>
    </div>
    <script>
      (function(){
        var payload = { type: 'google-calendar-connected', success: $postSuccess };
        try {
          if (window.opener && !window.opener.closed) { window.opener.postMessage(payload, '*'); }
          else if (window.parent && window.parent !== window) { window.parent.postMessage(payload, '*'); }
        } catch(e){}
        var ok = document.getElementById('ok');
        if (ok) ok.addEventListener('click', function(){
          try { window.close(); } catch(e) {}
          setTimeout(function(){ location.replace('about:blank'); }, 150);
        });
      })();
    </script>
  </body>
 </html>
                """
            )
            return template.safe_substitute(
                title=title,
                accent=accent,
                safe_msg=safe_msg,
                postSuccess=("true" if success else "false"),
            )

        if not code:
            return HTMLResponse(content=html(False, "Missing authorization code. Please retry."), media_type="text/html")
        if not self.oauth_client_id or not self.oauth_client_secret:
            return HTMLResponse(content=html(False, "Server is missing Google OAuth credentials. Contact support."), media_type="text/html")
        user_id = state or await self._get_user_id_from_context() or ""
        try:
            tokens = await self._exchange_code_for_tokens(code)
            await self._save_user_tokens(user_id, tokens)
            return HTMLResponse(content=html(True, "Your Google Calendar is now connected."), media_type="text/html")
        except Exception as e:
            return HTMLResponse(content=html(False, f"{str(e)}"), media_type="text/html")

    # ---------------- Tools ----------------
    # @tool(description="Initialize Google Calendar authorization. Returns a URL the user should open to grant access.")
    # async def init_calendar_auth(self) -> str:
    #     user_id = await self._get_user_id_from_context()
    #     if not user_id:
    #         return "❌ Unable to resolve user identity for authorization"
    #     auth_url = self._build_auth_url(user_id)
    #     log_tool_execution(self.agent.name, 'google_calendar.init_calendar_auth', 'success', {"user_id": user_id})
    #     return f"Open this link to authorize access: {auth_url}"

    @tool(description="Connects to Google calendar and lists upcoming calendar events.")
    async def list_events(self, max_results: int = 10) -> str:
        user_id = await self._get_user_id_from_context()
        if not user_id:
            return "❌ Missing user identity"
        tokens = await self._load_user_tokens(user_id)
        if not tokens or not tokens.get('access_token'):
            auth_url = self._build_auth_url(user_id)
            log_tool_execution(self.agent.name, 'google_calendar.init_calendar_auth', 'success', {"user_id": user_id})
            return f"Open this link to authorize access: {auth_url}"
            # return "❌ Not authorized. Call init_calendar_auth() first."
        try:
            access_token = tokens.get('access_token')
            headers = {"Authorization": f"Bearer {access_token}"}
            params = {
                "maxResults": int(max_results),
                "singleEvents": True,
                "orderBy": "startTime",
                "timeMin": datetime.now(timezone.utc).isoformat(),
            }
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get("https://www.googleapis.com/calendar/v3/calendars/primary/events", headers=headers, params=params)
                if resp.status_code == 401 and tokens.get('refresh_token'):
                    refreshed = await self._refresh_access_token(tokens['refresh_token'])
                    if 'access_token' in refreshed:
                        tokens['access_token'] = refreshed['access_token']
                        await self._save_user_tokens(user_id, tokens)
                        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
                        resp = await client.get("https://www.googleapis.com/calendar/v3/calendars/primary/events", headers=headers, params=params)
                if resp.status_code == 401:
                    return "❌ Token expired or invalid. Re-run init_calendar_auth()."
                if resp.status_code == 403:
                    try:
                        return f"❌ Permission error (403). Details: {resp.json()}"
                    except Exception:
                        return f"❌ Permission error (403)."
                resp.raise_for_status()
                data = resp.json()
                items = data.get('items', [])
                if not items:
                    return "(no upcoming events)"
                lines = []
                for ev in items[: int(max_results)]:
                    start = (ev.get('start') or {}).get('dateTime') or (ev.get('start') or {}).get('date') or 'unknown'
                    lines.append(f"- {start} | {ev.get('summary', '(no title)')}")
                return "\n".join(lines)
        except Exception as e:
            return f"❌ Failed to list events: {e}"


