import os
import json
import base64
import hashlib
import hmac
import urllib.parse
import secrets
from typing import Any, Dict, Optional, List, Union
from datetime import datetime, timezone

import httpx
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, prompt, http, hook
from webagents.server.context.context_vars import get_context
from webagents.utils.logging import get_logger, log_skill_event, log_tool_execution


class XComSkill(Skill):
    """Simplified X.com (Twitter) OAuth 1.0a integration for multitenant applications.

    This skill provides:
    - OAuth 1.0a User Context authentication with per-user rate limits
    - User subscription monitoring (follow specific X users)
    - Webhook-based post monitoring with relevance checking
    - Automatic notifications via notification skill integration
    - Secure credential storage via auth/KV skills

    Core workflow:
    1. Users authenticate via OAuth 1.0a
    2. Subscribe to specific X users to monitor
    3. Webhook receives posts from subscribed users
    4. Agent checks post relevance against instructions
    5. Relevant posts trigger notifications to owner
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config or {}, scope="all")
        self.logger = None
        
        # OAuth 1.0a credentials
        self.api_key = os.getenv("X_API_KEY", "")
        self.api_secret = os.getenv("X_API_SECRET", "")
        
        # OAuth endpoints
        self.request_token_url = "https://api.x.com/oauth/request_token"
        self.authorize_url = "https://api.x.com/oauth/authorize"
        self.access_token_url = "https://api.x.com/oauth/access_token"
        
        # API base URL
        self.api_base = "https://api.x.com/2"
        
        # OAuth callback path
        self.oauth_redirect_path = "/oauth/x/callback"
        
        # Base URL for this agent
        env_agents = os.getenv("AGENTS_BASE_URL")
        base_root = (env_agents or "http://localhost:2224").rstrip('/')
        if base_root.endswith("/agents"):
            self.agent_base_url = base_root
        else:
            self.agent_base_url = base_root + "/agents"
    
    def get_dependencies(self) -> List[str]:
        """Skill dependencies"""
        return ['auth', 'kv', 'notifications']

    async def initialize(self, agent) -> None:
        self.agent = agent
        self.logger = get_logger('skill.x_com', agent.name)
        log_skill_event(agent.name, 'x_com', 'initialized', {})

    # ---------------- OAuth 1.0a Helpers ----------------
    def _redirect_uri(self) -> str:
        """Generate the OAuth callback URI"""
        base = self.agent_base_url.rstrip('/')
        return f"{base}/{self.agent.name}{self.oauth_redirect_path}"

    def _generate_nonce(self) -> str:
        """Generate a random nonce for OAuth"""
        return secrets.token_urlsafe(32)

    def _generate_timestamp(self) -> str:
        """Generate timestamp for OAuth"""
        return str(int(datetime.now(timezone.utc).timestamp()))

    def _percent_encode(self, string: str) -> str:
        """Percent encode string for OAuth"""
        return urllib.parse.quote(str(string), safe='')

    def _generate_signature_base_string(self, method: str, url: str, params: Dict[str, str]) -> str:
        """Generate OAuth signature base string"""
        # Sort parameters
        sorted_params = sorted(params.items())
        param_string = '&'.join([f"{self._percent_encode(k)}={self._percent_encode(v)}" for k, v in sorted_params])
        
        return f"{method.upper()}&{self._percent_encode(url)}&{self._percent_encode(param_string)}"

    def _generate_signature(self, method: str, url: str, params: Dict[str, str], 
                          token_secret: str = "") -> str:
        """Generate OAuth signature"""
        base_string = self._generate_signature_base_string(method, url, params)
        signing_key = f"{self._percent_encode(self.api_secret)}&{self._percent_encode(token_secret)}"
        
        signature = hmac.new(
            signing_key.encode('utf-8'),
            base_string.encode('utf-8'),
            hashlib.sha1
        ).digest()
        
        return base64.b64encode(signature).decode('utf-8')

    def _build_auth_header(self, method: str, url: str, params: Dict[str, str], 
                          oauth_params: Dict[str, str], token_secret: str = "") -> str:
        """Build OAuth authorization header"""
        # Combine all parameters for signature
        all_params = {**params, **oauth_params}
        signature = self._generate_signature(method, url, all_params, token_secret)
        oauth_params['oauth_signature'] = signature
        
        # Build header
        header_params = []
        for key, value in sorted(oauth_params.items()):
            header_params.append(f'{self._percent_encode(key)}="{self._percent_encode(value)}"')
        
        return f"OAuth {', '.join(header_params)}"

    async def _get_auth_skill(self):
        """Get authentication skill"""
        return self.agent.skills.get("auth")

    async def _get_kv_skill(self):
        """Get key-value storage skill"""
        return self.agent.skills.get("kv") or self.agent.skills.get("json_storage")

    async def _get_notification_skill(self):
        """Get notification skill"""
        return self.agent.skills.get("notifications")

    def _token_filename(self, user_id: str) -> str:
        """Generate filename for user tokens"""
        return f"x_tokens_{user_id}.json"

    async def _save_user_tokens(self, user_id: str, tokens: Dict[str, Any]) -> None:
        """Save user tokens securely using KV skill"""
        kv_skill = await self._get_kv_skill()
        if kv_skill and hasattr(kv_skill, 'kv_set'):
            try:
                # Use KV skill for secure, persistent storage
                result = await kv_skill.kv_set(
                    key=self._token_filename(user_id), 
                    value=json.dumps(tokens), 
                    namespace="x_com_auth"
                )
                if "✅" in str(result):  # KV skill returns "✅ Saved" on success
                    return
            except Exception as e:
                self.logger.warning(f"Failed to save tokens to KV skill: {e}")
        
        # Fallback: in-memory storage (not recommended for production)
        setattr(self.agent, '_x_tokens', getattr(self.agent, '_x_tokens', {}))
        self.agent._x_tokens[user_id] = tokens

    async def _load_user_tokens(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Load user tokens from KV skill"""
        kv_skill = await self._get_kv_skill()
        if kv_skill and hasattr(kv_skill, 'kv_get'):
            try:
                # Load from KV skill
                stored = await kv_skill.kv_get(
                    key=self._token_filename(user_id), 
                    namespace="x_com_auth"
                )
                if isinstance(stored, str) and stored.strip() and stored.startswith('{'):
                    return json.loads(stored)
            except Exception as e:
                self.logger.warning(f"Failed to load tokens from KV skill: {e}")
        
        # Fallback: in-memory storage
        mem = getattr(self.agent, '_x_tokens', {})
        return mem.get(user_id)

    async def _get_user_id_from_context(self) -> Optional[str]:
        """Get user ID from request context via auth skill"""
        try:
            ctx = get_context()
            if not ctx:
                return None
            auth = getattr(ctx, 'auth', None) or (ctx and ctx.get('auth'))
            return getattr(auth, 'user_id', None)
        except Exception:
            return None

    async def _get_authenticated_user_id(self) -> Optional[str]:
        """Get authenticated user ID, ensuring proper auth"""
        auth_skill = await self._get_auth_skill()
        if not auth_skill:
            return await self._get_user_id_from_context()
        
        try:
            ctx = get_context()
            if not ctx or not ctx.auth or not ctx.auth.authenticated:
                return None
            return ctx.auth.user_id
        except Exception:
            return None

    # ---------------- User Subscription Management ----------------
    async def _save_subscriptions(self, owner_user_id: str, subscriptions: Dict[str, Any]) -> None:
        """Save user subscriptions for an owner"""
        kv_skill = await self._get_kv_skill()
        if kv_skill and hasattr(kv_skill, 'kv_set'):
            try:
                await kv_skill.kv_set(
                    key=f"x_subscriptions_{owner_user_id}",
                    value=json.dumps(subscriptions),
                    namespace="x_com_subscriptions"
                )
            except Exception as e:
                self.logger.error(f"Failed to save subscriptions: {e}")

    async def _load_subscriptions(self, owner_user_id: str) -> Dict[str, Any]:
        """Load user subscriptions for an owner"""
        kv_skill = await self._get_kv_skill()
        if kv_skill and hasattr(kv_skill, 'kv_get'):
            try:
                stored = await kv_skill.kv_get(
                    key=f"x_subscriptions_{owner_user_id}",
                    namespace="x_com_subscriptions"
                )
                if isinstance(stored, str) and stored.strip() and stored.startswith('{'):
                    return json.loads(stored)
            except Exception as e:
                self.logger.error(f"Failed to load subscriptions: {e}")
        return {
            'users': {},  # username -> {user_id, instructions, active}
            'webhook_active': False,
            'created_at': datetime.now(timezone.utc).isoformat()
        }

    async def _register_webhook_with_x(self, webhook_url: str, user_tokens: Dict[str, str]) -> Dict[str, Any]:
        """Register webhook with X.com Account Activity API"""
        try:
            # First, create webhook environment if not exists
            env_response = await self._make_authenticated_request(
                'POST', '/1.1/account_activity/all/webhooks.json',
                {'url': webhook_url},
                user_tokens
            )
            
            webhook_id = env_response.get('id')
            
            # Subscribe user to webhook
            subscription_response = await self._make_authenticated_request(
                'POST', f'/1.1/account_activity/all/{webhook_id}/subscriptions.json',
                {},
                user_tokens
            )
            
            return {
                'webhook_id': webhook_id,
                'webhook_url': webhook_url,
                'subscription_active': True,
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to register webhook with X.com: {e}")
            raise

    async def _verify_webhook_signature(self, signature: str, timestamp: str, body: str) -> bool:
        """Verify X.com webhook signature"""
        try:
            # X.com uses HMAC-SHA256 for webhook signatures
            expected_signature = hmac.new(
                self.api_secret.encode('utf-8'),
                f"{timestamp}.{body}".encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Compare signatures securely
            return hmac.compare_digest(signature, expected_signature)
        except Exception as e:
            self.logger.error(f"Webhook signature verification failed: {e}")
            return False

    async def _process_webhook_event(self, event_data: Dict[str, Any], owner_user_id: str) -> None:
        """Process incoming webhook event from subscribed users"""
        try:
            # Focus only on tweet creation events from subscribed users
            if 'tweet_create_events' in event_data:
                await self._handle_subscribed_user_tweets(event_data['tweet_create_events'], owner_user_id)
            
        except Exception as e:
            self.logger.error(f"Error processing webhook event: {e}")

    async def _handle_subscribed_user_tweets(self, tweet_events: List[Dict], owner_user_id: str) -> None:
        """Handle tweet creation events from subscribed users"""
        subscriptions = await self._load_subscriptions(owner_user_id)
        subscribed_users = subscriptions.get('users', {})
        
        if not subscribed_users:
            return
        
        for tweet in tweet_events:
            tweet_user = tweet.get('user', {}).get('screen_name', '').lower()
            tweet_text = tweet.get('text', '')
            tweet_id = tweet.get('id_str', '')
            
            # Check if this tweet is from a subscribed user
            if tweet_user in subscribed_users:
                user_config = subscribed_users[tweet_user]
                if user_config.get('active', True):
                    # Check relevance against instructions
                    is_relevant = await self._check_post_relevance(
                        tweet_text, 
                        user_config.get('instructions', ''),
                        tweet_user,
                        owner_user_id
                    )
                    
                    if is_relevant:
                        await self._send_post_notification(
                            tweet_text, 
                            tweet_user, 
                            tweet_id, 
                            owner_user_id
                        )

    async def _check_post_relevance(self, tweet_text: str, instructions: str, username: str, owner_user_id: str) -> bool:
        """Check if a post is relevant based on agent instructions"""
        if not instructions.strip():
            # If no specific instructions, consider all posts relevant
            return True
        
        try:
            # Get the agent's LLM skill for relevance checking
            llm_skill = self.agent.skills.get("llm")
            
            prompt = f"""
You are helping monitor X.com posts for relevance. 

INSTRUCTIONS: {instructions}

POST from @{username}: {tweet_text}

Based on the instructions above, is this post relevant and worth notifying the owner about?
Consider:
- Does it match the monitoring criteria?
- Is it actionable or important?
- Would the owner want to know about this?

Respond with only "YES" or "NO".
"""
            
            if llm_skill and hasattr(llm_skill, 'generate'):
                response = await llm_skill.generate(prompt)
                result = response.get('text', '').strip().upper() if isinstance(response, dict) else str(response).strip().upper()
                return result.startswith('YES')
            else:
                # Fallback: simple keyword matching if no LLM available
                instructions_lower = instructions.lower()
                tweet_lower = tweet_text.lower()
                
                # Simple relevance check based on keyword overlap
                instruction_words = set(instructions_lower.split())
                tweet_words = set(tweet_lower.split())
                
                # If at least 20% of instruction keywords appear in tweet, consider relevant
                if instruction_words:
                    overlap = len(instruction_words & tweet_words)
                    return overlap / len(instruction_words) >= 0.2
                
                return True  # Default to relevant if no clear criteria
                
        except Exception as e:
            self.logger.error(f"Error checking post relevance: {e}")
            return True  # Default to relevant on error

    async def _send_post_notification(self, tweet_text: str, username: str, tweet_id: str, owner_user_id: str) -> None:
        """Send notification about relevant post using notification skill"""
        try:
            notification_skill = await self._get_notification_skill()
            if not notification_skill:
                self.logger.warning("Notification skill not available")
                return
            
            # Truncate tweet text for notification
            display_text = tweet_text[:100] + "..." if len(tweet_text) > 100 else tweet_text
            
            title = f"📱 New post from @{username}"
            body = f"Post: {display_text}\n\nTweet ID: {tweet_id}"
            
            # Use the notification skill to send notification
            if hasattr(notification_skill, 'send_notification'):
                result = await notification_skill.send_notification(
                    title=title,
                    body=body,
                    tag=f"x_com_post_{username}",
                    type="agent_update",
                    priority="normal"
                )
                self.logger.info(f"Notification sent for @{username} post: {result}")
            else:
                self.logger.warning("Notification skill does not have send_notification method")
                
        except Exception as e:
            self.logger.error(f"Failed to send post notification: {e}")



    async def _get_request_token(self) -> Dict[str, str]:
        """Step 1: Get OAuth request token"""
        oauth_params = {
            'oauth_consumer_key': self.api_key,
            'oauth_nonce': self._generate_nonce(),
            'oauth_signature_method': 'HMAC-SHA1',
            'oauth_timestamp': self._generate_timestamp(),
            'oauth_version': '1.0',
            'oauth_callback': self._redirect_uri()
        }
        
        auth_header = self._build_auth_header('POST', self.request_token_url, {}, oauth_params)
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                self.request_token_url,
                headers={'Authorization': auth_header}
            )
            response.raise_for_status()
            
            # Parse response
            data = urllib.parse.parse_qs(response.text)
            return {
                'oauth_token': data['oauth_token'][0],
                'oauth_token_secret': data['oauth_token_secret'][0],
                'oauth_callback_confirmed': data.get('oauth_callback_confirmed', ['false'])[0]
            }

    def _build_authorize_url(self, oauth_token: str, user_id: str) -> str:
        """Step 2: Build authorization URL"""
        params = {
            'oauth_token': oauth_token,
            'state': user_id  # Include user_id for callback correlation
        }
        return f"{self.authorize_url}?{urllib.parse.urlencode(params)}"

    async def _get_access_token(self, oauth_token: str, oauth_token_secret: str, 
                               oauth_verifier: str) -> Dict[str, str]:
        """Step 3: Exchange request token for access token"""
        oauth_params = {
            'oauth_consumer_key': self.api_key,
            'oauth_nonce': self._generate_nonce(),
            'oauth_signature_method': 'HMAC-SHA1',
            'oauth_timestamp': self._generate_timestamp(),
            'oauth_version': '1.0',
            'oauth_token': oauth_token,
            'oauth_verifier': oauth_verifier
        }
        
        auth_header = self._build_auth_header(
            'POST', self.access_token_url, {}, oauth_params, oauth_token_secret
        )
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                self.access_token_url,
                headers={'Authorization': auth_header}
            )
            response.raise_for_status()
            
            # Parse response
            data = urllib.parse.parse_qs(response.text)
            return {
                'oauth_token': data['oauth_token'][0],
                'oauth_token_secret': data['oauth_token_secret'][0],
                'user_id': data.get('user_id', [''])[0],
                'screen_name': data.get('screen_name', [''])[0]
            }

    async def _make_authenticated_request(self, method: str, endpoint: str, 
                                        params: Dict[str, Any] = None, 
                                        user_tokens: Dict[str, str] = None) -> Dict[str, Any]:
        """Make authenticated API request using user tokens"""
        if not user_tokens:
            raise ValueError("User tokens required for authenticated requests")
        
        url = f"{self.api_base}{endpoint}"
        params = params or {}
        
        # Convert params to strings for OAuth signature
        str_params = {k: str(v) for k, v in params.items()}
        
        oauth_params = {
            'oauth_consumer_key': self.api_key,
            'oauth_nonce': self._generate_nonce(),
            'oauth_signature_method': 'HMAC-SHA1',
            'oauth_timestamp': self._generate_timestamp(),
            'oauth_version': '1.0',
            'oauth_token': user_tokens['oauth_token']
        }
        
        auth_header = self._build_auth_header(
            method, url, str_params, oauth_params, user_tokens['oauth_token_secret']
        )
        
        headers = {
            'Authorization': auth_header,
            'Content-Type': 'application/json'
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            if method.upper() == 'GET':
                response = await client.get(url, params=params, headers=headers)
            elif method.upper() == 'POST':
                response = await client.post(url, json=params, headers=headers)
            else:
                response = await client.request(method, url, json=params, headers=headers)
            
            # Check rate limits
            remaining = response.headers.get('x-rate-limit-remaining')
            reset = response.headers.get('x-rate-limit-reset')
            
            if remaining:
                self.logger.info(f"Rate limit remaining: {remaining}, resets at: {reset}")
            
            response.raise_for_status()
            return response.json()

    # ---------------- Prompts ----------------
    @prompt(priority=40, scope=["owner", "all"])
    def x_com_prompt(self) -> str:
        return (
            "Ultra-minimal X.com integration with just 3 tools: "
            "x_subscribe() to monitor users and get notifications, "
            "x_post() to tweet, and x_manage() to view/manage subscriptions. "
            "Authentication is handled automatically. Uses notification skill for alerts."
        )

    # ---------------- HTTP Callback Handler ----------------
    @http(subpath="/oauth/x/callback", method="get", scope=["all"])
    async def oauth_callback(self, oauth_token: str = None, oauth_verifier: str = None, 
                           state: str = None) -> Dict[str, Any]:
        """Handle OAuth callback"""
        from fastapi.responses import HTMLResponse
        
        def html(success: bool, message: str) -> str:
            from string import Template
            color_ok = "#1DA1F2"  # Twitter blue
            color_err = "#dc2626"  # red-600
            accent = color_ok if success else color_err
            title = "X.com connected" if success else "X.com connection failed"
            safe_msg = (message or '').replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('$','$$')
            
            template = Template("""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>WebAgents – X.com</title>
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
      .x-icon { width: 44px; height: 44px; }
    </style>
  </head>
  <body>
    <div class="card">
      <div class="icon">
        <svg class="x-icon" viewBox="0 0 24 24" fill="currentColor">
          <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
        </svg>
      </div>
      <h1>$title</h1>
      <p>$safe_msg</p>
      <button id="ok">OK</button>
    </div>
    <script>
      (function(){
        var payload = { type: 'x-com-connected', success: $postSuccess };
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
</html>""")
            
            return template.safe_substitute(
                title=title,
                accent=accent,
                safe_msg=safe_msg,
                postSuccess=("true" if success else "false"),
            )

        if not oauth_token or not oauth_verifier:
            return HTMLResponse(
                content=html(False, "Missing OAuth parameters. Please retry authentication."), 
                media_type="text/html"
            )

        if not self.api_key or not self.api_secret:
            return HTMLResponse(
                content=html(False, "Server missing X.com API credentials. Contact support."), 
                media_type="text/html"
            )

        user_id = state or await self._get_user_id_from_context() or ""
        
        try:
            # Load request token secret from temporary storage
            temp_tokens = getattr(self.agent, '_temp_x_tokens', {})
            token_secret = temp_tokens.get(oauth_token, {}).get('oauth_token_secret')
            
            if not token_secret:
                return HTMLResponse(
                    content=html(False, "Invalid OAuth state. Please restart authentication."), 
                    media_type="text/html"
                )

            # Exchange for access token
            access_tokens = await self._get_access_token(oauth_token, token_secret, oauth_verifier)
            
            # Save access tokens
            await self._save_user_tokens(user_id, access_tokens)
            
            # Clean up temporary tokens
            if oauth_token in temp_tokens:
                del temp_tokens[oauth_token]
            
            return HTMLResponse(
                content=html(True, "Your X.com account is now connected and ready to use."), 
                media_type="text/html"
            )
            
        except Exception as e:
            return HTMLResponse(
                content=html(False, f"Authentication failed: {str(e)}"), 
                media_type="text/html"
            )

    # ---------------- Webhook HTTP Handler ----------------
    @http(subpath="/webhook/x/events", method="post", scope=["all"])
    async def webhook_handler(self, request: Request) -> JSONResponse:
        """Handle incoming webhook events from X.com"""
        try:
            # Get request headers and body
            signature = request.headers.get('x-twitter-webhooks-signature')
            timestamp = request.headers.get('x-twitter-webhooks-timestamp')
            
            if not signature or not timestamp:
                raise HTTPException(status_code=400, detail="Missing webhook signature headers")
            
            # Read request body
            body = await request.body()
            body_text = body.decode('utf-8')
            
            # Verify webhook signature
            if not await self._verify_webhook_signature(signature, timestamp, body_text):
                raise HTTPException(status_code=401, detail="Invalid webhook signature")
            
            # Parse event data
            try:
                event_data = json.loads(body_text)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON payload")
            
            # Extract owner user ID from event data
            # For X.com webhooks, we need to map the webhook to the owner
            # This is simplified - in production you'd have a proper mapping system
            owner_user_id = event_data.get('for_user_id')  # X.com provides this
            
            if owner_user_id:
                # Process the webhook event
                await self._process_webhook_event(event_data, owner_user_id)
            
            # Return success response
            return JSONResponse(
                content={"status": "success", "message": "Webhook processed"}, 
                status_code=200
            )
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Webhook handler error: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @http(subpath="/webhook/x/challenge", method="get", scope=["all"])
    async def webhook_challenge(self, crc_token: str = None) -> JSONResponse:
        """Handle X.com webhook challenge (CRC - Challenge Response Check)"""
        if not crc_token:
            raise HTTPException(status_code=400, detail="Missing crc_token parameter")
        
        try:
            # Generate response using HMAC-SHA256
            response_token = base64.b64encode(
                hmac.new(
                    self.api_secret.encode('utf-8'),
                    crc_token.encode('utf-8'),
                    hashlib.sha256
                ).digest()
            ).decode('utf-8')
            
            return JSONResponse(
                content={"response_token": f"sha256={response_token}"},
                status_code=200
            )
            
        except Exception as e:
            self.logger.error(f"Webhook challenge error: {e}")
            raise HTTPException(status_code=500, detail="Challenge generation failed")

    # ---------------- Minimal Tools (3 Only) ----------------
    
    @tool(description="Subscribe to X.com users and get notified when they post relevant content. Handles authentication automatically.", scope="owner")
    async def x_subscribe(self, username: str, instructions: str = "Notify me about all posts") -> str:
        """Subscribe to monitor posts from a specific X.com user with automatic authentication"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        # Check if authenticated, if not provide auth URL
        tokens = await self._load_user_tokens(user_id)
        if not tokens:
            if not self.api_key or not self.api_secret:
                return "❌ X.com API credentials not configured"
            
            try:
                # Auto-generate auth URL
                request_tokens = await self._get_request_token()
                if not hasattr(self.agent, '_temp_x_tokens'):
                    self.agent._temp_x_tokens = {}
                self.agent._temp_x_tokens[request_tokens['oauth_token']] = request_tokens
                auth_url = self._build_authorize_url(request_tokens['oauth_token'], user_id)
                
                return f"🔗 First, authorize X.com access: {auth_url}\n\nThen run this command again to subscribe to @{username}"
            except Exception as e:
                return f"❌ Authentication setup failed: {str(e)}"
        
        try:
            # Clean username
            username = username.lstrip('@').lower()
            
            # Validate username exists on X.com
            user_info = await self._make_authenticated_request(
                'GET', '/2/users/by/username/' + username,
                {'user.fields': 'id,name,username'},
                tokens
            )
            
            if not user_info.get('data'):
                return f"❌ User @{username} not found on X.com"
            
            x_user_data = user_info['data']
            display_name = x_user_data['name']
            
            # Load and update subscriptions
            subscriptions = await self._load_subscriptions(user_id)
            subscriptions['users'][username] = {
                'user_id': x_user_data['id'],
                'display_name': display_name,
                'instructions': instructions.strip(),
                'active': True,
                'subscribed_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Set up webhook if needed
            if not subscriptions.get('webhook_active'):
                webhook_url = f"{self.agent_base_url}/{self.agent.name}/webhook/x/events"
                webhook_info = await self._register_webhook_with_x(webhook_url, tokens)
                subscriptions['webhook_active'] = True
                subscriptions['webhook_id'] = webhook_info['webhook_id']
            
            await self._save_subscriptions(user_id, subscriptions)
            
            return f"✅ Subscribed to @{username} ({display_name})!\n🔔 Instructions: {instructions}\n\nYou'll get notifications when they post relevant content."
            
        except Exception as e:
            return f"❌ Failed to subscribe to @{username}: {str(e)}"

    @tool(description="Post a tweet to X.com. Handles authentication automatically.")
    async def x_post(self, text: str) -> str:
        """Post a tweet with automatic authentication handling"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        # Check authentication
        tokens = await self._load_user_tokens(user_id)
        if not tokens:
            if not self.api_key or not self.api_secret:
                return "❌ X.com API credentials not configured"
            
            try:
                # Auto-generate auth URL
                request_tokens = await self._get_request_token()
                if not hasattr(self.agent, '_temp_x_tokens'):
                    self.agent._temp_x_tokens = {}
                self.agent._temp_x_tokens[request_tokens['oauth_token']] = request_tokens
                auth_url = self._build_authorize_url(request_tokens['oauth_token'], user_id)
                
                return f"🔗 First, authorize X.com access: {auth_url}\n\nThen run this command again to post your tweet."
            except Exception as e:
                return f"❌ Authentication setup failed: {str(e)}"
        
        try:
            response = await self._make_authenticated_request(
                'POST', '/tweets', 
                {'text': text}, 
                tokens
            )
            
            tweet_id = response.get('data', {}).get('id', 'unknown')
            return f"✅ Tweet posted! ID: {tweet_id}"
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                return "❌ Rate limit exceeded. Please wait before posting again."
            elif e.response.status_code == 401:
                return "❌ Authentication expired. Please re-authorize X.com access."
            else:
                return f"❌ Failed to post tweet: {e.response.status_code}"
        except Exception as e:
            return f"❌ Error posting tweet: {str(e)}"

    @tool(description="View your X.com subscriptions and manage them.", scope="owner")
    async def x_manage(self, action: str = "list", username: str = None) -> str:
        """Manage X.com subscriptions: list, unsubscribe"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        subscriptions = await self._load_subscriptions(user_id)
        users = subscriptions.get('users', {})
        
        if action.lower() == "list":
            if not users:
                return "📭 No subscriptions yet.\n\nUse x_subscribe(username, instructions) to start monitoring X.com users."
            
            result = ["📋 Your X.com Subscriptions:\n"]
            for username, config in users.items():
                display_name = config.get('display_name', username)
                instructions = config.get('instructions', 'Monitor all posts')
                result.append(f"• @{username} ({display_name})")
                result.append(f"  📝 {instructions}\n")
            
            result.append("💡 Use x_manage('unsubscribe', 'username') to remove a subscription")
            return "\n".join(result)
        
        elif action.lower() == "unsubscribe":
            if not username:
                return "❌ Please specify username to unsubscribe from"
            
            username = username.lstrip('@').lower()
            if username not in users:
                return f"❌ Not subscribed to @{username}"
            
            display_name = users[username].get('display_name', username)
            del users[username]
            await self._save_subscriptions(user_id, subscriptions)
            
            return f"✅ Unsubscribed from @{username} ({display_name})"
        
        else:
            return "❌ Invalid action. Use 'list' or 'unsubscribe'"
    async def get_webhook_config(self) -> str:
        """Get current webhook monitoring configuration"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        config = await self._load_webhook_config(user_id)
        if not config:
            return "❌ No webhook monitoring configured. Use setup_webhook_monitoring() first."
        
        status = "🟢 Active" if config.get('active', False) else "🔴 Inactive"
        keywords = config.get('keywords', [])
        
        return f"""📡 Webhook Monitoring Configuration

Status: {status}
Webhook URL: {config.get('webhook_url', 'Not set')}
Keywords: {', '.join(keywords) if keywords else 'All posts'}
Mentions only: {'Yes' if config.get('mentions_only', False) else 'No'}
Notifications: {'Enabled' if config.get('send_notifications', True) else 'Disabled'}
Created: {config.get('created_at', 'Unknown')}"""

    @tool(description="Update webhook monitoring configuration.", scope="owner")
    async def update_webhook_config(self, keywords: List[str] = None, mentions_only: bool = None, 
                                  send_notifications: bool = None) -> str:
        """Update webhook monitoring configuration"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        config = await self._load_webhook_config(user_id)
        if not config:
            return "❌ No webhook configured. Use setup_webhook_monitoring() first."
        
        # Update configuration
        if keywords is not None:
            config['keywords'] = keywords
        if mentions_only is not None:
            config['mentions_only'] = mentions_only
        if send_notifications is not None:
            config['send_notifications'] = send_notifications
        
        config['updated_at'] = datetime.now(timezone.utc).isoformat()
        
        await self._save_webhook_config(user_id, config)
        
        return "✅ Webhook configuration updated successfully!"

    @tool(description="Disable webhook monitoring.", scope="owner")
    async def disable_webhook_monitoring(self) -> str:
        """Disable webhook monitoring"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        config = await self._load_webhook_config(user_id)
        if not config:
            return "❌ No webhook configured"
        
        try:
            # Deactivate subscription with X.com
            tokens = await self._load_user_tokens(user_id)
            if tokens and config.get('webhook_id'):
                await self._make_authenticated_request(
                    'DELETE', 
                    f"/1.1/account_activity/all/{config['webhook_id']}/subscriptions.json",
                    {},
                    tokens
                )
            
            # Update configuration
            config['active'] = False
            config['disabled_at'] = datetime.now(timezone.utc).isoformat()
            await self._save_webhook_config(user_id, config)
            
            return "✅ Webhook monitoring disabled successfully!"
            
        except Exception as e:
            return f"❌ Failed to disable webhook monitoring: {str(e)}"

    @tool(description="Get recent notifications from webhook events.", scope="owner")
    async def get_notifications(self, limit: int = 10) -> str:
        """Get recent notifications from webhook events"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        try:
            kv_skill = await self._get_kv_skill()
            if not kv_skill:
                return "❌ KV skill not available"
            
            notifications_data = await kv_skill.kv_get(
                key=f"notifications_{user_id}",
                namespace="x_com_notifications"
            )
            
            if not notifications_data or not notifications_data.strip():
                return "📭 No notifications yet"
            
            try:
                notifications = json.loads(notifications_data)
            except:
                return "❌ Error reading notifications"
            
            if not notifications:
                return "📭 No notifications yet"
            
            # Get recent notifications
            recent = notifications[-limit:]
            result = ["🔔 Recent Notifications:\n"]
            
            for i, notif in enumerate(reversed(recent), 1):
                timestamp = notif.get('timestamp', 'Unknown time')
                event_type = notif.get('event_type', 'unknown')
                message = notif.get('message', 'No message')
                read_status = "✅" if notif.get('read', False) else "🔴"
                
                result.append(f"{i}. {read_status} [{event_type}] {timestamp}")
                result.append(f"   {message}\n")
            
            return "\n".join(result)
            
        except Exception as e:
            return f"❌ Error getting notifications: {str(e)}"

    @tool(description="Mark notifications as read.", scope="owner")
    async def mark_notifications_read(self) -> str:
        """Mark all notifications as read"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        try:
            kv_skill = await self._get_kv_skill()
            if not kv_skill:
                return "❌ KV skill not available"
            
            notifications_data = await kv_skill.kv_get(
                key=f"notifications_{user_id}",
                namespace="x_com_notifications"
            )
            
            if not notifications_data or not notifications_data.strip():
                return "📭 No notifications to mark as read"
            
            try:
                notifications = json.loads(notifications_data)
            except:
                return "❌ Error reading notifications"
            
            # Mark all as read
            for notif in notifications:
                notif['read'] = True
            
            await kv_skill.kv_set(
                key=f"notifications_{user_id}",
                value=json.dumps(notifications),
                namespace="x_com_notifications"
            )
            
            return "✅ All notifications marked as read"
            
        except Exception as e:
            return f"❌ Error marking notifications as read: {str(e)}"
