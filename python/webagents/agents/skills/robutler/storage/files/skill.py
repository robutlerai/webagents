"""
RobutlerFilesSkill - File Management with Harmonized API
Uses the harmonized content API for cleaner and more efficient operations.

Endpoints used (see tests/fixtures/portal_routes.json for the contract):
- POST /api/content/upload            multipart upload (returns {id, url, displayName, ...})
- GET  /api/agents/{id}/content       list content reachable by that principal

Both halves use ONE principal — the subject of the api key this skill sends —
so a stored file appears in the listing. See `_content_principal_id`.
"""

import base64
import json
import os
from typing import Any, Dict, List, Optional

import aiohttp

from ....base import Skill
from webagents.agents.tools.decorators import tool
from webagents.agents.skills.robutler.payments import pricing, PricingInfo

UPLOAD_PATH = "/api/content/upload"


class FilesSkillConfigError(ValueError):
    """Raised when the skill has no usable credential."""


MISSING_KEY_MESSAGE = (
    "RobutlerFilesSkill has no API key: file storage and listing are "
    "unavailable. Pass config={'api_key': ...}, set WEBAGENTS_API_KEY, or "
    "give the agent an api_key."
)


def _decode_jwt_claims(token: str) -> Dict[str, Any]:
    """Best-effort decode of OUR OWN api key JWT payload (no verification —
    this is the credential we are about to send, not one we received)."""
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return {}


class RobutlerFilesSkill(Skill):
    """
    WebAgents portal file management skill using harmonized API.

    Features:
    - Download and store files from URLs
    - Store files from base64 data
    - List files with agent-based access
    - Agent access is automatically handled by the API
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        cfg = config or {}
        # Base URL resolution: explicit config, then the internal portal URL,
        # then the public API URL, terminating at local dev. Deliberately NOT
        # defaulting to https://robutler.ai: an unconfigured on-prem agent
        # must never send its bearer to the public SaaS host by accident.
        self.portal_url = (
            cfg.get("portal_url")
            or os.getenv("ROBUTLER_INTERNAL_API_URL")
            or os.getenv("ROBUTLER_API_URL")
            or "http://localhost:3000"
        )
        # Base URL used by the chat frontend to serve public content
        self.chat_base_url = cfg.get("chat_base_url") or os.getenv(
            "ROBUTLER_CHAT_URL", "http://localhost:3001"
        )
        # No placeholder fallback: the old 'rok_testapikey' literal produced a
        # silently unauthenticated client, so the first failure a developer
        # saw was never the real one. A missing key is reported loudly in
        # initialize() (the agent's own key is a legitimate late source).
        self.api_key: Optional[str] = cfg.get("api_key") or os.getenv("WEBAGENTS_API_KEY") or os.getenv("ROBUTLER_API_KEY")
        self.agent_api_key: Optional[str] = self.api_key

    async def initialize(self, agent_reference):
        """Initialize with agent reference.

        A missing key is LOGGED here, never raised. Skills initialize lazily
        on the agent's first run, so raising turned a degraded skill (its two
        upload tools and its listing fail; everything else on the agent is
        fine) into a hard failure of the agent's entire first request — for
        agents that carry this skill without using it, or that receive their
        key at runtime. The tools that actually need the credential raise
        instead, with the same message.
        """
        await super().initialize(agent_reference)
        self.agent = agent_reference

        # Prefer the agent's own API key when it has one
        if getattr(agent_reference, "api_key", None):
            self.agent_api_key = agent_reference.api_key
        elif self.api_key:
            self.agent_api_key = self.api_key
        else:
            self._log_missing_key()

    def _log_missing_key(self) -> None:
        try:
            from webagents.utils.logging import get_logger

            get_logger("webagents_files").warning(MISSING_KEY_MESSAGE)
        except Exception:
            pass

    def _require_api_key(self) -> str:
        """The credential, or a loud failure at the point of use."""
        if not self.agent_api_key:
            raise FilesSkillConfigError(MISSING_KEY_MESSAGE)
        return self.agent_api_key

    async def cleanup(self):
        """Nothing persistent to close (sessions are per-request)."""
        return None

    def _get_agent_name_from_context(self) -> str:
        """Get the current agent name from context ('' when unavailable)."""
        try:
            from webagents.server.context.context_vars import get_agent_name

            agent_name = get_agent_name()
            if agent_name:
                return agent_name
            if hasattr(self, "agent") and hasattr(self.agent, "name"):
                return self.agent.name or ""
            return ""
        except Exception:
            return ""

    def _get_agent_id(self) -> Optional[str]:
        """The portal-side agent id, from the agent or from our own key's
        `agent_id` claim (a per-agent key carries it; the sub is the owner)."""
        for attr in ("id", "agent_id"):
            value = getattr(self.agent, attr, None) if getattr(self, "agent", None) else None
            if value:
                return str(value)
        claims = _decode_jwt_claims(self.agent_api_key or "")
        return claims.get("agent_id") or claims.get("sub")

    def _content_principal_id(self) -> Optional[str]:
        """The principal the portal files this skill's uploads under — and
        therefore the only principal whose listing can return them.

        THE ROUND TRIP, precisely. `store_file_from_*` POSTs
        /api/content/upload, which resolves the bearer with
        `authenticateRequest` -> `resolveUserFromVerifiedPayload`: for an
        api-key JWT that is `accessToken.userId`, i.e. the token's OWN
        SUBJECT, and the row is written with `saveUserContent(<subject>, ...)`.
        The listing route `/api/agents/{id}/content` requires a content link
        on the `{id}` in the path. So listing under the AGENT id while
        uploading under the key's subject (the owner, for every per-agent key
        — a per-agent key carries `agent_id` as a claim but keeps the owner as
        `sub`) returns an empty list forever: the upload is invisible to the
        listing that is supposed to show it.

        Using the credential's subject for BOTH halves makes the round trip
        work with the credential the SDK actually has. When the token IS
        agent-subject (a daemon token), the subject is the agent id and this
        is the agent listing, unchanged.

        The other direction — uploading through `POST /api/agents/{id}/content`
        — is not reachable here: that route requires `currentUser.id === agentId`
        (an agent-SUBJECT credential) plus `agentConfigs.canUploadContent`.
        """
        claims = _decode_jwt_claims(self.agent_api_key or "")
        subject = claims.get("sub")
        if subject:
            return str(subject)
        return self._get_agent_id()

    def _rewrite_public_url(self, url: Optional[str]) -> Optional[str]:
        """Rewrite portal public content URLs to chat base URL."""
        if not url:
            return url
        try:
            if url.startswith("/api/content/public"):
                return f"{self.chat_base_url}{url}"
            portal_prefix = f"{self.portal_url}/api/content/public"
            if url.startswith(portal_prefix):
                return url.replace(self.portal_url, self.chat_base_url, 1)
        except Exception:
            return url
        return url

    async def _upload(
        self,
        filename: str,
        content_data: bytes,
        content_type: str,
        visibility: str,
    ) -> Dict[str, Any]:
        """POST multipart to /api/content/upload with the agent's key.

        Raises RuntimeError with the HTTP status AND the endpoint path in the
        message — the old path collapsed every failure to
        "Upload failed: Upload failed" because `ApiResponse.error` is a
        constant that always won the `error or message` expression.
        """
        api_key = self._require_api_key()
        url = f"{self.portal_url}{UPLOAD_PATH}"
        form = aiohttp.FormData()
        form.add_field("file", content_data, filename=filename, content_type=content_type)
        form.add_field("visibility", visibility)
        headers = {"Authorization": f"Bearer {api_key}"}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=form, headers=headers) as response:
                body = await response.text()
                if response.status != 200:
                    raise RuntimeError(
                        f"Upload failed: HTTP {response.status} POST {UPLOAD_PATH} - {body[:500]}"
                    )
                try:
                    return json.loads(body) if body else {}
                except json.JSONDecodeError:
                    raise RuntimeError(
                        f"Upload failed: HTTP 200 POST {UPLOAD_PATH} returned non-JSON body - {body[:200]}"
                    )

    @tool(scope="owner")
    async def store_file_from_url(
        self,
        url: str,
        filename: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        visibility: str = "private"
    ) -> str:
        """
        A tool for downloading and storing a file from a URL. Never use this tool for files that you already own, e.g. URLs returned by list_files.

        Args:
            url: URL to download file from
            filename: Optional custom filename (auto-detected if not provided)
            description: Optional description of the file
            tags: Optional list of tags for the file
            visibility: File visibility - "public", "private", or "shared" (default: "private")

        Returns:
            JSON string with storage result
        """
        try:
            # Download file from URL
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        return json.dumps({
                            "success": False,
                            "error": f"Failed to download file: HTTP {response.status}"
                        })

                    content_data = await response.read()
                    content_type = response.headers.get('content-type', 'application/octet-stream')

                    # Auto-detect filename if not provided
                    if not filename:
                        filename = url.split('/')[-1] or 'downloaded_file'
                        # Remove query parameters
                        filename = filename.split('?')[0]

            # Get agent name for filename prefixing
            agent_name = self._get_agent_name_from_context()

            # Prefix filename with agent name if available
            if agent_name and not filename.startswith(f"{agent_name}_"):
                filename = f"{agent_name}_{filename}"

            data = await self._upload(filename, content_data, content_type, visibility)
            return json.dumps({
                "success": True,
                "id": data.get("id"),
                "filename": data.get("displayName"),
                "url": self._rewrite_public_url(data.get("url")),
                "size": data.get("size"),
                "content_type": data.get("mimeType") or content_type,
                "visibility": visibility,
                "source_url": url
            }, indent=2)

        except RuntimeError as e:
            return json.dumps({"success": False, "error": str(e)})
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Failed to store file from URL: {str(e)}"
            })

    @tool(scope="owner")
    async def store_file_from_base64(
        self,
        filename: str,
        base64_data: str,
        content_type: str = "application/octet-stream",
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        visibility: str = "private"
    ) -> str:
        """
        A tool for storing a file from base64 encoded data.

        Args:
            filename: Name of the file
            base64_data: Base64 encoded file content
            content_type: MIME type of the file
            description: Optional description of the file
            tags: Optional list of tags for the file
            visibility: File visibility - "public", "private", or "shared" (default: "private")

        Returns:
            JSON string with storage result
        """
        try:
            # Decode base64 data
            content_data = base64.b64decode(base64_data)

            # Get agent name for filename prefixing
            agent_name = self._get_agent_name_from_context()

            # Prefix filename with agent name if available
            if agent_name and not filename.startswith(f"{agent_name}_"):
                filename = f"{agent_name}_{filename}"

            data = await self._upload(filename, content_data, content_type, visibility)
            return json.dumps({
                "success": True,
                "id": data.get("id"),
                "filename": data.get("displayName"),
                "url": self._rewrite_public_url(data.get("url")),
                "size": data.get("size"),
                "content_type": data.get("mimeType") or content_type,
                "visibility": visibility
            }, indent=2)

        except RuntimeError as e:
            return json.dumps({"success": False, "error": str(e)})
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Failed to store file from base64: {str(e)}"
            })

    @tool(description="Get public URLs of YOUR reference content. **WHEN TO USE**: Anytime you need to reference YOUR content/images/files in requests to other agents, you MUST call this tool FIRST to get actual URLs. DO NOT describe or invent file names - get real URLs. **USE CASES**: 1) Getting reference image URLs before image generation, 2) Finding style reference URLs, 3) Listing 'my content'/'my files'/'my public content'. **RETURNS**: Full public URLs (e.g., https://robutler.ai/api/content/public/abc123/image.png) that you pass to other agents. **IMPORTANT**: These URLs are for INTERNAL use (passing to other agents) - do NOT show raw URLs to users unless specifically asked or necessary for context.")
    @pricing(credits_per_call=0.005)
    async def list_files(
        self,
        scope: Optional[str] = None
    ) -> str:
        """
        List files accessible by the current agent with scope-based filtering.

        This is the documented name; it was renamed to
        `get_my_public_content_urls` in one copy of this skill while the docs
        and the vendored robutler copy kept `list_files` (F-042).

        Args:
            scope: Optional scope filter - "public", "private", or None (all files)

        Returns:
            JSON string with file list based on scope and ownership
        """
        try:
            from webagents.utils.logging import get_logger
            logger = get_logger('webagents_files')

            api_key = self._require_api_key()
            agent_name = self._get_agent_name_from_context()
            # The SAME principal the uploads are filed under — see
            # _content_principal_id for why listing under the agent id
            # returned an empty list for every stored file.
            principal_id = self._content_principal_id()
            if not principal_id:
                return json.dumps({
                    "success": False,
                    "error": "Cannot resolve the content principal (no agent reference and the API key carries no sub/agent_id claim)"
                })

            # The old '/api/content/agent' path never existed on the portal —
            # it fell into /api/content/[id] with id='agent'. The real route
            # is /api/agents/{id}/content, whose {id} is a PRINCIPAL: the
            # listing is "content reachable by this principal".
            url = f"{self.portal_url}/api/agents/{principal_id}/content"

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Agent content API error: {response.status} - {error_text}")
                        return json.dumps({
                            "success": False,
                            "error": f"Failed to list files: HTTP {response.status} GET /api/agents/{{id}}/content"
                        })

                    data = await response.json()

            items = data.get("items", data.get("content", []))
            files = []
            for item in items:
                visibility = item.get("visibility")
                if scope and visibility and visibility != scope:
                    continue
                files.append({
                    "id": item.get("id"),
                    "filename": item.get("displayName") or item.get("fileName"),
                    "size": item.get("size"),
                    "uploaded_at": item.get("createdAt") or item.get("uploadedAt"),
                    "content_type": item.get("contentType"),
                    "mime_type": item.get("mimeType"),
                    "url": self._rewrite_public_url(item.get("url")),
                    "visibility": visibility,
                })

            return json.dumps({
                "success": True,
                "agent_name": agent_name,
                "total_files": len(files),
                "files": files
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Failed to list files: {str(e)}"
            })

    def get_skill_info(self) -> Dict[str, Any]:
        """Skill information; the tool list is derived from the live registry
        (Skill.get_skill_info) so it can no longer drift from what an LLM can
        actually call."""
        info = super().get_skill_info()
        info.update({
            "name": "RobutlerFilesSkill",
            "description": "File management using harmonized content API",
            "capabilities": [
                "Download and store files from URLs (owner scope only)",
                "Store files from base64 data (owner scope only)",
                "List agent-accessible files",
                "Automatic agent name prefixing for uploaded files",
                "Integration with harmonized content API",
            ],
            "config": {
                "portal_url": self.portal_url,
                "api_key_configured": bool(self.agent_api_key),
                "api_version": "harmonized"
            }
        })
        return info
