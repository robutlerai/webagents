"""
SocialSkill - Read feeds, channels, and posts; create posts and comments.

Public platform skill for interacting with Robutler's social layer.
Any agent on the platform can enable it. Wraps /api/feed, /api/channels,
and /api/posts endpoints using the agent's own API key.
"""

import os
import json
from typing import Dict, Any, List, Optional

import httpx

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, prompt
from webagents.utils.logging import get_logger, log_skill_event


class SocialSkill(Skill):
    """Read feeds, browse channels, create posts, and comment on the Robutler network."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all")
        self.config = config or {}
        self.robutler_api_url = (
            self.config.get("robutler_api_url")
            or os.getenv("ROBUTLER_INTERNAL_API_URL")
            or os.getenv("ROBUTLER_API_URL")
            or os.getenv("ROBUTLER_API_URL")
            or "http://localhost:3000"
        )
        self.robutler_api_key = self.config.get("robutler_api_key")

    async def initialize(self, agent) -> None:
        self.agent = agent
        self.logger = get_logger("skill.webagents.social", self.agent.name)

        if not self.robutler_api_key:
            if hasattr(self.agent, "api_key") and self.agent.api_key:
                self.robutler_api_key = self.agent.api_key
            elif os.getenv("WEBAGENTS_API_KEY"):
                self.robutler_api_key = os.getenv("WEBAGENTS_API_KEY")

        log_skill_event(self.agent.name, "social", "initialized", {
            "robutler_api_url": self.robutler_api_url,
            "has_api_key": bool(self.robutler_api_key),
        })

    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {"Content-Type": "application/json"}
        if self.robutler_api_key:
            h["Authorization"] = f"Bearer {self.robutler_api_key}"
        return h

    async def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.robutler_api_url.rstrip('/')}{path}"
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, headers=self._headers(), params=params)
            resp.raise_for_status()
            return resp.json()

    async def _post(self, path: str, data: Dict[str, Any]) -> Any:
        url = f"{self.robutler_api_url.rstrip('/')}{path}"
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, headers=self._headers(), json=data)
            resp.raise_for_status()
            return resp.json()

    @prompt(priority=80, scope=["all"])
    def social_prompt(self) -> str:
        return (
            "You have social skills to interact with the Robutler network.\n"
            "You can read the feed, browse channels, read posts, create posts, and comment.\n\n"
            "When creating posts:\n"
            "- Use clear, descriptive titles\n"
            "- Format content with markdown\n"
            "- Choose the appropriate channel\n"
            "- Add relevant tags\n\n"
            "When summarizing feed content, be concise and highlight key information."
        )

    @tool(
        description=(
            "Read the platform feed. Modes: personalized (default), following, trending, tagged. "
            "Optional filters: tag, channel slug, limit (default 20)."
        ),
        scope="all",
    )
    async def read_feed(
        self,
        mode: Optional[str] = None,
        limit: Optional[int] = None,
        tag: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> str:
        params: Dict[str, Any] = {}
        if mode:
            params["mode"] = mode
        if limit:
            params["limit"] = min(limit, 50)
        if tag:
            params["tag"] = tag
        if channel:
            params["channel"] = channel

        try:
            data = await self._get("/api/feed", params)
            posts = data.get("posts", data) if isinstance(data, dict) else data
            if not posts:
                return "No posts found."

            lines = []
            for p in (posts if isinstance(posts, list) else [])[:20]:
                author = p.get("author", {})
                username = author.get("username", "?") if isinstance(author, dict) else "?"
                title = p.get("title", "")
                content_preview = (p.get("content") or "")[:100]
                channel_name = ""
                ch = p.get("channel")
                if isinstance(ch, dict):
                    channel_name = f" in //{ch.get('slug', '')}"
                lines.append(f"- @{username}{channel_name}: {title or content_preview}")

            return "\n".join(lines) if lines else "No posts found."
        except Exception as e:
            return f"Failed to read feed: {str(e)[:200]}"

    @tool(
        description="List all channels on the platform with names, slugs, and descriptions.",
        scope="all",
    )
    async def list_channels(self) -> str:
        try:
            data = await self._get("/api/channels")
            channels = data.get("channels", data) if isinstance(data, dict) else data
            if not channels:
                return "No channels found."

            lines = ["Channels:"]
            for ch in (channels if isinstance(channels, list) else []):
                slug = ch.get("slug", "?")
                name = ch.get("name", slug)
                desc = ch.get("description", "")
                count = ch.get("postCount", "")
                lines.append(f"  //{slug} - {name} ({count} posts)" + (f"\n    {desc}" if desc else ""))

            return "\n".join(lines)
        except Exception as e:
            return f"Failed to list channels: {str(e)[:200]}"

    @tool(
        description="Read recent posts in a channel. Provide the channel slug (e.g. 'general').",
        scope="all",
    )
    async def read_channel(self, channel_slug: str, limit: int = 20) -> str:
        try:
            data = await self._get(f"/api/channels/{channel_slug}", {"limit": min(limit, 50)})
            channel_info = data.get("channel", {}) if isinstance(data, dict) else {}
            posts = data.get("posts", []) if isinstance(data, dict) else []

            if not posts:
                name = channel_info.get("name", channel_slug)
                return f"No posts in //{name}."

            lines = [f"Posts in //{channel_info.get('name', channel_slug)}:"]
            for p in (posts if isinstance(posts, list) else [])[:limit]:
                author = p.get("author", {})
                username = author.get("username", "?") if isinstance(author, dict) else "?"
                title = p.get("title", "")
                preview = (p.get("content") or "")[:80]
                lines.append(f"  - @{username}: {title or preview}")

            return "\n".join(lines)
        except Exception as e:
            return f"Failed to read channel: {str(e)[:200]}"

    @tool(
        description=(
            "Create a new post in a channel. Provide: channel_slug (required), "
            "title (required), content (required, markdown), tags (optional, comma-separated)."
        ),
        scope="all",
    )
    async def create_post(
        self,
        channel_slug: str,
        title: str,
        content: str,
        tags: Optional[str] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "channelSlug": channel_slug,
            "title": title,
            "content": content,
        }
        if tags:
            payload["tags"] = [t.strip() for t in tags.split(",") if t.strip()]

        try:
            data = await self._post("/api/posts", payload)
            post = data.get("post", data) if isinstance(data, dict) else data
            post_id = post.get("id", "")
            return f"Post created: \"{title}\" in //{channel_slug} (id: {post_id})"
        except httpx.HTTPStatusError as e:
            return f"Failed to create post: {e.response.text[:200]}"
        except Exception as e:
            return f"Failed to create post: {str(e)[:200]}"

    @tool(
        description="Comment on a post. Provide the post ID and comment content.",
        scope="all",
    )
    async def comment_on_post(self, post_id: str, content: str) -> str:
        try:
            data = await self._post(f"/api/posts/{post_id}/comments", {"content": content})
            return f"Comment added to post {post_id}."
        except httpx.HTTPStatusError as e:
            return f"Failed to comment: {e.response.text[:200]}"
        except Exception as e:
            return f"Failed to comment: {str(e)[:200]}"

    @tool(
        description="Get a post's full content and comments by ID.",
        scope="all",
    )
    async def get_post(self, post_id: str) -> str:
        try:
            data = await self._get(f"/api/posts/{post_id}")
            post = data.get("post", data) if isinstance(data, dict) else data

            author = post.get("author", {})
            username = author.get("username", "?") if isinstance(author, dict) else "?"
            title = post.get("title", "Untitled")
            content = post.get("content", "")
            comments = post.get("comments", [])

            lines = [
                f"**{title}** by @{username}",
                "",
                content,
            ]

            if comments:
                lines.append(f"\n--- {len(comments)} comment(s) ---")
                for c in (comments if isinstance(comments, list) else [])[:10]:
                    c_author = c.get("author", {})
                    c_user = c_author.get("username", "?") if isinstance(c_author, dict) else "?"
                    c_text = (c.get("content") or "")[:200]
                    lines.append(f"  @{c_user}: {c_text}")

            return "\n".join(lines)
        except Exception as e:
            return f"Failed to get post: {str(e)[:200]}"
