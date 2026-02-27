"""Tests for SocialSkill"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from webagents.agents.skills.robutler.social.skill import SocialSkill


@pytest.fixture
def skill():
    s = SocialSkill({"roborum_api_url": "http://test-api:3000", "robutler_api_key": "test-key"})
    agent = MagicMock()
    agent.name = "test-agent"
    agent.api_key = "test-key"

    import asyncio
    asyncio.get_event_loop().run_until_complete(s.initialize(agent))
    return s


class TestSocialSkillInit:
    def test_default_config(self):
        s = SocialSkill()
        assert s.roborum_api_url == "http://localhost:3000"
        assert s.robutler_api_key is None

    def test_custom_config(self):
        s = SocialSkill({"roborum_api_url": "https://example.com", "robutler_api_key": "abc"})
        assert s.roborum_api_url == "https://example.com"
        assert s.robutler_api_key == "abc"

    def test_headers(self, skill):
        h = skill._headers()
        assert h["Authorization"] == "Bearer test-key"
        assert h["Content-Type"] == "application/json"


class TestReadFeed:
    @pytest.mark.asyncio
    async def test_read_feed_success(self, skill):
        mock_posts = {
            "posts": [
                {
                    "author": {"username": "alice"},
                    "title": "Hello World",
                    "content": "Test post content",
                    "channel": {"slug": "general"},
                }
            ]
        }
        with patch.object(skill, "_get", new_callable=AsyncMock, return_value=mock_posts):
            result = await skill.read_feed()
            assert "@alice" in result
            assert "Hello World" in result

    @pytest.mark.asyncio
    async def test_read_feed_empty(self, skill):
        with patch.object(skill, "_get", new_callable=AsyncMock, return_value={"posts": []}):
            result = await skill.read_feed()
            assert "No posts found" in result

    @pytest.mark.asyncio
    async def test_read_feed_error(self, skill):
        with patch.object(skill, "_get", new_callable=AsyncMock, side_effect=Exception("Connection refused")):
            result = await skill.read_feed()
            assert "Failed to read feed" in result

    @pytest.mark.asyncio
    async def test_read_feed_with_filters(self, skill):
        with patch.object(skill, "_get", new_callable=AsyncMock, return_value={"posts": []}) as mock_get:
            await skill.read_feed(mode="trending", limit=5, tag="ai")
            mock_get.assert_called_once_with("/api/feed", {"mode": "trending", "limit": 5, "tag": "ai"})


class TestListChannels:
    @pytest.mark.asyncio
    async def test_list_channels_success(self, skill):
        mock_data = {
            "channels": [
                {"slug": "general", "name": "General", "description": "General discussion", "postCount": 42},
                {"slug": "tech", "name": "Tech", "description": "", "postCount": 10},
            ]
        }
        with patch.object(skill, "_get", new_callable=AsyncMock, return_value=mock_data):
            result = await skill.list_channels()
            assert "//general" in result
            assert "42 posts" in result

    @pytest.mark.asyncio
    async def test_list_channels_empty(self, skill):
        with patch.object(skill, "_get", new_callable=AsyncMock, return_value={"channels": []}):
            result = await skill.list_channels()
            assert "No channels found" in result


class TestReadChannel:
    @pytest.mark.asyncio
    async def test_read_channel_success(self, skill):
        mock_data = {
            "channel": {"name": "General", "slug": "general"},
            "posts": [
                {"author": {"username": "bob"}, "title": "Test", "content": "Hello"},
            ],
        }
        with patch.object(skill, "_get", new_callable=AsyncMock, return_value=mock_data):
            result = await skill.read_channel("general")
            assert "//General" in result
            assert "@bob" in result


class TestCreatePost:
    @pytest.mark.asyncio
    async def test_create_post_success(self, skill):
        mock_data = {"post": {"id": "abc-123"}}
        with patch.object(skill, "_post", new_callable=AsyncMock, return_value=mock_data):
            result = await skill.create_post("general", "My Title", "Content here", tags="ai,ml")
            assert "Post created" in result
            assert "abc-123" in result

    @pytest.mark.asyncio
    async def test_create_post_failure(self, skill):
        resp = MagicMock()
        resp.text = "Channel not found"
        with patch.object(
            skill, "_post", new_callable=AsyncMock,
            side_effect=httpx.HTTPStatusError("404", request=MagicMock(), response=resp),
        ):
            result = await skill.create_post("nonexistent", "Title", "Content")
            assert "Failed" in result


class TestCommentOnPost:
    @pytest.mark.asyncio
    async def test_comment_success(self, skill):
        with patch.object(skill, "_post", new_callable=AsyncMock, return_value={}):
            result = await skill.comment_on_post("post-123", "Great post!")
            assert "Comment added" in result


class TestGetPost:
    @pytest.mark.asyncio
    async def test_get_post_success(self, skill):
        mock_data = {
            "post": {
                "author": {"username": "charlie"},
                "title": "Deep Dive",
                "content": "Long form content here...",
                "comments": [
                    {"author": {"username": "dave"}, "content": "Nice!"},
                ],
            }
        }
        with patch.object(skill, "_get", new_callable=AsyncMock, return_value=mock_data):
            result = await skill.get_post("post-456")
            assert "Deep Dive" in result
            assert "@charlie" in result
            assert "@dave" in result
            assert "1 comment" in result
