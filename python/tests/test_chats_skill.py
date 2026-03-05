"""
Unit tests for ChatsSkill (Robutler chat metadata enrichment + unreads).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from webagents.agents.skills.robutler.chats import ChatsSkill


def _make_aiohttp_session(responses):
    """Create a mock aiohttp.ClientSession that returns different responses per URL pattern."""
    mock_session = MagicMock()
    call_count = {"n": 0}

    def make_resp(resp_data, ok=True, status=200):
        mock_resp = AsyncMock()
        mock_resp.ok = ok
        mock_resp.status = status
        mock_resp.json = AsyncMock(return_value=resp_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)
        return mock_resp

    resp_list = [make_resp(**r) for r in responses]

    def side_effect(*args, **kwargs):
        idx = min(call_count["n"], len(resp_list) - 1)
        call_count["n"] += 1
        return resp_list[idx]

    mock_session.get = MagicMock(side_effect=side_effect)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    return mock_session


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.metadata = {}
    agent.api_key = "test-api-key"
    agent.name = "test-agent"
    return agent


@pytest.fixture
def robutler_chats_response():
    return {
        "chats": [
            {
                "id": "chat-uuid-1",
                "type": "dm",
                "name": None,
                "participants": [{"username": "alice"}, {"username": "bot"}],
                "lastMessageAt": "2025-02-05T12:00:00Z",
            },
            {
                "id": "chat-uuid-2",
                "type": "group",
                "name": "Team",
                "participants": [{"username": "bob"}, {"username": "alice"}, {"username": "bot"}],
                "lastMessageAt": "2025-02-05T11:00:00Z",
            },
        ]
    }


@pytest.fixture
def unreads_response():
    return {
        "unreads": [
            {
                "chat_id": "chat-uuid-1",
                "unread_count": 3,
                "last_read_at": "2025-02-05T10:00:00Z",
                "chat_type": "dm",
                "last_message_at": "2025-02-05T12:00:00Z",
            },
            {
                "chat_id": "chat-uuid-2",
                "unread_count": 1,
                "last_read_at": None,
                "chat_type": "group",
                "last_message_at": "2025-02-05T11:00:00Z",
            },
        ]
    }


@pytest.mark.asyncio
async def test_initialize_sets_chats_metadata(mock_agent, robutler_chats_response, unreads_response):
    skill = ChatsSkill(config={"robutler_url": "https://api.robutler.test"})
    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_session = _make_aiohttp_session([
            {"resp_data": robutler_chats_response},
            {"resp_data": unreads_response},
        ])
        mock_session_cls.return_value = mock_session

        await skill.initialize(mock_agent)

    assert "chats" in mock_agent.metadata
    chats = mock_agent.metadata["chats"]
    assert len(chats) == 2
    assert mock_agent.chats == chats

    chat1 = next(c for c in chats if c["id"] == "chat-uuid-1")
    assert chat1["type"] == "dm"
    assert chat1["url"] == "https://api.robutler.test/chats/chat-uuid-1"
    assert "completions" in chat1["transports"]
    assert chat1["transports"]["completions"] == "https://api.robutler.test/api/chats/chat-uuid-1/completions"
    assert chat1["transports"]["uamp"] == "wss://api.robutler.test/chats/chat-uuid-1/uamp"
    assert set(chat1["participants"]) == {"alice", "bot"}
    assert chat1["last_message_at"] == "2025-02-05T12:00:00Z"

    chat2 = next(c for c in chats if c["id"] == "chat-uuid-2")
    assert chat2["name"] == "Team"
    assert set(chat2["participants"]) == {"bob", "alice", "bot"}


@pytest.mark.asyncio
async def test_initialize_skips_when_no_api_key(mock_agent):
    mock_agent.api_key = None
    with patch.dict("os.environ", {}, clear=False):
        skill = ChatsSkill(config={})
        await skill.initialize(mock_agent)
    assert "chats" not in mock_agent.metadata


@pytest.mark.asyncio
async def test_initialize_uses_env_api_key_when_agent_has_none(mock_agent, robutler_chats_response, unreads_response):
    mock_agent.api_key = None
    skill = ChatsSkill(config={"robutler_url": "https://api.robutler.test"})
    with patch.dict("os.environ", {"WEBAGENTS_API_KEY": "env-key"}, clear=False):
        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = _make_aiohttp_session([
                {"resp_data": robutler_chats_response},
                {"resp_data": unreads_response},
            ])
            mock_session_cls.return_value = mock_session

            await skill.initialize(mock_agent)

    assert mock_agent.metadata["chats"]
    # First call should use env key
    call_args = mock_session.get.call_args_list[0]
    assert call_args[1]["headers"]["Authorization"] == "Bearer env-key"


@pytest.mark.asyncio
async def test_initialize_skips_chat_without_id(mock_agent, unreads_response):
    skill = ChatsSkill(config={"robutler_url": "https://api.robutler.test"})
    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_session = _make_aiohttp_session([
            {"resp_data": {"chats": [{"type": "dm"}]}},  # no id
            {"resp_data": unreads_response},
        ])
        mock_session_cls.return_value = mock_session

        await skill.initialize(mock_agent)

    assert mock_agent.metadata["chats"] == []


@pytest.mark.asyncio
async def test_initialize_handles_api_error_gracefully(mock_agent):
    skill = ChatsSkill(config={"robutler_url": "https://api.robutler.test"})
    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_session = _make_aiohttp_session([
            {"resp_data": {}, "ok": False, "status": 401},
        ])
        mock_session_cls.return_value = mock_session

        await skill.initialize(mock_agent)

    # Should not set chats on error (exception caught, logger.warning called)
    assert "chats" not in mock_agent.metadata or mock_agent.metadata.get("chats") is None


# ------------------------------------------------------------------
# Unreads tests
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_initialize_fetches_unreads(mock_agent, robutler_chats_response, unreads_response):
    skill = ChatsSkill(config={"robutler_url": "https://api.robutler.test"})
    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_session = _make_aiohttp_session([
            {"resp_data": robutler_chats_response},
            {"resp_data": unreads_response},
        ])
        mock_session_cls.return_value = mock_session

        await skill.initialize(mock_agent)

    assert len(skill._cached_unreads) == 2
    assert skill._cached_unreads[0]["chat_id"] == "chat-uuid-1"
    assert skill._cached_unreads[0]["unread_count"] == 3


@pytest.mark.asyncio
async def test_get_unreads_tool_returns_cached(mock_agent, robutler_chats_response, unreads_response):
    skill = ChatsSkill(config={"robutler_url": "https://api.robutler.test"})
    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_session = _make_aiohttp_session([
            {"resp_data": robutler_chats_response},
            {"resp_data": unreads_response},
        ])
        mock_session_cls.return_value = mock_session

        await skill.initialize(mock_agent)

    # Call get_unreads without refresh — should use cache, no new API call
    result = await skill.get_unreads(refresh=False)
    assert "chat-uuid-1" in result
    assert "3 unread" in result
    assert "chat-uuid-2" in result
    assert "1 unread" in result


@pytest.mark.asyncio
async def test_get_unreads_tool_empty(mock_agent, robutler_chats_response):
    skill = ChatsSkill(config={"robutler_url": "https://api.robutler.test"})
    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_session = _make_aiohttp_session([
            {"resp_data": robutler_chats_response},
            {"resp_data": {"unreads": []}},
        ])
        mock_session_cls.return_value = mock_session

        await skill.initialize(mock_agent)

    result = await skill.get_unreads()
    assert result == "No unread messages."


@pytest.mark.asyncio
async def test_get_unreads_tool_refresh(mock_agent, robutler_chats_response, unreads_response):
    skill = ChatsSkill(config={"robutler_url": "https://api.robutler.test"})

    # Initialize with empty unreads, then refresh with data
    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_session = _make_aiohttp_session([
            {"resp_data": robutler_chats_response},
            {"resp_data": {"unreads": []}},
        ])
        mock_session_cls.return_value = mock_session
        await skill.initialize(mock_agent)

    assert skill._cached_unreads == []

    # Now call refresh
    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_session = _make_aiohttp_session([
            {"resp_data": unreads_response},
        ])
        mock_session_cls.return_value = mock_session

        result = await skill.get_unreads(refresh=True)

    assert "3 unread" in result
    assert len(skill._cached_unreads) == 2


@pytest.mark.asyncio
async def test_refresh_chats_tool(mock_agent, robutler_chats_response, unreads_response):
    skill = ChatsSkill(config={"robutler_url": "https://api.robutler.test"})
    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_session = _make_aiohttp_session([
            {"resp_data": robutler_chats_response},
            {"resp_data": unreads_response},
        ])
        mock_session_cls.return_value = mock_session
        await skill.initialize(mock_agent)

    # Call refresh_chats with updated data
    updated_chats = {
        "chats": [
            {"id": "chat-uuid-3", "type": "dm", "name": "New Chat", "participants": [], "lastMessageAt": None},
        ]
    }
    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_session = _make_aiohttp_session([
            {"resp_data": updated_chats},
        ])
        mock_session_cls.return_value = mock_session

        result = await skill.refresh_chats()

    assert "1 chats" in result
    assert "chat-uuid-3" in result
    assert len(mock_agent.metadata["chats"]) == 1
    assert mock_agent.chats[0]["id"] == "chat-uuid-3"


@pytest.mark.asyncio
async def test_get_unreads_no_api_key(mock_agent):
    mock_agent.api_key = None
    env_clear = {"WEBAGENTS_API_KEY": "", "SERVICE_TOKEN": "", "ROBUTLER_API_URL": "", "ROBUTLER_INTERNAL_API_URL": ""}
    skill = ChatsSkill(config={})
    with patch.dict("os.environ", env_clear):
        await skill.initialize(mock_agent)

    result = await skill.get_unreads()
    assert "No API key" in result


@pytest.mark.asyncio
async def test_cleanup_cancels_poll_task(mock_agent, robutler_chats_response, unreads_response):
    skill = ChatsSkill(config={
        "robutler_url": "https://api.robutler.test",
        "poll_unreads": True,
        "poll_interval": 3600,  # Long interval so it doesn't fire
    })
    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_session = _make_aiohttp_session([
            {"resp_data": robutler_chats_response},
            {"resp_data": unreads_response},
        ])
        mock_session_cls.return_value = mock_session
        await skill.initialize(mock_agent)

    assert skill._poll_task is not None
    assert not skill._poll_task.done()

    await skill.cleanup()
    assert skill._poll_task.done()
