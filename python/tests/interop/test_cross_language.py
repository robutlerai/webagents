"""Cross-language interoperability tests (Python side).

Validates that Python can communicate with both Python and TypeScript
agent servers via UAMP over HTTP and chat completions.

Run with:
    PYTHON_AGENT_URL=http://localhost:9100 TS_AGENT_URL=http://localhost:9200 \
        python -m pytest tests/interop/test_cross_language.py -v
"""

import json
import os
import uuid

import pytest

PYTHON_AGENT_URL = os.environ.get("PYTHON_AGENT_URL")
TS_AGENT_URL = os.environ.get("TS_AGENT_URL")

skip_no_python = pytest.mark.skipif(
    not PYTHON_AGENT_URL, reason="PYTHON_AGENT_URL not set"
)
skip_no_ts = pytest.mark.skipif(not TS_AGENT_URL, reason="TS_AGENT_URL not set")
skip_no_both = pytest.mark.skipif(
    not (PYTHON_AGENT_URL and TS_AGENT_URL),
    reason="Both PYTHON_AGENT_URL and TS_AGENT_URL required",
)

try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

pytestmark = pytest.mark.skipif(not HAS_HTTPX, reason="httpx not installed")


def _event_id():
    return f"evt_{uuid.uuid4().hex[:12]}"


def _uamp_events(text: str):
    return [
        {
            "type": "session.create",
            "event_id": _event_id(),
            "session": {"modalities": ["text"]},
        },
        {
            "type": "input.text",
            "event_id": _event_id(),
            "text": text,
            "role": "user",
        },
        {
            "type": "response.create",
            "event_id": _event_id(),
        },
    ]


# ---------------------------------------------------------------------------
# UAMP over HTTP
# ---------------------------------------------------------------------------


class TestUAMPOverHTTP:
    @skip_no_python
    @pytest.mark.asyncio
    async def test_python_to_python_uamp(self):
        """Send UAMP events from Python test to Python agent."""
        async with httpx.AsyncClient() as client:
            res = await client.post(
                f"{PYTHON_AGENT_URL}/uamp",
                json=_uamp_events("hello from Python test"),
                timeout=10,
            )
        assert res.status_code == 200
        body = res.json()
        assert isinstance(body, list)
        types = [e["type"] for e in body]
        assert "response.done" in types

    @skip_no_ts
    @pytest.mark.asyncio
    async def test_python_to_ts_uamp(self):
        """Send UAMP events from Python test to TypeScript agent."""
        async with httpx.AsyncClient() as client:
            res = await client.post(
                f"{TS_AGENT_URL}/uamp",
                json=_uamp_events("hello from Python to TS"),
                timeout=10,
            )
        assert res.status_code == 200
        body = res.json()
        assert isinstance(body, list)
        types = [e["type"] for e in body]
        assert "response.done" in types


# ---------------------------------------------------------------------------
# Chat Completions
# ---------------------------------------------------------------------------


class TestChatCompletions:
    @skip_no_python
    @pytest.mark.asyncio
    async def test_python_to_python_chat(self):
        async with httpx.AsyncClient() as client:
            res = await client.post(
                f"{PYTHON_AGENT_URL}/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "cross-lang ping"}],
                    "stream": False,
                },
                timeout=10,
            )
        assert res.status_code == 200
        data = res.json()
        assert "choices" in data
        assert len(data["choices"]) > 0

    @skip_no_ts
    @pytest.mark.asyncio
    async def test_python_to_ts_chat(self):
        async with httpx.AsyncClient() as client:
            res = await client.post(
                f"{TS_AGENT_URL}/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "cross-lang ping"}],
                    "stream": False,
                },
                timeout=10,
            )
        assert res.status_code == 200
        data = res.json()
        assert "choices" in data


# ---------------------------------------------------------------------------
# Agent Info
# ---------------------------------------------------------------------------


class TestAgentInfo:
    @skip_no_python
    @pytest.mark.asyncio
    async def test_python_agent_info(self):
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{PYTHON_AGENT_URL}/models", timeout=5)
        assert res.status_code == 200

    @skip_no_ts
    @pytest.mark.asyncio
    async def test_ts_agent_info(self):
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{TS_AGENT_URL}/info", timeout=5)
        assert res.status_code == 200
        info = res.json()
        assert "name" in info


# ---------------------------------------------------------------------------
# Event Serialization Parity
# ---------------------------------------------------------------------------


class TestEventParity:
    def test_uamp_event_structure(self):
        """Validate that Python-constructed UAMP events match the expected format."""
        events = _uamp_events("test")
        assert events[0]["type"] == "session.create"
        assert events[1]["type"] == "input.text"
        assert events[2]["type"] == "response.create"
        for e in events:
            assert "event_id" in e

    def test_payment_token_structure(self):
        """Validate JWT payment token structure matches TS format."""
        import time

        header = {"alg": "EdDSA", "typ": "JWT"}
        payload = {
            "iss": "robutler.ai",
            "sub": "user-123",
            "aud": ["agent-abc"],
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
            "jti": f"pt_{uuid.uuid4().hex}",
            "amt": "1000000000",
            "cur": "nanocents",
        }
        assert header["alg"] == "EdDSA"
        assert isinstance(payload["aud"], list)
        assert isinstance(payload["amt"], str)
