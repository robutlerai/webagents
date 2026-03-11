"""
Tests for LLMProxySkill - UAMP LLM Proxy via WebSocket

Tests the LLM proxy skill that connects daemon agents to the platform's
UAMP LLM proxy for completions without direct API keys.
"""

import pytest
import json
import os
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from webagents.agents.skills.core.llm.proxy.skill import (
    LLMProxySkill,
    LLMProxyError,
    PaymentRequiredError,
    DEFAULT_PROXY_URL,
    DEFAULT_MODEL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockWebSocket:
    """Fake WebSocket that records sent messages and replays scripted responses."""

    def __init__(self, responses=None):
        self.sent: list[str] = []
        self._responses = list(responses or [])
        self._closed = False

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def recv(self) -> str:
        if not self._responses:
            await asyncio.sleep(60)
            raise asyncio.TimeoutError()
        return self._responses.pop(0)

    async def close(self) -> None:
        self._closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


def _event(event_type: str, **kwargs) -> str:
    return json.dumps({'type': event_type, **kwargs})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLLMProxySkillInit:
    """Test creation and default configuration."""

    def test_creates_with_default_config(self):
        skill = LLMProxySkill()
        assert skill.proxy_url == DEFAULT_PROXY_URL
        assert skill.model == DEFAULT_MODEL
        assert skill.temperature == 0.7
        assert skill.max_tokens is None
        assert skill.payment_token is None

    def test_creates_with_custom_config(self):
        skill = LLMProxySkill({
            'proxy_url': 'ws://custom:9000/llm',
            'model': 'claude-4',
            'temperature': 0.3,
            'max_tokens': 2048,
            'payment_token': 'tok_abc',
        })
        assert skill.proxy_url == 'ws://custom:9000/llm'
        assert skill.model == 'claude-4'
        assert skill.temperature == 0.3
        assert skill.max_tokens == 2048
        assert skill.payment_token == 'tok_abc'

    def test_uses_env_var_for_proxy_url(self):
        with patch.dict(os.environ, {'ROBUTLER_LLM_PROXY_URL': 'ws://env:4000/llm'}):
            skill = LLMProxySkill()
            assert skill.proxy_url == 'ws://env:4000/llm'

    def test_config_proxy_url_takes_precedence_over_env(self):
        with patch.dict(os.environ, {'ROBUTLER_LLM_PROXY_URL': 'ws://env:4000/llm'}):
            skill = LLMProxySkill({'proxy_url': 'ws://explicit:5000/llm'})
            assert skill.proxy_url == 'ws://explicit:5000/llm'

    def test_uses_env_var_for_payment_token(self):
        with patch.dict(os.environ, {'ROBUTLER_PAYMENT_TOKEN': 'env_tok_123'}):
            skill = LLMProxySkill()
            assert skill.payment_token == 'env_tok_123'


class TestLLMProxySkillConnect:
    """Test WebSocket connection and session flow."""

    @pytest.mark.asyncio
    async def test_connects_to_proxy(self):
        ws = MockWebSocket(responses=[
            _event('session.created', session_id='sess_1'),
            _event('response.created', response_id='resp_1'),
            _event('response.done', response={
                'output': [],
                'usage': {'input_tokens': 5, 'output_tokens': 3, 'total_tokens': 8},
            }),
        ])

        with patch('webagents.agents.skills.core.llm.proxy.skill.websockets.client.connect',
                    return_value=ws):
            skill = LLMProxySkill()
            result = await skill.chat_completion(
                messages=[{'role': 'user', 'content': 'Hello'}],
            )

        assert result['object'] == 'chat.completion'
        assert len(ws.sent) >= 2  # session.create + input.text + response.create

        session_create = json.loads(ws.sent[0])
        assert session_create['type'] == 'session.create'
        assert session_create['session']['modalities'] == ['text']

    @pytest.mark.asyncio
    async def test_streams_response(self):
        ws = MockWebSocket(responses=[
            _event('session.created', session_id='sess_1'),
            _event('response.created', response_id='resp_1'),
            _event('response.delta', delta={'type': 'text', 'text': 'Hello'}),
            _event('response.delta', delta={'type': 'text', 'text': ' World'}),
            _event('response.done', response={
                'output': [],
                'usage': {'input_tokens': 5, 'output_tokens': 2, 'total_tokens': 7},
            }),
        ])

        chunks = []
        with patch('webagents.agents.skills.core.llm.proxy.skill.websockets.client.connect',
                    return_value=ws):
            skill = LLMProxySkill()
            async for chunk in skill.chat_completion_stream(
                messages=[{'role': 'user', 'content': 'Hi'}],
            ):
                chunks.append(chunk)

        text_chunks = [
            c['choices'][0]['delta']['content']
            for c in chunks
            if c.get('choices', [{}])[0].get('delta', {}).get('content')
        ]
        assert text_chunks == ['Hello', ' World']

        final = chunks[-1]
        assert final['choices'][0]['finish_reason'] == 'stop'


class TestLLMProxySkillCancel:
    """Test abort / cancel behaviour."""

    @pytest.mark.asyncio
    async def test_handles_cancel(self):
        ws = MockWebSocket(responses=[
            _event('session.created', session_id='sess_1'),
            _event('response.created', response_id='resp_1'),
            _event('response.delta', delta={'type': 'text', 'text': 'partial'}),
            _event('response.cancelled'),
        ])

        chunks = []
        with patch('webagents.agents.skills.core.llm.proxy.skill.websockets.client.connect',
                    return_value=ws):
            skill = LLMProxySkill()
            async for chunk in skill.chat_completion_stream(
                messages=[{'role': 'user', 'content': 'cancel me'}],
            ):
                chunks.append(chunk)

        assert any(c['choices'][0].get('finish_reason') == 'stop' for c in chunks)

    @pytest.mark.asyncio
    async def test_cancel_response_sends_event(self):
        ws = MockWebSocket()
        skill = LLMProxySkill()
        await skill.cancel_response(ws)

        assert len(ws.sent) == 1
        cancel_msg = json.loads(ws.sent[0])
        assert cancel_msg['type'] == 'response.cancel'


class TestLLMProxySkillPayment:
    """Test payment handling."""

    @pytest.mark.asyncio
    async def test_payment_required_with_token_submits(self):
        ws = MockWebSocket(responses=[
            _event('session.created', session_id='sess_1'),
            _event('response.created', response_id='resp_1'),
            _event('payment.required', requirements={'amount': '0.50', 'currency': 'USD'}),
            _event('response.delta', delta={'type': 'text', 'text': 'paid result'}),
            _event('response.done', response={'output': [], 'usage': {}}),
        ])

        with patch('webagents.agents.skills.core.llm.proxy.skill.websockets.client.connect',
                    return_value=ws):
            skill = LLMProxySkill({'payment_token': 'tok_pay'})
            result = await skill.chat_completion(
                messages=[{'role': 'user', 'content': 'buy'}],
            )

        payment_sent = [json.loads(m) for m in ws.sent if 'payment.submit' in m]
        assert len(payment_sent) == 1
        assert payment_sent[0]['payment']['token'] == 'tok_pay'
        assert payment_sent[0]['payment']['amount'] == '0.50'
        assert result['choices'][0]['message']['content'] == 'paid result'

    @pytest.mark.asyncio
    async def test_payment_required_without_token_raises(self):
        ws = MockWebSocket(responses=[
            _event('session.created', session_id='sess_1'),
            _event('response.created', response_id='resp_1'),
            _event('payment.required', requirements={'amount': '1.00', 'currency': 'USD'}),
        ])

        with patch('webagents.agents.skills.core.llm.proxy.skill.websockets.client.connect',
                    return_value=ws):
            skill = LLMProxySkill()  # no payment_token
            with pytest.raises(PaymentRequiredError):
                await skill.chat_completion(
                    messages=[{'role': 'user', 'content': 'pay'}],
                )


class TestLLMProxySkillErrors:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_response_error_raises(self):
        ws = MockWebSocket(responses=[
            _event('session.created', session_id='sess_1'),
            _event('response.created', response_id='resp_1'),
            _event('response.error', error={'code': 'rate_limit', 'message': 'Too fast'}),
        ])

        with patch('webagents.agents.skills.core.llm.proxy.skill.websockets.client.connect',
                    return_value=ws):
            skill = LLMProxySkill()
            with pytest.raises(LLMProxyError, match='Too fast'):
                await skill.chat_completion(
                    messages=[{'role': 'user', 'content': 'spam'}],
                )

    @pytest.mark.asyncio
    async def test_payment_error_raises(self):
        ws = MockWebSocket(responses=[
            _event('session.created', session_id='sess_1'),
            _event('response.created', response_id='resp_1'),
            _event('payment.error', code='insufficient_funds', message='Not enough'),
        ])

        with patch('webagents.agents.skills.core.llm.proxy.skill.websockets.client.connect',
                    return_value=ws):
            skill = LLMProxySkill()
            with pytest.raises(LLMProxyError, match='Not enough'):
                await skill.chat_completion(
                    messages=[{'role': 'user', 'content': 'buy'}],
                )


class TestLLMProxySkillHelpers:
    """Test internal helper methods."""

    def test_build_extensions_includes_payment_token(self):
        skill = LLMProxySkill({'payment_token': 'tok_ext'})
        ext = skill._build_extensions('gpt-4')
        assert ext['X-Payment-Token'] == 'tok_ext'
        assert ext['model'] == 'gpt-4'

    def test_build_extensions_temperature(self):
        skill = LLMProxySkill({'temperature': 0.2})
        ext = skill._build_extensions('auto/balanced')
        assert ext['temperature'] == 0.2

    def test_convert_tools_openai_format(self):
        tools = [{
            'type': 'function',
            'function': {
                'name': 'search',
                'description': 'Search the web',
                'parameters': {'type': 'object', 'properties': {'q': {'type': 'string'}}},
            },
        }]
        result = LLMProxySkill._convert_tools(tools)
        assert len(result) == 1
        assert result[0]['name'] == 'search'
        assert result[0]['description'] == 'Search the web'

    def test_delta_to_openai_chunk_text(self):
        chunk = LLMProxySkill._delta_to_openai_chunk(
            {'type': 'text', 'text': 'hello'},
            'resp_1', 'gpt-4', 1,
        )
        assert chunk is not None
        assert chunk['choices'][0]['delta']['content'] == 'hello'

    def test_delta_to_openai_chunk_tool_call(self):
        chunk = LLMProxySkill._delta_to_openai_chunk(
            {'type': 'tool_call', 'tool_call': {'id': 'tc1', 'name': 'fn', 'arguments': '{}'}},
            'resp_1', 'gpt-4', 1,
        )
        assert chunk is not None
        assert chunk['choices'][0]['delta']['tool_calls'][0]['function']['name'] == 'fn'

    def test_delta_to_openai_chunk_unknown_returns_none(self):
        chunk = LLMProxySkill._delta_to_openai_chunk(
            {'type': 'audio', 'data': 'binary'},
            'resp_1', 'gpt-4', 1,
        )
        assert chunk is None
