"""
The LLM proxy skill's wire fixes from the 2026-09-27 real-model pass (the
chat-fixes lane):

  * the tools travel in `response.create` (`response.tools`), where the
    platform and the TypeScript client have always put them; in
    `session.tools` the platform never read them, so every Python agent on
    Robutler's models ran with no tools and the model guessed;
  * every tool call keeps its own `index`, keyed by call id, so two parallel
    calls are not glued into one (`dir_list({}{})`);
  * a call id keeps the platform's `|ts:` thought-signature suffix through
    the chunk and back into the next request;
  * an empty completion whose finish reason is MALFORMED_FUNCTION_CALL or
    UNEXPECTED_TOOL_CALL is sent once more, on a fresh socket, and the last
    chunk says why the provider stopped (`webagents_finish`);
  * a thinking event with no content is not sent as a literal `<think></think>`.
"""

import asyncio
import json
from unittest.mock import patch

import pytest

from webagents.agents.skills.core.llm.proxy.skill import LLMProxySkill

CONNECT = 'webagents.agents.skills.core.llm.proxy.skill.websockets.client.connect'


class MockWebSocket:
    """A socket that records what was sent and replays scripted events."""

    def __init__(self, responses):
        self.sent = []
        self._responses = list(responses)
        self.closed = False

    async def send(self, data):
        self.sent.append(data)

    async def recv(self):
        if not self._responses:
            await asyncio.sleep(60)
            raise asyncio.TimeoutError()
        return self._responses.pop(0)

    async def close(self):
        self.closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


def _event(event_type, **fields):
    return json.dumps({'type': event_type, **fields})


def _delta_call(call_id, name, arguments):
    return _event('response.delta', delta={'type': 'tool_call', 'tool_call': {'id': call_id, 'name': name, 'arguments': arguments}})


TOOLS = [
    {'type': 'function', 'function': {'name': 'list_directory', 'description': 'List a folder', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}}}},
    {'type': 'function', 'function': {'name': 'read_file', 'description': 'Read a file', 'parameters': {'type': 'object'}}},
]


def _run(skill, messages, sockets, tools=None):
    """Every chunk of one `chat_completion_stream`, with `sockets` handed out in order."""

    async def go():
        with patch(CONNECT, side_effect=list(sockets)):
            return [chunk async for chunk in skill.chat_completion_stream(messages, tools=tools)]

    return asyncio.run(go())


def _frames(ws):
    return [json.loads(frame) for frame in ws.sent]


class TestToolsReachTheModel:
    def test_the_tools_go_in_response_create_where_the_platform_reads_them(self):
        ws = MockWebSocket([
            _event('session.created', session_id='s1'),
            _event('response.created', response_id='r1'),
            _event('response.done', response={'output': [], 'finish_reason': 'STOP'}),
        ])
        _run(LLMProxySkill({'platform_token': 'jwt'}), [{'role': 'user', 'content': 'list the folder'}], [ws], tools=TOOLS)
        session_create, response_create = _frames(ws)
        assert session_create['type'] == 'session.create' and 'tools' not in session_create['session']
        assert response_create['type'] == 'response.create'
        # The OpenAI function shape, unchanged: the platform's adapters read
        # `function.name` (the first proof run got Google's 400 "Unknown name
        # \"undefined\" at 'tools[0]'" from the flattened shape).
        assert response_create['response']['tools'] == TOOLS
        assert [t['function']['name'] for t in response_create['response']['tools']] == ['list_directory', 'read_file']

    def test_no_tools_means_no_tools_field(self):
        ws = MockWebSocket([
            _event('session.created', session_id='s1'),
            _event('response.created', response_id='r1'),
            _event('response.done', response={'output': []}),
        ])
        _run(LLMProxySkill({'platform_token': 'jwt'}), [{'role': 'user', 'content': 'hi'}], [ws])
        assert 'tools' not in _frames(ws)[1]['response']


class TestParallelToolCallsKeepTheirIndex:
    def test_two_calls_get_two_indexes_and_reconstruct_as_two_calls(self):
        ws = MockWebSocket([
            _event('session.created', session_id='s1'),
            _event('response.created', response_id='r1'),
            _delta_call('c1', 'list_directory', '{"path": "."}'),
            _delta_call('c2', 'read_file', '{"path": "notes.md"}'),
            _event('response.done', response={'output': [], 'finish_reason': 'STOP'}),
        ])
        chunks = _run(LLMProxySkill({'platform_token': 'jwt'}), [{'role': 'user', 'content': 'go'}], [ws], tools=TOOLS)
        calls = [tc for c in chunks for tc in (c['choices'][0]['delta'].get('tool_calls') or [])]
        assert [(tc['index'], tc['id']) for tc in calls] == [(0, 'c1'), (1, 'c2')]

        from webagents.agents.core.base_agent import BaseAgent

        agent = BaseAgent(name='t', instructions='', skills={}, scopes=['all'])
        rebuilt = agent._reconstruct_response_from_chunks(chunks)
        tool_calls = rebuilt['choices'][0]['message']['tool_calls']
        assert [(tc['function']['name'], tc['function']['arguments']) for tc in tool_calls] == [
            ('list_directory', '{"path": "."}'),
            ('read_file', '{"path": "notes.md"}'),
        ]

    def test_the_same_call_id_keeps_its_index_and_the_tool_call_event_shape_agrees(self):
        ws = MockWebSocket([
            _event('session.created', session_id='s1'),
            _event('response.created', response_id='r1'),
            _event('tool.call', call_id='a', name='read_file', arguments='{"path": "x"}'),
            _delta_call('a', 'read_file', ''),
            _event('tool.call', call_id='b', name='read_file', arguments='{"path": "y"}'),
            _event('response.done', response={'output': []}),
        ])
        chunks = _run(LLMProxySkill({'platform_token': 'jwt'}), [{'role': 'user', 'content': 'go'}], [ws], tools=TOOLS)
        calls = [tc for c in chunks for tc in (c['choices'][0]['delta'].get('tool_calls') or [])]
        assert [(tc['index'], tc['id']) for tc in calls] == [(0, 'a'), (0, 'a'), (1, 'b')]


class TestTheCallIdRoundTrip:
    def test_the_thought_signature_suffix_survives_the_chunk_and_the_next_request(self):
        signed = 'call_7|ts:CqEBChk'
        ws = MockWebSocket([
            _event('session.created', session_id='s1'),
            _event('response.created', response_id='r1'),
            _delta_call(signed, 'list_directory', '{}'),
            _event('response.done', response={'output': []}),
        ])
        chunks = _run(LLMProxySkill({'platform_token': 'jwt'}), [{'role': 'user', 'content': 'go'}], [ws], tools=TOOLS)
        [call] = [tc for c in chunks for tc in (c['choices'][0]['delta'].get('tool_calls') or [])]
        assert call['id'] == signed

        # The next request carries the id back unchanged, on the assistant's
        # call and on the tool result: Gemini answers 400 without it.
        conversation = [
            {'role': 'user', 'content': 'go'},
            {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': signed, 'type': 'function', 'function': {'name': 'list_directory', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': signed, 'content': 'AGENT.md'},
        ]
        ws2 = MockWebSocket([
            _event('session.created', session_id='s2'),
            _event('response.created', response_id='r2'),
            _event('response.delta', delta={'type': 'text', 'text': 'One file.'}),
            _event('response.done', response={'output': [], 'finish_reason': 'STOP'}),
        ])
        _run(LLMProxySkill({'platform_token': 'jwt'}), conversation, [ws2], tools=TOOLS)
        messages = _frames(ws2)[1]['response']['messages']
        assert messages[1]['tool_calls'][0]['id'] == signed
        assert messages[2]['tool_call_id'] == signed


class TestAnEmptyCompletionIsRetriedOnce:
    def _empty(self, session_id, finish, blocked=False, usage=None):
        response = {'output': [], 'finish_reason': finish}
        if blocked:
            response['finish_blocked'] = True
        if usage:
            response['usage'] = usage
        return MockWebSocket([
            _event('session.created', session_id=session_id),
            _event('response.created', response_id=f'r-{session_id}'),
            _event('response.done', response=response),
        ])

    def test_a_malformed_tool_call_is_sent_once_more_on_a_fresh_socket(self):
        first = self._empty('s1', 'MALFORMED_FUNCTION_CALL', usage={'input_tokens': 100, 'output_tokens': 0, 'total_tokens': 100})
        second = MockWebSocket([
            _event('session.created', session_id='s2'),
            _event('response.created', response_id='r2'),
            _delta_call('c1', 'list_directory', '{"path": "."}'),
            _event('response.done', response={'output': [], 'finish_reason': 'STOP', 'usage': {'input_tokens': 100, 'output_tokens': 9, 'total_tokens': 109}}),
        ])
        chunks = _run(LLMProxySkill({'platform_token': 'jwt'}), [{'role': 'user', 'content': 'list the folder'}], [first, second], tools=TOOLS)
        assert first.closed and second.closed
        # Both sockets got the same request, tools included.
        assert _frames(first)[1]['response']['tools'] == _frames(second)[1]['response']['tools']
        calls = [tc for c in chunks for tc in (c['choices'][0]['delta'].get('tool_calls') or [])]
        assert [tc['function']['name'] for tc in calls] == ['list_directory']
        last = chunks[-1]
        assert last['choices'][0]['finish_reason'] == 'stop'
        assert last['webagents_finish'] == {'reason': 'STOP', 'blocked': False, 'retried': True}

    def test_a_second_empty_answer_says_so_and_is_not_retried_again(self):
        sockets = [self._empty('s1', 'MALFORMED_FUNCTION_CALL'), self._empty('s2', 'MALFORMED_FUNCTION_CALL')]
        chunks = _run(LLMProxySkill({'platform_token': 'jwt'}), [{'role': 'user', 'content': 'go'}], sockets, tools=TOOLS)
        assert len(chunks) == 1
        assert chunks[0]['webagents_finish'] == {'reason': 'MALFORMED_FUNCTION_CALL', 'blocked': False, 'retried': True}

    def test_an_unexpected_tool_call_is_retried_too(self):
        sockets = [self._empty('s1', 'UNEXPECTED_TOOL_CALL'), self._empty('s2', 'STOP')]
        chunks = _run(LLMProxySkill({'platform_token': 'jwt'}), [{'role': 'user', 'content': 'go'}], sockets, tools=TOOLS)
        assert chunks[-1]['webagents_finish'] == {'reason': 'STOP', 'blocked': False, 'retried': True}

    def test_an_empty_stop_is_not_retried(self):
        # Measured to repeat: a retry would only bill a second prompt.
        chunks = _run(LLMProxySkill({'platform_token': 'jwt'}), [{'role': 'user', 'content': 'go'}], [self._empty('s1', 'STOP')])
        assert len(chunks) == 1
        assert chunks[0]['webagents_finish'] == {'reason': 'STOP', 'blocked': False, 'retried': False}

    def test_a_blocked_prompt_is_not_retried(self):
        chunks = _run(LLMProxySkill({'platform_token': 'jwt'}), [{'role': 'user', 'content': 'go'}], [self._empty('s1', 'MALFORMED_FUNCTION_CALL', blocked=True)])
        assert len(chunks) == 1
        assert chunks[0]['webagents_finish'] == {'reason': 'MALFORMED_FUNCTION_CALL', 'blocked': True, 'retried': False}

    def test_a_completion_that_said_something_is_not_retried_whatever_the_reason(self):
        ws = MockWebSocket([
            _event('session.created', session_id='s1'),
            _event('response.created', response_id='r1'),
            _event('response.delta', delta={'type': 'text', 'text': 'Let me look.'}),
            _event('response.done', response={'output': [], 'finish_reason': 'MALFORMED_FUNCTION_CALL'}),
        ])
        chunks = _run(LLMProxySkill({'platform_token': 'jwt'}), [{'role': 'user', 'content': 'go'}], [ws], tools=TOOLS)
        assert [c['choices'][0]['delta'].get('content') for c in chunks] == ['Let me look.', None]
        assert chunks[-1]['webagents_finish']['retried'] is False

    def test_no_reason_reported_is_said_as_none(self):
        ws = MockWebSocket([
            _event('session.created', session_id='s1'),
            _event('response.created', response_id='r1'),
            _event('response.done', response={'output': []}),
        ])
        chunks = _run(LLMProxySkill({'platform_token': 'jwt'}), [{'role': 'user', 'content': 'go'}], [ws])
        assert chunks[-1]['webagents_finish'] == {'reason': None, 'blocked': False, 'retried': False}


class TestThinkingEvents:
    def test_an_empty_thinking_event_is_not_sent_as_a_literal_think_tag(self):
        ws = MockWebSocket([
            _event('session.created', session_id='s1'),
            _event('response.created', response_id='r1'),
            _event('thinking', content=''),
            _event('thinking', content='Listing first.'),
            _event('response.delta', delta={'type': 'text', 'text': 'Done.'}),
            _event('response.done', response={'output': [], 'finish_reason': 'STOP'}),
        ])
        chunks = _run(LLMProxySkill({'platform_token': 'jwt'}), [{'role': 'user', 'content': 'go'}], [ws])
        contents = [c['choices'][0]['delta'].get('content') for c in chunks if c['choices'][0]['delta'].get('content')]
        assert contents == ['<think>Listing first.</think>', 'Done.']
        assert '<think></think>' not in contents


def test_the_retry_list_is_the_one_both_sdks_share():
    import pathlib

    from webagents.agents.skills.core.llm.proxy.skill import RETRY_FINISH_REASONS

    fixture = json.loads((pathlib.Path(__file__).parent / 'fixtures' / 'cli' / 'chat_fixes_empty_reply.json').read_text())
    assert list(RETRY_FINISH_REASONS) == fixture['retry_finish_reasons']
