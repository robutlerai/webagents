"""
LLM Proxy Skill - WebAgents V2.0

Connects to the UAMP LLM proxy via WebSocket for daemon agents
running inside the Kubernetes cluster. The proxy handles provider
selection, API keys, and metered billing - callers only need a
payment token.

Protocol flow (per request):
  1. Connect WS → send session.create (with payment token)
  2. Receive session.created
  3. Send input.text + response.create
  4. Receive response.delta* → response.done
  5. Handle payment.required / payment.error if balance is low

The skill exposes the same chat_completion / chat_completion_stream
interface as every other LLM skill and registers itself as a handoff.
"""

import os
import json
import uuid
import asyncio
import time
from typing import Dict, Any, List, Optional, AsyncGenerator, TYPE_CHECKING

try:
    import websockets
    import websockets.client
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

if TYPE_CHECKING:
    from webagents.agents.core.base_agent import BaseAgent

from webagents.agents.skills.base import Skill, Handoff
from webagents.utils.logging import get_logger, log_skill_event


DEFAULT_PROXY_URL = 'wss://robutler.ai/llm'
DEFAULT_MODEL = 'auto/balanced'
UAMP_VERSION = '1.0'
CONNECT_TIMEOUT = 10.0
RESPONSE_TIMEOUT = 120.0


def _event_id() -> str:
    return str(uuid.uuid4())


def _base_event(event_type: str) -> Dict[str, Any]:
    return {
        'type': event_type,
        'event_id': _event_id(),
        'timestamp': int(time.time() * 1000),
    }


class LLMProxyError(Exception):
    """Raised when the LLM proxy returns an error event."""

    def __init__(self, code: str, message: str, details: Any = None):
        self.code = code
        self.details = details
        super().__init__(message)


class PaymentRequiredError(LLMProxyError):
    """Raised when the proxy demands payment before continuing."""

    def __init__(self, requirements: Dict[str, Any]):
        self.requirements = requirements
        super().__init__(
            'payment_required',
            f"Payment required: {requirements.get('amount')} {requirements.get('currency')}",
            requirements,
        )


def _purchase_pointer_of(requirements: Any) -> Optional[str]:
    """The purchase URL of a PURCHASE POINTER: an `mpp` entry of
    `requirements.schemes` that names `purchase_url` and has NO `challenge`
    member (the portal's lib/payments/purchase-pointer.ts). This socket has no
    door, so it never sends an entry that carries a challenge; one that did
    is not a pointer and is left to the token path."""
    schemes = requirements.get('schemes') if isinstance(requirements, dict) else None
    for entry in schemes if isinstance(schemes, list) else []:
        if not isinstance(entry, dict) or entry.get('scheme') != 'mpp':
            continue
        url = entry.get('purchase_url')
        if isinstance(url, str) and url.strip() and entry.get('challenge') is None:
            return url.strip()
    return None


class LLMProxySkill(Skill):
    """
    LLM skill that delegates completions to the platform's UAMP LLM proxy.

    Designed for agents that use the platform's centralized LLM proxy
    instead of calling provider APIs directly.  The proxy URL defaults
    to ``wss://robutler.ai/llm`` and can be overridden with the
    ``ROBUTLER_LLM_PROXY_URL`` environment variable (e.g.
    ``ws://localhost:3000/llm`` for k8s deployments) or the
    ``proxy_url`` config key.
    """

    async def _follow_purchase_pointer(self, requirements: Any, call: Any) -> bool:
        """Follow the platform's purchase pointer when a buyer is configured
        (the `mpp_buyer` note in `__init__`). True when the buyer bought. A
        refusal is the buyer's policy speaking (it has already reported it
        through `on_refusal`), and the token path that follows is unchanged
        either way."""
        follow = getattr(self.mpp_buyer, 'purchase_at', None) if self.mpp_buyer else None
        url = _purchase_pointer_of(requirements) if follow is not None else None
        if url is None:
            return False
        outcome = await follow(url, source_url=self.proxy_url, call=call)
        if getattr(outcome, 'ok', False):
            return True
        self.logger.info(
            f"LLM proxy: purchase pointer not followed: {getattr(outcome, 'reason', None) or 'unknown'}"
        )
        return False

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope='all')
        self.config = config or {}

        self.proxy_url: str = (
            self.config.get('proxy_url')
            or os.environ.get('ROBUTLER_LLM_PROXY_URL')
            or DEFAULT_PROXY_URL
        )
        self.model: str = self.config.get('model', DEFAULT_MODEL)
        self.temperature: float = self.config.get('temperature', 0.7)
        self.max_tokens: Optional[int] = self.config.get('max_tokens')
        self.payment_token: Optional[str] = (
            self.config.get('payment_token')
            or os.environ.get('ROBUTLER_PAYMENT_TOKEN')
        )
        self.connect_timeout: float = self.config.get('connect_timeout', CONNECT_TIMEOUT)
        self.response_timeout: float = self.config.get('response_timeout', RESPONSE_TIMEOUT)
        # The MPP buyer (2026-09-19), an `MppBuyer` from
        # `payments_x402.mpp_buyer`, set once by the operator; the TypeScript
        # `LLMProxySkillConfig.mppBuyer`. When the token this session pays
        # with runs dry, the platform's `/llm` socket names where to buy more:
        # a `payment.required` whose `mpp` entry carries `purchase_url` and no
        # challenge (nobody on that socket was verified, so none could be
        # minted). With a buyer that has `purchase_at` the pointer is followed
        # (the buyer's own signed request to the purchase URL, paid under its
        # policy) and the token is then submitted as before, against the
        # funded balance. `MppBuyer` follows a pointer only when its policy
        # sets `daily_cap_cents`; with none it refuses
        # (`pointer_needs_daily_cap`) and the token path runs as it always
        # did. Duck-typed, so this skill never imports the payment
        # module; without a buyer every path is exactly what it was.
        self.mpp_buyer: Optional[Any] = self.config.get('mpp_buyer')

        self.agent: Optional['BaseAgent'] = None
        self.logger = get_logger('skill.llm.proxy', 'init')

        if not WEBSOCKETS_AVAILABLE:
            raise ImportError(
                'websockets library not available. Install with: pip install websockets'
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self, agent: 'BaseAgent') -> None:
        self.agent = agent
        self.logger = get_logger('skill.llm.proxy', agent.name)

        agent.register_handoff(
            Handoff(
                target=f"proxy_{self.model.replace('/', '_').replace('-', '_')}",
                description=f"UAMP LLM proxy completion handler ({self.model})",
                scope='all',
                metadata={
                    'function': self.chat_completion_stream,
                    'priority': 10,
                    'is_generator': True,
                },
            ),
            source='llm_proxy',
        )

        self.logger.info(f"Registered LLM proxy handoff (model={self.model}, url={self.proxy_url})")
        log_skill_event(agent.name, 'llm_proxy', 'initialized', {
            'model': self.model,
            'proxy_url': self.proxy_url,
        })

    # ------------------------------------------------------------------
    # Public API (mirrors other LLM skills)
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Non-streaming completion: collects the full response and returns it."""
        if stream:
            raise ValueError('Use chat_completion_stream() for streaming responses')

        content_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        usage: Dict[str, int] = {}
        response_id = ''
        finish_reason = 'stop'

        async for chunk in self.chat_completion_stream(messages, model=model, tools=tools, **kwargs):
            choices = chunk.get('choices', [])
            if not choices:
                continue
            delta = choices[0].get('delta', {})
            if delta.get('content'):
                content_parts.append(delta['content'])
            if delta.get('tool_calls'):
                tool_calls.extend(delta['tool_calls'])
            if choices[0].get('finish_reason'):
                finish_reason = choices[0]['finish_reason']
            if chunk.get('usage'):
                usage = chunk['usage']
            if chunk.get('id'):
                response_id = chunk['id']

        message: Dict[str, Any] = {
            'role': 'assistant',
            'content': ''.join(content_parts),
        }
        if tool_calls:
            message['tool_calls'] = tool_calls

        return {
            'id': response_id or f'proxy-{_event_id()}',
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': model or self.model,
            'choices': [{
                'index': 0,
                'message': message,
                'finish_reason': finish_reason,
            }],
            'usage': usage,
        }

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming completion via the UAMP LLM proxy.

        Opens a fresh WebSocket per request, runs the full
        session.create → input.text → response.create flow,
        then yields OpenAI-format streaming chunks.
        """
        target_model = model or self.model
        chunk_index = 0

        async with websockets.client.connect(
            self.proxy_url,
            open_timeout=self.connect_timeout,
            close_timeout=5,
        ) as ws:
            # 1. session.create --------------------------------------------------
            session_create = {
                **_base_event('session.create'),
                'uamp_version': UAMP_VERSION,
                'session': {
                    'modalities': ['text'],
                    'extensions': self._build_extensions(target_model, **kwargs),
                },
            }
            if tools:
                session_create['session']['tools'] = self._convert_tools(tools)
            await ws.send(json.dumps(session_create))

            # Wait for session.created
            await self._wait_for_event(ws, 'session.created')

            # 2. input.text for each message + response.create -------------------
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if isinstance(content, list):
                    content = ' '.join(
                        part.get('text', '') for part in content if part.get('type') == 'text'
                    )
                text_event = {
                    **_base_event('input.text'),
                    'text': content,
                    'role': role if role in ('user', 'system') else 'user',
                }
                await ws.send(json.dumps(text_event))

            response_create = {
                **_base_event('response.create'),
            }
            await ws.send(json.dumps(response_create))

            # 3. Consume response events -----------------------------------------
            response_id: Optional[str] = None
            done = False
            # One key per response: a buyer that counts purchases per call
            # (`max_purchases_per_call`) keys the count on it.
            purchase_call = object()

            while not done:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=self.response_timeout)
                except asyncio.TimeoutError:
                    self.logger.error('Timed out waiting for proxy response')
                    raise LLMProxyError('timeout', 'Response timeout from LLM proxy')

                event = json.loads(raw)
                event_type = event.get('type', '')

                if event_type == 'response.created':
                    response_id = event.get('response_id', '')

                elif event_type == 'response.delta':
                    chunk_index += 1
                    delta = event.get('delta', {})
                    oai_chunk = self._delta_to_openai_chunk(
                        delta, response_id or '', target_model, chunk_index,
                    )
                    if oai_chunk:
                        yield oai_chunk

                elif event_type == 'response.done':
                    done = True
                    resp = event.get('response', {})
                    usage = resp.get('usage')
                    final_chunk: Dict[str, Any] = {
                        'id': response_id or '',
                        'object': 'chat.completion.chunk',
                        'model': target_model,
                        'choices': [{
                            'index': 0,
                            'delta': {},
                            'finish_reason': 'stop',
                        }],
                    }
                    if usage:
                        final_chunk['usage'] = {
                            'prompt_tokens': usage.get('input_tokens', 0),
                            'completion_tokens': usage.get('output_tokens', 0),
                            'total_tokens': usage.get('total_tokens', 0),
                        }
                    yield final_chunk

                elif event_type == 'response.error':
                    err = event.get('error', {})
                    raise LLMProxyError(
                        err.get('code', 'unknown'),
                        err.get('message', 'Unknown proxy error'),
                        err.get('details'),
                    )

                elif event_type == 'payment.required':
                    reqs = event.get('requirements', {})
                    await self._follow_purchase_pointer(reqs, purchase_call)
                    if self.payment_token:
                        submit = {
                            **_base_event('payment.submit'),
                            'payment': {
                                'scheme': 'token',
                                'amount': reqs.get('amount', '0'),
                                'token': self.payment_token,
                            },
                        }
                        await ws.send(json.dumps(submit))
                    else:
                        raise PaymentRequiredError(reqs)

                elif event_type == 'payment.error':
                    raise LLMProxyError(
                        event.get('code', 'payment_error'),
                        event.get('message', 'Payment error'),
                    )

                elif event_type == 'response.cancelled':
                    done = True
                    yield {
                        'id': response_id or '',
                        'object': 'chat.completion.chunk',
                        'model': target_model,
                        'choices': [{
                            'index': 0,
                            'delta': {},
                            'finish_reason': 'stop',
                        }],
                    }

                elif event_type == 'tool.call':
                    chunk_index += 1
                    yield {
                        'id': response_id or '',
                        'object': 'chat.completion.chunk',
                        'model': target_model,
                        'choices': [{
                            'index': 0,
                            'delta': {
                                'tool_calls': [{
                                    'index': 0,
                                    'id': event.get('call_id', ''),
                                    'type': 'function',
                                    'function': {
                                        'name': event.get('name', ''),
                                        'arguments': event.get('arguments', ''),
                                    },
                                }],
                            },
                            'finish_reason': None,
                        }],
                    }

                elif event_type == 'thinking':
                    chunk_index += 1
                    text = event.get('content', '')
                    yield {
                        'id': response_id or '',
                        'object': 'chat.completion.chunk',
                        'model': target_model,
                        'choices': [{
                            'index': 0,
                            'delta': {'content': f'<think>{text}</think>'},
                            'finish_reason': None,
                        }],
                    }

                # session.created, payment.accepted, pong, etc. → ignore

    # ------------------------------------------------------------------
    # Abort helper
    # ------------------------------------------------------------------

    async def cancel_response(self, ws: Any) -> None:
        """Send response.cancel on an open WebSocket."""
        cancel_event = {**_base_event('response.cancel')}
        await ws.send(json.dumps(cancel_event))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_extensions(self, model: str, **kwargs: Any) -> Dict[str, Any]:
        extensions: Dict[str, Any] = {}
        if self.payment_token:
            extensions['X-Payment-Token'] = self.payment_token
        extensions['model'] = model
        if kwargs.get('temperature') is not None:
            extensions['temperature'] = kwargs['temperature']
        elif self.temperature is not None:
            extensions['temperature'] = self.temperature
        if kwargs.get('max_tokens') or self.max_tokens:
            extensions['max_tokens'] = kwargs.get('max_tokens') or self.max_tokens
        return extensions

    @staticmethod
    def _convert_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-format tool defs to UAMP ToolDefinition shape."""
        uamp_tools = []
        for tool in tools:
            if tool.get('type') == 'function':
                func = tool.get('function', {})
                uamp_tools.append({
                    'name': func.get('name', ''),
                    'description': func.get('description', ''),
                    'parameters': func.get('parameters', {}),
                })
            else:
                uamp_tools.append(tool)
        return uamp_tools

    @staticmethod
    def _delta_to_openai_chunk(
        delta: Dict[str, Any],
        response_id: str,
        model: str,
        index: int,
    ) -> Optional[Dict[str, Any]]:
        """Convert a UAMP response.delta payload to an OpenAI streaming chunk."""
        delta_type = delta.get('type', '')
        oai_delta: Dict[str, Any] = {}

        if delta_type == 'text' and delta.get('text'):
            oai_delta['content'] = delta['text']
        elif delta_type == 'tool_call' and delta.get('tool_call'):
            tc = delta['tool_call']
            oai_delta['tool_calls'] = [{
                'index': 0,
                'id': tc.get('id', ''),
                'type': 'function',
                'function': {
                    'name': tc.get('name', ''),
                    'arguments': tc.get('arguments', ''),
                },
            }]
        else:
            return None

        return {
            'id': response_id,
            'object': 'chat.completion.chunk',
            'model': model,
            'choices': [{
                'index': 0,
                'delta': oai_delta,
                'finish_reason': None,
            }],
        }

    async def _wait_for_event(
        self,
        ws: Any,
        expected_type: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Block until we receive a specific event type (or error/timeout)."""
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise LLMProxyError('timeout', f'Timed out waiting for {expected_type}')
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            event = json.loads(raw)
            if event.get('type') == expected_type:
                return event
            if event.get('type') == 'session.error':
                err = event.get('error', {})
                raise LLMProxyError(
                    err.get('code', 'session_error'),
                    err.get('message', 'Session error'),
                )
            if event.get('type') == 'response.error':
                err = event.get('error', {})
                raise LLMProxyError(
                    err.get('code', 'unknown'),
                    err.get('message', 'Error during session setup'),
                )
