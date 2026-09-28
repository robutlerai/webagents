"""
LLM Proxy Skill - WebAgents V2.0

Connects to the UAMP LLM proxy via WebSocket for daemon agents
running inside the Kubernetes cluster. The proxy handles provider
selection, API keys, and metered billing - callers only need a
payment token.

Protocol flow (per request):
  1. Connect WS → send session.create (with a payment token, or the
     signed-in person's platform bearer as `Authorization`)
  2. Receive session.created
  3. Send response.create carrying the whole conversation (`messages`)
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

#: Finish reasons worth one more request when the completion was empty: Gemini's
#: verdicts on a tool call it could not form, or a tool it was never given. A
#: sampling accident, not a property of the prompt (the TypeScript proxy skill
#: keeps the same list).
RETRY_FINISH_REASONS = ('MALFORMED_FUNCTION_CALL', 'UNEXPECTED_TOOL_CALL')

#: Yielded by `_stream_once` in place of its last chunk when the request is to
#: be sent once more (never leaves `chat_completion_stream`).
_RETRY = object()

#: Seconds to wait before sending again after the platform went away.
SERVER_AWAY_RETRY_DELAY_S = 1.0
#: WebSocket close codes that mean the server is going away, not that the request was bad.
SERVER_AWAY_CODES = (1001, 1012)


def server_went_away(error: BaseException) -> bool:
    """A websocket closed by the server with 1001 (going away) or 1012 (restart)."""
    try:
        from websockets.exceptions import ConnectionClosed
    except Exception:  # noqa: BLE001 - no websockets, no socket to have closed
        return False
    if not isinstance(error, ConnectionClosed):
        return False
    received = getattr(error, "rcvd", None)
    code = getattr(received, "code", None) if received is not None else getattr(error, "code", None)
    return code in SERVER_AWAY_CODES


def _event_id() -> str:
    return str(uuid.uuid4())


def _base_event(event_type: str) -> Dict[str, Any]:
    return {
        'type': event_type,
        'event_id': _event_id(),
        'timestamp': int(time.time() * 1000),
    }


#: The HTTP status a served agent answers when the platform's LLM proxy refused
#: (2026-09-27): `serve` used to answer a refused non-streaming call with a 500
#: and a traceback. Anything else the proxy refuses is a bad gateway.
_PROXY_STATUS = {
    'payment_required': 402,
    'unauthorized': 401,
    'invalid_request': 400,
    'rate_limit': 429,
    'rate_limited': 429,
    'timeout': 504,
}


class LLMProxyError(Exception):
    """Raised when the LLM proxy returns an error event."""

    def __init__(self, code: str, message: str, details: Any = None):
        self.code = code
        self.details = details
        super().__init__(message)

    @property
    def status_code(self) -> int:
        """The status `serve` answers with (`server/core/app.py`, `_refusal`);
        the message is the platform's own sentence, written for the caller."""
        return _PROXY_STATUS.get(self.code, 502)

    def to_dict(self) -> Dict[str, Any]:
        message = str(self)
        if 'session.create' in message or 'session.extensions' in message:
            message = REFUSED_CREDENTIAL_MESSAGE
        return {'error': {'code': self.code, 'message': message}}


#: What a caller of a served agent reads when its request carried no payment
#: token and the agent runs on Robutler's models (S-327, 2026-09-28): the
#: turn is the caller's, so the caller pays, and nothing is dialled without a
#: token. The TypeScript proxy skill says the same (`CALLERS_PAY_REFUSAL`);
#: the words are the shared fixture `cli/final_sdk_serve_model.json`.
CALLERS_PAY_REFUSAL = (
    "This agent runs on Robutler's models, and each caller pays for its own turns: "
    "send a payment token in the X-Payment-Token header."
)

#: A served caller never reads the platform's protocol wording (B11,
#: 2026-09-28): `session.create requires X-Payment-Token ...` named a socket
#: event the caller never sent. `to_dict` (what `serve` answers) says this
#: instead; `str(error)` keeps the platform's words for the chat, whose
#: failure presentation reads them (fixture `cli/failure_presentation.json`).
REFUSED_CREDENTIAL_MESSAGE = "Robutler did not accept the credential this agent's model runs on."


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
        # THE SIGNED-IN PERSON (2026-09-24). A CLI agent with no provider key
        # runs here on the token `webagents login` stored: sent as
        # `Authorization` when there is no payment token, and funded by the
        # platform from that person's credits (`lib/llm/cli-bearer-funding.ts`).
        # A string, or a callable read on every request so a fresh login
        # reaches an agent the daemon already built. Config only, never the
        # environment: the shell skill passes its environment to every command.
        self.platform_token: Any = self.config.get('platform_token')
        # OTHER CALLERS' TURNS (S-327, 2026-09-28). `serve`, `mcp serve` and
        # the daemon build this skill with `callers_pay`: each call is paid by
        # the payment token the caller's request carries (or the one given to
        # this process), never by a sign-in, even one configured in code, and
        # a call with no token is refused with 402 before anything is dialled.
        self.callers_pay: bool = bool(self.config.get('callers_pay'))
        if self.callers_pay:
            self.platform_token = None
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
        session.create → response.create flow, then yields OpenAI-format
        streaming chunks.

        ONE RETRY ON A BROKEN TOOL CALL (2026-09-27). Gemini answers a turn it
        could not turn into a tool call with an EMPTY completion and the
        finish reason MALFORMED_FUNCTION_CALL (UNEXPECTED_TOOL_CALL for a tool
        it was never given). That is a sampling accident, not a property of
        the prompt, so the request is sent once more. The platform sends the
        reason in `response.done` (`finish_reason`), and this skill puts it on
        its last chunk as `webagents_finish`, so the chat can say what
        happened when the second attempt is empty too. Nothing else is
        retried: an empty `STOP` was measured to repeat (the platform's own
        note on it), and a retry there only bills a second prompt.
        """
        target_model = model or self.model
        for attempt in (1, 2):
            stream = self._stream_once(messages, target_model, tools, may_retry=attempt == 1, **kwargs)
            retry = False
            yielded = False
            try:
                async for chunk in stream:
                    if chunk is _RETRY:
                        retry = True
                        break
                    yielded = True
                    yield chunk
            except Exception as error:  # noqa: BLE001 - only a server going away is retried; the rest re-raised
                # ONE RETRY WHEN THE PLATFORM GOES AWAY (2026-09-28). A deploy or
                # restart drains the socket with 1001 (going away) or 1012
                # (restart); the turn failed with the raw close words even when
                # nothing had been said yet. Before any output the request is sent
                # once more, after a moment for the next server to take over.
                if attempt == 1 and not yielded and server_went_away(error):
                    self.logger.warning(
                        f'The platform closed the connection ({error}) before {target_model} answered; '
                        'sending the request once more'
                    )
                    await asyncio.sleep(SERVER_AWAY_RETRY_DELAY_S)
                    retry = True
                else:
                    raise
            finally:
                await stream.aclose()
            if not retry:
                return

    async def _stream_once(
        self,
        messages: List[Dict[str, Any]],
        target_model: str,
        tools: Optional[List[Dict[str, Any]]],
        may_retry: bool,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        """One request on one socket. Yields `_RETRY` in place of the last
        chunk when the completion was empty for a reason worth one more try."""
        extensions = self._build_extensions(target_model, **kwargs)
        if self.callers_pay and 'X-Payment-Token' not in extensions:
            # Nothing to pay with, and never the sign-in (S-327): refused
            # here, with the caller's words, rather than by the platform with
            # its protocol's (B11).
            raise LLMProxyError('payment_required', CALLERS_PAY_REFUSAL)
        chunk_index = 0
        # THE INDEX IS PER CALL (2026-09-27). Every tool call was sent with
        # `index: 0`, and the agent joins argument fragments by index
        # (`_reconstruct_response_from_chunks`), so two parallel calls became
        # one call with both argument strings glued together: `dir_list({}{})`.
        # The platform sends each call whole, keyed by its id; an id gets the
        # next free index the first time it is seen.
        call_indexes: Dict[str, int] = {}
        # Whether any text or tool call was yielded. Thinking alone is not an
        # answer, and it is what an empty completion often consists of.
        produced = False

        def tool_index(call_id: str) -> int:
            return call_indexes.setdefault(call_id, len(call_indexes))

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
                    'extensions': extensions,
                },
            }
            await ws.send(json.dumps(session_create))

            # Wait for session.created
            try:
                await self._wait_for_event(ws, 'session.created')
            except LLMProxyError as exc:
                raise self._explain_refused_sign_in(exc, session_create) from exc

            # 2. response.create, carrying the whole conversation --------------
            #
            # ONE EVENT, ROLES KEPT (2026-09-24). This sent one `input.text` per
            # message with every role but user and system rewritten to user,
            # and the platform keeps only the LAST `input.text` it is sent
            # (`handleInputText` overwrites `pendingInput`): every request
            # reached the model as the final message alone, no instructions,
            # no history, no tool results. The platform reads
            # `response.messages` whole, which is how the TypeScript client
            # (`uamp/client.ts`, `sendResponse`) has always sent it.
            response_create = {
                **_base_event('response.create'),
                'response': {
                    'model': target_model,
                    'messages': [self._wire_message(msg) for msg in messages],
                },
            }
            # TOOLS GO WITH THE RESPONSE (2026-09-27). They went in
            # `session.tools`, which the platform never read (it reads
            # `session.extensions.tools` and `response.tools`), so every agent
            # on Robutler's models ran with NO tools: the model guessed names
            # and arguments, and Gemini ended most such turns with
            # MALFORMED_FUNCTION_CALL. The TypeScript client has always sent
            # `response.tools`, and every deployed platform reads it.
            if tools:
                response_create['response']['tools'] = self._convert_tools(tools)
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
                    tc = delta.get('tool_call') if delta.get('type') == 'tool_call' else None
                    oai_chunk = self._delta_to_openai_chunk(
                        delta, response_id or '', target_model, chunk_index,
                        tool_index=tool_index(str(tc.get('id', ''))) if isinstance(tc, dict) else 0,
                    )
                    if oai_chunk:
                        produced = True
                        yield oai_chunk

                elif event_type == 'response.done':
                    done = True
                    resp = event.get('response', {})
                    usage = resp.get('usage')
                    finish = resp.get('finish_reason')
                    blocked = resp.get('finish_blocked') is True
                    if may_retry and not produced and finish in RETRY_FINISH_REASONS and not blocked:
                        self.logger.warning(
                            f'{target_model} returned an empty completion ({finish}); sending the request once more'
                        )
                        yield _RETRY
                        return
                    final_chunk: Dict[str, Any] = {
                        'id': response_id or '',
                        'object': 'chat.completion.chunk',
                        'model': target_model,
                        'choices': [{
                            'index': 0,
                            'delta': {},
                            'finish_reason': 'stop',
                        }],
                        # Why the provider stopped, for the chat's line when
                        # the reply is empty (`cli/repl/failures.py`). `reason`
                        # is the provider's own word, `blocked` says the PROMPT
                        # was refused, and `retried` that this was the second
                        # attempt.
                        'webagents_finish': {
                            'reason': finish if isinstance(finish, str) and finish else None,
                            'blocked': blocked,
                            'retried': not may_retry,
                        },
                    }
                    if usage:
                        final_chunk['usage'] = {
                            'prompt_tokens': usage.get('input_tokens', 0),
                            'completion_tokens': usage.get('output_tokens', 0),
                            'total_tokens': usage.get('total_tokens', 0),
                        }
                        # The platform's cost, when it reports one, for the
                        # chat footer (plan item 2.4, `llm/pricing.py`).
                        if isinstance(usage.get('cost'), dict):
                            final_chunk['usage']['cost'] = dict(usage['cost'])
                        # THE CHARGE ITSELF (B4, 2026-09-28): `total_cost`, the
                        # credits the platform's settle deducted for this call,
                        # which the chat shows as it is, not as an estimate.
                        charged = usage.get('total_cost')
                        if isinstance(charged, (int, float)) and not isinstance(charged, bool):
                            final_chunk['usage']['total_cost'] = charged
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
                    _pay_token = self._resolve_payment_token()
                    if _pay_token:
                        submit = {
                            **_base_event('payment.submit'),
                            'payment': {
                                'scheme': 'token',
                                'amount': reqs.get('amount', '0'),
                                'token': _pay_token,
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
                    produced = True
                    yield {
                        'id': response_id or '',
                        'object': 'chat.completion.chunk',
                        'model': target_model,
                        'choices': [{
                            'index': 0,
                            'delta': {
                                'tool_calls': [{
                                    'index': tool_index(str(event.get('call_id', ''))),
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
                    if not text:
                        # A thinking event with no content (a stage marker)
                        # was sent as a literal `<think></think>`, which ACP
                        # forwarded to the editor as the whole reply
                        # (2026-09-27).
                        continue
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

    @staticmethod
    def _wire_message(msg: Dict[str, Any]) -> Dict[str, Any]:
        """An OpenAI-format message as the platform's `response.messages` takes it."""
        content = msg.get('content', '')
        if isinstance(content, list):
            content = ' '.join(
                part.get('text', '') for part in content if isinstance(part, dict) and part.get('type') == 'text'
            )
        out: Dict[str, Any] = {'role': msg.get('role', 'user'), 'content': content}
        if msg.get('tool_calls'):
            out['tool_calls'] = msg['tool_calls']
        if msg.get('tool_call_id'):
            out['tool_call_id'] = msg['tool_call_id']
        if msg.get('name'):
            out['name'] = msg['name']
        return out

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

    def _resolve_payment_token(self) -> Optional[str]:
        """The token to pay this call with, PER REQUEST first.

        S-206 (2026-09-21). `self.payment_token` is resolved ONCE at
        construction, from `config['payment_token']` or
        `ROBUTLER_PAYMENT_TOKEN`. That is fine for an agent the operator funds
        up front and wrong for one served through the MACHINE DOOR: the door
        mints a token per call and puts it on the REQUEST, so a statically
        configured skill has nothing to pay with, dials the rail bare, and the
        rail hangs up (observed on the TypeScript side as a 4001 close, the
        work delivered and the door's reservation never drawn down).

        The per-request carrier is `context.payments.payment_token`, the same
        one the UCP skill reads. Fall back to the constructed value so an
        operator-funded agent is unchanged.
        """
        try:
            context = self.get_context()
        except Exception:
            context = None
        if context is not None:
            payments = getattr(context, 'payments', None)
            token = getattr(payments, 'payment_token', None) if payments else None
            if token:
                return token
            if self.callers_pay:
                # The caller's own token on its request, as the TypeScript
                # server puts it on the run's context (`createContextFromHono`):
                # a served agent with no payment skill never read it, so its
                # callers could not pay for their own turns (S-327).
                token = self._request_payment_token(context)
                if token:
                    return token
        return self.payment_token

    @staticmethod
    def _request_payment_token(context: Any) -> Optional[str]:
        request = getattr(context, 'request', None)
        headers = getattr(request, 'headers', None)
        if headers is None:
            return None
        try:
            token = headers.get('x-payment-token') or headers.get('X-Payment-Token')
        except Exception:  # noqa: BLE001 - a request without readable headers carries no token
            return None
        return token or None

    def _resolve_platform_token(self) -> Optional[str]:
        token = self.platform_token() if callable(self.platform_token) else self.platform_token
        return token or None

    def _explain_refused_sign_in(self, exc: LLMProxyError, session_create: Dict[str, Any]) -> LLMProxyError:
        """A plain reason when a server that predates CLI sign-ins refuses one (2026-09-24).

        A server that funds the `webagents login` bearer names it in this
        refusal; one that does not asks for `X-Payment-Token` alone, which
        tells a person at the terminal nothing they can act on.
        """
        extensions = session_create.get('session', {}).get('extensions', {})
        message = str(exc)
        sent_sign_in = 'Authorization' in extensions and 'X-Payment-Token' not in extensions
        if sent_sign_in and 'X-Payment-Token' in message and 'Bearer' not in message:
            return LLMProxyError(
                exc.code,
                f'Robutler at {self.proxy_url} does not run models for a CLI sign-in. '
                'Use a provider key: webagents secrets set OPENAI_API_KEY.',
            )
        return exc

    def _build_extensions(self, model: str, **kwargs: Any) -> Dict[str, Any]:
        extensions: Dict[str, Any] = {}
        _pay_token = self._resolve_payment_token()
        if _pay_token:
            extensions['X-Payment-Token'] = _pay_token
        else:
            _platform_token = self._resolve_platform_token()
            if _platform_token:
                extensions['Authorization'] = f'Bearer {_platform_token}'
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
        """The tools as the platform's `response.tools` takes them: the OpenAI
        function shape (`{type: 'function', function: {name, description,
        parameters}}`), which is what the TypeScript agent sends
        (`getToolDefinitions`) and what the platform's provider adapters read
        (`t.function.name`). This used to flatten each tool to the bare
        `{name, description, parameters}` shape, which nothing ever read while
        the tools went in `session.tools`; the first time they reached a
        provider, Google answered 400 "Unknown name \"undefined\" at
        'tools[0]'" (2026-09-27). A bare tool is wrapped, never flattened."""
        wire: List[Dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get('type') == 'function' and isinstance(tool.get('function'), dict):
                wire.append(tool)
            elif tool.get('name'):
                wire.append({
                    'type': 'function',
                    'function': {
                        'name': tool['name'],
                        'description': tool.get('description', ''),
                        'parameters': tool.get('parameters', {}),
                    },
                })
            else:
                wire.append(tool)
        return wire

    @staticmethod
    def _delta_to_openai_chunk(
        delta: Dict[str, Any],
        response_id: str,
        model: str,
        index: int,
        tool_index: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """Convert a UAMP response.delta payload to an OpenAI streaming chunk.
        `tool_index` is the call's own index in the turn (see `_stream_once`)."""
        delta_type = delta.get('type', '')
        oai_delta: Dict[str, Any] = {}

        if delta_type == 'text' and delta.get('text'):
            oai_delta['content'] = delta['text']
        elif delta_type == 'tool_call' and delta.get('tool_call'):
            tc = delta['tool_call']
            oai_delta['tool_calls'] = [{
                'index': tool_index,
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
