"""
UAMP Transport Skill - WebAgents V2.0

Native UAMP (Universal Agentic Message Protocol) WebSocket transport.
Provides direct UAMP event communication for real-time agent messaging.

Protocol: https://uamp.dev/
"""

import asyncio
import json
import uuid
import time
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import websocket
from webagents.uamp import (
    # Session events
    SessionCreateEvent,
    SessionCreatedEvent,
    CapabilitiesEvent,
    # Input events
    InputTextEvent,
    InputAudioEvent,
    InputImageEvent,
    InputTypingEvent,
    # Response events
    ResponseCreateEvent,
    ResponseCreatedEvent,
    ResponseDeltaEvent,
    ResponseDoneEvent,
    ResponseErrorEvent,
    # Tool events
    ToolCallEvent,
    ToolResultEvent,
    # Progress events
    ProgressEvent,
    ThinkingEvent,
    # Presence events
    PresenceTypingEvent,
    # Payment events
    PaymentRequiredEvent,
    PaymentSubmitEvent,
    PaymentAcceptedEvent,
    PaymentBalanceEvent,
    PaymentErrorEvent,
    PaymentScheme,
    PaymentRequirements,
    # Utility events
    PingEvent,
    PongEvent,
    # Types
    ContentDelta,
    ContentItem,
    UsageStats,
    ResponseOutput,
    Session,
    ModelCapabilities,
)

if TYPE_CHECKING:
    from webagents.agents.core.base_agent import BaseAgent
    from fastapi import WebSocket

try:
    from webagents.agents.skills.robutler.payments.exceptions import (
        PaymentError,
        PaymentTokenRequiredError,
    )
except ImportError:
    PaymentError = None  # type: ignore[misc, assignment]
    PaymentTokenRequiredError = None  # type: ignore[misc, assignment]

# Payment error codes that are retryable via a fresh payment token
_RETRYABLE_PAYMENT_CODES = frozenset({
    "PAYMENT_TOKEN_REQUIRED",
    "INSUFFICIENT_TOKEN_BALANCE",
    "PAYMENT_TOKEN_INVALID",
    "PAYMENT_CHARGING_FAILED",
})


class SessionClosedError(Exception):
    """Raised when a UAMP session closes while waiting for an event."""
    pass


@dataclass
class UAMPSession:
    """UAMP session state."""
    id: str = field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:12]}")
    modalities: List[str] = field(default_factory=lambda: ["text"])
    instructions: str = ""
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    status: str = "active"
    
    # Payment tracking
    payment_balance: Optional[str] = None
    payment_currency: str = "USD"
    payment_token_expires_at: Optional[int] = None
    payment_token: Optional[str] = None  # Pre-loaded or from payment.submit

    # Runtime state
    conversation: List[Dict[str, Any]] = field(default_factory=list)

    # Pending event futures: event_type -> list of futures waiting for that event
    _pending_events: Dict[str, List[asyncio.Future]] = field(default_factory=dict, repr=False)
    # WebSocket reference (set by transport after creation)
    _ws: Optional[Any] = field(default=None, repr=False)
    # Current background response generation task (so main loop stays unblocked)
    _response_task: Optional[asyncio.Task] = field(default=None, repr=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "modalities": self.modalities,
            "instructions": self.instructions,
            "created_at": self.created_at,
            "status": self.status,
        }

    async def wait_for_event(self, event_type: str, timeout: float = 60.0) -> Any:
        """Wait for a specific UAMP event type to arrive on this session.

        Args:
            event_type: The UAMP event type string (e.g. 'payment.submit').
            timeout: Maximum seconds to wait before raising asyncio.TimeoutError.

        Returns:
            The received event dict.

        Raises:
            asyncio.TimeoutError: If the event is not received within *timeout*.
            SessionClosedError: If the session closes while waiting.
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending_events.setdefault(event_type, []).append(fut)
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.CancelledError:
            raise SessionClosedError(f"Session {self.id} closed while waiting for '{event_type}'")
        finally:
            # Clean up the future from the pending list
            pending = self._pending_events.get(event_type, [])
            if fut in pending:
                pending.remove(fut)

    def _resolve_event(self, event_type: str, event_data: Any) -> bool:
        """Resolve the first pending future for *event_type*. Returns True if resolved."""
        pending = self._pending_events.get(event_type, [])
        while pending:
            fut = pending.pop(0)
            if not fut.done():
                fut.set_result(event_data)
                return True
        return False

    def _on_close(self) -> None:
        """Cancel all pending event futures and the response task (session is closing)."""
        for futures_list in self._pending_events.values():
            for fut in futures_list:
                if not fut.done():
                    fut.cancel()
        self._pending_events.clear()
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
            self._response_task = None

    async def send_event(self, event) -> None:
        """Send a UAMP event to the client over this session's WebSocket.

        Args:
            event: A UAMP event object with a ``to_dict()`` method.
        """
        if self._ws is not None:
            data = event.to_dict() if hasattr(event, 'to_dict') else event
            await self._ws.send_json(data)


class UAMPTransportSkill(Skill):
    """
    Native UAMP WebSocket transport.
    
    Implements direct UAMP event protocol over WebSocket for:
    - Session management
    - Multimodal input (text, audio, image)
    - Streaming responses
    - Tool calls
    - Typing indicators
    - Payment token management
    
    Example:
        agent = BaseAgent(
            name="my-agent",
            skills=[UAMPTransportSkill()]
        )
        
        # Connect via WebSocket to /agents/my-agent/uamp
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all")
        self._sessions: Dict[str, UAMPSession] = {}
        # Map connection IDs to session IDs for presence
        self._connections: Dict[str, str] = {}  # ws_id -> session_id
        # Session ID -> Future that receives payment token from payment.submit (for payment negotiation)
        self._pending_payment_futures: Dict[str, asyncio.Future] = {}
    
    async def initialize(self, agent: 'BaseAgent') -> None:
        """Initialize the UAMP transport."""
        self.agent = agent
        from webagents.utils.logging import get_logger
        self.logger = get_logger('skill.transport.uamp', agent.name)
    
    @websocket("/uamp")
    async def uamp_session(self, ws: 'WebSocket') -> None:
        """
        WebSocket endpoint for native UAMP protocol.
        
        Handles bidirectional UAMP event communication.
        """
        await ws.accept()
        
        ws_id = f"ws_{uuid.uuid4().hex[:12]}"
        session: Optional[UAMPSession] = None
        
        try:
            async for message in ws.iter_json():
                event_type = message.get("type", "")
                event_id = message.get("event_id", f"evt_{uuid.uuid4().hex[:12]}")
                
                # Handle session creation
                if event_type == "session.create":
                    session = await self._handle_session_create(ws, message)
                    session._ws = ws
                    self._connections[ws_id] = session.id
                    continue
                
                # Require session for all other events
                if not session:
                    await self._send_error(ws, "session_required", "Session must be created first")
                    continue
                
                # Resolve any pending wait_for_event futures for this event type
                session._resolve_event(event_type, message)

                # Route events
                handlers = {
                    "session.update": self._handle_session_update,
                    "capabilities.query": self._handle_capabilities_query,
                    "input.text": self._handle_input,
                    "input.audio": self._handle_input_audio,
                    "input.image": self._handle_input_image,
                    "input.video": self._handle_input_video,
                    "input.file": self._handle_input_file,
                    "input.typing": self._handle_input_typing,
                    "response.create": self._handle_response_create,
                    "response.cancel": self._handle_response_cancel,
                    "tool.result": self._handle_tool_result,
                    "payment.submit": self._handle_payment_submit,
                    "ping": self._handle_ping,
                }
                
                handler = handlers.get(event_type)
                if handler:
                    await handler(ws, session, message)
                else:
                    await self._send_error(ws, "unknown_event", f"Unknown event type: {event_type}")
                    
        except Exception as e:
            # Connection closed or error
            pass
        finally:
            # Cleanup: cancel any pending payment negotiation for this session
            if session:
                fut = self._pending_payment_futures.pop(session.id, None)
                if fut is not None and not fut.done():
                    fut.cancel()
                session._on_close()
                self._sessions.pop(session.id, None)
            self._connections.pop(ws_id, None)
    
    async def _handle_session_create(
        self,
        ws: 'WebSocket',
        event: Dict[str, Any]
    ) -> UAMPSession:
        """Create a new UAMP session."""
        session_config = event.get("session", {})
        
        session = UAMPSession(
            modalities=session_config.get("modalities", ["text"]),
            instructions=session_config.get("instructions", ""),
        )
        self._sessions[session.id] = session
        
        # Send session.created
        created_event = SessionCreatedEvent(
            uamp_version="1.0",
            session=Session(
                id=session.id,
                created_at=session.created_at,
                status=session.status,
            )
        )
        await ws.send_json(created_event.to_dict())
        
        # Send capabilities
        capabilities_event = CapabilitiesEvent(
            capabilities=ModelCapabilities(
                model_id=getattr(self.agent, 'model', 'unknown'),
                provider=getattr(self.agent, 'provider', 'webagents'),
                modalities=session.modalities,
                supports_streaming=True,
            )
        )
        await ws.send_json(capabilities_event.to_dict())
        
        # Send initial balance if configured
        if hasattr(self, 'payment_config') and self.payment_config:
            balance_event = PaymentBalanceEvent(
                balance=session.payment_balance or "0",
                currency=session.payment_currency,
            )
            await ws.send_json(balance_event.to_dict())
        
        return session
    
    async def _handle_session_update(
        self,
        ws: 'WebSocket',
        session: UAMPSession,
        event: Dict[str, Any]
    ) -> None:
        """Update session configuration. Supports payment_token for pre-loading tokens."""
        session_config = event.get("session", {})
        payment_token = event.get("payment_token") or session_config.get("payment_token")

        if "modalities" in session_config:
            session.modalities = session_config["modalities"]
        if "instructions" in session_config:
            session.instructions = session_config["instructions"]
        if payment_token:
            session.payment_token = payment_token

        await self._send_event(ws, "session.updated", {
            "session": session.to_dict()
        })
    
    async def _handle_capabilities_query(
        self,
        ws: 'WebSocket',
        session: UAMPSession,
        event: Dict[str, Any]
    ) -> None:
        """Handle capabilities query."""
        capabilities_event = CapabilitiesEvent(
            capabilities=ModelCapabilities(
                model_id=getattr(self.agent, 'model', 'unknown'),
                provider=getattr(self.agent, 'provider', 'webagents'),
                modalities=session.modalities,
                supports_streaming=True,
            )
        )
        await ws.send_json(capabilities_event.to_dict())
    
    def _spawn_response(self, ws: 'WebSocket', session: UAMPSession, session_id: str | None = None, messages: list | None = None, payment_token: str | None = None) -> None:
        """Spawn _generate_response as a background task.

        This is critical: the main WS message loop must stay unblocked so it
        can receive follow-up events (e.g. ``payment.submit``) while the
        response is being generated.  If we ``await`` _generate_response
        directly, the message loop deadlocks during payment negotiation.
        """
        # Cancel any in-flight response for this session
        if session._response_task and not session._response_task.done():
            session._response_task.cancel()

        async def _safe_generate() -> None:
            try:
                await self._generate_response(ws, session, session_id=session_id, messages=messages, payment_token=payment_token)
            except asyncio.CancelledError:
                pass
            except Exception:
                # Already handled inside _generate_response; guard against leaks
                pass

        session._response_task = asyncio.ensure_future(_safe_generate())

    async def _handle_input(
        self,
        ws: 'WebSocket',
        session: UAMPSession,
        event: Dict[str, Any]
    ) -> None:
        """Handle input (text, multimodal) and generate response.
        
        Supports stateless mode: if event contains 'messages' array, uses it
        as the authoritative conversation context (platform sends full history).
        Falls back to session.conversation for backward compatibility.
        """
        text = event.get("text", "")
        role = event.get("role", "user")
        session_id = event.get("session_id")
        messages = event.get("messages")  # Stateless: full conversation from platform
        payment_token = event.get("payment_token")
        
        if messages:
            # Stateless mode: use platform-provided messages as conversation context
            session.conversation = list(messages)
        else:
            # Backward compat: append text to session.conversation
            session.conversation.append({
                "role": role,
                "content": text,
            })
        
        # If user message, trigger response (background -- keeps WS loop free)
        if role == "user":
            self._spawn_response(ws, session, session_id=session_id, messages=messages, payment_token=payment_token)
    
    async def _handle_input_audio(
        self,
        ws: 'WebSocket',
        session: UAMPSession,
        event: Dict[str, Any]
    ) -> None:
        """Handle audio input.

        Receives audio data (base64) and sends it through the LLM as an
        ``input_audio`` content part.  Models that support audio natively
        (e.g. Gemini) will process it directly; for others, litellm will
        fall back to text-only processing.
        """
        audio_data = event.get("audio", "")
        audio_format = event.get("format", "webm")
        is_final = event.get("is_final", False)

        if not audio_data:
            return

        # Accumulate audio chunks in session; send to LLM when is_final or
        # when we receive a standalone chunk (non-streaming push-to-talk).
        if not hasattr(session, '_audio_chunks'):
            session._audio_chunks = []  # type: ignore[attr-defined]

        session._audio_chunks.append(audio_data)  # type: ignore[attr-defined]

        # If is_final or there is no prior chunk (single-shot), flush.
        # For streaming voice calls, the browser sends is_final=true when
        # silence is detected.  As a fallback, we also flush if a chunk
        # arrives > 2 s after the last one (handled externally by timer).
        if not is_final and len(session._audio_chunks) > 1:  # type: ignore[attr-defined]
            # Still accumulating – wait for is_final
            return

        # Merge all chunks into one base64 payload
        merged_audio = "".join(session._audio_chunks)  # type: ignore[attr-defined]
        session._audio_chunks = []  # type: ignore[attr-defined]

        # Add audio to conversation as multimodal content (OpenAI input_audio format)
        session.conversation.append({
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": merged_audio,
                        "format": audio_format,
                    },
                }
            ],
        })

        # Generate response (background -- keeps WS loop free)
        self._spawn_response(ws, session)
    
    async def _handle_input_image(
        self,
        ws: 'WebSocket',
        session: UAMPSession,
        event: Dict[str, Any]
    ) -> None:
        """Handle image input."""
        image = event.get("image", "")
        detail = event.get("detail", "auto")
        
        # Add image to conversation as multimodal content
        session.conversation.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image, "detail": detail}}
            ]
        })
    
    async def _handle_input_video(
        self,
        ws: 'WebSocket',
        session: UAMPSession,
        event: Dict[str, Any]
    ) -> None:
        """Handle video input.

        Accepts a URL or base64-encoded video. Models with native video
        support (e.g. Gemini) will process the URL directly; others will
        see the URL as context in a text fallback.
        """
        video = event.get("video", "")
        video_url = video if isinstance(video, str) else video.get("url", "")

        # Use image_url format -- Gemini and compatible models accept
        # video URLs through the same multimodal content part.
        session.conversation.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": video_url}},
            ]
        })

    async def _handle_input_file(
        self,
        ws: 'WebSocket',
        session: UAMPSession,
        event: Dict[str, Any]
    ) -> None:
        """Handle file/document input.

        Accepts a URL (typically a signed content URL) and passes it to
        the LLM. Modern models (GPT-4o, Gemini, Claude) can process
        PDFs and other documents from URLs. The platform also sends
        extracted text alongside the URL as a fallback.
        """
        file_ref = event.get("file", "")
        file_url = file_ref if isinstance(file_ref, str) else file_ref.get("url", "")
        filename = event.get("filename", "document")
        mime_type = event.get("mime_type", "application/octet-stream")

        content_parts = []

        # Include the document URL for models that can fetch it
        if file_url:
            content_parts.append(
                {"type": "image_url", "image_url": {"url": file_url}}
            )

        # If no URL, note the attachment as text
        if not content_parts:
            content_parts.append(
                {"type": "text", "text": f"[Document attached: {filename} ({mime_type})]"}
            )

        session.conversation.append({
            "role": "user",
            "content": content_parts,
        })

    async def _handle_input_typing(
        self,
        ws: 'WebSocket',
        session: UAMPSession,
        event: Dict[str, Any]
    ) -> None:
        """Handle typing indicator."""
        is_typing = event.get("is_typing", True)
        conversation_id = event.get("conversation_id")
        
        # Broadcast to other participants in the conversation
        # This would integrate with a presence system
        typing_event = PresenceTypingEvent(
            user_id=getattr(self.agent, 'id', 'agent'),
            username=getattr(self.agent, 'name', 'Agent'),
            is_typing=is_typing,
            conversation_id=conversation_id,
        )
        # In a multi-user scenario, broadcast to other connections
        # For now, just acknowledge
    
    async def _handle_response_create(
        self,
        ws: 'WebSocket',
        session: UAMPSession,
        event: Dict[str, Any]
    ) -> None:
        """Explicitly request response generation."""
        self._spawn_response(ws, session)
    
    async def _handle_response_cancel(
        self,
        ws: 'WebSocket',
        session: UAMPSession,
        event: Dict[str, Any]
    ) -> None:
        """Cancel current response."""
        if session._response_task and not session._response_task.done():
            session._response_task.cancel()
            session._response_task = None
        await self._send_event(ws, "response.cancelled", {
            "response_id": event.get("response_id")
        })
    
    async def _handle_tool_result(
        self,
        ws: 'WebSocket',
        session: UAMPSession,
        event: Dict[str, Any]
    ) -> None:
        """Handle tool execution result from client."""
        call_id = event.get("call_id", "")
        result = event.get("result", "")
        is_error = event.get("is_error", False)
        
        # Add tool result to conversation
        session.conversation.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": result,
        })
        
        # Continue response generation with tool result (background)
        self._spawn_response(ws, session)
    
    async def _handle_payment_submit(
        self,
        ws: 'WebSocket',
        session: UAMPSession,
        event: Dict[str, Any]
    ) -> None:
        """Handle payment token submission. Resolves pending payment negotiation if active."""
        payment = event.get("payment", {})
        scheme = payment.get("scheme", "token")
        token = payment.get("token")
        amount = payment.get("amount", "0")

        if token:
            session.payment_token = token
            # Resolve any pending payment.required wait (so _generate_response can retry)
            fut = self._pending_payment_futures.pop(session.id, None)
            if fut is not None and not fut.done():
                fut.set_result(token)

        # Legacy balance tracking (optional)
        payment_id = f"pay_{uuid.uuid4().hex[:12]}"
        session.payment_balance = amount
        session.payment_token_expires_at = int(time.time() * 1000) + (3600 * 1000)  # 1 hour

        accepted_event = PaymentAcceptedEvent(
            payment_id=payment_id,
            balance_remaining=amount,
            expires_at=session.payment_token_expires_at,
        )
        await ws.send_json(accepted_event.to_dict())
    
    async def _handle_ping(
        self,
        ws: 'WebSocket',
        session: UAMPSession,
        event: Dict[str, Any]
    ) -> None:
        """Handle keepalive ping."""
        pong = PongEvent()
        await ws.send_json(pong.to_dict())
    
    async def _generate_response(
        self,
        ws: 'WebSocket',
        session: UAMPSession,
        session_id: str | None = None,
        messages: list | None = None,
        payment_token: str | None = None,
    ) -> None:
        """Generate and stream response. Handles payment.required / payment.submit negotiation.
        
        Args:
            session_id: Interaction session_id to echo on all outgoing events.
            messages: Stateless conversation context from platform (if provided).
            payment_token: Payment token for this interaction.
        """
        response_id = f"resp_{uuid.uuid4().hex[:12]}"

        # Helper to attach session_id to outgoing events
        def _attach_session_id(event_obj):
            if session_id is not None and hasattr(event_obj, 'session_id'):
                event_obj.session_id = session_id
            elif session_id is not None and hasattr(event_obj, 'to_dict'):
                pass  # handled in to_dict via BaseEvent.session_id
            return event_obj

        # Set transport-agnostic payment token from event or session
        context = self.get_context()
        if context is not None:
            if hasattr(context, 'usage'):
                context.usage = []
            # Prefer per-request payment_token, fall back to session-level
            effective_token = payment_token or session.payment_token
            if effective_token:
                context.payment_token = effective_token

        # Check payment balance if required (legacy optional check)
        if hasattr(self, "requires_payment") and self.requires_payment:
            if not session.payment_balance or float(session.payment_balance) <= 0:
                required_event = PaymentRequiredEvent(
                    response_id=response_id,
                    requirements=PaymentRequirements(
                        amount="1.00",
                        currency=session.payment_currency,
                        schemes=[PaymentScheme(scheme="token", network="robutler")],
                        reason="llm_usage",
                    ),
                )
                await ws.send_json(required_event.to_dict())
                return

        # Send response.created
        created_event = ResponseCreatedEvent(response_id=response_id, session_id=session_id)
        await ws.send_json(created_event.to_dict())

        messages = self._build_messages(session)
        full_text = ""
        payment_negotiated = False

        try:
            while True:
                try:
                    async for chunk in self.execute_handoff(messages):
                        # Handle tool_call events from the agentic loop
                        if isinstance(chunk, dict) and chunk.get("type") == "tool_call":
                            tc_event = ResponseDeltaEvent(
                                response_id=response_id,
                                delta=ContentDelta(
                                    type="tool_call",
                                    tool_call={
                                        "id": chunk.get("call_id", ""),
                                        "name": chunk.get("name", ""),
                                        "arguments": chunk.get("arguments", ""),
                                    },
                                ),
                                session_id=session_id,
                            )
                            await ws.send_json(tc_event.to_dict())
                            continue
                        # Handle tool_result events
                        if isinstance(chunk, dict) and chunk.get("type") == "tool_result":
                            tr_event = ResponseDeltaEvent(
                                response_id=response_id,
                                delta=ContentDelta(
                                    type="tool_result",
                                    tool_result={
                                        "call_id": chunk.get("id", ""),
                                        "result": chunk.get("result", ""),
                                        "status": chunk.get("status", "success"),
                                    },
                                ),
                                session_id=session_id,
                            )
                            await ws.send_json(tr_event.to_dict())
                            continue
                        if isinstance(chunk, dict) and chunk.get("type") == "tool_progress":
                            tp_call_id = chunk.get("call_id", "")
                            tp_text = chunk.get("text", "")
                            self.logger.debug(f"📡 UAMP sending tool_progress: call_id={tp_call_id} text_len={len(tp_text)}")
                            tp_event = ResponseDeltaEvent(
                                response_id=response_id,
                                delta=ContentDelta(
                                    type="tool_progress",
                                    tool_progress={
                                        "call_id": tp_call_id,
                                        "text": tp_text,
                                    },
                                ),
                                session_id=session_id,
                            )
                            await ws.send_json(tp_event.to_dict())
                            continue
                        delta_text = self._extract_delta_text(chunk)
                        if delta_text:
                            full_text += delta_text
                            delta_event = ResponseDeltaEvent(
                                response_id=response_id,
                                delta=ContentDelta(type="text", text=delta_text),
                                session_id=session_id,
                            )
                            await ws.send_json(delta_event.to_dict())
                    break
                except Exception as e:
                    # Check for retryable payment errors (token required, insufficient
                    # balance on token, invalid/expired token).  These all mean "ask the
                    # client for a (new) payment token, then retry".
                    is_retryable_payment = False
                    if PaymentError is not None and isinstance(e, PaymentError):
                        error_code = getattr(e, "error_code", "")
                        is_retryable_payment = error_code in _RETRYABLE_PAYMENT_CODES

                    if is_retryable_payment:
                        # Build payment.required from error context
                        err_ctx = getattr(e, "context", None) or {}
                        accepts = err_ctx.get("accepts") or []
                        amount = "0.01"
                        schemes_list = [PaymentScheme(scheme="token", network="robutler")]

                        # Try to extract amount from error context
                        if err_ctx.get("required_balance"):
                            amount = str(err_ctx["required_balance"])
                        if accepts:
                            first = accepts[0] if isinstance(accepts, list) else {}
                            if isinstance(first, dict):
                                amount = str(first.get("maxAmountRequired") or first.get("amount") or amount)
                                schemes_list = [
                                    PaymentScheme(
                                        scheme=first.get("scheme", "token"),
                                        network=first.get("network", "robutler"),
                                        max_amount=first.get("maxAmountRequired"),
                                    )
                                ]

                        required_event = PaymentRequiredEvent(
                            response_id=response_id,
                            session_id=session_id,
                            requirements=PaymentRequirements(
                                amount=amount,
                                currency=session.payment_currency,
                                schemes=schemes_list,
                                reason="llm_usage",
                            ),
                        )
                        await ws.send_json(required_event.to_dict())

                        # Wait for payment.submit (client sends token via the WS
                        # message loop, which is now free because _generate_response
                        # runs as a background task).
                        fut: asyncio.Future[str] = asyncio.get_running_loop().create_future()
                        self._pending_payment_futures[session.id] = fut
                        try:
                            token = await asyncio.wait_for(fut, timeout=60.0)
                        except asyncio.TimeoutError:
                            self._pending_payment_futures.pop(session.id, None)
                            error_event = ResponseErrorEvent(
                                response_id=response_id,
                                session_id=session_id,
                                error={"code": "payment_timeout", "message": "Payment token not received in time"},
                            )
                            await ws.send_json(error_event.to_dict())
                            return
                        self._pending_payment_futures.pop(session.id, None)
                        session.payment_token = token
                        if context is not None:
                            context.payment_token = token
                        payment_negotiated = True
                        continue  # Retry execute_handoff
                    raise
        except asyncio.CancelledError:
            self.logger.info(f"[uamp] Response {response_id} cancelled")
            if full_text:
                session.conversation.append({"role": "assistant", "content": full_text})
            return
        except Exception as e:
            error_event = ResponseErrorEvent(
                response_id=response_id,
                session_id=session_id,
                error={
                    "code": "generation_error",
                    "message": str(e),
                },
            )
            await ws.send_json(error_event.to_dict())
            return

        # Add assistant response to conversation
        session.conversation.append({"role": "assistant", "content": full_text})

        # Send done
        done_event = ResponseDoneEvent(
            response_id=response_id,
            session_id=session_id,
            response=ResponseOutput(
                id=response_id,
                status="completed",
                output=[ContentItem(type="text", text=full_text)],
                usage=UsageStats(
                    input_tokens=0,
                    output_tokens=len(full_text.split()),
                    total_tokens=len(full_text.split()),
                ),
            ),
        )
        await ws.send_json(done_event.to_dict())

        if payment_negotiated:
            payment_id = f"pay_{uuid.uuid4().hex[:12]}"
            accepted_event = PaymentAcceptedEvent(
                payment_id=payment_id,
                balance_remaining=session.payment_balance or "0",
                expires_at=session.payment_token_expires_at,
            )
            await ws.send_json(accepted_event.to_dict())

        # Update payment balance if tracking
        if session.payment_balance:
            new_balance = max(0, float(session.payment_balance) - 0.01)
            session.payment_balance = str(new_balance)
            balance_event = PaymentBalanceEvent(
                balance=session.payment_balance,
                currency=session.payment_currency,
                low_balance_warning=new_balance < 1.0,
                expires_at=session.payment_token_expires_at,
            )
            await ws.send_json(balance_event.to_dict())
    
    def _build_messages(self, session: UAMPSession) -> List[Dict[str, Any]]:
        """Build messages list from session conversation."""
        messages: List[Dict[str, Any]] = []

        # Add system instructions
        if session.instructions:
            messages.append({
                "role": "system",
                "content": session.instructions,
            })

        messages.extend(session.conversation)
        return messages
    
    def _extract_delta_text(self, chunk: Dict[str, Any]) -> str:
        """Extract text from streaming chunk."""
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            return delta.get("content", "")
        return ""
    
    async def _send_event(
        self,
        ws: 'WebSocket',
        event_type: str,
        data: Dict[str, Any]
    ) -> None:
        """Send a generic event."""
        event = {
            "type": event_type,
            "event_id": f"evt_{uuid.uuid4().hex[:12]}",
            "timestamp": int(time.time() * 1000),
            **data
        }
        await ws.send_json(event)
    
    async def _send_error(
        self,
        ws: 'WebSocket',
        code: str,
        message: str
    ) -> None:
        """Send an error event."""
        await self._send_event(ws, "error", {
            "error": {
                "code": code,
                "message": message,
            }
        })
