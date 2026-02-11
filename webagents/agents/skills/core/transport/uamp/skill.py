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
    from webagents.agents.skills.robutler.payments.exceptions import PaymentTokenRequiredError
except ImportError:
    PaymentTokenRequiredError = None  # type: ignore[misc, assignment]


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
        """Cancel all pending event futures (session is closing)."""
        for futures_list in self._pending_events.values():
            for fut in futures_list:
                if not fut.done():
                    fut.cancel()
        self._pending_events.clear()

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
                    "input.text": self._handle_input_text,
                    "input.audio": self._handle_input_audio,
                    "input.image": self._handle_input_image,
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
    
    async def _handle_input_text(
        self,
        ws: 'WebSocket',
        session: UAMPSession,
        event: Dict[str, Any]
    ) -> None:
        """Handle text input and generate response."""
        text = event.get("text", "")
        role = event.get("role", "user")
        
        # Add to conversation
        session.conversation.append({
            "role": role,
            "content": text,
        })
        
        # If user message, trigger response
        if role == "user":
            await self._generate_response(ws, session)
    
    async def _handle_input_audio(
        self,
        ws: 'WebSocket',
        session: UAMPSession,
        event: Dict[str, Any]
    ) -> None:
        """Handle audio input."""
        # For audio, we'd transcribe and then process as text
        # This is a placeholder - full implementation would use STT
        await self._send_event(ws, "transcript.delta", {
            "response_id": f"resp_{uuid.uuid4().hex[:12]}",
            "transcript": "[audio transcription not implemented]"
        })
    
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
        await self._generate_response(ws, session)
    
    async def _handle_response_cancel(
        self,
        ws: 'WebSocket',
        session: UAMPSession,
        event: Dict[str, Any]
    ) -> None:
        """Cancel current response."""
        # Would need to track active responses to cancel
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
        
        # Continue response generation with tool result
        await self._generate_response(ws, session)
    
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
        session: UAMPSession
    ) -> None:
        """Generate and stream response. Handles payment.required / payment.submit negotiation."""
        response_id = f"resp_{uuid.uuid4().hex[:12]}"

        # Set transport-agnostic payment token from session if present (pre-loaded or from prior payment.submit)
        context = self.get_context()
        if context is not None and session.payment_token:
            context.payment_token = session.payment_token

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
        created_event = ResponseCreatedEvent(response_id=response_id)
        await ws.send_json(created_event.to_dict())

        messages = self._build_messages(session)
        full_text = ""
        payment_negotiated = False

        try:
            while True:
                try:
                    async for chunk in self.execute_handoff(messages):
                        delta_text = self._extract_delta_text(chunk)
                        if delta_text:
                            full_text += delta_text
                            delta_event = ResponseDeltaEvent(
                                response_id=response_id,
                                delta=ContentDelta(type="text", text=delta_text),
                            )
                            await ws.send_json(delta_event.to_dict())
                    break
                except Exception as e:
                    if PaymentTokenRequiredError is not None and isinstance(e, PaymentTokenRequiredError):
                        # Build payment.required from error context
                        accepts = (getattr(e, "context", None) or {}).get("accepts") or []
                        amount = "0.01"
                        schemes_list = [PaymentScheme(scheme="token", network="robutler")]
                        if accepts:
                            first = accepts[0] if isinstance(accepts, list) else {}
                            if isinstance(first, dict):
                                amount = str(first.get("maxAmountRequired", amount))
                                schemes_list = [
                                    PaymentScheme(
                                        scheme=first.get("scheme", "token"),
                                        network=first.get("network", "robutler"),
                                        max_amount=first.get("maxAmountRequired"),
                                    )
                                ]
                        required_event = PaymentRequiredEvent(
                            response_id=response_id,
                            requirements=PaymentRequirements(
                                amount=amount,
                                currency=session.payment_currency,
                                schemes=schemes_list,
                                reason="llm_usage",
                            ),
                        )
                        await ws.send_json(required_event.to_dict())

                        # Wait for payment.submit (client sends token)
                        fut: asyncio.Future[str] = asyncio.get_running_loop().create_future()
                        self._pending_payment_futures[session.id] = fut
                        try:
                            token = await asyncio.wait_for(fut, timeout=60.0)
                        except asyncio.TimeoutError:
                            self._pending_payment_futures.pop(session.id, None)
                            error_event = ResponseErrorEvent(
                                response_id=response_id,
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
        except Exception as e:
            error_event = ResponseErrorEvent(
                response_id=response_id,
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
        messages = []
        
        # Add system instructions
        if session.instructions:
            messages.append({
                "role": "system",
                "content": session.instructions,
            })
        
        # Add conversation history
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
