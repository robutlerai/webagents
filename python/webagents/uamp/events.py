"""
UAMP Event Definitions.

All UAMP communication is event-based. Events flow bidirectionally between clients and agents.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Literal
import uuid
import time

from .types import (
    SessionConfig,
    Session,
    ContentDelta,
    ContentItem,
    UsageStats,
    ResponseOutput,
    ToolDefinition,
    ModelCapabilities,
    ClientCapabilities,
)


def generate_event_id() -> str:
    """Generate a unique event ID."""
    return f"evt_{uuid.uuid4().hex[:12]}"


def current_timestamp() -> int:
    """Get current timestamp in milliseconds."""
    return int(time.time() * 1000)


# =============================================================================
# Base Event
# =============================================================================

@dataclass
class BaseEvent:
    """Base event structure shared by all events."""
    type: str
    event_id: str = field(default_factory=generate_event_id)
    timestamp: Optional[int] = field(default_factory=current_timestamp)
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        result = {
            "type": self.type,
            "event_id": self.event_id,
        }
        if self.timestamp is not None:
            result["timestamp"] = self.timestamp
        if self.session_id is not None:
            result["session_id"] = self.session_id
        return result


# =============================================================================
# Session Events
# =============================================================================

@dataclass
class SessionCreateEvent(BaseEvent):
    """Client event to create a new session.
    
    Can include client_capabilities (unified Capabilities format) to inform
    the agent what the client can render and handle.
    """
    type: Literal["session.create"] = "session.create"
    uamp_version: str = "1.0"
    session: Optional[SessionConfig] = None
    client_capabilities: Optional[ClientCapabilities] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["uamp_version"] = self.uamp_version
        if self.session:
            result["session"] = {
                "modalities": self.session.modalities,
                "instructions": self.session.instructions,
                "tools": [t.function if hasattr(t, 'function') else t for t in (self.session.tools or [])],
                "extensions": self.session.extensions,
            }
            if self.session.response_format is not None:
                result["session"]["response_format"] = self.session.response_format
        if self.client_capabilities:
            cap = self.client_capabilities
            cap_dict = {
                "id": cap.id,
                "provider": cap.provider,
                "modalities": cap.modalities,
                "supports_streaming": cap.supports_streaming,
            }
            if cap.widgets:
                cap_dict["widgets"] = cap.widgets
            if cap.extensions:
                cap_dict["extensions"] = cap.extensions
            result["client_capabilities"] = cap_dict
        return result


@dataclass
class SessionCreatedEvent(BaseEvent):
    """Server event confirming session creation."""
    type: Literal["session.created"] = "session.created"
    uamp_version: str = "1.0"
    session: Optional[Session] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["uamp_version"] = self.uamp_version
        if self.session:
            result["session"] = {
                "id": self.session.id,
                "created_at": self.session.created_at,
                "status": self.session.status,
            }
        return result


@dataclass
class SessionUpdateEvent(BaseEvent):
    """Client event to update session configuration."""
    type: Literal["session.update"] = "session.update"
    session: Optional[SessionConfig] = None


@dataclass
class SessionUpdatedEvent(BaseEvent):
    """Server event confirming session update."""
    type: Literal["session.updated"] = "session.updated"
    session: Optional[Session] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.session:
            result["session"] = {
                "id": self.session.id,
                "created_at": self.session.created_at,
                "status": self.session.status,
            }
        return result


@dataclass
class CapabilitiesQueryEvent(BaseEvent):
    """Client event to query model/agent capabilities.
    
    Can be sent at any time to get current capabilities.
    Server responds with CapabilitiesEvent.
    """
    type: Literal["capabilities.query"] = "capabilities.query"
    model: Optional[str] = None  # Query for specific model, or current if omitted


@dataclass
class ClientCapabilitiesEvent(BaseEvent):
    """Client event announcing client capabilities.
    
    Uses unified Capabilities structure - same format as model/agent.
    
    Example:
        ClientCapabilitiesEvent(
            capabilities=Capabilities(
                id="web-app",
                provider="robutler",
                modalities=["text", "image", "audio"],
                widgets=["chart", "form"],
                extensions={"supports_html": True}
            )
        )
    """
    type: Literal["client.capabilities"] = "client.capabilities"
    capabilities: Optional[ClientCapabilities] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.capabilities:
            cap = self.capabilities
            cap_dict = {
                "id": cap.id,
                "provider": cap.provider,
                "modalities": cap.modalities,
                "supports_streaming": cap.supports_streaming,
            }
            if cap.widgets:
                cap_dict["widgets"] = cap.widgets
            if cap.provides:
                cap_dict["provides"] = cap.provides
            if cap.extensions:
                cap_dict["extensions"] = cap.extensions
            result["capabilities"] = cap_dict
        return result


@dataclass
class CapabilitiesEvent(BaseEvent):
    """Server event announcing model/agent capabilities.
    
    Sent after session.created to inform clients what the model supports.
    Enables clients to adapt their UI (show/hide image upload, audio button, etc.)
    """
    type: Literal["capabilities"] = "capabilities"
    capabilities: Optional[ModelCapabilities] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.capabilities:
            cap = self.capabilities
            cap_dict = {
                "model_id": cap.model_id,
                "provider": cap.provider,
                "modalities": cap.modalities,
                "supports_streaming": cap.supports_streaming,
                "supports_thinking": cap.supports_thinking,
                "supports_caching": cap.supports_caching,
            }
            if cap.context_window:
                cap_dict["context_window"] = cap.context_window
            if cap.max_output_tokens:
                cap_dict["max_output_tokens"] = cap.max_output_tokens
            if cap.image:
                cap_dict["image"] = {
                    "formats": cap.image.formats,
                    "detail_levels": cap.image.detail_levels,
                }
                if cap.image.max_size_bytes:
                    cap_dict["image"]["max_size_bytes"] = cap.image.max_size_bytes
                if cap.image.max_images_per_request:
                    cap_dict["image"]["max_images_per_request"] = cap.image.max_images_per_request
            if cap.audio:
                cap_dict["audio"] = {
                    "input_formats": cap.audio.input_formats,
                    "output_formats": cap.audio.output_formats,
                    "supports_realtime": cap.audio.supports_realtime,
                }
                if cap.audio.voices:
                    cap_dict["audio"]["voices"] = cap.audio.voices
            if cap.file:
                cap_dict["file"] = {
                    "supported_mime_types": cap.file.supported_mime_types,
                    "supports_pdf": cap.file.supports_pdf,
                    "supports_code": cap.file.supports_code,
                }
                if cap.file.max_size_bytes:
                    cap_dict["file"]["max_size_bytes"] = cap.file.max_size_bytes
            if cap.tools:
                cap_dict["tools"] = {
                    "supports_tools": cap.tools.supports_tools,
                    "supports_parallel_tools": cap.tools.supports_parallel_tools,
                    "built_in_tools": cap.tools.built_in_tools,
                }
            if cap.extensions:
                cap_dict["extensions"] = cap.extensions
            result["capabilities"] = cap_dict
        return result


# =============================================================================
# Input Events
# =============================================================================

@dataclass
class SessionEndEvent(BaseEvent):
    """Either side ends a session."""
    type: Literal["session.end"] = "session.end"
    reason: Optional[str] = None  # 'user_left', 'timeout', 'daemon_takeover', etc.

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.reason:
            result["reason"] = self.reason
        return result


@dataclass
class SessionErrorEvent(BaseEvent):
    """Session-level error."""
    type: Literal["session.error"] = "session.error"
    error: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["error"] = self.error
        return result


@dataclass
class InputTextEvent(BaseEvent):
    """Client event for text input."""
    type: Literal["input.text"] = "input.text"
    text: str = ""
    role: str = "user"  # user, system

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["text"] = self.text
        result["role"] = self.role
        return result


@dataclass
class InputAudioEvent(BaseEvent):
    """Client event for audio input."""
    type: Literal["input.audio"] = "input.audio"
    audio: str = ""  # Base64 encoded
    format: str = "pcm16"
    is_final: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["audio"] = self.audio
        result["format"] = self.format
        result["is_final"] = self.is_final
        return result


@dataclass
class InputImageEvent(BaseEvent):
    """Client event for image input."""
    type: Literal["input.image"] = "input.image"
    image: Union[str, Dict[str, str]] = ""  # Base64 or {"url": "..."}
    format: Optional[str] = None  # jpeg, png, webp, gif
    detail: Optional[str] = None  # low, high, auto

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["image"] = self.image
        if self.format:
            result["format"] = self.format
        if self.detail:
            result["detail"] = self.detail
        return result


@dataclass
class InputVideoEvent(BaseEvent):
    """Client event for video input."""
    type: Literal["input.video"] = "input.video"
    video: Union[str, Dict[str, str]] = ""  # Base64 or {"url": "..."}
    format: Optional[str] = None  # mp4, webm


@dataclass
class InputFileEvent(BaseEvent):
    """Client event for file input."""
    type: Literal["input.file"] = "input.file"
    file: Union[str, Dict[str, str]] = ""  # Base64 or {"url": "..."}
    filename: str = ""
    mime_type: str = ""


# =============================================================================
# Response Events
# =============================================================================

@dataclass
class InputAudioCommittedEvent(BaseEvent):
    """Client event indicating audio input buffer was committed (realtime mode)."""
    type: Literal["input.audio_committed"] = "input.audio_committed"
    duration_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        return result


@dataclass
class ResponseCreateEvent(BaseEvent):
    """Client event to request a response."""
    type: Literal["response.create"] = "response.create"
    response: Optional[Dict[str, Any]] = None  # Override options
    response_format: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.response:
            result["response"] = self.response
        if self.response_format is not None:
            result["response_format"] = self.response_format
        return result


@dataclass
class ResponseCancelEvent(BaseEvent):
    """Client event to cancel a response."""
    type: Literal["response.cancel"] = "response.cancel"
    response_id: Optional[str] = None


@dataclass
class ResponseCreatedEvent(BaseEvent):
    """Server event confirming response started."""
    type: Literal["response.created"] = "response.created"
    response_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["response_id"] = self.response_id
        return result


@dataclass
class ResponseDeltaEvent(BaseEvent):
    """Server event for streaming content."""
    type: Literal["response.delta"] = "response.delta"
    response_id: str = ""
    delta: Optional[ContentDelta] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["response_id"] = self.response_id
        if self.delta:
            delta_dict = {"type": self.delta.type}
            if self.delta.text is not None:
                delta_dict["text"] = self.delta.text
            if self.delta.audio is not None:
                delta_dict["audio"] = self.delta.audio
            if self.delta.tool_call is not None:
                delta_dict["tool_call"] = self.delta.tool_call
            if self.delta.tool_result is not None:
                delta_dict["tool_result"] = self.delta.tool_result
            if self.delta.tool_progress is not None:
                delta_dict["tool_progress"] = self.delta.tool_progress
            result["delta"] = delta_dict
        return result


@dataclass
class ResponseDoneEvent(BaseEvent):
    """Server event for response completion."""
    type: Literal["response.done"] = "response.done"
    response_id: str = ""
    response: Optional[ResponseOutput] = None
    signature: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["response_id"] = self.response_id
        if self.response:
            response_dict = {
                "id": self.response.id,
                "status": self.response.status,
                "output": [
                    {
                        "type": item.type,
                        "text": item.text,
                        "audio": item.audio,
                        "image": item.image,
                        "tool_call": item.tool_call,
                        "tool_result": item.tool_result,
                    }
                    for item in self.response.output
                ],
            }
            if self.response.usage:
                response_dict["usage"] = {
                    "input_tokens": self.response.usage.input_tokens,
                    "output_tokens": self.response.usage.output_tokens,
                    "total_tokens": self.response.usage.total_tokens,
                }
            result["response"] = response_dict
        if self.signature is not None:
            result["signature"] = self.signature
        return result


@dataclass
class ResponseCancelledEvent(BaseEvent):
    """Server event confirming response cancellation."""
    type: Literal["response.cancelled"] = "response.cancelled"
    response_id: str = ""
    partial_output: Optional[List[ContentItem]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["response_id"] = self.response_id
        if self.partial_output is not None:
            result["partial_output"] = [
                {"type": item.type, "text": item.text}
                for item in self.partial_output
            ]
        return result


@dataclass
class ResponseErrorEvent(BaseEvent):
    """Server event for response error."""
    type: Literal["response.error"] = "response.error"
    response_id: Optional[str] = None
    error: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.response_id:
            result["response_id"] = self.response_id
        result["error"] = self.error
        return result


# =============================================================================
# Tool Events
# =============================================================================

@dataclass
class ToolCallEvent(BaseEvent):
    """Server event requesting tool execution."""
    type: Literal["tool.call"] = "tool.call"
    call_id: str = ""
    name: str = ""
    arguments: str = ""  # JSON string

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["call_id"] = self.call_id
        result["name"] = self.name
        result["arguments"] = self.arguments
        return result


@dataclass
class ToolResultEvent(BaseEvent):
    """Client event providing tool result."""
    type: Literal["tool.result"] = "tool.result"
    call_id: str = ""
    result: str = ""  # JSON string
    is_error: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["call_id"] = self.call_id
        result["result"] = self.result
        if self.is_error:
            result["is_error"] = True
        return result


@dataclass
class ToolCallDoneEvent(BaseEvent):
    """Server event indicating tool call completed."""
    type: Literal["tool.call_done"] = "tool.call_done"
    call_id: str = ""


# =============================================================================
# Audio Events
# =============================================================================

@dataclass
class AudioDeltaEvent(BaseEvent):
    """Server event for streaming audio."""
    type: Literal["audio.delta"] = "audio.delta"
    response_id: str = ""
    audio: str = ""  # Base64 encoded chunk

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["response_id"] = self.response_id
        result["audio"] = self.audio
        return result


@dataclass
class TranscriptDeltaEvent(BaseEvent):
    """Server event for real-time transcription."""
    type: Literal["transcript.delta"] = "transcript.delta"
    response_id: str = ""
    transcript: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["response_id"] = self.response_id
        result["transcript"] = self.transcript
        return result


@dataclass
class AudioDoneEvent(BaseEvent):
    """Server event indicating audio output completed for a response."""
    type: Literal["audio.done"] = "audio.done"
    response_id: str = ""
    duration_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["response_id"] = self.response_id
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        return result


@dataclass
class TranscriptDoneEvent(BaseEvent):
    """Server event indicating transcript completed."""
    type: Literal["transcript.done"] = "transcript.done"
    response_id: str = ""
    transcript: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["response_id"] = self.response_id
        result["transcript"] = self.transcript
        return result


# =============================================================================
# Progress Events
# =============================================================================

@dataclass
class ProgressEvent(BaseEvent):
    """Server event for long-running operation status updates."""
    type: Literal["progress"] = "progress"
    target: str = "response"  # tool, response, upload, reasoning
    target_id: Optional[str] = None
    stage: Optional[str] = None
    message: Optional[str] = None
    percent: Optional[int] = None
    step: Optional[int] = None
    total_steps: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["target"] = self.target
        if self.target_id:
            result["target_id"] = self.target_id
        if self.stage:
            result["stage"] = self.stage
        if self.message:
            result["message"] = self.message
        if self.percent is not None:
            result["percent"] = self.percent
        if self.step is not None:
            result["step"] = self.step
        if self.total_steps is not None:
            result["total_steps"] = self.total_steps
        return result


@dataclass
class ThinkingEvent(BaseEvent):
    """Server event for reasoning content (like extended thinking)."""
    type: Literal["thinking"] = "thinking"
    content: str = ""
    stage: Optional[str] = None  # analyzing, planning, reflecting
    redacted: bool = False
    is_delta: bool = False  # True = append, False = complete thought

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["content"] = self.content
        if self.stage:
            result["stage"] = self.stage
        if self.redacted:
            result["redacted"] = self.redacted
        if self.is_delta:
            result["is_delta"] = self.is_delta
        return result


# =============================================================================
# Usage Events
# =============================================================================

@dataclass
class UsageDeltaEvent(BaseEvent):
    """Server event for incremental usage updates."""
    type: Literal["usage.delta"] = "usage.delta"
    response_id: str = ""
    delta: Optional[UsageStats] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["response_id"] = self.response_id
        if self.delta:
            result["delta"] = {
                "input_tokens": self.delta.input_tokens,
                "output_tokens": self.delta.output_tokens,
                "total_tokens": self.delta.total_tokens,
            }
        return result


# =============================================================================
# Presence Events
# =============================================================================

@dataclass
class InputTypingEvent(BaseEvent):
    """Client event indicating user typing status."""
    type: Literal["input.typing"] = "input.typing"
    is_typing: bool = True
    chat_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["is_typing"] = self.is_typing
        if self.chat_id:
            result["chat_id"] = self.chat_id
        return result


@dataclass
class PresenceTypingEvent(BaseEvent):
    """Server event broadcasting typing status from another participant."""
    type: Literal["presence.typing"] = "presence.typing"
    user_id: str = ""
    username: Optional[str] = None
    is_typing: bool = True
    chat_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["user_id"] = self.user_id
        if self.username:
            result["username"] = self.username
        result["is_typing"] = self.is_typing
        if self.chat_id:
            result["chat_id"] = self.chat_id
        return result


# =============================================================================
# Payment Events
# =============================================================================

@dataclass
class PaymentScheme:
    """A payment scheme option."""
    scheme: str  # 'token', 'crypto', 'card', 'ap2'
    network: Optional[str] = None  # 'robutler', 'ethereum', 'base', etc.
    address: Optional[str] = None  # For crypto payments
    min_amount: Optional[str] = None
    max_amount: Optional[str] = None


@dataclass
class PaymentRequirements:
    """Payment requirements details."""
    amount: str
    currency: str
    schemes: List[PaymentScheme]
    expires_at: Optional[int] = None
    reason: Optional[str] = None  # 'llm_usage', 'tool_call', 'api_access'
    # UCP/AP2 extension
    ap2: Optional[Dict[str, Any]] = None


@dataclass
class PaymentRequiredEvent(BaseEvent):
    """Server event indicating payment is required to continue."""
    type: Literal["payment.required"] = "payment.required"
    response_id: Optional[str] = None
    requirements: Optional[PaymentRequirements] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.response_id:
            result["response_id"] = self.response_id
        if self.requirements:
            req = self.requirements
            req_dict = {
                "amount": req.amount,
                "currency": req.currency,
                "schemes": [
                    {k: v for k, v in {
                        "scheme": s.scheme,
                        "network": s.network,
                        "address": s.address,
                        "min_amount": s.min_amount,
                        "max_amount": s.max_amount,
                    }.items() if v is not None}
                    for s in req.schemes
                ],
            }
            if req.expires_at:
                req_dict["expires_at"] = req.expires_at
            if req.reason:
                req_dict["reason"] = req.reason
            if req.ap2:
                req_dict["ap2"] = req.ap2
            result["requirements"] = req_dict
        return result


@dataclass
class PaymentData:
    """Payment submission data."""
    scheme: str
    amount: str
    network: Optional[str] = None
    token: Optional[str] = None
    proof: Optional[str] = None
    ap2_credential: Optional[Dict[str, Any]] = None


@dataclass
class PaymentSubmitEvent(BaseEvent):
    """Client event submitting payment token or proof."""
    type: Literal["payment.submit"] = "payment.submit"
    payment: Optional[PaymentData] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.payment:
            p = self.payment
            payment_dict = {
                "scheme": p.scheme,
                "amount": p.amount,
            }
            if p.network:
                payment_dict["network"] = p.network
            if p.token:
                payment_dict["token"] = p.token
            if p.proof:
                payment_dict["proof"] = p.proof
            if p.ap2_credential:
                payment_dict["ap2_credential"] = p.ap2_credential
            result["payment"] = payment_dict
        return result


@dataclass
class PaymentAcceptedEvent(BaseEvent):
    """Server event confirming payment was accepted."""
    type: Literal["payment.accepted"] = "payment.accepted"
    payment_id: str = ""
    balance_remaining: Optional[str] = None
    expires_at: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["payment_id"] = self.payment_id
        if self.balance_remaining:
            result["balance_remaining"] = self.balance_remaining
        if self.expires_at:
            result["expires_at"] = self.expires_at
        return result


@dataclass
class PaymentBalanceEvent(BaseEvent):
    """Server event for balance update notification."""
    type: Literal["payment.balance"] = "payment.balance"
    balance: str = "0"
    currency: str = "USD"
    low_balance_warning: bool = False
    estimated_remaining: Optional[int] = None
    expires_at: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["balance"] = self.balance
        result["currency"] = self.currency
        if self.low_balance_warning:
            result["low_balance_warning"] = self.low_balance_warning
        if self.estimated_remaining is not None:
            result["estimated_remaining"] = self.estimated_remaining
        if self.expires_at:
            result["expires_at"] = self.expires_at
        return result


@dataclass
class PaymentErrorEvent(BaseEvent):
    """Server event for payment error."""
    type: Literal["payment.error"] = "payment.error"
    code: str = "payment_failed"  # insufficient_balance, token_expired, token_invalid, payment_failed, rate_limited, mandate_revoked
    message: str = ""
    balance_required: Optional[str] = None
    balance_current: Optional[str] = None
    can_retry: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["code"] = self.code
        result["message"] = self.message
        if self.balance_required:
            result["balance_required"] = self.balance_required
        if self.balance_current:
            result["balance_current"] = self.balance_current
        result["can_retry"] = self.can_retry
        return result


# =============================================================================
# Conversation Item Events (OpenAI Realtime compatibility)
# =============================================================================

@dataclass
class ConversationItemCreateEvent(BaseEvent):
    """Client event to add an item to the conversation."""
    type: Literal["conversation.item.create"] = "conversation.item.create"
    item: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.item:
            result["item"] = self.item
        return result


@dataclass
class ConversationItemDeleteEvent(BaseEvent):
    """Client event to delete an item from the conversation."""
    type: Literal["conversation.item.delete"] = "conversation.item.delete"
    item_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["item_id"] = self.item_id
        return result


@dataclass
class ConversationItemTruncateEvent(BaseEvent):
    """Client event to truncate a conversation item."""
    type: Literal["conversation.item.truncate"] = "conversation.item.truncate"
    item_id: str = ""
    content_index: int = 0
    audio_end_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["item_id"] = self.item_id
        result["content_index"] = self.content_index
        if self.audio_end_ms is not None:
            result["audio_end_ms"] = self.audio_end_ms
        return result


# =============================================================================
# Utility Events
# =============================================================================

@dataclass
class PingEvent(BaseEvent):
    """Client event for keepalive."""
    type: Literal["ping"] = "ping"


@dataclass
class PongEvent(BaseEvent):
    """Server event for keepalive response."""
    type: Literal["pong"] = "pong"


@dataclass
class RateLimitEvent(BaseEvent):
    """Server event for rate limit notification."""
    type: Literal["rate_limit"] = "rate_limit"
    limit: int = 0
    remaining: int = 0
    reset_at: int = 0  # Unix timestamp

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["limit"] = self.limit
        result["remaining"] = self.remaining
        result["reset_at"] = self.reset_at
        return result


# =============================================================================
# Type Aliases
# =============================================================================

ClientEvent = Union[
    SessionCreateEvent,
    SessionUpdateEvent,
    SessionEndEvent,
    CapabilitiesQueryEvent,
    ClientCapabilitiesEvent,
    InputTextEvent,
    InputAudioEvent,
    InputAudioCommittedEvent,
    InputImageEvent,
    InputVideoEvent,
    InputFileEvent,
    InputTypingEvent,
    ToolResultEvent,
    ResponseCreateEvent,
    ResponseCancelEvent,
    PaymentSubmitEvent,
    ConversationItemCreateEvent,
    ConversationItemDeleteEvent,
    ConversationItemTruncateEvent,
    PingEvent,
]

ServerEvent = Union[
    SessionCreatedEvent,
    SessionUpdatedEvent,
    SessionEndEvent,
    SessionErrorEvent,
    CapabilitiesEvent,
    ResponseCreatedEvent,
    ResponseDeltaEvent,
    ResponseDoneEvent,
    ResponseCancelledEvent,
    ResponseErrorEvent,
    ToolCallEvent,
    ToolCallDoneEvent,
    AudioDeltaEvent,
    AudioDoneEvent,
    TranscriptDeltaEvent,
    TranscriptDoneEvent,
    UsageDeltaEvent,
    ProgressEvent,
    ThinkingEvent,
    PresenceTypingEvent,
    PaymentRequiredEvent,
    PaymentAcceptedEvent,
    PaymentBalanceEvent,
    PaymentErrorEvent,
    RateLimitEvent,
    PongEvent,
]
