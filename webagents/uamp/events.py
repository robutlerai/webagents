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

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        result = {
            "type": self.type,
            "event_id": self.event_id,
        }
        if self.timestamp is not None:
            result["timestamp"] = self.timestamp
        return result


# =============================================================================
# Session Events
# =============================================================================

@dataclass
class SessionCreateEvent(BaseEvent):
    """Client event to create a new session."""
    type: Literal["session.create"] = "session.create"
    uamp_version: str = "1.0"
    session: Optional[SessionConfig] = None

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


# =============================================================================
# Input Events
# =============================================================================

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
class ResponseCreateEvent(BaseEvent):
    """Client event to request a response."""
    type: Literal["response.create"] = "response.create"
    response: Optional[Dict[str, Any]] = None  # Override options

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.response:
            result["response"] = self.response
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
            result["delta"] = delta_dict
        return result


@dataclass
class ResponseDoneEvent(BaseEvent):
    """Server event for response completion."""
    type: Literal["response.done"] = "response.done"
    response_id: str = ""
    response: Optional[ResponseOutput] = None

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
    InputTextEvent,
    InputAudioEvent,
    InputImageEvent,
    InputVideoEvent,
    InputFileEvent,
    ToolResultEvent,
    ResponseCreateEvent,
    ResponseCancelEvent,
    PingEvent,
]

ServerEvent = Union[
    SessionCreatedEvent,
    ResponseCreatedEvent,
    ResponseDeltaEvent,
    ResponseDoneEvent,
    ResponseErrorEvent,
    ToolCallEvent,
    ToolCallDoneEvent,
    AudioDeltaEvent,
    TranscriptDeltaEvent,
    UsageDeltaEvent,
    ProgressEvent,
    ThinkingEvent,
    RateLimitEvent,
    PongEvent,
]
