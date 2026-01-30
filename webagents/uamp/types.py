"""
UMP Type Definitions.

Type definitions for UMP protocol elements.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class Modality(str, Enum):
    """Supported modalities."""
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"
    FILE = "file"


class AudioFormat(str, Enum):
    """Supported audio formats."""
    PCM16 = "pcm16"
    G711_ULAW = "g711_ulaw"
    G711_ALAW = "g711_alaw"
    MP3 = "mp3"
    OPUS = "opus"
    WAV = "wav"
    WEBM = "webm"
    AAC = "aac"


@dataclass
class VoiceConfig:
    """Provider-agnostic voice configuration."""
    provider: Optional[str] = None
    voice_id: Optional[str] = None
    name: Optional[str] = None
    speed: Optional[float] = None
    pitch: Optional[float] = None
    language: Optional[str] = None
    extensions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnDetectionConfig:
    """Generic turn detection configuration."""
    type: str = "server_vad"  # server_vad, client_vad, push_to_talk, none
    threshold: Optional[float] = None
    silence_duration_ms: Optional[int] = None
    prefix_padding_ms: Optional[int] = None
    extensions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolDefinition:
    """Standard tool/function definition."""
    type: str = "function"
    function: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionConfig:
    """Full session configuration."""
    modalities: List[str] = field(default_factory=lambda: ["text"])
    instructions: Optional[str] = None
    tools: Optional[List[ToolDefinition]] = None
    voice: Optional[VoiceConfig] = None
    input_audio_format: Optional[str] = None
    output_audio_format: Optional[str] = None
    turn_detection: Optional[TurnDetectionConfig] = None
    extensions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentDelta:
    """Content delta in streaming response."""
    type: str  # text, audio, tool_call
    text: Optional[str] = None
    audio: Optional[str] = None  # Base64
    tool_call: Optional[Dict[str, Any]] = None


@dataclass
class ContentItem:
    """Content item in response output."""
    type: str  # text, audio, image, tool_call, tool_result
    text: Optional[str] = None
    audio: Optional[str] = None  # Base64
    image: Optional[str] = None  # Base64 or URL
    tool_call: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None


@dataclass
class CostInfo:
    """Cost tracking information."""
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"


@dataclass
class AudioUsage:
    """Audio-specific usage information."""
    input_seconds: Optional[float] = None
    output_seconds: Optional[float] = None


@dataclass
class UsageStats:
    """Token and cost tracking."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: Optional[int] = None
    cost: Optional[CostInfo] = None
    audio: Optional[AudioUsage] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseOutput:
    """Response output structure."""
    id: str = ""
    status: str = "completed"  # completed, cancelled, failed
    output: List[ContentItem] = field(default_factory=list)
    usage: Optional[UsageStats] = None


@dataclass
class Session:
    """Full session object."""
    id: str = ""
    created_at: int = 0  # Unix timestamp
    config: Optional[SessionConfig] = None
    status: str = "active"  # active, closed
