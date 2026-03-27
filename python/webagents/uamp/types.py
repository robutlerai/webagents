"""
UAMP Type Definitions.

Type definitions for the Universal Agentic Message Protocol (UAMP).
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


@dataclass
class ImageCapabilities:
    """Image input capabilities."""
    formats: List[str] = field(default_factory=lambda: ["jpeg", "png", "gif", "webp"])
    max_size_bytes: Optional[int] = None  # e.g., 20MB
    max_pixels: Optional[int] = None  # e.g., 20M pixels
    detail_levels: List[str] = field(default_factory=lambda: ["auto", "low", "high"])
    max_images_per_request: Optional[int] = None


@dataclass
class AudioCapabilities:
    """Audio input/output capabilities."""
    input_formats: List[str] = field(default_factory=lambda: ["pcm16", "wav"])
    output_formats: List[str] = field(default_factory=lambda: ["pcm16", "wav"])
    sample_rates: List[int] = field(default_factory=lambda: [24000])
    max_duration_seconds: Optional[int] = None
    supports_realtime: bool = False
    voices: List[str] = field(default_factory=list)


@dataclass
class FileCapabilities:
    """File input capabilities."""
    supported_mime_types: List[str] = field(default_factory=list)
    max_size_bytes: Optional[int] = None
    supports_pdf: bool = False
    supports_code: bool = True
    supports_structured_data: bool = True  # JSON, CSV, etc.


@dataclass
class ToolCapabilities:
    """Tool/function calling capabilities."""
    supports_tools: bool = True
    supports_parallel_tools: bool = True
    supports_streaming_tools: bool = True
    max_tools_per_request: Optional[int] = None
    built_in_tools: List[str] = field(default_factory=list)  # web_search, code_interpreter, etc.


@dataclass
class ModelCapabilities:
    """Full model capability declaration."""
    model_id: str = ""
    provider: str = ""
    
    # Modalities
    modalities: List[str] = field(default_factory=lambda: ["text"])
    
    # Detailed capabilities by modality
    image: Optional[ImageCapabilities] = None
    audio: Optional[AudioCapabilities] = None
    file: Optional[FileCapabilities] = None
    tools: Optional[ToolCapabilities] = None
    
    # Model features
    supports_streaming: bool = True
    supports_thinking: bool = False  # Extended thinking / reasoning
    supports_caching: bool = False  # Context caching
    context_window: Optional[int] = None  # Max input tokens
    max_output_tokens: Optional[int] = None
    
    # Content types that may appear in responses
    output_content_types: List[str] = field(default_factory=list)
    
    # Extensions for provider-specific features
    extensions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Capabilities:
    """Unified capability declaration for models, clients, and agents.
    
    Same structure for all - context determines interpretation:
    - Model: what the LLM can process
    - Client: what the client can render/handle  
    - Agent: what the agent provides (model + skills)
    
    Examples:
        # Model capabilities
        Capabilities(
            id="gpt-4o",
            provider="openai",
            modalities=["text", "image"],
            image=ImageCapabilities(formats=["jpeg", "png"]),
            tools=ToolCapabilities(built_in_tools=["web_search"]),
            supports_streaming=True,
            context_window=128000
        )
        
        # Client capabilities
        Capabilities(
            id="web-app",
            provider="robutler",
            modalities=["text", "image", "audio"],
            supports_streaming=True,
            widgets=["chart", "form"],
            extensions={"supports_html": True}
        )
        
        # Agent capabilities
        Capabilities(
            id="my-agent",
            provider="webagents",
            modalities=["text", "image"],
            tools=ToolCapabilities(built_in_tools=["web_search", "render_chart"]),
            provides=["chart", "tts"],
            endpoints=["/api/data"]
        )
    """
    # Identity (model_id, client_id, or agent_id depending on context)
    id: str = ""
    provider: str = ""
    
    # Core modalities
    modalities: List[str] = field(default_factory=lambda: ["text"])
    
    # Detailed capabilities by modality (same as ModelCapabilities)
    image: Optional[ImageCapabilities] = None
    audio: Optional[AudioCapabilities] = None
    file: Optional[FileCapabilities] = None
    tools: Optional[ToolCapabilities] = None
    
    # Features
    supports_streaming: bool = True
    supports_thinking: bool = False
    supports_caching: bool = False
    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None
    
    # Agent/client extensions (optional, used when relevant)
    provides: List[str] = field(default_factory=list)  # What capabilities are provided
    widgets: List[str] = field(default_factory=list)   # Available/supported widgets
    endpoints: List[str] = field(default_factory=list) # HTTP endpoints (agent)
    
    # Content types that may appear in responses (e.g. ["text", "image"])
    output_content_types: List[str] = field(default_factory=list)
    
    # Extensions for context-specific features
    extensions: Dict[str, Any] = field(default_factory=dict)


# Aliases for clarity in code
ClientCapabilities = Capabilities
AgentCapabilities = Capabilities


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
    response_format: Optional[Dict[str, Any]] = None
    extensions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentDelta:
    """Content delta in streaming response."""
    type: str  # text, audio, tool_call, tool_result, tool_progress
    text: Optional[str] = None
    audio: Optional[str] = None  # Base64
    tool_call: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None
    tool_progress: Optional[Dict[str, Any]] = None


@dataclass
class ContentItem:
    """Content item in response output."""
    type: str  # text, audio, image, video, file, tool_call, tool_result
    text: Optional[str] = None
    audio: Optional[str] = None  # Base64
    image: Optional[str] = None  # Base64 or URL
    video: Optional[str] = None  # Base64 or URL
    file: Optional[str] = None  # Base64 or URL
    filename: Optional[str] = None
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    duration_ms: Optional[int] = None
    content_id: Optional[str] = None
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
