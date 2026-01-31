"""
OpenAI Realtime API UAMP Adapter.

Converts between OpenAI Realtime WebSocket protocol and UAMP events.
https://platform.openai.com/docs/guides/realtime

The Realtime API is already very similar to UAMP, so this adapter is thin.
"""

from typing import List, Dict, Any, Optional
import uuid

from webagents.uamp import (
    SessionCreateEvent,
    SessionCreatedEvent,
    SessionUpdateEvent,
    InputTextEvent,
    InputAudioEvent,
    ResponseCreateEvent,
    ResponseCancelEvent,
    ResponseDeltaEvent,
    ResponseDoneEvent,
    AudioDeltaEvent,
    TranscriptDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    ServerEvent,
    ClientEvent,
    SessionConfig,
    Session,
    ContentDelta,
    ContentItem,
    UsageStats,
    ResponseOutput,
    VoiceConfig,
    TurnDetectionConfig,
)


class RealtimeUAMPAdapter:
    """
    OpenAI Realtime API adapter.
    
    The Realtime API event structure inspired UAMP, so this adapter
    is largely a 1:1 mapping with some naming adjustments.
    """
    
    def to_uamp(self, event: Dict[str, Any]) -> Optional[ClientEvent]:
        """
        Convert Realtime client event to UAMP event.
        
        Args:
            event: OpenAI Realtime client event
            
        Returns:
            UAMP ClientEvent or None
        """
        event_type = event.get("type", "")
        
        if event_type == "session.update":
            session_config = event.get("session", {})
            return SessionUpdateEvent(
                session=SessionConfig(
                    modalities=session_config.get("modalities", ["text"]),
                    voice=VoiceConfig(
                        name=session_config.get("voice", "alloy")
                    ) if session_config.get("voice") else None,
                    instructions=session_config.get("instructions"),
                    turn_detection=TurnDetectionConfig(
                        type=session_config.get("turn_detection", {}).get("type", "server_vad"),
                        threshold=session_config.get("turn_detection", {}).get("threshold"),
                        silence_duration_ms=session_config.get("turn_detection", {}).get("silence_duration_ms"),
                    ) if session_config.get("turn_detection") else None,
                )
            )
            
        elif event_type == "input_audio_buffer.append":
            return InputAudioEvent(
                audio=event.get("audio", ""),
                format="pcm16"
            )
            
        elif event_type == "conversation.item.create":
            item = event.get("item", {})
            if item.get("type") == "message":
                content = item.get("content", [])
                for part in content:
                    if part.get("type") == "text":
                        return InputTextEvent(
                            text=part.get("text", ""),
                            role=item.get("role", "user")
                        )
                    elif part.get("type") == "input_audio":
                        return InputAudioEvent(
                            audio=part.get("audio", ""),
                            format="pcm16"
                        )
            elif item.get("type") == "function_call_output":
                return ToolResultEvent(
                    call_id=item.get("call_id", ""),
                    result=item.get("output", ""),
                    is_error=False
                )
            
        elif event_type == "response.create":
            return ResponseCreateEvent()
            
        elif event_type == "response.cancel":
            return ResponseCancelEvent()
        
        return None
    
    def from_uamp(self, event: ServerEvent) -> Optional[Dict[str, Any]]:
        """
        Convert UAMP server event to Realtime event.
        
        Args:
            event: UAMP ServerEvent
            
        Returns:
            OpenAI Realtime event dict or None
        """
        if isinstance(event, SessionCreatedEvent):
            session = event.session
            config = session.config if session else None
            return {
                "type": "session.created",
                "event_id": event.event_id,
                "session": {
                    "id": session.id if session else "",
                    "modalities": config.modalities if config else ["text"],
                    "voice": config.voice.name if config and config.voice else "alloy",
                    "instructions": config.instructions if config else "",
                }
            }
            
        elif isinstance(event, ResponseDeltaEvent):
            if event.delta:
                if event.delta.type == "text":
                    return {
                        "type": "response.text.delta",
                        "event_id": event.event_id,
                        "response_id": event.response_id,
                        "delta": event.delta.text or ""
                    }
                elif event.delta.type == "audio":
                    return {
                        "type": "response.audio.delta",
                        "event_id": event.event_id,
                        "response_id": event.response_id,
                        "delta": event.delta.audio or ""
                    }
                elif event.delta.type == "tool_call":
                    tc = event.delta.tool_call or {}
                    return {
                        "type": "response.function_call_arguments.delta",
                        "event_id": event.event_id,
                        "response_id": event.response_id,
                        "call_id": tc.get("id", ""),
                        "delta": tc.get("arguments", "")
                    }
                    
        elif isinstance(event, AudioDeltaEvent):
            return {
                "type": "response.audio.delta",
                "event_id": event.event_id,
                "response_id": event.response_id,
                "delta": event.audio
            }
            
        elif isinstance(event, TranscriptDeltaEvent):
            return {
                "type": "response.audio_transcript.delta",
                "event_id": event.event_id,
                "response_id": event.response_id,
                "delta": event.transcript
            }
            
        elif isinstance(event, ToolCallEvent):
            return {
                "type": "response.function_call_arguments.done",
                "event_id": event.event_id,
                "call_id": event.call_id,
                "name": event.name,
                "arguments": event.arguments
            }
            
        elif isinstance(event, ResponseDoneEvent):
            output = []
            if event.response:
                for item in event.response.output:
                    output.append({
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": item.type, "text": item.text}] if item.text else []
                    })
            
            usage = None
            if event.response and event.response.usage:
                usage = {
                    "total_tokens": event.response.usage.total_tokens,
                    "input_tokens": event.response.usage.input_tokens,
                    "output_tokens": event.response.usage.output_tokens
                }
            
            return {
                "type": "response.done",
                "event_id": event.event_id,
                "response": {
                    "id": event.response_id,
                    "object": "realtime.response",
                    "status": "completed",
                    "output": output,
                    "usage": usage
                }
            }
        
        return None
    
    def create_session_created(self, session_id: str, config: Dict[str, Any]) -> SessionCreatedEvent:
        """Helper to create a SessionCreatedEvent from config."""
        return SessionCreatedEvent(
            session=Session(
                id=session_id,
                config=SessionConfig(
                    modalities=config.get("modalities", ["text"]),
                    voice=VoiceConfig(name=config.get("voice", "alloy")),
                    instructions=config.get("instructions", ""),
                )
            )
        )
