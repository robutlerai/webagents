"""
Google A2A Protocol UAMP Adapter.

Converts between A2A (Agent2Agent) Protocol and UAMP events.
https://google.github.io/A2A/specification/
"""

from typing import List, Dict, Any, Optional
import json
import uuid

from webagents.uamp import (
    SessionCreateEvent,
    SessionCreatedEvent,
    InputTextEvent,
    InputImageEvent,
    InputFileEvent,
    ResponseCreateEvent,
    ResponseDeltaEvent,
    ResponseDoneEvent,
    ProgressEvent,
    ServerEvent,
    ClientEvent,
    SessionConfig,
    ContentDelta,
    ContentItem,
    UsageStats,
    ResponseOutput,
)


class A2AUAMPAdapter:
    """
    Google A2A Protocol adapter.
    
    Converts A2A task messages (with parts) to/from UAMP events.
    
    A2A Message Format:
        {
            "role": "user" | "agent",
            "parts": [
                {"type": "text", "text": "..."},
                {"type": "file", "file": {"name": "...", "mimeType": "...", "data": "base64..."}},
                {"type": "data", "data": {...}, "mimeType": "application/json"}
            ]
        }
    """
    
    def to_uamp(self, request: Dict[str, Any]) -> List[ClientEvent]:
        """
        Convert A2A task request to UAMP events.
        
        Args:
            request: A2A task request with message/messages
            
        Returns:
            List of UAMP ClientEvent objects
        """
        events: List[ClientEvent] = []
        
        # Create session with text modality (extend based on parts)
        modalities = ["text"]
        session_config = SessionConfig(
            modalities=modalities,
            extensions={
                "protocol": "a2a",
                "protocol_version": "0.2.1",
            }
        )
        events.append(SessionCreateEvent(session=session_config))
        
        # Handle single message
        message = request.get("message")
        if message:
            events.extend(self._message_to_events(message))
        
        # Handle message list
        messages = request.get("messages") or []
        for msg in messages:
            events.extend(self._message_to_events(msg))
        
        # Request response
        events.append(ResponseCreateEvent())
        
        return events
    
    def _message_to_events(self, message: Dict[str, Any]) -> List[ClientEvent]:
        """Convert a single A2A message to UAMP events."""
        events: List[ClientEvent] = []
        role = message.get("role", "user")
        
        # A2A uses "agent" role, UAMP uses "assistant"
        if role == "agent":
            role = "assistant"
        
        parts = message.get("parts", [])
        
        for part in parts:
            part_type = part.get("type", "text")
            
            if part_type == "text" or "text" in part:
                events.append(InputTextEvent(
                    text=part.get("text", ""),
                    role=role
                ))
                
            elif part_type == "file":
                file_data = part.get("file", {})
                mime_type = file_data.get("mimeType", "")
                
                if mime_type.startswith("image/"):
                    # Image file - use data URL or direct URL
                    if "data" in file_data:
                        image_data = f"data:{mime_type};base64,{file_data['data']}"
                    else:
                        image_data = {"url": file_data.get("uri", "")}
                    
                    events.append(InputImageEvent(
                        image=image_data,
                        detail="auto"
                    ))
                else:
                    # Generic file
                    if "data" in file_data:
                        file_content = file_data.get("data", "")
                    else:
                        file_content = {"url": file_data.get("uri", "")}
                    
                    events.append(InputFileEvent(
                        file=file_content,
                        filename=file_data.get("name", "unnamed"),
                        mime_type=mime_type
                    ))
                    
            elif part_type == "data":
                # Structured data - convert to text
                data = part.get("data", {})
                mime_type = part.get("mimeType", "application/json")
                events.append(InputTextEvent(
                    text=f"[Data ({mime_type})]: {json.dumps(data)}",
                    role=role
                ))
        
        return events
    
    def from_uamp(self, events: List[ServerEvent]) -> Dict[str, Any]:
        """
        Convert UAMP events to A2A task result.
        
        Args:
            events: List of UAMP ServerEvent objects
            
        Returns:
            A2A task result dict
        """
        parts: List[Dict[str, Any]] = []
        content_buffer: List[str] = []
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        
        for event in events:
            if isinstance(event, ResponseDeltaEvent):
                if event.delta and event.delta.type == "text" and event.delta.text:
                    content_buffer.append(event.delta.text)
                    
            elif isinstance(event, ResponseDoneEvent):
                # Finalize content
                if content_buffer:
                    parts.append({
                        "type": "text",
                        "text": "".join(content_buffer)
                    })
                
                # Add any output items from response
                if event.response:
                    for item in event.response.output:
                        if item.type == "text" and item.text and not content_buffer:
                            parts.append({
                                "type": "text",
                                "text": item.text
                            })
        
        # If we only have buffered content but no ResponseDone
        if content_buffer and not parts:
            parts.append({
                "type": "text",
                "text": "".join(content_buffer)
            })
        
        return {
            "id": task_id,
            "status": "completed",
            "message": {
                "role": "agent",
                "parts": parts
            }
        }
    
    def from_uamp_streaming(self, event: ServerEvent) -> Optional[Dict[str, Any]]:
        """
        Convert UAMP event to A2A SSE event.
        
        Args:
            event: Single UAMP ServerEvent
            
        Returns:
            A2A SSE event dict or None
        """
        if isinstance(event, ResponseDeltaEvent):
            if event.delta and event.delta.type == "text" and event.delta.text:
                return {
                    "event": "task.message",
                    "data": {
                        "role": "agent",
                        "parts": [{"type": "text", "text": event.delta.text}]
                    }
                }
                
        elif isinstance(event, ProgressEvent):
            return {
                "event": "task.progress",
                "data": {
                    "stage": event.stage,
                    "message": event.message,
                    "percent": event.percent
                }
            }
            
        elif isinstance(event, ResponseDoneEvent):
            return {
                "event": "task.completed",
                "data": {
                    "status": "completed"
                }
            }
        
        return None
    
    def to_sse(self, event_type: str, data: Dict[str, Any]) -> str:
        """Format as SSE event string."""
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
