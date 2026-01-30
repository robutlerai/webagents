"""
Realtime Transport Skill - WebAgents V2.0

OpenAI Realtime API implementation with WebSocket support.
https://platform.openai.com/docs/guides/realtime

Uses UAMP (Universal Agentic Message Protocol) for internal message representation.
"""

import json
import uuid
import time
import base64
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import websocket
from webagents.uamp import (
    ResponseDeltaEvent,
    ResponseDoneEvent,
    ContentDelta,
    ContentItem,
    UsageStats,
    ResponseOutput,
)
from .uamp_adapter import RealtimeUAMPAdapter

if TYPE_CHECKING:
    from webagents.agents.core.base_agent import BaseAgent
    from fastapi import WebSocket


@dataclass
class RealtimeSession:
    """Realtime session state"""
    id: str = field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:12]}")
    voice: str = "alloy"
    modalities: List[str] = field(default_factory=lambda: ["text"])
    instructions: str = ""
    input_audio_format: str = "pcm16"
    output_audio_format: str = "pcm16"
    turn_detection: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    
    # Runtime state
    audio_buffer: bytes = field(default_factory=bytes)
    conversation: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "voice": self.voice,
            "modalities": self.modalities,
            "instructions": self.instructions,
            "input_audio_format": self.input_audio_format,
            "output_audio_format": self.output_audio_format,
            "turn_detection": self.turn_detection
        }
    
    def update(self, config: Dict[str, Any]) -> None:
        """Update session configuration"""
        if "voice" in config:
            self.voice = config["voice"]
        if "modalities" in config:
            self.modalities = config["modalities"]
        if "instructions" in config:
            self.instructions = config["instructions"]
        if "input_audio_format" in config:
            self.input_audio_format = config["input_audio_format"]
        if "output_audio_format" in config:
            self.output_audio_format = config["output_audio_format"]
        if "turn_detection" in config:
            self.turn_detection = config["turn_detection"]


class RealtimeTransportSkill(Skill):
    """
    OpenAI Realtime API transport over WebSocket.
    
    Implements real-time bidirectional communication with support for:
    - Text and audio modalities
    - Session management
    - Conversation item handling
    - Response streaming
    
    Example:
        agent = BaseAgent(
            name="my-agent",
            skills=[RealtimeTransportSkill()]
        )
        
        # Connect via WebSocket to /agents/my-agent/realtime
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all")
        self._sessions: Dict[str, RealtimeSession] = {}
        self._adapter = RealtimeUAMPAdapter()
    
    async def initialize(self, agent: 'BaseAgent') -> None:
        """Initialize the Realtime transport"""
        self.agent = agent
    
    @websocket("/realtime")
    async def realtime_session(self, ws: 'WebSocket') -> None:
        """
        WebSocket endpoint for OpenAI Realtime API.
        
        Handles bidirectional communication with event-based protocol.
        """
        await ws.accept()
        
        # Create session
        session = RealtimeSession()
        self._sessions[session.id] = session
        
        # Send session created event
        await self._send_event(ws, "session.created", {
            "session": session.to_dict()
        })
        
        try:
            # Process incoming events
            async for message in ws.iter_json():
                await self._handle_event(ws, session, message)
        except Exception as e:
            # Connection closed or error
            pass
        finally:
            # Cleanup session
            self._sessions.pop(session.id, None)
    
    async def _handle_event(
        self,
        ws: 'WebSocket',
        session: RealtimeSession,
        event: Dict[str, Any]
    ) -> None:
        """Handle incoming WebSocket event"""
        event_type = event.get("type", "")
        event_id = event.get("event_id", str(uuid.uuid4()))
        
        handlers = {
            "session.update": self._handle_session_update,
            "input_audio_buffer.append": self._handle_audio_append,
            "input_audio_buffer.commit": self._handle_audio_commit,
            "input_audio_buffer.clear": self._handle_audio_clear,
            "conversation.item.create": self._handle_item_create,
            "conversation.item.delete": self._handle_item_delete,
            "conversation.item.truncate": self._handle_item_truncate,
            "response.create": self._handle_response_create,
            "response.cancel": self._handle_response_cancel,
        }
        
        handler = handlers.get(event_type)
        if handler:
            await handler(ws, session, event)
        else:
            await self._send_event(ws, "error", {
                "type": "invalid_event",
                "message": f"Unknown event type: {event_type}"
            })
    
    async def _handle_session_update(
        self,
        ws: 'WebSocket',
        session: RealtimeSession,
        event: Dict[str, Any]
    ) -> None:
        """Update session configuration"""
        session_config = event.get("session", {})
        session.update(session_config)
        
        await self._send_event(ws, "session.updated", {
            "session": session.to_dict()
        })
    
    async def _handle_audio_append(
        self,
        ws: 'WebSocket',
        session: RealtimeSession,
        event: Dict[str, Any]
    ) -> None:
        """Append audio to input buffer"""
        audio_b64 = event.get("audio", "")
        if audio_b64:
            try:
                audio_bytes = base64.b64decode(audio_b64)
                session.audio_buffer += audio_bytes
            except Exception:
                pass
    
    async def _handle_audio_commit(
        self,
        ws: 'WebSocket',
        session: RealtimeSession,
        event: Dict[str, Any]
    ) -> None:
        """Commit audio buffer as conversation item"""
        if session.audio_buffer:
            # In a full implementation, this would transcribe the audio
            # For now, we'll just acknowledge the commit
            item_id = f"item_{uuid.uuid4().hex[:12]}"
            
            # Add placeholder item to conversation
            session.conversation.append({
                "id": item_id,
                "type": "message",
                "role": "user",
                "content": [{"type": "input_audio", "audio": "[audio data]"}]
            })
            
            await self._send_event(ws, "input_audio_buffer.committed", {
                "item_id": item_id
            })
            
            # Clear buffer
            session.audio_buffer = bytes()
            
            # Note: In production, you'd transcribe and trigger response here
    
    async def _handle_audio_clear(
        self,
        ws: 'WebSocket',
        session: RealtimeSession,
        event: Dict[str, Any]
    ) -> None:
        """Clear audio input buffer"""
        session.audio_buffer = bytes()
        await self._send_event(ws, "input_audio_buffer.cleared", {})
    
    async def _handle_item_create(
        self,
        ws: 'WebSocket',
        session: RealtimeSession,
        event: Dict[str, Any]
    ) -> None:
        """Create a conversation item"""
        item = event.get("item", {})
        item_id = item.get("id", f"item_{uuid.uuid4().hex[:12]}")
        item["id"] = item_id
        
        session.conversation.append(item)
        
        await self._send_event(ws, "conversation.item.created", {
            "item": item
        })
    
    async def _handle_item_delete(
        self,
        ws: 'WebSocket',
        session: RealtimeSession,
        event: Dict[str, Any]
    ) -> None:
        """Delete a conversation item"""
        item_id = event.get("item_id", "")
        session.conversation = [
            item for item in session.conversation
            if item.get("id") != item_id
        ]
        
        await self._send_event(ws, "conversation.item.deleted", {
            "item_id": item_id
        })
    
    async def _handle_item_truncate(
        self,
        ws: 'WebSocket',
        session: RealtimeSession,
        event: Dict[str, Any]
    ) -> None:
        """Truncate conversation item (audio) at specified index"""
        item_id = event.get("item_id", "")
        content_index = event.get("content_index", 0)
        audio_end_ms = event.get("audio_end_ms", 0)
        
        # Find and truncate item
        for item in session.conversation:
            if item.get("id") == item_id:
                # Mark as truncated
                item["truncated"] = True
                item["truncate_audio_end_ms"] = audio_end_ms
                break
        
        await self._send_event(ws, "conversation.item.truncated", {
            "item_id": item_id,
            "content_index": content_index,
            "audio_end_ms": audio_end_ms
        })
    
    async def _handle_response_create(
        self,
        ws: 'WebSocket',
        session: RealtimeSession,
        event: Dict[str, Any]
    ) -> None:
        """Create and stream a response with full event coverage"""
        response_id = f"resp_{uuid.uuid4().hex[:12]}"
        output_index = 0
        
        # Convert conversation to OpenAI messages
        messages = self._conversation_to_messages(session)
        
        # Add system instructions if set
        if session.instructions:
            messages.insert(0, {"role": "system", "content": session.instructions})
        
        # Emit response created
        await self._send_event(ws, "response.created", {
            "response": {
                "id": response_id,
                "object": "realtime.response",
                "status": "in_progress",
                "status_details": None,
                "output": [],
                "usage": None
            }
        })
        
        try:
            # Stream response through handoff
            item_id = f"item_{uuid.uuid4().hex[:12]}"
            full_text = ""
            
            # Emit output item added
            await self._send_event(ws, "response.output_item.added", {
                "response_id": response_id,
                "output_index": output_index,
                "item": {
                    "id": item_id,
                    "object": "realtime.item",
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": []
                }
            })
            
            # Emit content part added
            content_index = 0
            await self._send_event(ws, "response.content_part.added", {
                "response_id": response_id,
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "part": {
                    "type": "text",
                    "text": ""
                }
            })
            
            async for chunk in self.execute_handoff(messages):
                delta_text = self._extract_delta_text(chunk)
                if delta_text:
                    full_text += delta_text
                    
                    # Send text delta
                    await self._send_event(ws, "response.text.delta", {
                        "response_id": response_id,
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": content_index,
                        "delta": delta_text
                    })
            
            # Send text done
            await self._send_event(ws, "response.text.done", {
                "response_id": response_id,
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "text": full_text
            })
            
            # Emit content part done
            await self._send_event(ws, "response.content_part.done", {
                "response_id": response_id,
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "part": {
                    "type": "text",
                    "text": full_text
                }
            })
            
            # Add to conversation
            session.conversation.append({
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": full_text}]
            })
            
            # Emit output item done
            await self._send_event(ws, "response.output_item.done", {
                "response_id": response_id,
                "output_index": output_index,
                "item": {
                    "id": item_id,
                    "object": "realtime.item",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "text", "text": full_text}]
                }
            })
            
            # Response done with usage
            await self._send_event(ws, "response.done", {
                "response": {
                    "id": response_id,
                    "object": "realtime.response",
                    "status": "completed",
                    "status_details": None,
                    "output": [{
                        "id": item_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": full_text}]
                    }],
                    "usage": {
                        "total_tokens": len(full_text.split()),
                        "input_tokens": sum(len(m.get("content", "").split()) for m in messages),
                        "output_tokens": len(full_text.split())
                    }
                }
            })
            
        except Exception as e:
            await self._send_event(ws, "response.done", {
                "response": {
                    "id": response_id,
                    "object": "realtime.response",
                    "status": "failed",
                    "status_details": {"type": "error", "error": str(e)},
                    "output": [],
                    "usage": None
                }
            })
    
    async def _handle_response_cancel(
        self,
        ws: 'WebSocket',
        session: RealtimeSession,
        event: Dict[str, Any]
    ) -> None:
        """Cancel current response generation"""
        # In a full implementation, this would cancel the streaming
        await self._send_event(ws, "response.cancelled", {})
    
    def _conversation_to_messages(
        self,
        session: RealtimeSession
    ) -> List[Dict[str, Any]]:
        """Convert Realtime conversation to OpenAI messages"""
        messages = []
        for item in session.conversation:
            if item.get("type") == "message":
                role = item.get("role", "user")
                content_parts = item.get("content", [])
                
                # Extract text content
                text_parts = []
                for part in content_parts:
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "input_audio":
                        text_parts.append("[audio input]")
                
                if text_parts:
                    messages.append({
                        "role": role,
                        "content": "\n".join(text_parts)
                    })
        
        return messages
    
    def _extract_delta_text(self, chunk: Dict[str, Any]) -> str:
        """Extract text delta from OpenAI streaming chunk"""
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            return delta.get("content", "")
        return ""
    
    def _openai_chunk_to_uamp(self, chunk: Dict[str, Any]) -> Optional[ResponseDeltaEvent]:
        """Convert OpenAI streaming chunk to UAMP ResponseDeltaEvent."""
        choices = chunk.get("choices", [])
        if not choices:
            return None
        
        delta = choices[0].get("delta", {})
        content = delta.get("content", "")
        
        if not content:
            return None
        
        return ResponseDeltaEvent(
            response_id=chunk.get("id", ""),
            delta=ContentDelta(type="text", text=content)
        )
    
    async def _send_event(
        self,
        ws: 'WebSocket',
        event_type: str,
        data: Dict[str, Any]
    ) -> None:
        """Send event to WebSocket client"""
        event = {
            "type": event_type,
            "event_id": f"evt_{uuid.uuid4().hex[:12]}",
            **data
        }
        await ws.send_json(event)
