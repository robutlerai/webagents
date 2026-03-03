"""
Agent Client Protocol (ACP) UAMP Adapter.

Converts between ACP JSON-RPC protocol and UAMP events.
https://agentclientprotocol.com/

Full UAMP integration for bidirectional message flow.
"""

from typing import List, Dict, Any, Optional
import json

from webagents.uamp import (
    SessionCreateEvent,
    SessionCreatedEvent,
    InputTextEvent,
    InputImageEvent,
    ResponseCreateEvent,
    ResponseCreatedEvent,
    ResponseDeltaEvent,
    ResponseDoneEvent,
    ResponseErrorEvent,
    ToolCallEvent,
    ToolResultEvent,
    ProgressEvent,
    ThinkingEvent,
    ServerEvent,
    ClientEvent,
    SessionConfig,
    ContentDelta,
    ContentItem,
    UsageStats,
    ResponseOutput,
)


class ACPUAMPAdapter:
    """
    Agent Client Protocol adapter with full UAMP integration.
    
    Converts ACP JSON-RPC 2.0 messages to/from UAMP events.
    ACP is used for IDE integration (Cursor, Zed, JetBrains).
    
    Supports both:
    - ACP v1 spec: session/prompt with ContentBlock[] prompt
    - Legacy: prompt/submit with messages array
    """
    
    def to_uamp(self, request: Dict[str, Any]) -> List[ClientEvent]:
        """
        Convert ACP JSON-RPC request to UAMP events.
        
        Handles:
        - session/prompt: ACP v1 format with ContentBlock[]
        - prompt/submit / chat/submit: Legacy format
        - tools/call: Convert to tool result
        
        Args:
            request: ACP JSON-RPC request
            
        Returns:
            List of UAMP ClientEvent objects
        """
        events: List[ClientEvent] = []
        
        method = request.get("method", "")
        params = request.get("params", {})
        
        # ACP v1: session/prompt with ContentBlock[] prompt
        if method == "session/prompt":
            session_config = SessionConfig(
                modalities=["text", "image"],
                extensions={
                    "protocol": "acp",
                    "protocol_version": "1",
                    "session_id": params.get("sessionId", ""),
                }
            )
            events.append(SessionCreateEvent(session=session_config))
            
            # Convert ACP ContentBlock[] prompt to UAMP events
            prompt = params.get("prompt", [])
            events.extend(self._content_blocks_to_uamp(prompt))
            
            # Request response
            events.append(ResponseCreateEvent())
        
        # Legacy: prompt/submit or chat/submit with messages array
        elif method in ("prompt/submit", "chat/submit"):
            session_config = SessionConfig(
                modalities=["text"],
                extensions={
                    "protocol": "acp",
                    "protocol_version": "1.0",
                }
            )
            events.append(SessionCreateEvent(session=session_config))
            
            # Convert messages
            messages = params.get("messages") or []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                # Handle tool results in conversation
                if role == "tool":
                    events.append(ToolResultEvent(
                        call_id=msg.get("tool_call_id", ""),
                        result=content if isinstance(content, str) else json.dumps(content),
                        is_error=False
                    ))
                else:
                    events.append(InputTextEvent(
                        text=content if isinstance(content, str) else json.dumps(content),
                        role=role
                    ))
            
            # Request response
            events.append(ResponseCreateEvent())
            
        elif method == "tools/call":
            # Tool result being provided
            events.append(ToolResultEvent(
                call_id=params.get("call_id", params.get("name", "")),
                result=json.dumps(params.get("arguments", {})),
                is_error=False
            ))
        
        return events
    
    def _content_blocks_to_uamp(self, content_blocks: List[Dict[str, Any]]) -> List[ClientEvent]:
        """
        Convert ACP ContentBlock[] to UAMP input events.
        
        ContentBlock types:
        - text: { type: "text", text: string }
        - image: { type: "image", data: base64, mimeType: string }
        - resource: { type: "resource", uri: string, mimeType: string, text?: string }
        """
        events: List[ClientEvent] = []
        
        for block in content_blocks:
            block_type = block.get("type", "text")
            
            if block_type == "text":
                events.append(InputTextEvent(
                    text=block.get("text", ""),
                    role="user"
                ))
            
            elif block_type == "image":
                # Image with base64 data
                data = block.get("data", "")
                mime_type = block.get("mimeType", "image/png")
                # Extract format from mime type
                fmt = mime_type.split("/")[-1] if "/" in mime_type else "png"
                events.append(InputImageEvent(
                    image=data,
                    format=fmt
                ))
            
            elif block_type == "resource":
                # Resource with URI
                uri = block.get("uri", "")
                text = block.get("text", "")
                mime_type = block.get("mimeType", "")
                
                if text:
                    # If text content is available, use it
                    events.append(InputTextEvent(
                        text=f"[Resource: {uri}]\n{text}",
                        role="user"
                    ))
                elif mime_type.startswith("image/"):
                    # Image resource
                    events.append(InputImageEvent(
                        image={"url": uri},
                        format=mime_type.split("/")[-1]
                    ))
                else:
                    # Generic resource reference
                    events.append(InputTextEvent(
                        text=f"[Resource: {uri} ({mime_type})]",
                        role="user"
                    ))
        
        return events
    
    def from_uamp(self, events: List[ServerEvent], request_id: Any = None) -> Dict[str, Any]:
        """
        Convert UAMP events to ACP JSON-RPC response.
        
        Args:
            events: List of UAMP ServerEvent objects
            request_id: Original JSON-RPC request ID
            
        Returns:
            ACP JSON-RPC response dict
        """
        content_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        
        for event in events:
            if isinstance(event, ResponseDeltaEvent):
                if event.delta and event.delta.type == "text" and event.delta.text:
                    content_parts.append(event.delta.text)
                elif event.delta and event.delta.type == "tool_call" and event.delta.tool_call:
                    tc = event.delta.tool_call
                    existing = next(
                        (t for t in tool_calls if t.get("id") == tc.get("id")),
                        None
                    )
                    if existing:
                        if tc.get("arguments"):
                            existing["arguments"] += tc["arguments"]
                    else:
                        tool_calls.append({
                            "id": tc.get("id", ""),
                            "name": tc.get("name", ""),
                            "arguments": tc.get("arguments", "")
                        })
                        
            elif isinstance(event, ResponseDoneEvent):
                if event.response:
                    for item in event.response.output:
                        if item.type == "text" and item.text and not content_parts:
                            content_parts.append(item.text)
        
        result = {
            "status": "complete",
            "content": "".join(content_parts) if content_parts else None,
        }
        
        if tool_calls:
            result["tool_calls"] = tool_calls
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result
        }
    
    def from_uamp_streaming(
        self, 
        event: ServerEvent, 
        session_id: str = "",
        request_id: Any = None
    ) -> Optional[Dict[str, Any]]:
        """
        Convert UAMP event to ACP v1 session/update notification for streaming.
        
        Args:
            event: Single UAMP ServerEvent
            session_id: ACP session ID
            request_id: Original request ID for reference
            
        Returns:
            ACP JSON-RPC notification dict or None
        """
        if isinstance(event, ResponseCreatedEvent):
            return self._session_update(session_id, {
                "sessionUpdate": "agent_response_started",
                "responseId": event.response_id
            })
        
        elif isinstance(event, ResponseDeltaEvent):
            if event.delta and event.delta.type == "text" and event.delta.text:
                return self._session_update(session_id, {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {
                        "type": "text",
                        "text": event.delta.text
                    }
                })
            elif event.delta and event.delta.type == "tool_call":
                tc = event.delta.tool_call or {}
                return self._session_update(session_id, {
                    "sessionUpdate": "tool_call",
                    "toolCallId": tc.get("id", ""),
                    "title": f"Calling {tc.get('name', 'tool')}",
                    "kind": "other",
                    "status": "pending"
                })
        
        elif isinstance(event, ToolCallEvent):
            return self._session_update(session_id, {
                "sessionUpdate": "tool_call",
                "toolCallId": event.call_id,
                "title": f"Calling {event.name}",
                "kind": "other",
                "status": "pending"
            })
                
        elif isinstance(event, ProgressEvent):
            return self._session_update(session_id, {
                "sessionUpdate": "progress",
                "stage": event.stage,
                "message": event.message,
                "percent": event.percent
            })
        
        elif isinstance(event, ThinkingEvent):
            return self._session_update(session_id, {
                "sessionUpdate": "thinking",
                "content": event.content,
                "stage": event.stage
            })
            
        elif isinstance(event, ResponseDoneEvent):
            # Don't emit notification - the final response handles completion
            return None
        
        elif isinstance(event, ResponseErrorEvent):
            return self._session_update(session_id, {
                "sessionUpdate": "error",
                "error": event.error
            })
        
        return None
    
    def _session_update(self, session_id: str, update: Dict[str, Any]) -> Dict[str, Any]:
        """Create a session/update notification."""
        return {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                **update
            }
        }
    
    def make_notification(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a JSON-RPC notification."""
        return {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
    
    def make_response(self, request_id: Any, result: Any) -> Dict[str, Any]:
        """Create a JSON-RPC response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result
        }
    
    def make_error(self, request_id: Any, code: int, message: str) -> Dict[str, Any]:
        """Create a JSON-RPC error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }
