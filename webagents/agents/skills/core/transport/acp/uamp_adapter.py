"""
Agent Client Protocol (ACP) UAMP Adapter.

Converts between ACP JSON-RPC protocol and UAMP events.
https://agentclientprotocol.com/
"""

from typing import List, Dict, Any, Optional
import json

from webagents.uamp import (
    SessionCreateEvent,
    SessionCreatedEvent,
    InputTextEvent,
    ResponseCreateEvent,
    ResponseDeltaEvent,
    ResponseDoneEvent,
    ToolCallEvent,
    ToolResultEvent,
    ProgressEvent,
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
    Agent Client Protocol adapter.
    
    Converts ACP JSON-RPC 2.0 messages to/from UAMP events.
    ACP is used for IDE integration (Cursor, Zed, JetBrains).
    """
    
    def to_uamp(self, request: Dict[str, Any]) -> List[ClientEvent]:
        """
        Convert ACP JSON-RPC request to UAMP events.
        
        Handles:
        - prompt/submit / chat/submit: Convert messages to UAMP
        - tools/call: Convert to tool result
        
        Args:
            request: ACP JSON-RPC request
            
        Returns:
            List of UAMP ClientEvent objects
        """
        events: List[ClientEvent] = []
        
        method = request.get("method", "")
        params = request.get("params", {})
        
        if method in ("prompt/submit", "chat/submit"):
            # Create session
            session_config = SessionConfig(
                modalities=["text"],
                extensions={
                    "protocol": "acp",
                    "protocol_version": "1.0",
                }
            )
            events.append(SessionCreateEvent(session=session_config))
            
            # Convert messages
            messages = params.get("messages", [])
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
    
    def from_uamp_streaming(self, event: ServerEvent, request_id: Any = None) -> Optional[Dict[str, Any]]:
        """
        Convert UAMP event to ACP JSON-RPC notification for streaming.
        
        Args:
            event: Single UAMP ServerEvent
            request_id: Original request ID for reference
            
        Returns:
            ACP JSON-RPC notification dict or None
        """
        if isinstance(event, ResponseDeltaEvent):
            if event.delta and event.delta.type == "text" and event.delta.text:
                return {
                    "jsonrpc": "2.0",
                    "method": "prompt/progress",
                    "params": {
                        "requestId": str(request_id),
                        "content": event.delta.text,
                        "role": "assistant"
                    }
                }
            elif event.delta and event.delta.type == "tool_call":
                tc = event.delta.tool_call or {}
                return {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "requestId": str(request_id),
                        "id": tc.get("id", ""),
                        "name": tc.get("name", ""),
                        "arguments": tc.get("arguments", "")
                    }
                }
                
        elif isinstance(event, ProgressEvent):
            return {
                "jsonrpc": "2.0",
                "method": "prompt/progress",
                "params": {
                    "requestId": str(request_id),
                    "stage": event.stage,
                    "message": event.message,
                    "percent": event.percent
                }
            }
            
        elif isinstance(event, ResponseDoneEvent):
            return {
                "jsonrpc": "2.0",
                "method": "prompt/done",
                "params": {
                    "requestId": str(request_id),
                    "status": "complete"
                }
            }
        
        return None
    
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
