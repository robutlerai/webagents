"""
OpenAI Chat Completions UAMP Adapter.

Converts between OpenAI Chat Completions format and UAMP events.
"""

from typing import List, Dict, Any, Optional
import json

from webagents.uamp import (
    SessionCreateEvent,
    InputTextEvent,
    InputImageEvent,
    InputAudioEvent,
    InputFileEvent,
    ResponseCreateEvent,
    ResponseDeltaEvent,
    ResponseDoneEvent,
    ToolResultEvent,
    ServerEvent,
    ClientEvent,
    SessionConfig,
    UsageStats,
)


class CompletionsUAMPAdapter:
    """
    OpenAI Chat Completions API adapter.
    
    Converts the familiar Chat Completions request/response format
    to/from UAMP events.
    """
    
    def to_uamp(self, request: Dict[str, Any]) -> List[ClientEvent]:
        """
        Convert Chat Completions request to UAMP events.
        
        Args:
            request: OpenAI Chat Completions request dict
            
        Returns:
            List of UAMP ClientEvent objects
        """
        events: List[ClientEvent] = []
        
        # Create session
        session_config = SessionConfig(
            modalities=["text"],
            tools=request.get("tools"),
            extensions={
                "model": request.get("model"),
                "temperature": request.get("temperature"),
                "max_tokens": request.get("max_tokens"),
            }
        )
        events.append(SessionCreateEvent(session=session_config))
        
        # Convert messages to input events
        for msg in request.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "tool":
                # Tool result message
                events.append(ToolResultEvent(
                    call_id=msg.get("tool_call_id", ""),
                    result=content if isinstance(content, str) else json.dumps(content),
                    is_error=False
                ))
            elif role == "assistant" and msg.get("tool_calls"):
                # Assistant message with tool calls - skip, this is history
                # In UAMP, tool calls come from the model response
                events.append(InputTextEvent(
                    text=content if isinstance(content, str) else "",
                    role="assistant"
                ))
            elif isinstance(content, list):
                # Multimodal content: array of {type, text?, image_url?, ...}
                for part in content:
                    part_type = part.get("type", "text") if isinstance(part, dict) else "text"
                    if part_type == "text":
                        text_val = part.get("text", "") if isinstance(part, dict) else str(part)
                        if text_val:
                            events.append(InputTextEvent(text=text_val, role=role))
                    elif part_type == "image_url":
                        image_url_obj = part.get("image_url", {})
                        url = image_url_obj.get("url", "") if isinstance(image_url_obj, dict) else str(image_url_obj)
                        detail = image_url_obj.get("detail") if isinstance(image_url_obj, dict) else None
                        if url:
                            # Check if base64 data URI or URL
                            if url.startswith("data:"):
                                # Extract base64 from data URI: data:image/png;base64,xxxx
                                comma_idx = url.find(",")
                                base64_data = url[comma_idx + 1:] if comma_idx >= 0 else url
                                # Extract format from MIME
                                fmt = None
                                if "image/" in url:
                                    fmt_part = url.split("image/")[1].split(";")[0].split(",")[0]
                                    if fmt_part in ("jpeg", "jpg", "png", "webp", "gif"):
                                        fmt = fmt_part
                                events.append(InputImageEvent(
                                    image=base64_data,
                                    format=fmt,
                                    detail=detail,
                                ))
                            else:
                                events.append(InputImageEvent(
                                    image={"url": url},
                                    detail=detail,
                                ))
                    elif part_type == "input_audio":
                        audio_data = part.get("input_audio", {})
                        events.append(InputAudioEvent(
                            audio=audio_data.get("data", ""),
                            format=audio_data.get("format", "wav"),
                        ))
            else:
                # Regular text message
                events.append(InputTextEvent(
                    text=content if isinstance(content, str) else json.dumps(content),
                    role=role
                ))
        
        # Request response
        events.append(ResponseCreateEvent())
        
        return events
    
    def from_uamp(self, events: List[ServerEvent]) -> Dict[str, Any]:
        """
        Convert UAMP events to Chat Completions response.
        
        Args:
            events: List of UAMP ServerEvent objects
            
        Returns:
            OpenAI Chat Completions response dict
        """
        # Collect content from delta events
        content_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        usage: Optional[UsageStats] = None
        response_id = ""
        
        for event in events:
            if isinstance(event, ResponseDeltaEvent):
                response_id = event.response_id
                if event.delta:
                    if event.delta.type == "text" and event.delta.text:
                        content_parts.append(event.delta.text)
                    elif event.delta.type == "tool_call" and event.delta.tool_call:
                        # Accumulate tool call
                        tc = event.delta.tool_call
                        # Find or create tool call entry
                        existing = next(
                            (t for t in tool_calls if t.get("id") == tc.get("id")),
                            None
                        )
                        if existing:
                            # Append to arguments
                            if tc.get("arguments"):
                                existing["function"]["arguments"] += tc["arguments"]
                        else:
                            tool_calls.append({
                                "id": tc.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": tc.get("name", ""),
                                    "arguments": tc.get("arguments", "")
                                }
                            })
            elif isinstance(event, ResponseDoneEvent):
                response_id = event.response_id
                if event.response:
                    usage = event.response.usage
                    # Also extract any content from output
                    for item in event.response.output:
                        if item.type == "text" and item.text:
                            if not content_parts:
                                content_parts.append(item.text)
        
        # Build response
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_parts) if content_parts else None,
        }
        
        if tool_calls:
            message["tool_calls"] = tool_calls
        
        response = {
            "id": response_id,
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop"
            }]
        }
        
        if usage:
            response["usage"] = {
                "prompt_tokens": usage.input_tokens,
                "completion_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            }
        
        return response
    
    def from_uamp_streaming(self, event: ServerEvent) -> Optional[str]:
        """
        Convert UAMP event to SSE chunk for streaming.
        
        Args:
            event: Single UAMP ServerEvent
            
        Returns:
            SSE data string or None
        """
        if isinstance(event, ResponseDeltaEvent):
            if event.delta:
                chunk: Dict[str, Any] = {
                    "id": event.response_id,
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": None
                    }]
                }
                
                if event.delta.type == "text" and event.delta.text:
                    chunk["choices"][0]["delta"]["content"] = event.delta.text
                elif event.delta.type == "tool_call" and event.delta.tool_call:
                    tc = event.delta.tool_call
                    chunk["choices"][0]["delta"]["tool_calls"] = [{
                        "index": 0,
                        "id": tc.get("id"),
                        "type": "function",
                        "function": {
                            "name": tc.get("name"),
                            "arguments": tc.get("arguments", "")
                        }
                    }]
                
                return f"data: {json.dumps(chunk)}\n\n"
        
        elif isinstance(event, ResponseDoneEvent):
            chunk = {
                "id": event.response_id,
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
        
        return None
    
    @staticmethod
    def messages_to_uamp(messages: List[Dict[str, Any]]) -> List[ClientEvent]:
        """
        Convenience method to convert just messages to UAMP events.
        
        Args:
            messages: List of message dicts
            
        Returns:
            List of UAMP ClientEvent objects
        """
        adapter = CompletionsUAMPAdapter()
        return adapter.to_uamp({"messages": messages})
