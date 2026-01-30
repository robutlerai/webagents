"""
Anthropic LLM UAMP Adapter.

Converts between UAMP events and Anthropic Messages API format.
Handles differences like:
- System prompt as separate parameter
- Different content block structure
- Tool call format differences
- Extended thinking support
"""

import json
import time
from typing import Dict, Any, List, Optional, Union, AsyncGenerator

from webagents.uamp import (
    # Input events
    InputTextEvent,
    InputImageEvent,
    InputAudioEvent,
    InputFileEvent,
    ToolResultEvent,
    # Output events
    ResponseCreatedEvent,
    ResponseDeltaEvent,
    ResponseDoneEvent,
    ResponseErrorEvent,
    ToolCallEvent,
    ThinkingEvent,
    CapabilitiesEvent,
    # Types
    ContentDelta,
    ContentItem,
    UsageStats,
    ResponseOutput,
    ModelCapabilities,
    ImageCapabilities,
    FileCapabilities,
    ToolCapabilities,
    ClientEvent,
    ServerEvent,
)


class AnthropicUAMPAdapter:
    """
    Adapter for converting between UAMP events and Anthropic Messages API.
    
    Direction:
    - to_anthropic: UAMP events → Anthropic messages/params
    - to_uamp: Anthropic response/events → UAMP events
    """
    
    # Model capability definitions
    MODEL_CAPABILITIES = {
        "claude-3-5-sonnet": ModelCapabilities(
            model_id="claude-3-5-sonnet-20241022",
            provider="anthropic",
            modalities=["text", "image"],
            supports_streaming=True,
            supports_thinking=True,
            supports_caching=True,
            context_window=200000,
            max_output_tokens=8192,
            image=ImageCapabilities(
                formats=["jpeg", "png", "gif", "webp"],
                max_size_bytes=20 * 1024 * 1024,  # 20MB
                max_images_per_request=20,
            ),
            file=FileCapabilities(
                supports_pdf=True,
                supported_mime_types=[
                    "application/pdf",
                    "text/plain",
                    "text/csv",
                    "text/html",
                    "application/json",
                ],
                max_size_bytes=32 * 1024 * 1024,  # 32MB for PDFs
            ),
            tools=ToolCapabilities(
                supports_tools=True,
                supports_parallel_tools=True,
                built_in_tools=["computer_use", "bash", "text_editor"],
            ),
        ),
        "claude-3-5-haiku": ModelCapabilities(
            model_id="claude-3-5-haiku-20241022",
            provider="anthropic",
            modalities=["text", "image"],
            supports_streaming=True,
            supports_thinking=True,
            context_window=200000,
            max_output_tokens=8192,
            image=ImageCapabilities(
                formats=["jpeg", "png", "gif", "webp"],
            ),
            tools=ToolCapabilities(
                supports_tools=True,
                supports_parallel_tools=True,
            ),
        ),
        "claude-3-opus": ModelCapabilities(
            model_id="claude-3-opus-20240229",
            provider="anthropic",
            modalities=["text", "image"],
            supports_streaming=True,
            context_window=200000,
            max_output_tokens=4096,
            image=ImageCapabilities(
                formats=["jpeg", "png", "gif", "webp"],
            ),
            tools=ToolCapabilities(
                supports_tools=True,
                supports_parallel_tools=True,
            ),
        ),
    }
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        self.model = model
    
    def get_capabilities(self, model: Optional[str] = None) -> ModelCapabilities:
        """Get capabilities for a model."""
        target_model = model or self.model
        # Find matching capability (prefix match)
        for model_prefix, caps in self.MODEL_CAPABILITIES.items():
            if target_model.startswith(model_prefix):
                return caps
        # Default capabilities
        return ModelCapabilities(
            model_id=target_model,
            provider="anthropic",
            modalities=["text", "image"],
            supports_streaming=True,
        )
    
    def to_anthropic(
        self,
        events: List[ClientEvent],
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Convert UAMP client events to Anthropic API parameters.
        
        Args:
            events: List of UAMP client events
            model: Model to use
            **kwargs: Additional Anthropic parameters
            
        Returns:
            Dict with Anthropic API parameters
        """
        messages = []
        system_prompt = None
        current_role = "user"
        current_content = []
        
        for event in events:
            if isinstance(event, InputTextEvent):
                if event.role == "system":
                    # Anthropic uses top-level system parameter
                    if system_prompt:
                        system_prompt += "\n" + event.text
                    else:
                        system_prompt = event.text
                else:
                    # Flush if role changes
                    role = "assistant" if event.role == "assistant" else "user"
                    if role != current_role and current_content:
                        messages.append({"role": current_role, "content": current_content})
                        current_content = []
                    
                    current_role = role
                    current_content.append({"type": "text", "text": event.text})
                    
            elif isinstance(event, InputImageEvent):
                # Images go with user messages
                if current_role != "user" and current_content:
                    messages.append({"role": current_role, "content": current_content})
                    current_content = []
                current_role = "user"
                
                image_content = self._convert_image(event)
                current_content.append(image_content)
                
            elif isinstance(event, InputFileEvent):
                # Handle files based on mime type
                if event.mime_type.startswith("image/"):
                    current_content.append(self._convert_file_as_image(event))
                elif event.mime_type == "application/pdf":
                    # Anthropic supports PDF via document blocks
                    current_content.append(self._convert_pdf(event))
                else:
                    # Other files as text
                    current_content.append({
                        "type": "text",
                        "text": f"[File: {event.filename} ({event.mime_type})]"
                    })
                    
            elif isinstance(event, ToolResultEvent):
                # Flush current content first
                if current_content:
                    messages.append({"role": current_role, "content": current_content})
                    current_content = []
                
                # Anthropic tool results go in user messages
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": event.call_id,
                        "content": event.result,
                        **({"is_error": True} if event.is_error else {}),
                    }]
                })
                current_role = "user"
        
        # Flush remaining content
        if current_content:
            messages.append({"role": current_role, "content": current_content})
        
        # Build params
        params = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": kwargs.pop("max_tokens", 4096),
        }
        
        if system_prompt:
            params["system"] = system_prompt
        
        # Pass through other params
        params.update(kwargs)
        
        return params
    
    def _convert_image(self, event: InputImageEvent) -> Dict[str, Any]:
        """Convert InputImageEvent to Anthropic image format."""
        if isinstance(event.image, str):
            if event.image.startswith("data:"):
                # Parse data URL
                import re
                match = re.match(r'data:([^;]+);base64,(.+)', event.image)
                if match:
                    media_type, data = match.groups()
                    return {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        }
                    }
            elif event.image.startswith("http"):
                # URL reference
                return {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": event.image,
                    }
                }
            else:
                # Assume raw base64
                fmt = event.format or "jpeg"
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": f"image/{fmt}",
                        "data": event.image,
                    }
                }
        else:
            # Dict with url
            return {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": event.image.get("url", ""),
                }
            }
    
    def _convert_file_as_image(self, event: InputFileEvent) -> Dict[str, Any]:
        """Convert file event to Anthropic image format."""
        data = event.file if isinstance(event.file, str) else event.file.get("data", "")
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": event.mime_type,
                "data": data,
            }
        }
    
    def _convert_pdf(self, event: InputFileEvent) -> Dict[str, Any]:
        """Convert PDF file to Anthropic document format."""
        data = event.file if isinstance(event.file, str) else event.file.get("data", "")
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": data,
            }
        }
    
    def to_uamp(
        self,
        response: Any,
        response_id: Optional[str] = None
    ) -> List[ServerEvent]:
        """
        Convert Anthropic Message response to UAMP events.
        
        Args:
            response: Anthropic Message object
            response_id: Optional response ID
            
        Returns:
            List of UAMP server events
        """
        events = []
        resp_id = response_id or getattr(response, "id", "")
        
        # ResponseCreated
        events.append(ResponseCreatedEvent(response_id=resp_id))
        
        # Extract content
        output_items = []
        
        for block in response.content:
            if block.type == "text":
                output_items.append(ContentItem(
                    type="text",
                    text=block.text
                ))
            elif block.type == "tool_use":
                events.append(ToolCallEvent(
                    call_id=block.id,
                    name=block.name,
                    arguments=json.dumps(block.input),
                ))
                output_items.append(ContentItem(
                    type="tool_call",
                    tool_call={
                        "id": block.id,
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    }
                ))
            elif block.type == "thinking":
                # Extended thinking block
                events.append(ThinkingEvent(
                    content=block.thinking,
                    redacted=getattr(block, "redacted", False),
                ))
        
        # Usage
        usage = None
        if hasattr(response, "usage"):
            usage = UsageStats(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )
        
        # ResponseDone
        events.append(ResponseDoneEvent(
            response_id=resp_id,
            response=ResponseOutput(
                id=resp_id,
                status="completed" if response.stop_reason == "end_turn" else response.stop_reason,
                output=output_items,
                usage=usage,
            )
        ))
        
        return events
    
    def to_uamp_streaming(
        self,
        event: Any,
        response_id: Optional[str] = None
    ) -> Optional[ServerEvent]:
        """
        Convert Anthropic streaming event to UAMP event.
        
        Args:
            event: Anthropic stream event
            response_id: Optional response ID
            
        Returns:
            UAMP ServerEvent or None
        """
        event_type = event.type
        
        if event_type == "message_start":
            resp_id = event.message.id
            return ResponseCreatedEvent(response_id=resp_id)
        
        elif event_type == "content_block_start":
            block = event.content_block
            if block.type == "tool_use":
                return ToolCallEvent(
                    call_id=block.id,
                    name=block.name,
                    arguments="",
                )
            elif block.type == "thinking":
                return ThinkingEvent(
                    content="",
                    is_delta=True,
                )
        
        elif event_type == "content_block_delta":
            delta = event.delta
            if delta.type == "text_delta":
                return ResponseDeltaEvent(
                    response_id=response_id or "",
                    delta=ContentDelta(type="text", text=delta.text)
                )
            elif delta.type == "input_json_delta":
                return ResponseDeltaEvent(
                    response_id=response_id or "",
                    delta=ContentDelta(
                        type="tool_call",
                        tool_call={"arguments": delta.partial_json}
                    )
                )
            elif delta.type == "thinking_delta":
                return ThinkingEvent(
                    content=delta.thinking,
                    is_delta=True,
                )
        
        elif event_type == "message_delta":
            if event.delta.stop_reason:
                return ResponseDoneEvent(
                    response_id=response_id or "",
                    response=ResponseOutput(
                        id=response_id or "",
                        status="completed" if event.delta.stop_reason == "end_turn" else event.delta.stop_reason,
                        output=[],
                    )
                )
        
        elif event_type == "message_stop":
            return ResponseDoneEvent(
                response_id=response_id or "",
                response=ResponseOutput(
                    id=response_id or "",
                    status="completed",
                    output=[],
                )
            )
        
        return None
    
    async def stream_to_uamp(
        self,
        stream: AsyncGenerator[Any, None]
    ) -> AsyncGenerator[ServerEvent, None]:
        """
        Convert Anthropic async stream to UAMP events.
        
        Args:
            stream: Async generator of Anthropic events
            
        Yields:
            UAMP ServerEvent objects
        """
        response_id = None
        
        async for event in stream:
            if event.type == "message_start":
                response_id = event.message.id
            
            uamp_event = self.to_uamp_streaming(event, response_id)
            if uamp_event:
                yield uamp_event
