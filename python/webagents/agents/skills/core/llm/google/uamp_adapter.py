"""
Google Gemini LLM UAMP Adapter.

Converts between UAMP events and Google GenAI (Gemini) API format.
Handles differences like:
- Content parts format
- Tool/function declaration format
- Multi-turn conversation structure
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
    InputVideoEvent,
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
    AudioCapabilities,
    FileCapabilities,
    ToolCapabilities,
    ClientEvent,
    ServerEvent,
)


class GoogleUAMPAdapter:
    """
    Adapter for converting between UAMP events and Google Gemini API.
    
    Direction:
    - to_gemini: UAMP events → Gemini contents/params
    - to_uamp: Gemini response/chunks → UAMP events
    """
    
    # Model capability definitions
    MODEL_CAPABILITIES = {
        "gemini-2.5-pro": ModelCapabilities(
            model_id="gemini-2.5-pro",
            provider="google",
            modalities=["text", "image", "audio", "video"],
            supports_streaming=True,
            supports_thinking=True,
            supports_caching=True,
            context_window=1048576,  # 1M tokens
            max_output_tokens=8192,
            image=ImageCapabilities(
                formats=["jpeg", "png", "gif", "webp"],
                max_images_per_request=16,
            ),
            audio=AudioCapabilities(
                input_formats=["wav", "mp3", "aiff", "aac", "ogg", "flac"],
                output_formats=["wav", "mp3"],
                supports_realtime=True,
            ),
            file=FileCapabilities(
                supports_pdf=True,
                supported_mime_types=[
                    "application/pdf",
                    "text/plain",
                    "text/csv",
                    "text/html",
                    "application/json",
                    "video/mp4",
                    "video/webm",
                ],
                max_size_bytes=2 * 1024 * 1024 * 1024,  # 2GB for video
            ),
            tools=ToolCapabilities(
                supports_tools=True,
                supports_parallel_tools=True,
                built_in_tools=["google_search", "code_execution"],
            ),
        ),
        "gemini-2.5-flash": ModelCapabilities(
            model_id="gemini-2.5-flash",
            provider="google",
            modalities=["text", "image", "audio", "video"],
            supports_streaming=True,
            supports_thinking=True,
            context_window=1048576,
            max_output_tokens=8192,
            image=ImageCapabilities(
                formats=["jpeg", "png", "gif", "webp"],
            ),
            audio=AudioCapabilities(
                input_formats=["wav", "mp3", "aiff", "aac", "ogg", "flac"],
            ),
            file=FileCapabilities(
                supports_pdf=True,
            ),
            tools=ToolCapabilities(
                supports_tools=True,
                supports_parallel_tools=True,
                built_in_tools=["google_search", "code_execution"],
            ),
        ),
        "gemini-2.0-flash": ModelCapabilities(
            model_id="gemini-2.0-flash",
            provider="google",
            modalities=["text", "image", "audio", "video"],
            supports_streaming=True,
            context_window=1048576,
            max_output_tokens=8192,
            image=ImageCapabilities(
                formats=["jpeg", "png", "gif", "webp"],
            ),
            tools=ToolCapabilities(
                supports_tools=True,
                supports_parallel_tools=True,
                built_in_tools=["google_search", "code_execution"],
            ),
        ),
    }
    
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = model
    
    def get_capabilities(self, model: Optional[str] = None) -> ModelCapabilities:
        """Get capabilities for a model."""
        target_model = model or self.model
        # Find matching capability
        for model_prefix, caps in self.MODEL_CAPABILITIES.items():
            if target_model.startswith(model_prefix):
                return caps
        # Default capabilities
        return ModelCapabilities(
            model_id=target_model,
            provider="google",
            modalities=["text", "image"],
            supports_streaming=True,
        )
    
    def to_gemini(
        self,
        events: List[ClientEvent],
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Convert UAMP client events to Gemini API parameters.
        
        Args:
            events: List of UAMP client events
            model: Model to use
            **kwargs: Additional Gemini parameters
            
        Returns:
            Dict with Gemini API parameters (contents, system_instruction, etc.)
        """
        contents = []
        system_instruction = None
        current_role = "user"
        current_parts = []
        
        for event in events:
            if isinstance(event, InputTextEvent):
                if event.role == "system":
                    # Gemini uses system_instruction parameter
                    if system_instruction:
                        system_instruction += "\n" + event.text
                    else:
                        system_instruction = event.text
                else:
                    # Flush if role changes
                    role = "model" if event.role == "assistant" else "user"
                    if role != current_role and current_parts:
                        contents.append({"role": current_role, "parts": current_parts})
                        current_parts = []
                    
                    current_role = role
                    current_parts.append({"text": event.text})
                    
            elif isinstance(event, InputImageEvent):
                # Images go with user content
                if current_role != "user" and current_parts:
                    contents.append({"role": current_role, "parts": current_parts})
                    current_parts = []
                current_role = "user"
                
                image_part = self._convert_image(event)
                current_parts.append(image_part)
                
            elif isinstance(event, InputAudioEvent):
                # Audio content
                if current_role != "user" and current_parts:
                    contents.append({"role": current_role, "parts": current_parts})
                    current_parts = []
                current_role = "user"
                
                current_parts.append({
                    "inline_data": {
                        "mime_type": f"audio/{event.format}",
                        "data": event.audio,
                    }
                })
                
            elif isinstance(event, InputVideoEvent):
                # Video content
                if current_role != "user" and current_parts:
                    contents.append({"role": current_role, "parts": current_parts})
                    current_parts = []
                current_role = "user"
                
                if isinstance(event.video, str) and event.video.startswith("http"):
                    current_parts.append({
                        "file_data": {
                            "file_uri": event.video,
                            "mime_type": f"video/{event.format or 'mp4'}",
                        }
                    })
                else:
                    data = event.video if isinstance(event.video, str) else event.video.get("data", "")
                    current_parts.append({
                        "inline_data": {
                            "mime_type": f"video/{event.format or 'mp4'}",
                            "data": data,
                        }
                    })
                
            elif isinstance(event, InputFileEvent):
                # Handle files based on mime type
                if event.mime_type.startswith("image/"):
                    current_parts.append(self._convert_file_as_inline(event))
                elif event.mime_type.startswith("video/"):
                    current_parts.append(self._convert_file_as_inline(event))
                elif event.mime_type.startswith("audio/"):
                    current_parts.append(self._convert_file_as_inline(event))
                elif event.mime_type == "application/pdf":
                    current_parts.append(self._convert_file_as_inline(event))
                else:
                    # Other files as text
                    current_parts.append({
                        "text": f"[File: {event.filename} ({event.mime_type})]"
                    })
                    
            elif isinstance(event, ToolResultEvent):
                # Flush current content
                if current_parts:
                    contents.append({"role": current_role, "parts": current_parts})
                    current_parts = []
                
                # Tool results in Gemini format
                contents.append({
                    "role": "user",
                    "parts": [{
                        "function_response": {
                            "name": event.call_id,  # Gemini uses name, we store call_id
                            "response": json.loads(event.result) if event.result.startswith("{") else {"result": event.result},
                        }
                    }]
                })
                current_role = "user"
        
        # Flush remaining content
        if current_parts:
            contents.append({"role": current_role, "parts": current_parts})
        
        # Build params
        params = {
            "model": model or self.model,
            "contents": contents,
        }
        
        if system_instruction:
            params["system_instruction"] = system_instruction
        
        # Pass through other params
        params.update(kwargs)
        
        return params
    
    def _convert_image(self, event: InputImageEvent) -> Dict[str, Any]:
        """Convert InputImageEvent to Gemini image part."""
        if isinstance(event.image, str):
            if event.image.startswith("data:"):
                # Parse data URL
                import re
                match = re.match(r'data:([^;]+);base64,(.+)', event.image)
                if match:
                    media_type, data = match.groups()
                    return {
                        "inline_data": {
                            "mime_type": media_type,
                            "data": data,
                        }
                    }
            elif event.image.startswith("http"):
                # URL - use file_data for external URLs
                return {
                    "file_data": {
                        "file_uri": event.image,
                        "mime_type": f"image/{event.format or 'jpeg'}",
                    }
                }
            else:
                # Assume raw base64
                fmt = event.format or "jpeg"
                return {
                    "inline_data": {
                        "mime_type": f"image/{fmt}",
                        "data": event.image,
                    }
                }
        else:
            # Dict with url
            url = event.image.get("url", "")
            if url.startswith("http"):
                return {
                    "file_data": {
                        "file_uri": url,
                        "mime_type": f"image/{event.format or 'jpeg'}",
                    }
                }
            else:
                return {
                    "inline_data": {
                        "mime_type": f"image/{event.format or 'jpeg'}",
                        "data": url,
                    }
                }
    
    def _convert_file_as_inline(self, event: InputFileEvent) -> Dict[str, Any]:
        """Convert file event to Gemini inline_data format."""
        data = event.file if isinstance(event.file, str) else event.file.get("data", "")
        return {
            "inline_data": {
                "mime_type": event.mime_type,
                "data": data,
            }
        }
    
    def to_uamp(
        self,
        response: Any,
        response_id: Optional[str] = None
    ) -> List[ServerEvent]:
        """
        Convert Gemini GenerateContentResponse to UAMP events.
        
        Args:
            response: Gemini response object
            response_id: Optional response ID
            
        Returns:
            List of UAMP server events
        """
        events = []
        resp_id = response_id or f"gemini_{int(time.time() * 1000)}"
        
        # ResponseCreated
        events.append(ResponseCreatedEvent(response_id=resp_id))
        
        # Extract content
        output_items = []
        
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    output_items.append(ContentItem(
                        type="text",
                        text=part.text
                    ))
                elif hasattr(part, "function_call"):
                    fc = part.function_call
                    events.append(ToolCallEvent(
                        call_id=fc.name,  # Gemini uses name as ID
                        name=fc.name,
                        arguments=json.dumps(dict(fc.args)),
                    ))
                    output_items.append(ContentItem(
                        type="tool_call",
                        tool_call={
                            "id": fc.name,
                            "name": fc.name,
                            "arguments": json.dumps(dict(fc.args)),
                        }
                    ))
                elif hasattr(part, "thought") and part.thought:
                    # Gemini thinking/reasoning
                    events.append(ThinkingEvent(
                        content=part.thought,
                    ))
        
        # Usage
        usage = None
        if hasattr(response, "usage_metadata"):
            um = response.usage_metadata
            usage = UsageStats(
                input_tokens=getattr(um, "prompt_token_count", 0),
                output_tokens=getattr(um, "candidates_token_count", 0),
                total_tokens=getattr(um, "total_token_count", 0),
            )
        
        # ResponseDone
        events.append(ResponseDoneEvent(
            response_id=resp_id,
            response=ResponseOutput(
                id=resp_id,
                status="completed",
                output=output_items,
                usage=usage,
            )
        ))
        
        return events
    
    def to_uamp_streaming(
        self,
        chunk: Any,
        response_id: Optional[str] = None
    ) -> Optional[ServerEvent]:
        """
        Convert Gemini streaming chunk to UAMP event.
        
        Args:
            chunk: Gemini GenerateContentResponse chunk
            response_id: Optional response ID
            
        Returns:
            UAMP ServerEvent or None
        """
        if not chunk.candidates:
            return None
        
        candidate = chunk.candidates[0]
        
        if not hasattr(candidate, "content") or not candidate.content.parts:
            return None
        
        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                return ResponseDeltaEvent(
                    response_id=response_id or "",
                    delta=ContentDelta(type="text", text=part.text)
                )
            elif hasattr(part, "function_call"):
                fc = part.function_call
                return ToolCallEvent(
                    call_id=fc.name,
                    name=fc.name,
                    arguments=json.dumps(dict(fc.args)),
                )
            elif hasattr(part, "thought") and part.thought:
                return ThinkingEvent(
                    content=part.thought,
                    is_delta=True,
                )
        
        # Check for finish
        if hasattr(candidate, "finish_reason") and candidate.finish_reason:
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
        Convert Gemini async stream to UAMP events.
        
        Args:
            stream: Async generator of Gemini chunks
            
        Yields:
            UAMP ServerEvent objects
        """
        response_id = f"gemini_{int(time.time() * 1000)}"
        first_chunk = True
        
        async for chunk in stream:
            if first_chunk:
                yield ResponseCreatedEvent(response_id=response_id)
                first_chunk = False
            
            event = self.to_uamp_streaming(chunk, response_id)
            if event:
                yield event
