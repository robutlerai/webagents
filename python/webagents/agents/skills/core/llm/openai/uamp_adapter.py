"""
OpenAI LLM UAMP Adapter.

Converts between UAMP events and OpenAI API format.
OpenAI's format heavily influenced UAMP, so conversion is mostly straightforward.
"""

import json
import time
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from dataclasses import asdict

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
    AudioCapabilities,
    FileCapabilities,
    ToolCapabilities,
    ClientEvent,
    ServerEvent,
)


class OpenAIUAMPAdapter:
    """
    Adapter for converting between UAMP events and OpenAI Chat Completions API.
    
    Direction:
    - to_openai: UAMP events → OpenAI messages/params
    - to_uamp: OpenAI response/chunks → UAMP events
    """

    # MIME types OpenAI accepts natively via file content parts
    FILE_TYPES = {
        "application/pdf",
        "text/plain", "text/html", "text/css", "text/csv", "text/markdown",
        "text/javascript", "text/x-python", "text/x-c", "text/x-c++", "text/x-java",
        "application/json",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }

    MIME_TO_DEFAULT_EXT = {
        "application/pdf": ".pdf",
        "text/plain": ".txt", "text/html": ".html", "text/css": ".css",
        "text/csv": ".csv", "text/markdown": ".md", "text/javascript": ".js",
        "application/json": ".json",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    }

    # Model capability definitions
    MODEL_CAPABILITIES = {
        "gpt-4o": ModelCapabilities(
            model_id="gpt-4o",
            provider="openai",
            modalities=["text", "image"],
            supports_streaming=True,
            supports_thinking=False,
            context_window=128000,
            max_output_tokens=16384,
            image=ImageCapabilities(
                formats=["jpeg", "png", "gif", "webp"],
                max_images_per_request=20,
                detail_levels=["auto", "low", "high"],
            ),
            file=FileCapabilities(
                supports_pdf=True,
                supported_mime_types=["application/pdf", "text/plain", "text/csv"],
            ),
            tools=ToolCapabilities(
                supports_tools=True,
                supports_parallel_tools=True,
                built_in_tools=["web_search", "code_interpreter"],
            ),
        ),
        "gpt-4o-mini": ModelCapabilities(
            model_id="gpt-4o-mini",
            provider="openai",
            modalities=["text", "image"],
            supports_streaming=True,
            context_window=128000,
            max_output_tokens=16384,
            image=ImageCapabilities(
                formats=["jpeg", "png", "gif", "webp"],
            ),
            tools=ToolCapabilities(
                supports_tools=True,
                supports_parallel_tools=True,
            ),
        ),
        "o1-preview": ModelCapabilities(
            model_id="o1-preview",
            provider="openai",
            modalities=["text"],
            supports_streaming=False,
            supports_thinking=True,
            context_window=128000,
            max_output_tokens=32768,
            tools=ToolCapabilities(
                supports_tools=False,
            ),
        ),
        "o1-mini": ModelCapabilities(
            model_id="o1-mini",
            provider="openai",
            modalities=["text"],
            supports_streaming=False,
            supports_thinking=True,
            context_window=128000,
            max_output_tokens=65536,
            tools=ToolCapabilities(
                supports_tools=False,
            ),
        ),
        "gpt-4o-audio-preview": ModelCapabilities(
            model_id="gpt-4o-audio-preview",
            provider="openai",
            modalities=["text", "audio"],
            supports_streaming=True,
            audio=AudioCapabilities(
                input_formats=["pcm16", "wav"],
                output_formats=["pcm16", "wav"],
                supports_realtime=True,
                voices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            ),
            tools=ToolCapabilities(
                supports_tools=True,
            ),
        ),
    }
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
    
    def get_capabilities(self, model: Optional[str] = None) -> ModelCapabilities:
        """Get capabilities for a model."""
        target_model = model or self.model
        # Find matching capability (prefix match for versioned models)
        for model_prefix, caps in self.MODEL_CAPABILITIES.items():
            if target_model.startswith(model_prefix):
                return caps
        # Default capabilities
        return ModelCapabilities(
            model_id=target_model,
            provider="openai",
            modalities=["text"],
            supports_streaming=True,
        )
    
    @staticmethod
    def convert_messages(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-format messages for OpenAI-compatible APIs.
        Mirrors the TypeScript convertMessages() in adapters/openai.ts.

        Strips content_items and other UAMP-specific fields, forwarding only
        standard OpenAI message properties.
        """
        result: List[Dict[str, Any]] = []
        for m in messages:
            clean: Dict[str, Any] = {"role": m.get("role", "user")}
            clean["content"] = m.get("content", "")
            if m.get("tool_calls"):
                clean["tool_calls"] = m["tool_calls"]
            if m.get("tool_call_id"):
                clean["tool_call_id"] = m["tool_call_id"]
            if m.get("name"):
                clean["name"] = m["name"]
            result.append(clean)
        return result

    def to_openai(
        self,
        events: List[ClientEvent],
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Convert UAMP client events to OpenAI API parameters.
        
        Args:
            events: List of UAMP client events
            model: Model to use
            **kwargs: Additional OpenAI parameters
            
        Returns:
            Dict with OpenAI API parameters (messages, model, etc.)
        """
        messages = []
        current_role = "user"
        current_content = []
        
        for event in events:
            if isinstance(event, InputTextEvent):
                # Flush previous content if role changes
                if event.role != current_role and current_content:
                    messages.append(self._build_message(current_role, current_content))
                    current_content = []
                
                current_role = event.role if event.role != "system" else "system"
                if event.role == "system":
                    # System messages are separate
                    messages.append({"role": "system", "content": event.text})
                else:
                    current_content.append({"type": "text", "text": event.text})
                    
            elif isinstance(event, InputImageEvent):
                # Images always go with user content
                if current_role != "user" and current_content:
                    messages.append(self._build_message(current_role, current_content))
                    current_content = []
                current_role = "user"
                
                image_content = self._convert_image(event)
                current_content.append(image_content)
                
            elif isinstance(event, InputAudioEvent):
                # Audio input (for realtime-capable models)
                if current_role != "user" and current_content:
                    messages.append(self._build_message(current_role, current_content))
                    current_content = []
                current_role = "user"
                
                current_content.append({
                    "type": "input_audio",
                    "input_audio": {
                        "data": event.audio,
                        "format": event.format,
                    }
                })
                
            elif isinstance(event, InputFileEvent):
                if event.mime_type.startswith("image/"):
                    current_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{event.mime_type};base64,{event.file}" 
                            if isinstance(event.file, str) and not event.file.startswith("http")
                            else event.file if isinstance(event.file, str)
                            else event.file.get("url", "")
                        }
                    })
                elif event.mime_type.startswith("video/"):
                    current_content.append({
                        "type": "text",
                        "text": "[Attached video — not supported by this model]"
                    })
                elif event.mime_type in self.FILE_TYPES:
                    data = event.file if isinstance(event.file, str) else event.file.get("data", "")
                    filename = event.filename or f"document{self.MIME_TO_DEFAULT_EXT.get(event.mime_type, '')}"
                    current_content.append({
                        "type": "file",
                        "file": {
                            "filename": filename,
                            "file_data": f"data:{event.mime_type};base64,{data}",
                        }
                    })
                else:
                    current_content.append({
                        "type": "text",
                        "text": f"[Attached file: {event.filename} ({event.mime_type}) — content not available inline]"
                    })
                    
            elif isinstance(event, ToolResultEvent):
                # Flush current content first
                if current_content:
                    messages.append(self._build_message(current_role, current_content))
                    current_content = []
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": event.call_id,
                    "content": event.result,
                })
                current_role = "user"
        
        # Flush remaining content
        if current_content:
            messages.append(self._build_message(current_role, current_content))
        
        # Build params
        params = {
            "model": model or self.model,
            "messages": messages,
            **kwargs
        }
        
        return params
    
    def _build_message(self, role: str, content: List[Dict]) -> Dict[str, Any]:
        """Build a message from content parts."""
        if len(content) == 1 and content[0].get("type") == "text":
            # Simple text message
            return {"role": role, "content": content[0]["text"]}
        else:
            # Multimodal message
            return {"role": role, "content": content}
    
    def _convert_image(self, event: InputImageEvent) -> Dict[str, Any]:
        """Convert InputImageEvent to OpenAI image_url format."""
        if isinstance(event.image, str):
            if event.image.startswith("data:") or event.image.startswith("http"):
                url = event.image
            else:
                # Assume base64, need to construct data URL
                fmt = event.format or "jpeg"
                url = f"data:image/{fmt};base64,{event.image}"
        else:
            # Dict with url key
            url = event.image.get("url", "")
        
        result = {
            "type": "image_url",
            "image_url": {"url": url}
        }
        
        if event.detail:
            result["image_url"]["detail"] = event.detail
            
        return result
    
    def to_uamp(
        self,
        response: Dict[str, Any],
        response_id: Optional[str] = None
    ) -> List[ServerEvent]:
        """
        Convert OpenAI response to UAMP events.
        
        Args:
            response: OpenAI ChatCompletion response
            response_id: Optional response ID to use
            
        Returns:
            List of UAMP server events
        """
        events = []
        resp_id = response_id or response.get("id", "")
        
        # ResponseCreated
        events.append(ResponseCreatedEvent(response_id=resp_id))
        
        # Extract content
        choices = response.get("choices", [])
        output_items = []
        
        for choice in choices:
            message = choice.get("message", {})
            
            # Text content
            if message.get("content"):
                output_items.append(ContentItem(
                    type="text",
                    text=message["content"]
                ))
            
            # Tool calls
            for tool_call in message.get("tool_calls", []) or []:
                events.append(ToolCallEvent(
                    call_id=tool_call.get("id", ""),
                    name=tool_call.get("function", {}).get("name", ""),
                    arguments=tool_call.get("function", {}).get("arguments", ""),
                ))
                output_items.append(ContentItem(
                    type="tool_call",
                    tool_call=tool_call
                ))
        
        # Usage
        usage = None
        if response.get("usage"):
            usage = UsageStats(
                input_tokens=response["usage"].get("prompt_tokens", 0),
                output_tokens=response["usage"].get("completion_tokens", 0),
                total_tokens=response["usage"].get("total_tokens", 0),
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
        chunk: Dict[str, Any],
        response_id: Optional[str] = None
    ) -> Optional[ServerEvent]:
        """
        Convert OpenAI streaming chunk to UAMP event.
        
        Args:
            chunk: OpenAI ChatCompletionChunk
            response_id: Optional response ID
            
        Returns:
            UAMP ServerEvent or None
        """
        resp_id = response_id or chunk.get("id", "")
        choices = chunk.get("choices", [])
        
        if not choices:
            return None
        
        delta = choices[0].get("delta", {})
        finish_reason = choices[0].get("finish_reason")
        
        # Text delta
        if delta.get("content"):
            return ResponseDeltaEvent(
                response_id=resp_id,
                delta=ContentDelta(type="text", text=delta["content"])
            )
        
        # Tool call delta
        if delta.get("tool_calls"):
            tool_call = delta["tool_calls"][0]
            if tool_call.get("function", {}).get("name"):
                # New tool call
                return ToolCallEvent(
                    call_id=tool_call.get("id", ""),
                    name=tool_call["function"]["name"],
                    arguments=tool_call["function"].get("arguments", ""),
                )
            elif tool_call.get("function", {}).get("arguments"):
                # Argument delta - emit as tool_call delta
                return ResponseDeltaEvent(
                    response_id=resp_id,
                    delta=ContentDelta(
                        type="tool_call",
                        tool_call={
                            "id": tool_call.get("id", ""),
                            "arguments": tool_call["function"]["arguments"]
                        }
                    )
                )
        
        # Finish
        if finish_reason:
            return ResponseDoneEvent(
                response_id=resp_id,
                response=ResponseOutput(
                    id=resp_id,
                    status="completed" if finish_reason == "stop" else finish_reason,
                    output=[],
                )
            )
        
        return None
    
    async def stream_to_uamp(
        self,
        stream: AsyncGenerator[Dict[str, Any], None]
    ) -> AsyncGenerator[ServerEvent, None]:
        """
        Convert OpenAI async stream to UAMP events.
        
        Args:
            stream: Async generator of OpenAI chunks
            
        Yields:
            UAMP ServerEvent objects
        """
        response_id = None
        first_chunk = True
        
        async for chunk in stream:
            if first_chunk:
                response_id = chunk.get("id", "")
                yield ResponseCreatedEvent(response_id=response_id)
                first_chunk = False
            
            event = self.to_uamp_streaming(chunk, response_id)
            if event:
                yield event
