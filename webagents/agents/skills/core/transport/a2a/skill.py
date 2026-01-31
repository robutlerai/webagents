"""
A2A Transport Skill - WebAgents V2.0

Google Agent2Agent Protocol implementation.
https://google.github.io/A2A/specification/

Uses UAMP (Universal Agentic Message Protocol) for internal message representation.
"""

import json
import uuid
import time
from typing import Dict, Any, List, Optional, AsyncGenerator, TYPE_CHECKING
from dataclasses import dataclass, field, asdict
from enum import Enum

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import http
from webagents.uamp import (
    InputTextEvent,
    InputImageEvent,
    InputFileEvent,
    ResponseDeltaEvent,
    ResponseDoneEvent,
    ContentDelta,
    ResponseOutput,
)
from .uamp_adapter import A2AUAMPAdapter

if TYPE_CHECKING:
    from webagents.agents.core.base_agent import BaseAgent


class TaskState(str, Enum):
    """A2A Task lifecycle states"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class A2ATask:
    """A2A Task representation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskState = TaskState.PENDING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class A2ATransportSkill(Skill):
    """
    Google Agent2Agent (A2A) Protocol transport.
    
    Implements the A2A specification for agent-to-agent communication:
    - Agent Card discovery at /.well-known/agent.json
    - Task creation and streaming at /tasks
    - Task status and cancellation
    
    Uses UAMP adapters for protocol conversion.
    
    Example:
        agent = BaseAgent(
            name="my-agent",
            skills=[A2ATransportSkill()]
        )
        
        # GET /agents/my-agent/.well-known/agent.json
        # POST /agents/my-agent/tasks
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all")
        self._tasks: Dict[str, A2ATask] = {}
        self._adapter = A2AUAMPAdapter()
    
    async def initialize(self, agent: 'BaseAgent') -> None:
        """Initialize the A2A transport"""
        self.agent = agent
    
    @http("/.well-known/agent.json", method="get")
    async def agent_card(self) -> Dict[str, Any]:
        """
        A2A Agent Card - capability discovery.
        
        Returns JSON describing the agent's identity, capabilities,
        authentication requirements, and supported interaction modes.
        """
        context = self.get_context()
        agent = context.agent if context else self.agent
        
        # Determine authentication requirements based on agent config
        auth_schemes = []
        if agent and hasattr(agent, 'api_key') and agent.api_key:
            auth_schemes.append({
                "type": "bearer",
                "description": "API key authentication via Bearer token"
            })
        if agent and hasattr(agent, 'scopes') and agent.scopes != "all":
            auth_schemes.append({
                "type": "oauth2",
                "description": "OAuth2 authentication for restricted endpoints"
            })
        
        # If no specific auth, allow anonymous
        if not auth_schemes:
            auth_schemes.append({
                "type": "none",
                "description": "No authentication required"
            })
        
        # Get model capabilities from active LLM skill
        model_caps = self._get_model_capabilities(agent)
        
        # Map UAMP modalities to A2A input/output modes
        input_modes = self._modalities_to_a2a_modes(model_caps.get("modalities", ["text"]))
        
        return {
            "name": agent.name if agent else "unknown",
            "description": agent.description if agent and hasattr(agent, 'description') else "",
            "url": f"/agents/{agent.name}" if agent else "",
            "version": "0.2.1",
            "protocolVersion": "0.2.1",
            "provider": {
                "organization": "WebAgents",
                "url": "https://github.com/robutlerai/webagents"
            },
            "capabilities": {
                "streaming": True,
                "pushNotifications": False,
                "stateTransitionHistory": True,
                "artifacts": True
            },
            "authentication": auth_schemes,
            "defaultInputModes": input_modes,
            "defaultOutputModes": ["text", "file", "data"],
            "skills": self._get_agent_skills(agent) if agent else [],
            # UAMP model capabilities (extension)
            "modelCapabilities": model_caps
        }
    
    def _get_agent_skills(self, agent) -> List[Dict[str, Any]]:
        """Get list of agent skills for Agent Card"""
        skills = []
        if hasattr(agent, 'get_all_tools'):
            for tool in agent.get_all_tools():
                if 'function' in tool:
                    skills.append({
                        "id": tool['function'].get('name', ''),
                        "name": tool['function'].get('name', ''),
                        "description": tool['function'].get('description', '')
                    })
        return skills[:10]  # Limit to 10 skills in card
    
    def _get_model_capabilities(self, agent) -> Dict[str, Any]:
        """Get UAMP model capabilities from agent's LLM skills.
        
        Discovers the active LLM skill and extracts its capabilities.
        """
        if not agent or not hasattr(agent, 'skills'):
            return {"modalities": ["text"], "supports_streaming": True}
        
        # Look for LLM skills with UAMP adapters
        for skill in agent.skills.values():
            if hasattr(skill, '_adapter') and hasattr(skill._adapter, 'get_capabilities'):
                caps = skill._adapter.get_capabilities()
                return {
                    "model_id": caps.model_id,
                    "provider": caps.provider,
                    "modalities": caps.modalities,
                    "supports_streaming": caps.supports_streaming,
                    "supports_thinking": caps.supports_thinking,
                    "context_window": caps.context_window,
                    "max_output_tokens": caps.max_output_tokens,
                    "image": {
                        "formats": caps.image.formats,
                        "detail_levels": caps.image.detail_levels,
                    } if caps.image else None,
                    "audio": {
                        "input_formats": caps.audio.input_formats,
                        "output_formats": caps.audio.output_formats,
                        "supports_realtime": caps.audio.supports_realtime,
                    } if caps.audio else None,
                    "file": {
                        "supports_pdf": caps.file.supports_pdf,
                        "supported_mime_types": caps.file.supported_mime_types,
                    } if caps.file else None,
                    "tools": {
                        "supports_tools": caps.tools.supports_tools,
                        "built_in_tools": caps.tools.built_in_tools,
                    } if caps.tools else None,
                }
        
        # Default capabilities if no LLM skill found
        return {"modalities": ["text"], "supports_streaming": True}
    
    def _modalities_to_a2a_modes(self, modalities: List[str]) -> List[str]:
        """Convert UAMP modalities to A2A input modes."""
        modes = ["text"]  # Always support text
        if "image" in modalities:
            modes.append("file")  # A2A uses 'file' for images
        if "file" in modalities:
            if "file" not in modes:
                modes.append("file")
        if "audio" in modalities or "video" in modalities:
            modes.append("data")
        return modes
    
    @http("/tasks", method="post")
    async def create_task(
        self,
        message: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Create and execute an A2A task with SSE streaming.
        
        Uses full UAMP flow:
        1. Convert A2A request to UAMP events via adapter
        2. Process through agent.process_uamp()
        3. Convert UAMP server events back to A2A format via adapter
        """
        from webagents.uamp import ResponseDeltaEvent, ResponseDoneEvent
        
        # Create task
        task = A2ATask()
        self._tasks[task.id] = task
        
        # Get agent reference
        context = self.get_context()
        agent = context.agent if context else self.agent
        
        if not agent:
            task.status = TaskState.FAILED
            task.error = "No agent available"
            yield self._sse_event("task.failed", {
                "id": task.id,
                "status": task.status.value,
                "error": task.error
            })
            return
        
        # Convert A2A request to UAMP events via adapter
        a2a_request = {"message": message, "messages": messages}
        uamp_events = self._adapter.to_uamp(a2a_request)
        
        # Store messages for history
        task.messages = self._uamp_to_openai_messages(uamp_events)
        
        # Emit task started event
        task.status = TaskState.RUNNING
        task.updated_at = time.time()
        yield self._sse_event("task.started", {
            "id": task.id,
            "status": task.status.value,
            "createdAt": task.created_at
        })
        
        try:
            # Process through agent's native UAMP method
            full_content = []
            async for uamp_event in agent.process_uamp(uamp_events):
                # Convert UAMP event to A2A format via adapter
                a2a_event = self._adapter.from_uamp_streaming(uamp_event)
                if a2a_event:
                    yield self._sse_event(a2a_event["event"], a2a_event["data"])
                    
                # Track content for result
                if isinstance(uamp_event, ResponseDeltaEvent):
                    if uamp_event.delta and uamp_event.delta.text:
                        full_content.append(uamp_event.delta.text)
            
            # Task completed
            task.status = TaskState.COMPLETED
            task.updated_at = time.time()
            task.result = {"content": "".join(full_content)}
            
            yield self._sse_event("task.completed", {
                "id": task.id,
                "status": task.status.value,
                "completedAt": task.updated_at
            })
            
        except Exception as e:
            # Task failed
            task.status = TaskState.FAILED
            task.updated_at = time.time()
            task.error = str(e)
            
            yield self._sse_event("task.failed", {
                "id": task.id,
                "status": task.status.value,
                "error": str(e)
            })
    
    @http("/tasks/{task_id}", method="get")
    async def get_task(self, task_id: str) -> Dict[str, Any]:
        """Get task status by ID"""
        task = self._tasks.get(task_id)
        if not task:
            return {"error": "Task not found", "task_id": task_id}
        
        return {
            "id": task.id,
            "status": task.status.value,
            "createdAt": task.created_at,
            "updatedAt": task.updated_at,
            "result": task.result,
            "error": task.error
        }
    
    @http("/tasks/{task_id}", method="delete")
    async def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """Cancel a running task"""
        task = self._tasks.get(task_id)
        if not task:
            return {"error": "Task not found", "task_id": task_id}
        
        if task.status == TaskState.RUNNING:
            task.status = TaskState.CANCELLED
            task.updated_at = time.time()
        
        return {
            "id": task.id,
            "status": task.status.value,
            "cancelled": task.status == TaskState.CANCELLED
        }
    
    @http("/tasks/{task_id}/artifacts", method="get")
    async def get_task_artifacts(self, task_id: str) -> Dict[str, Any]:
        """Get artifacts generated by a task"""
        task = self._tasks.get(task_id)
        if not task:
            return {"error": "Task not found", "task_id": task_id}
        
        # Artifacts are stored in task result
        artifacts = []
        if task.result:
            # Convert result to artifact format
            artifacts.append({
                "id": f"artifact_{task.id}",
                "type": "text",
                "mimeType": "text/plain",
                "data": task.result.get("content", ""),
                "createdAt": task.updated_at
            })
        
        return {
            "task_id": task_id,
            "artifacts": artifacts
        }
    
    def _uamp_to_openai_messages(self, uamp_events: List) -> List[Dict[str, Any]]:
        """Extract OpenAI-compatible messages from UAMP events.
        
        This bridges UAMP events to the handoff system which currently
        expects OpenAI message format.
        """
        openai_messages = []
        content_parts = []
        current_role = "user"
        has_multimodal = False
        
        for event in uamp_events:
            if isinstance(event, InputTextEvent):
                # If role changes, flush accumulated content
                if current_role != event.role and content_parts:
                    openai_messages.append({
                        "role": current_role,
                        "content": self._build_content(content_parts, has_multimodal)
                    })
                    content_parts = []
                    has_multimodal = False
                
                current_role = event.role
                content_parts.append({"type": "text", "text": event.text})
                
            elif isinstance(event, InputImageEvent):
                has_multimodal = True
                if isinstance(event.image, str) and event.image.startswith("data:"):
                    # Data URL
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": event.image}
                    })
                elif isinstance(event.image, dict) and "url" in event.image:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": event.image["url"]}
                    })
                    
            elif isinstance(event, InputFileEvent):
                # Non-image files as text description
                content_parts.append({
                    "type": "text",
                    "text": f"[File: {event.filename} ({event.mime_type})]"
                })
        
        # Flush remaining content
        if content_parts:
            openai_messages.append({
                "role": current_role,
                "content": self._build_content(content_parts, has_multimodal)
            })
        
        return openai_messages
    
    def _build_content(self, parts: List[Dict[str, Any]], has_multimodal: bool) -> Any:
        """Build OpenAI content from parts."""
        if has_multimodal:
            return parts
        elif len(parts) == 1:
            return parts[0].get("text", "")
        else:
            return "\n".join(p.get("text", "") for p in parts)
    
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
    
    def _sse_event(self, event_type: str, data: Dict[str, Any]) -> str:
        """Format SSE event"""
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
