"""
A2A Transport Skill - WebAgents V2.0

Google Agent2Agent Protocol implementation.
https://google.github.io/A2A/specification/
"""

import json
import uuid
import time
from typing import Dict, Any, List, Optional, AsyncGenerator, TYPE_CHECKING
from dataclasses import dataclass, field, asdict
from enum import Enum

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import http

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
            "defaultInputModes": ["text", "file", "data"],
            "defaultOutputModes": ["text", "file", "data"],
            "skills": self._get_agent_skills(agent) if agent else []
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
    
    @http("/tasks", method="post")
    async def create_task(
        self,
        message: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Create and execute an A2A task with SSE streaming.
        
        Accepts A2A message format and streams task events.
        """
        # Create task
        task = A2ATask()
        self._tasks[task.id] = task
        
        # Convert A2A message format to OpenAI format
        openai_messages = self._a2a_to_openai(message, messages)
        task.messages = openai_messages
        
        # Emit task started event
        task.status = TaskState.RUNNING
        task.updated_at = time.time()
        yield self._sse_event("task.started", {
            "id": task.id,
            "status": task.status.value,
            "createdAt": task.created_at
        })
        
        try:
            # Stream responses through handoff
            full_content = []
            async for chunk in self.execute_handoff(openai_messages):
                # Convert chunk to A2A message format
                a2a_message = self._openai_chunk_to_a2a(chunk)
                if a2a_message:
                    yield self._sse_event("task.message", a2a_message)
                    
                    # Accumulate content
                    if "parts" in a2a_message:
                        for part in a2a_message["parts"]:
                            if part.get("type") == "text":
                                full_content.append(part.get("text", ""))
            
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
    
    def _a2a_to_openai(
        self,
        message: Optional[Dict[str, Any]],
        messages: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Convert A2A message format to OpenAI format"""
        openai_messages = []
        
        # Handle single message
        if message:
            parts = message.get("parts", [])
            content = self._parts_to_content(parts)
            role = message.get("role", "user")
            # A2A uses "agent" role, OpenAI uses "assistant"
            if role == "agent":
                role = "assistant"
            openai_messages.append({"role": role, "content": content})
        
        # Handle message list
        if messages:
            for msg in messages:
                parts = msg.get("parts", [])
                content = self._parts_to_content(parts)
                role = msg.get("role", "user")
                if role == "agent":
                    role = "assistant"
                openai_messages.append({"role": role, "content": content})
        
        return openai_messages
    
    def _parts_to_content(self, parts: List[Dict[str, Any]]) -> Any:
        """Convert A2A parts to OpenAI content format
        
        Supports:
        - TextPart: {"type": "text", "text": "..."}
        - FilePart: {"type": "file", "file": {"name": "...", "mimeType": "...", "data": "base64..."}}
        - DataPart: {"type": "data", "data": {...}, "mimeType": "application/json"}
        """
        content_parts = []
        has_multimodal = False
        
        for part in parts:
            part_type = part.get("type", "text")
            
            if part_type == "text" or "text" in part:
                text = part.get("text", "")
                content_parts.append({"type": "text", "text": text})
                
            elif part_type == "file":
                has_multimodal = True
                file_data = part.get("file", {})
                mime_type = file_data.get("mimeType", "")
                
                # Handle image files as image_url for vision models
                if mime_type.startswith("image/"):
                    if "data" in file_data:
                        # Inline base64 data
                        data_url = f"data:{mime_type};base64,{file_data['data']}"
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": data_url}
                        })
                    elif "uri" in file_data:
                        # URL reference
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": file_data["uri"]}
                        })
                else:
                    # Non-image file - include as text description
                    name = file_data.get("name", "unnamed")
                    content_parts.append({
                        "type": "text",
                        "text": f"[File: {name} ({mime_type})]"
                    })
                    
            elif part_type == "data":
                # Structured JSON data
                data = part.get("data", {})
                mime_type = part.get("mimeType", "application/json")
                content_parts.append({
                    "type": "text",
                    "text": f"[Data ({mime_type})]: {json.dumps(data)}"
                })
        
        # Return simple string for text-only, list for multimodal
        if not has_multimodal and len(content_parts) == 1:
            return content_parts[0].get("text", "")
        elif not has_multimodal:
            return "\n".join(p.get("text", "") for p in content_parts)
        else:
            return content_parts
    
    def _openai_chunk_to_a2a(self, chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert OpenAI streaming chunk to A2A message format"""
        choices = chunk.get("choices", [])
        if not choices:
            return None
        
        delta = choices[0].get("delta", {})
        content = delta.get("content", "")
        
        if not content:
            return None
        
        return {
            "role": "agent",
            "parts": [{"type": "text", "text": content}]
        }
    
    def _sse_event(self, event_type: str, data: Dict[str, Any]) -> str:
        """Format SSE event"""
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
