"""
FastAPI Request/Response Models - WebAgents V2.0

Pydantic models for OpenAI-compatible API endpoints and server responses.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """OpenAI-compatible chat message"""
    role: str = Field(..., description="Message role: 'system', 'user', 'assistant', or 'tool'")
    content: Optional[Union[str, List[Dict[str, Any]]]] = Field(None, description="Message content (string or array of content parts for multimodal)")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool calls in the message")
    tool_call_id: Optional[str] = Field(None, description="Tool call ID for tool responses")


class ToolFunction(BaseModel):
    """OpenAI-compatible tool function definition"""
    name: str = Field(..., description="Function name")
    description: str = Field(..., description="Function description")
    parameters: Dict[str, Any] = Field(..., description="JSON schema for function parameters")


class Tool(BaseModel):
    """OpenAI-compatible tool definition"""
    type: str = Field("function", description="Tool type")
    function: ToolFunction = Field(..., description="Function definition")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str = Field(..., description="Model name")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    tools: Optional[List[Tool]] = Field(None, description="Available tools")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Tool choice strategy")
    temperature: Optional[float] = Field(None, ge=0, le=2, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, gt=0, description="Maximum tokens to generate")
    stream: bool = Field(False, description="Whether to stream the response")
    top_p: Optional[float] = Field(None, ge=0, le=1, description="Nucleus sampling parameter")
    frequency_penalty: Optional[float] = Field(None, ge=-2, le=2, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(None, ge=-2, le=2, description="Presence penalty")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")


class Usage(BaseModel):
    """OpenAI-compatible usage statistics"""
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total number of tokens used")


class ChatCompletionChoice(BaseModel):
    """OpenAI-compatible chat completion choice"""
    index: int = Field(..., description="Choice index")
    message: ChatMessage = Field(..., description="Generated message")
    finish_reason: str = Field(..., description="Reason for completion finish")


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response"""
    id: str = Field(..., description="Completion ID")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: List[ChatCompletionChoice] = Field(..., description="List of completion choices")
    usage: Usage = Field(..., description="Token usage statistics")


class AgentInfoResponse(BaseModel):
    """Agent information response"""
    name: str = Field(..., description="Agent name")
    instructions: str = Field(..., description="Agent instructions")
    model: str = Field(..., description="Model identifier")
    endpoints: Dict[str, str] = Field(..., description="Available endpoints")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Health status")
    version: str = Field(..., description="Server version")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")
    agents_count: int = Field(..., description="Number of registered agents")
    dynamic_agents_enabled: bool = Field(..., description="Whether dynamic agents are enabled")


class AgentListResponse(BaseModel):
    """Agent list response"""
    agents: List[Dict[str, Any]] = Field(..., description="List of available agents")
    total_count: int = Field(..., description="Total number of agents")


class ServerStatsResponse(BaseModel):
    """Server statistics response"""
    server: Dict[str, Any] = Field(..., description="Server information")
    agents: Dict[str, Any] = Field(..., description="Agent statistics")
    performance: Optional[Dict[str, Any]] = Field(None, description="Performance metrics") 