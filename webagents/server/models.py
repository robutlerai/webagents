"""
OpenAI-Compatible Request/Response Models for WebAgents V2.0 Server
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime


class ChatMessage(BaseModel):
    """OpenAI-compatible chat message"""
    role: str = Field(..., description="Role of the message sender")
    content: Optional[str] = Field(None, description="Message content (can be null for tool calls)")
    name: Optional[str] = Field(None, description="Name of the sender")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool calls in message")
    tool_call_id: Optional[str] = Field(None, description="Tool call ID for tool responses")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: Optional[str] = Field(None, description="Model name (agent name used as model)")
    messages: List[ChatMessage] = Field(..., description="List of messages in conversation")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="External tools to make available")
    temperature: Optional[float] = Field(None, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens in response")
    top_p: Optional[float] = Field(None, description="Nucleus sampling parameter")
    frequency_penalty: Optional[float] = Field(None, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(None, description="Presence penalty")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")


class OpenAIUsage(BaseModel):
    """OpenAI-compatible usage information"""
    prompt_tokens: int = Field(..., description="Number of prompt tokens")
    completion_tokens: int = Field(..., description="Number of completion tokens")
    total_tokens: int = Field(..., description="Total number of tokens")


class OpenAIChoice(BaseModel):
    """OpenAI-compatible choice in response"""
    index: int = Field(..., description="Index of the choice")
    message: ChatMessage = Field(..., description="Generated message")
    finish_reason: Optional[str] = Field(None, description="Reason for finishing")
    
    
class OpenAIResponse(BaseModel):
    """OpenAI-compatible chat completion response"""
    id: str = Field(..., description="Unique completion ID")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: List[OpenAIChoice] = Field(..., description="List of choices")
    usage: OpenAIUsage = Field(..., description="Token usage information")
    system_fingerprint: Optional[str] = Field(None, description="System fingerprint")


class OpenAIStreamChunk(BaseModel):
    """OpenAI-compatible streaming chunk"""
    id: str = Field(..., description="Unique completion ID")
    object: str = Field(default="chat.completion.chunk", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: List[Dict[str, Any]] = Field(..., description="Streaming choices with delta")
    usage: Optional[OpenAIUsage] = Field(None, description="Token usage (final chunk only)")


class AgentInfoResponse(BaseModel):
    """Agent information response"""
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    capabilities: List[str] = Field(..., description="List of agent capabilities")
    skills: List[str] = Field(..., description="List of configured skills")
    tools: List[str] = Field(..., description="List of available tools")
    model: Optional[str] = Field(None, description="Primary LLM model")
    pricing: Dict[str, Any] = Field(default_factory=dict, description="Pricing information")
    

class ServerInfo(BaseModel):
    """Server information response"""
    message: str = Field(default="WebAgents V2 Server", description="Server message")
    version: str = Field(default="2.0.0", description="Server version")
    agents: List[str] = Field(..., description="List of available agents")
    endpoints: Dict[str, str] = Field(..., description="Available endpoints")
    

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Health status")
    version: str = Field(default="2.0.0", description="Server version")
    timestamp: datetime = Field(..., description="Health check timestamp")
    agents: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="Agent health status") 