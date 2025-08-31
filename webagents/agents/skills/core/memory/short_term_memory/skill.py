"""
Short-term Memory Skill - WebAgents V2.0

Manages conversation context, message filtering, and immediate memory operations.
Handles recent message history, context windowing, and conversation summarization.
"""

import time
from typing import Dict, Any, List, Optional
from collections import deque
from dataclasses import dataclass

from ....base import Skill
from .....tools.decorators import tool, hook
from webagents.utils.logging import get_logger, log_skill_event, timer


@dataclass
class MessageContext:
    """Represents a message with context metadata"""
    role: str
    content: str
    timestamp: float
    token_count: Optional[int] = None
    importance: float = 1.0  # 0.0 to 1.0, higher = more important
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ShortTermMemorySkill(Skill):
    """
    Short-term memory skill for conversation context and message filtering
    
    Features:
    - Message history management with configurable window size
    - Intelligent message filtering and prioritization
    - Context summarization for long conversations
    - Token count tracking and optimization
    - Conversation state management
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all")
        
        # Configuration
        self.max_messages = config.get('max_messages', 50) if config else 50
        self.max_tokens = config.get('max_tokens', 4000) if config else 4000
        self.importance_threshold = config.get('importance_threshold', 0.3) if config else 0.3
        
        # Message storage
        self.message_history = deque(maxlen=self.max_messages)
        self.conversation_summary = ""
        self.total_tokens = 0
        
        # State tracking
        self.conversation_id = None
        self.last_activity = time.time()
    
    async def initialize(self, agent: 'BaseAgent') -> None:
        """Initialize short-term memory skill"""
        from webagents.utils.logging import get_logger, log_skill_event
        
        self.agent = agent
        self.logger = get_logger('skill.memory.short_term', agent.name)
        
        log_skill_event(agent.name, 'short_term_memory', 'initialized', {
            'max_messages': self.max_messages,
            'max_tokens': self.max_tokens,
            'importance_threshold': self.importance_threshold
        })
    
    @hook("on_connection", priority=15)
    async def setup_memory_context(self, context) -> Any:
        """Setup memory context for new connections"""
        self.logger.debug("Setting up memory context for new connection")
        
        # Initialize conversation tracking
        self.conversation_id = context.get("request_id") 
        self.last_activity = time.time()
        
        # Store memory state in context for other skills to access
        context.set("memory_state", {
            "conversation_id": self.conversation_id,
            "message_count": len(self.message_history),
            "total_tokens": self.total_tokens
        })
        
        return context
    
    @hook("on_message", priority=20)
    async def process_message_memory(self, context) -> Any:
        """Process and store new messages in memory"""
        messages = context.get("messages", [])
        
        if not messages:
            return context
        
        # Process the latest message
        latest_message = messages[-1]
        
        with timer("message_processing", self.agent.name):
            await self._add_message_to_memory(
                role=latest_message.get("role", "user"),
                content=latest_message.get("content", ""),
                metadata={"source": "conversation"}
            )
        
        # Update context with memory information
        context.set("memory_stats", {
            "messages_stored": len(self.message_history),
            "total_tokens": self.total_tokens,
            "last_activity": self.last_activity
        })
        
        return context
    
    @tool(description="Add a message to short-term memory with importance weighting")
    async def add_message(self, role: str, content: str, importance: float = 1.0, 
                         metadata: Optional[Dict[str, Any]] = None, context=None) -> str:
        """Add a message to short-term memory"""
        
        await self._add_message_to_memory(role, content, importance, metadata or {})
        
        if context:
            context.track_usage(1, "short_term_memory_storage")
        
        return f"Message stored in short-term memory (importance: {importance})"
    
    @tool(description="Retrieve recent conversation history")
    async def get_recent_messages(self, count: int = 10, min_importance: float = 0.0, 
                                 context=None) -> List[Dict[str, Any]]:
        """Retrieve recent messages from short-term memory"""
        
        # Filter messages by importance and recency
        filtered_messages = [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "importance": msg.importance,
                "metadata": msg.metadata
            }
            for msg in list(self.message_history)[-count:]
            if msg.importance >= min_importance
        ]
        
        self.logger.info(f"Retrieved {len(filtered_messages)} recent messages")
        
        if context:
            context.track_usage(0.5, "short_term_memory_retrieval")
        
        return filtered_messages
    
    @tool(description="Get conversation summary for context compression")
    async def get_conversation_summary(self, context=None) -> str:
        """Get a summary of the current conversation"""
        
        if not self.conversation_summary and len(self.message_history) > 5:
            # Generate summary from recent messages
            with timer("conversation_summarization", self.agent.name):
                await self._generate_summary()
        
        if context:
            context.track_usage(2, "conversation_summarization")
        
        return self.conversation_summary or "No conversation summary available"
    
    @tool(description="Clear short-term memory and start fresh")
    async def clear_memory(self, keep_summary: bool = True, context=None) -> str:
        """Clear short-term memory"""
        
        messages_cleared = len(self.message_history)
        
        if not keep_summary:
            self.conversation_summary = ""
        
        self.message_history.clear()
        self.total_tokens = 0
        self.last_activity = time.time()
        
        self.logger.info(f"Cleared {messages_cleared} messages from short-term memory")
        
        if context:
            context.track_usage(1, "memory_clearing")
        
        return f"Cleared {messages_cleared} messages from short-term memory"
    
    @tool(description="Get memory statistics and health info")
    async def get_memory_stats(self, context=None) -> Dict[str, Any]:
        """Get current memory statistics"""
        
        stats = {
            "message_count": len(self.message_history),
            "total_tokens": self.total_tokens,
            "max_messages": self.max_messages,
            "max_tokens": self.max_tokens,
            "memory_utilization": len(self.message_history) / self.max_messages,
            "token_utilization": self.total_tokens / self.max_tokens,
            "conversation_id": self.conversation_id,
            "last_activity": self.last_activity,
            "has_summary": bool(self.conversation_summary)
        }
        
        if context:
            context.track_usage(0.1, "memory_stats_retrieval")
        
        return stats
    
    # Private helper methods
    
    async def _add_message_to_memory(self, role: str, content: str, 
                                   importance: float = 1.0, metadata: Dict[str, Any] = None):
        """Internal method to add message to memory"""
        
        # Estimate token count (rough approximation)
        token_count = len(content.split()) * 1.3  # Rough tokens per word
        
        message = MessageContext(
            role=role,
            content=content,
            timestamp=time.time(),
            token_count=int(token_count),
            importance=importance,
            metadata=metadata or {}
        )
        
        # Add to memory
        self.message_history.append(message)
        self.total_tokens += message.token_count
        self.last_activity = time.time()
        
        # Check if we need to compress memory
        if self._needs_compression():
            await self._compress_memory()
        
        self.logger.debug(f"Added message to memory: {len(content)} chars, {token_count} tokens")
    
    def _needs_compression(self) -> bool:
        """Check if memory needs compression"""
        return (
            len(self.message_history) >= self.max_messages * 0.9 or
            self.total_tokens >= self.max_tokens * 0.9
        )
    
    async def _compress_memory(self):
        """Compress memory by removing less important messages"""
        
        self.logger.info("Compressing short-term memory")
        
        # Convert to list for easier manipulation
        messages = list(self.message_history)
        
        # Sort by importance (keep most important)
        messages.sort(key=lambda m: m.importance, reverse=True)
        
        # Keep top 70% by importance, but always keep recent messages
        keep_count = int(self.max_messages * 0.7)
        recent_count = min(10, len(messages) // 4)
        
        # Always keep recent messages regardless of importance
        recent_messages = list(self.message_history)[-recent_count:]
        important_messages = messages[:keep_count - recent_count]
        
        # Combine and remove duplicates
        kept_messages = []
        seen_content = set()
        
        for msg in important_messages + recent_messages:
            if msg.content not in seen_content:
                kept_messages.append(msg)
                seen_content.add(msg.content)
        
        # Update memory
        self.message_history.clear()
        self.message_history.extend(kept_messages)
        
        # Recalculate token count
        self.total_tokens = sum(msg.token_count for msg in self.message_history if msg.token_count)
        
        self.logger.info(f"Memory compressed: kept {len(kept_messages)} messages, {self.total_tokens} tokens")
    
    async def _generate_summary(self):
        """Generate a summary of the conversation"""
        
        if len(self.message_history) < 3:
            return
        
        # Simple extractive summarization (in production, could use LLM)
        messages = list(self.message_history)
        
        # Get key messages based on importance and recency
        key_messages = [
            msg for msg in messages
            if msg.importance > 0.7 or msg in messages[-5:]
        ]
        
        if key_messages:
            summary_parts = []
            current_topic = ""
            
            for msg in key_messages[-10:]:  # Last 10 key messages
                if len(msg.content) > 20:  # Ignore very short messages
                    summary_parts.append(f"{msg.role}: {msg.content[:100]}...")
            
            self.conversation_summary = "\n".join(summary_parts)
            self.logger.debug(f"Generated conversation summary: {len(self.conversation_summary)} chars")
    
    # Context integration methods
    
    def get_context_messages(self, max_tokens: int = None) -> List[Dict[str, Any]]:
        """Get messages formatted for LLM context"""
        
        target_tokens = max_tokens or self.max_tokens
        messages = []
        current_tokens = 0
        
        # Add messages from newest to oldest until we hit token limit
        for message in reversed(self.message_history):
            if current_tokens + message.token_count <= target_tokens:
                messages.append({
                    "role": message.role,
                    "content": message.content
                })
                current_tokens += message.token_count
            else:
                break
        
        # Reverse to get chronological order
        messages.reverse()
        
        # If we have a summary and not many messages fit, prepend summary
        if len(messages) < 3 and self.conversation_summary:
            messages.insert(0, {
                "role": "system",
                "content": f"Previous conversation summary: {self.conversation_summary}"
            })
        
        return messages 