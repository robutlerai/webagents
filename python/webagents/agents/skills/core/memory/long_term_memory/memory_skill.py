"""
LongTermMemorySkill - Persistent Memory Management

Automatically extracts and stores key facts, preferences, and context
from conversations for future reference using WebAgents portal storage
via dependencies.
"""

import json
import uuid
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from ....base import Skill
from .....tools.decorators import tool, prompt


@dataclass
class MemoryItem:
    """Represents a single memory item"""
    id: str
    content: str                    # The actual memory content
    category: str                   # Type of memory (preference, fact, context, etc.)
    importance: int                 # 1-10 importance score
    source: str                     # Where this memory came from
    tags: List[str]                # Keywords for searching
    created_at: str                # ISO timestamp
    last_accessed: Optional[str] = None    # When last used
    access_count: int = 0          # How often accessed

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()


class LongTermMemorySkill(Skill):
    """
    Long-term memory management skill with webagents portal integration via dependencies.
    
    Features:
    - Automatic memory extraction from conversations
    - Categorized memory storage (preferences, facts, context)
    - Importance scoring and prioritization
    - Searchable memory retrieval
    - Memory cleanup and maintenance
    - Integration with webagents portal storage via dependencies
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_memories = config.get('max_memories', 100) if config else 100
        self.auto_extract = config.get('auto_extract', True) if config else True
        self.use_webagents_storage = config.get('use_webagents_storage', True) if config else True
        self.fallback_file = config.get('fallback_file', '/tmp/agent_memory.json') if config else '/tmp/agent_memory.json'
        self.memories: Dict[str, MemoryItem] = {}
        self.agent_name = config.get('agent_name', 'default_agent') if config else 'default_agent'
        
        # Dependencies
        self.dependencies = config.get('dependencies', {}) if config else {}
        self.storage = None

    async def initialize(self, agent_reference):
        """Initialize with agent reference and dependencies"""
        await super().initialize(agent_reference)
        self.agent = agent_reference
        
        # Extract agent name from agent reference if available
        if hasattr(agent_reference, 'name'):
            self.agent_name = agent_reference.name
        
        # Get JSON storage from dependencies
        if 'webagents.json_storage' in self.dependencies:
            self.storage = self.dependencies['webagents.json_storage']
        elif hasattr(agent_reference, 'skills') and 'json_storage' in agent_reference.skills:
            # Fallback to agent skills
            self.storage = agent_reference.skills['json_storage']
        elif hasattr(agent_reference, 'skills') and 'storage' in agent_reference.skills:
            # Legacy fallback for old storage skill
            self.storage = agent_reference.skills['storage']
        
        # Load existing memories
        await self._load_memories()

    @prompt(priority=15, scope="all")
    def memory_extraction_guidance(self, context) -> str:
        """Guide the LLM on when and how to extract memories"""
        return """
LONG-TERM MEMORY EXTRACTION GUIDANCE:

Automatically extract and store important information from conversations for future reference:

WHAT TO MEMORIZE:
1. **User Preferences** - Coding style, frameworks, tools they prefer
2. **Project Context** - Current projects, technologies being used
3. **Key Facts** - Important decisions, requirements, constraints
4. **Work Patterns** - How they like to organize code, test preferences
5. **Domain Knowledge** - Specific business rules, technical details
6. **Recurring Themes** - Common problems, frequent requests

WHEN TO EXTRACT MEMORIES:
- User mentions preferences ("I prefer pytest over unittest")
- Important project decisions are made
- Technical requirements are established
- User provides context about their workflow
- Key facts emerge that would be useful later

HOW TO EXTRACT:
Use extract_key_memories() when you notice:
- Statements about preferences or requirements
- Important project context or decisions
- Technical specifications or constraints
- Workflow patterns or methodologies
- Domain-specific knowledge

MEMORY CATEGORIES:
- "preference" - User likes/dislikes, preferred tools/methods
- "project" - Current work, technologies, requirements
- "fact" - Important decisions, specifications, constraints
- "workflow" - How user organizes work, testing approaches
- "domain" - Business rules, technical knowledge, context

EXAMPLE TRIGGERS:
- "I always use pytest for testing" → extract_key_memories()
- "This project uses React and TypeScript" → extract_key_memories()
- "We need to support Python 3.8+" → extract_key_memories()
- "I organize tests in a dedicated tests/ folder" → extract_key_memories()
"""

    @tool
    async def extract_key_memories(
        self,
        conversation_context: str,
        focus_area: Optional[str] = None
    ) -> str:
        """
        Extract key memories from conversation context.
        
        Args:
            conversation_context: Recent conversation or context to analyze
            focus_area: Optional area to focus on (preferences, project, workflow)
            
        Returns:
            JSON string with extracted memories
        """
        try:
            # Analyze the conversation context and extract key information
            extracted_memories = self._analyze_and_extract(conversation_context, focus_area)
            
            saved_memories = []
            for memory_data in extracted_memories:
                memory_id = await self._save_memory_item(
                    content=memory_data['content'],
                    category=memory_data['category'],
                    importance=memory_data['importance'],
                    source="conversation_extraction",
                    tags=memory_data.get('tags', [])
                )
                saved_memories.append({
                    "id": memory_id,
                    "content": memory_data['content'],
                    "category": memory_data['category']
                })
            
            # Save to persistent storage
            await self._save_memories()
            
            return json.dumps({
                "extracted_count": len(saved_memories),
                "memories": saved_memories,
                "status": "success"
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"Failed to extract memories: {str(e)}",
                "extracted_count": 0
            })

    @tool
    async def save_memory(
        self,
        content: str,
        category: str = "fact",
        importance: int = 5,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Manually save a specific memory.
        
        Args:
            content: The memory content to store
            category: Memory category (preference, project, fact, workflow, domain)
            importance: Importance score 1-10
            tags: Optional tags for searching
            
        Returns:
            JSON string with memory details
        """
        try:
            memory_id = await self._save_memory_item(
                content=content,
                category=category,
                importance=importance,
                source="manual_entry",
                tags=tags or []
            )
            
            # Save to persistent storage
            await self._save_memories()
            
            return json.dumps({
                "memory_id": memory_id,
                "content": content,
                "category": category,
                "importance": importance,
                "status": "saved"
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"Failed to save memory: {str(e)}",
                "content": content
            })

    @tool
    async def list_memories(
        self,
        category: Optional[str] = None,
        min_importance: int = 1,
        search_tags: Optional[List[str]] = None
    ) -> str:
        """
        List stored memories with optional filtering.
        
        Args:
            category: Filter by category
            min_importance: Minimum importance score
            search_tags: Filter by tags
            
        Returns:
            JSON string with memory list
        """
        try:
            filtered_memories = []
            
            for memory in self.memories.values():
                # Apply filters
                if category and memory.category != category:
                    continue
                if memory.importance < min_importance:
                    continue
                if search_tags and not any(tag in memory.tags for tag in search_tags):
                    continue
                
                filtered_memories.append({
                    "id": memory.id,
                    "content": memory.content,
                    "category": memory.category,
                    "importance": memory.importance,
                    "tags": memory.tags,
                    "created_at": memory.created_at,
                    "access_count": memory.access_count
                })
            
            # Sort by importance and access count
            filtered_memories.sort(key=lambda x: (x['importance'], x['access_count']), reverse=True)
            
            storage_location = "webagents_json_storage" if self.storage else "local_file"
            
            return json.dumps({
                "total_memories": len(self.memories),
                "filtered_count": len(filtered_memories),
                "memories": filtered_memories,
                "categories": list(set(m.category for m in self.memories.values())),
                "storage_location": storage_location
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"Failed to list memories: {str(e)}",
                "total_memories": len(self.memories)
            })

    @tool
    async def search_memories(self, query: str, max_results: int = 10) -> str:
        """
        Search memories by content, tags, or category.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            JSON string with search results
        """
        try:
            query_lower = query.lower()
            matches = []
            
            for memory in self.memories.values():
                score = 0
                
                # Content matching
                if query_lower in memory.content.lower():
                    score += 3
                
                # Tag matching
                if any(query_lower in tag.lower() for tag in memory.tags):
                    score += 2
                
                # Category matching
                if query_lower in memory.category.lower():
                    score += 1
                
                if score > 0:
                    # Update access tracking
                    memory.last_accessed = datetime.utcnow().isoformat()
                    memory.access_count += 1
                    
                    matches.append({
                        "id": memory.id,
                        "content": memory.content,
                        "category": memory.category,
                        "importance": memory.importance,
                        "tags": memory.tags,
                        "score": score
                    })
            
            # Sort by score and importance
            matches.sort(key=lambda x: (x['score'], x['importance']), reverse=True)
            matches = matches[:max_results]
            
            # Save updated access counts
            await self._save_memories()
            
            return json.dumps({
                "query": query,
                "total_matches": len(matches),
                "memories": matches
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"Failed to search memories: {str(e)}",
                "query": query
            })

    @tool
    async def delete_memory(self, memory_id: str) -> str:
        """
        Delete a specific memory.
        
        Args:
            memory_id: ID of memory to delete
            
        Returns:
            JSON string with deletion result
        """
        try:
            if memory_id not in self.memories:
                return json.dumps({
                    "error": f"Memory {memory_id} not found",
                    "memory_id": memory_id
                })
            
            memory = self.memories[memory_id]
            del self.memories[memory_id]
            
            # Save to persistent storage
            await self._save_memories()
            
            return json.dumps({
                "memory_id": memory_id,
                "deleted_content": memory.content,
                "status": "deleted"
            })
            
        except Exception as e:
            return json.dumps({
                "error": f"Failed to delete memory: {str(e)}",
                "memory_id": memory_id
            })

    @tool
    async def get_memory_stats(self) -> str:
        """
        Get statistics about stored memories.
        
        Returns:
            JSON string with memory statistics
        """
        try:
            storage_location = "webagents_json_storage" if self.storage else "local_file"
            
            if not self.memories:
                return json.dumps({
                    "total_memories": 0,
                    "categories": {},
                    "importance_distribution": {},
                    "storage_location": storage_location,
                    "agent_name": self.agent_name
                })
            
            # Category breakdown
            categories = {}
            importance_dist = {}
            
            for memory in self.memories.values():
                categories[memory.category] = categories.get(memory.category, 0) + 1
                importance_dist[memory.importance] = importance_dist.get(memory.importance, 0) + 1
            
            # Most accessed memories
            top_memories = sorted(
                self.memories.values(),
                key=lambda x: x.access_count,
                reverse=True
            )[:5]
            
            return json.dumps({
                "total_memories": len(self.memories),
                "categories": categories,
                "importance_distribution": importance_dist,
                "most_accessed": [
                    {
                        "content": m.content[:100] + "..." if len(m.content) > 100 else m.content,
                        "category": m.category,
                        "access_count": m.access_count
                    }
                    for m in top_memories
                ],
                "storage_location": storage_location,
                "agent_name": self.agent_name,
                "max_memories": self.max_memories
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"Failed to get memory stats: {str(e)}",
                "total_memories": len(self.memories)
            })

    def _analyze_and_extract(self, context: str, focus_area: Optional[str] = None) -> List[Dict[str, Any]]:
        """Analyze conversation context and extract key memories"""
        memories = []
        
        # Simple extraction patterns for now (could be enhanced with LLM)
        context_lower = context.lower()
        
        # Extract preferences
        if "prefer" in context_lower or "like to" in context_lower or "always use" in context_lower:
            memories.append({
                "content": self._extract_preference_from_context(context),
                "category": "preference",
                "importance": 7,
                "tags": ["preference", "workflow"]
            })
        
        # Extract project context
        if any(word in context_lower for word in ["project", "using", "building", "working on"]):
            memories.append({
                "content": self._extract_project_context(context),
                "category": "project",
                "importance": 6,
                "tags": ["project", "technology"]
            })
        
        # Extract technical requirements
        if any(word in context_lower for word in ["requirement", "must", "need to", "should"]):
            memories.append({
                "content": self._extract_requirements(context),
                "category": "fact",
                "importance": 8,
                "tags": ["requirement", "constraint"]
            })
        
        return [m for m in memories if m["content"]]  # Filter out empty content

    def _extract_preference_from_context(self, context: str) -> str:
        """Extract preference statements from context"""
        sentences = context.split('.')
        for sentence in sentences:
            if any(word in sentence.lower() for word in ["prefer", "like to", "always use", "usually"]):
                return sentence.strip()
        return ""

    def _extract_project_context(self, context: str) -> str:
        """Extract project-related information"""
        sentences = context.split('.')
        for sentence in sentences:
            if any(word in sentence.lower() for word in ["project", "using", "building", "working on"]):
                return sentence.strip()
        return ""

    def _extract_requirements(self, context: str) -> str:
        """Extract requirement statements"""
        sentences = context.split('.')
        for sentence in sentences:
            if any(word in sentence.lower() for word in ["requirement", "must", "need to", "should"]):
                return sentence.strip()
        return ""

    async def _save_memory_item(
        self,
        content: str,
        category: str,
        importance: int,
        source: str,
        tags: List[str]
    ) -> str:
        """Save a memory item and return its ID"""
        memory_id = str(uuid.uuid4())
        
        memory = MemoryItem(
            id=memory_id,
            content=content,
            category=category,
            importance=max(1, min(10, importance)),  # Clamp to 1-10
            source=source,
            tags=tags,
            created_at=datetime.utcnow().isoformat()
        )
        
        self.memories[memory_id] = memory
        
        # Clean up if we exceed max memories
        if len(self.memories) > self.max_memories:
            self._cleanup_old_memories()
        
        return memory_id

    def _cleanup_old_memories(self):
        """Remove least important/accessed memories when limit is exceeded"""
        # Sort by importance and access count (ascending)
        sorted_memories = sorted(
            self.memories.items(),
            key=lambda x: (x[1].importance, x[1].access_count)
        )
        
        # Remove the least important ones
        to_remove = len(self.memories) - self.max_memories + 10  # Remove a few extra
        for i in range(to_remove):
            if i < len(sorted_memories):
                memory_id = sorted_memories[i][0]
                del self.memories[memory_id]

    async def _load_memories(self):
        """Load memories from storage"""
        try:
            if self.use_webagents_storage and self.storage:
                # Try to load from webagents JSON storage
                result = await self.storage.retrieve_json_data(f"{self.agent_name}_memory.json")
                result_data = json.loads(result)
                
                if result_data.get("success") and "data" in result_data:
                    data = result_data["data"]
                    for memory_data in data.get('memories', []):
                        memory = MemoryItem(**memory_data)
                        self.memories[memory.id] = memory
                    return
            
            # Fallback to local file
            if os.path.exists(self.fallback_file):
                with open(self.fallback_file, 'r') as f:
                    data = json.load(f)
                    for memory_data in data.get('memories', []):
                        memory = MemoryItem(**memory_data)
                        self.memories[memory.id] = memory
        except Exception as e:
            # If loading fails, start with empty memories
            self.memories = {}

    async def _save_memories(self):
        """Save memories to storage"""
        try:
            data = {
                "memories": [asdict(memory) for memory in self.memories.values()],
                "metadata": {
                    "total_count": len(self.memories),
                    "last_updated": datetime.utcnow().isoformat(),
                    "max_memories": self.max_memories,
                    "agent_name": self.agent_name
                }
            }
            
            if self.use_webagents_storage and self.storage:
                # Try to save to webagents JSON storage
                result = await self.storage.store_json_data(
                    f"{self.agent_name}_memory.json",
                    data,
                    f"Long-term memory storage for {self.agent_name}"
                )
                result_data = json.loads(result)
                if result_data.get("success"):
                    return
            
            # Fallback to local file
            os.makedirs(os.path.dirname(self.fallback_file), exist_ok=True)
            with open(self.fallback_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            # Log error but don't fail the operation
            pass

    def get_skill_info(self) -> Dict[str, Any]:
        """Get comprehensive skill information"""
        return {
            "name": "LongTermMemorySkill",
            "description": "Persistent long-term memory with webagents JSON storage via dependencies",
            "version": "3.0.0",
            "capabilities": [
                "Automatic memory extraction from conversations",
                "Categorized memory storage (preferences, facts, etc.)",
                "Searchable memory retrieval",
                "Importance-based prioritization",
                "Memory cleanup and maintenance",
                "WebAgents JSON storage integration via dependencies"
            ],
            "tools": [
                "extract_key_memories",
                "save_memory",
                "list_memories",
                "search_memories",
                "delete_memory",
                "get_memory_stats"
            ],
            "total_memories": len(self.memories),
            "categories": list(set(m.category for m in self.memories.values())) if self.memories else [],
            "config": {
                "use_webagents_storage": self.use_webagents_storage,
                "agent_name": self.agent_name,
                "max_memories": self.max_memories,
                "auto_extract": self.auto_extract,
                "storage_available": self.storage is not None,
                "dependencies": list(self.dependencies.keys())
            }
        } 