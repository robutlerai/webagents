"""
Test suite for VectorMemorySkill - Semantic vector memory with Milvus integration
"""

import pytest
import json
import sys
import os
from unittest.mock import AsyncMock, patch, MagicMock

# Add webagents to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.asyncio
async def test_vector_memory_skill_import():
    """Test that VectorMemorySkill can be imported successfully"""
    from webagents.agents.skills.core.vector_memory import VectorMemorySkill, VectorMemoryItem
    assert VectorMemorySkill is not None
    assert VectorMemoryItem is not None
    print("✅ VectorMemorySkill import successful")


@pytest.mark.asyncio
async def test_vector_memory_skill_initialization():
    """Test VectorMemorySkill initialization"""
    from webagents.agents.skills.core.vector_memory import VectorMemorySkill
    
    # Default initialization
    skill = VectorMemorySkill({'agent_name': 'test_agent'})
    assert skill is not None
    assert skill.agent_name == 'test_agent'
    assert skill.milvus_host == 'localhost'
    assert skill.milvus_port == 19530
    assert skill.milvus_collection == 'agent_memory'
    assert skill.default_scope == 'owner'
    
    # Custom configuration
    config = {
        'agent_name': 'custom_agent',
        'milvus_host': 'custom-host',
        'milvus_port': 19531,
        'milvus_collection': 'custom_memory',
        'embedding_model': 'text-embedding-ada-002',
        'default_scope': 'shared',
        'max_memories': 500
    }
    skill_custom = VectorMemorySkill(config)
    assert skill_custom.agent_name == 'custom_agent'
    assert skill_custom.milvus_host == 'custom-host'
    assert skill_custom.milvus_port == 19531
    assert skill_custom.milvus_collection == 'custom_memory'
    assert skill_custom.embedding_model == 'text-embedding-ada-002'
    assert skill_custom.default_scope == 'shared'
    assert skill_custom.max_memories == 500
    
    # Cleanup
    await skill.cleanup()
    await skill_custom.cleanup()
    
    print("✅ VectorMemorySkill initialization test passed")


@pytest.mark.asyncio
async def test_vector_memory_item():
    """Test VectorMemoryItem dataclass"""
    from webagents.agents.skills.core.vector_memory import VectorMemoryItem
    
    # Test with all fields
    memory = VectorMemoryItem(
        id="test-id",
        content="Test memory content",
        category="test",
        importance=8,
        source="test_source",
        tags=["tag1", "tag2"],
        scope="owner",
        embedding=[0.1, 0.2, 0.3]
    )
    
    assert memory.id == "test-id"
    assert memory.content == "Test memory content"
    assert memory.category == "test"
    assert memory.importance == 8
    assert memory.source == "test_source"
    assert memory.tags == ["tag1", "tag2"]
    assert memory.scope == "owner"
    assert memory.access_count == 0
    assert memory.embedding == [0.1, 0.2, 0.3]
    assert memory.created_at is not None  # Should be auto-generated
    
    # Test with minimal fields
    memory_minimal = VectorMemoryItem(
        id="minimal-id",
        content="Minimal content",
        category="minimal",
        importance=5,
        source="minimal_source",
        tags=[]
    )
    
    assert memory_minimal.scope == "owner"  # Default value
    assert memory_minimal.access_count == 0  # Default value
    assert memory_minimal.created_at is not None
    
    print("✅ VectorMemoryItem test passed")


@pytest.mark.asyncio 
async def test_store_vector_memory_without_dependencies():
    """Test storing vector memory when dependencies are not available"""
    from webagents.agents.skills.core.vector_memory import VectorMemorySkill
    
    skill = VectorMemorySkill({'agent_name': 'test_agent'})
    
    try:
        # Test storing without Milvus/OpenAI available
        result = await skill.store_vector_memory(
            content="Test memory content",
            category="test",
            importance=8,
            tags=["test"]
        )
        
        result_data = json.loads(result)
        
        # Should fail gracefully when dependencies not available
        assert result_data["success"] == False
        assert "requirements not met" in result_data["error"]
        
        print("✅ Store vector memory without dependencies test passed")
    finally:
        await skill.cleanup()


@pytest.mark.asyncio
async def test_search_vector_memories_without_dependencies():
    """Test searching vector memories when dependencies are not available"""
    from webagents.agents.skills.core.vector_memory import VectorMemorySkill
    
    skill = VectorMemorySkill({'agent_name': 'test_agent'})
    
    try:
        # Test searching without Milvus/OpenAI available
        result = await skill.search_vector_memories(
            query="test query",
            limit=5
        )
        
        result_data = json.loads(result)
        
        # Should fail gracefully when dependencies not available
        assert result_data["success"] == False
        assert "requirements not met" in result_data["error"]
        
        print("✅ Search vector memories without dependencies test passed")
    finally:
        await skill.cleanup()


@pytest.mark.asyncio
async def test_vector_memory_with_mocked_dependencies():
    """Test VectorMemorySkill with mocked Milvus and OpenAI"""
    from webagents.agents.skills.core.vector_memory import VectorMemorySkill
    
    # Mock the imports to simulate dependencies being available
    with patch('webagents.agents.skills.core.vector_memory.vector_memory_skill.MILVUS_AVAILABLE', True), \
         patch('webagents.agents.skills.core.vector_memory.vector_memory_skill.openai') as mock_openai:
        
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [MagicMock()]
        mock_embedding_response.data[0].embedding = [0.1, 0.2, 0.3] * 512  # 1536 dimensions
        mock_client.embeddings.create.return_value = mock_embedding_response
        mock_openai.OpenAI.return_value = mock_client
        
        skill = VectorMemorySkill({
            'agent_name': 'test_agent',
            'openai_api_key': 'test_key'
        })
        
        # Mock Milvus operations
        with patch.object(skill, '_connect_to_milvus', return_value=True), \
             patch.object(skill, '_check_requirements', return_value=True), \
             patch.object(skill, '_insert_memory', return_value=True):
            
            skill.connected = True
            skill.openai_client = mock_client
            
            try:
                # Test storing memory
                result = await skill.store_vector_memory(
                    content="Test memory with mocks",
                    category="test",
                    importance=7,
                    tags=["mock", "test"]
                )
                
                result_data = json.loads(result)
                assert result_data["success"] == True
                assert "memory_id" in result_data
                assert result_data["category"] == "test"
                assert result_data["importance"] == 7
                assert result_data["scope"] == "owner"  # Default scope
                
                print("✅ Vector memory with mocked dependencies test passed")
            finally:
                await skill.cleanup()


@pytest.mark.asyncio
async def test_vector_memory_search_with_mocks():
    """Test vector memory search with mocked dependencies"""
    from webagents.agents.skills.core.vector_memory import VectorMemorySkill
    
    # Mock the imports
    with patch('webagents.agents.skills.core.vector_memory.vector_memory_skill.MILVUS_AVAILABLE', True), \
         patch('webagents.agents.skills.core.vector_memory.vector_memory_skill.openai') as mock_openai:
        
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [MagicMock()]
        mock_embedding_response.data[0].embedding = [0.1, 0.2, 0.3] * 512
        mock_client.embeddings.create.return_value = mock_embedding_response
        mock_openai.OpenAI.return_value = mock_client
        
        skill = VectorMemorySkill({
            'agent_name': 'test_agent',
            'openai_api_key': 'test_key'
        })
        
        # Mock search results
        mock_search_results = [
            {
                "id": "test-memory-1",
                "content": "This is a test memory about databases",
                "category": "knowledge",
                "importance": 8,
                "tags": ["database", "sql"],
                "scope": "owner",
                "created_at": "2024-01-01T00:00:00Z",
                "access_count": 2,
                "score": 0.85
            },
            {
                "id": "test-memory-2", 
                "content": "Another memory about API design",
                "category": "knowledge",
                "importance": 7,
                "tags": ["api", "design"],
                "scope": "owner",
                "created_at": "2024-01-02T00:00:00Z",
                "access_count": 1,
                "score": 0.72
            }
        ]
        
        with patch.object(skill, '_connect_to_milvus', return_value=True), \
             patch.object(skill, '_check_requirements', return_value=True), \
             patch.object(skill, '_search_memories', return_value=mock_search_results), \
             patch.object(skill, '_update_access_counts', return_value=None):
            
            skill.connected = True
            skill.openai_client = mock_client
            
            try:
                # Test searching memories
                result = await skill.search_vector_memories(
                    query="database information",
                    limit=5,
                    category="knowledge",
                    min_importance=5
                )
                
                result_data = json.loads(result)
                assert result_data["success"] == True
                assert result_data["query"] == "database information"
                assert result_data["total_results"] == 2
                assert len(result_data["memories"]) == 2
                assert result_data["search_scope"] == "owner"
                
                # Check first result
                first_memory = result_data["memories"][0]
                assert first_memory["id"] == "test-memory-1"
                assert first_memory["category"] == "knowledge"
                assert first_memory["importance"] == 8
                assert first_memory["similarity_score"] == 0.85
                
                print("✅ Vector memory search with mocks test passed")
            finally:
                await skill.cleanup()


@pytest.mark.asyncio
async def test_list_vector_memories():
    """Test listing vector memories"""
    from webagents.agents.skills.core.vector_memory import VectorMemorySkill
    
    skill = VectorMemorySkill({'agent_name': 'test_agent'})
    
    # Mock list results
    mock_list_results = [
        {
            "id": "memory-1",
            "content": "Memory about Python",
            "category": "knowledge",
            "importance": 9,
            "tags": ["python", "programming"],
            "scope": "owner",
            "created_at": "2024-01-01T00:00:00Z",
            "access_count": 5
        },
        {
            "id": "memory-2",
            "content": "Memory about JavaScript",
            "category": "knowledge", 
            "importance": 7,
            "tags": ["javascript", "web"],
            "scope": "owner",
            "created_at": "2024-01-02T00:00:00Z",
            "access_count": 3
        }
    ]
    
    with patch.object(skill, '_check_requirements', return_value=True), \
         patch.object(skill, '_list_memories', return_value=mock_list_results):
        
        try:
            result = await skill.list_vector_memories(
                category="knowledge",
                min_importance=5,
                limit=10
            )
            
            result_data = json.loads(result)
            assert result_data["success"] == True
            assert result_data["total_memories"] == 2
            assert len(result_data["memories"]) == 2
            
            # Check filters
            filters = result_data["filters"]
            assert filters["category"] == "knowledge"
            assert filters["min_importance"] == 5
            assert filters["scope"] == "owner"  # Default scope
            assert filters["limit"] == 10
            
            print("✅ List vector memories test passed")
        finally:
            await skill.cleanup()


@pytest.mark.asyncio
async def test_delete_vector_memory():
    """Test deleting vector memory"""
    from webagents.agents.skills.core.vector_memory import VectorMemorySkill
    
    skill = VectorMemorySkill({'agent_name': 'test_agent'})
    
    with patch.object(skill, '_check_requirements', return_value=True), \
         patch.object(skill, '_delete_memory', return_value=True):
        
        try:
            result = await skill.delete_vector_memory("test-memory-id")
            
            result_data = json.loads(result)
            assert result_data["success"] == True
            assert result_data["memory_id"] == "test-memory-id"
            assert "deleted successfully" in result_data["message"]
            
            print("✅ Delete vector memory test passed")
        finally:
            await skill.cleanup()


@pytest.mark.asyncio
async def test_get_vector_memory_stats():
    """Test getting vector memory statistics"""
    from webagents.agents.skills.core.vector_memory import VectorMemorySkill
    
    skill = VectorMemorySkill({
        'agent_name': 'test_agent',
        'milvus_collection': 'test_collection',
        'embedding_model': 'text-embedding-3-small'
    })
    
    # Mock stats
    mock_stats = {
        "total_memories": 10,
        "categories": {"knowledge": 5, "conversation": 3, "context": 2},
        "scope_breakdown": {"owner": 8, "shared": 2},
        "collection_total": 25
    }
    
    with patch.object(skill, '_check_requirements', return_value=True), \
         patch.object(skill, '_get_memory_stats', return_value=mock_stats):
        
        try:
            result = await skill.get_vector_memory_stats()
            
            result_data = json.loads(result)
            assert result_data["success"] == True
            assert result_data["agent_name"] == "test_agent"
            assert result_data["milvus_collection"] == "test_collection"
            assert result_data["embedding_model"] == "text-embedding-3-small"
            assert result_data["default_scope"] == "owner"
            assert result_data["total_memories"] == 10
            assert result_data["categories"]["knowledge"] == 5
            assert result_data["scope_breakdown"]["owner"] == 8
            
            print("✅ Get vector memory stats test passed")
        finally:
            await skill.cleanup()


@pytest.mark.asyncio
async def test_skill_info():
    """Test getting skill information"""
    from webagents.agents.skills.core.vector_memory import VectorMemorySkill
    
    skill = VectorMemorySkill({'agent_name': 'test_agent'})
    
    try:
        info = skill.get_skill_info()
        
        # Check required fields
        assert "name" in info
        assert "description" in info
        assert "version" in info
        assert "capabilities" in info
        assert "tools" in info
        assert "config" in info
        
        # Check specific values
        assert info["name"] == "VectorMemorySkill"
        assert info["version"] == "1.0.0"
        assert len(info["tools"]) == 5
        assert "Milvus" in info["description"]
        assert info["config"]["agent_name"] == "test_agent"
        assert info["config"]["default_scope"] == "owner"
        
        print("✅ Skill info test passed")
    finally:
        await skill.cleanup()


@pytest.mark.asyncio
async def test_vector_memory_guidance_prompt():
    """Test vector memory guidance prompt"""
    from webagents.agents.skills.core.vector_memory import VectorMemorySkill
    
    skill = VectorMemorySkill({'agent_name': 'test_agent'})
    
    try:
        # Test the prompt method
        guidance = skill.vector_memory_guidance({})
        
        assert isinstance(guidance, str)
        assert len(guidance) > 100  # Should be substantial guidance
        assert "VECTOR MEMORY SYSTEM GUIDANCE" in guidance
        assert "WHEN TO STORE MEMORIES" in guidance
        assert "WHEN TO SEARCH MEMORIES" in guidance
        assert "MEMORY CATEGORIES" in guidance
        assert "owner" in guidance  # Should mention default scope
        assert "search_vector_memories" in guidance
        assert "store_vector_memory" in guidance
        
        print("✅ Vector memory guidance prompt test passed")
    finally:
        await skill.cleanup()


if __name__ == "__main__":
    # Run tests directly
    import asyncio
    
    async def run_vector_memory_tests():
        print("🧪 Running VectorMemorySkill Tests...")
        
        tests = [
            test_vector_memory_skill_import,
            test_vector_memory_skill_initialization,
            test_vector_memory_item,
            test_store_vector_memory_without_dependencies,
            test_search_vector_memories_without_dependencies,
            test_vector_memory_with_mocked_dependencies,
            test_vector_memory_search_with_mocks,
            test_list_vector_memories,
            test_delete_vector_memory,
            test_get_vector_memory_stats,
            test_skill_info,
            test_vector_memory_guidance_prompt
        ]
        
        for test in tests:
            try:
                await test()
            except Exception as e:
                print(f"❌ {test.__name__} failed: {e}")
                continue
        
        print("🎯 VectorMemorySkill Tests Complete!")
    
    asyncio.run(run_vector_memory_tests()) 