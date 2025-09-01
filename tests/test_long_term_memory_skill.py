"""
Test suite for LongTermMemorySkill - Persistent memory management with webagents storage
"""

import pytest
import json
import sys
import os
import tempfile
from unittest.mock import patch, AsyncMock

# Add webagents to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.mark.asyncio
async def test_long_term_memory_skill_import():
    """Test that LongTermMemorySkill can be imported successfully"""
    from webagents.agents.skills.core.long_term_memory import LongTermMemorySkill
    assert LongTermMemorySkill is not None
    print("✅ LongTermMemorySkill import successful")


@pytest.mark.asyncio
async def test_long_term_memory_skill_initialization():
    """Test LongTermMemorySkill initialization"""
    from webagents.agents.skills.core.long_term_memory import LongTermMemorySkill
    
    # Create temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_file = f.name
    
    try:
        # Default initialization
        skill = LongTermMemorySkill({'fallback_file': temp_file})
        assert skill is not None
        assert skill.max_memories == 100
        assert skill.auto_extract == True
        assert skill.use_webagents_storage == True
        assert len(skill.memories) == 0
        
        # Custom configuration
        config = {
            'fallback_file': temp_file,
            'max_memories': 50,
            'auto_extract': False,
            'use_webagents_storage': False,
            'agent_name': 'test_agent'
        }
        skill_custom = LongTermMemorySkill(config)
        assert skill_custom.max_memories == 50
        assert skill_custom.auto_extract == False
        assert skill_custom.use_webagents_storage == False
        assert skill_custom.agent_name == 'test_agent'
        
        print("✅ LongTermMemorySkill initialization test passed")
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)


@pytest.mark.asyncio
async def test_save_memory():
    """Test saving a memory manually"""
    from webagents.agents.skills.core.long_term_memory import LongTermMemorySkill
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_file = f.name
    
    try:
        skill = LongTermMemorySkill({
            'fallback_file': temp_file,
            'use_webagents_storage': False  # Use local storage for testing
        })
        
        # Save a memory
        result = await skill.save_memory(
            content="User prefers pytest for testing",
            category="preference",
            importance=8,
            tags=["testing", "pytest", "preference"]
        )
        
        result_data = json.loads(result)
        
        # Check result structure
        assert "memory_id" in result_data
        assert "content" in result_data
        assert "category" in result_data
        assert "importance" in result_data
        assert "status" in result_data
        
        # Check values
        assert result_data["content"] == "User prefers pytest for testing"
        assert result_data["category"] == "preference"
        assert result_data["importance"] == 8
        assert result_data["status"] == "saved"
        
        # Verify memory was stored
        memory_id = result_data["memory_id"]
        assert memory_id in skill.memories
        assert skill.memories[memory_id].content == "User prefers pytest for testing"
        
        print("✅ Save memory test passed")
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


@pytest.mark.asyncio
async def test_list_memories():
    """Test listing stored memories"""
    from webagents.agents.skills.core.long_term_memory import LongTermMemorySkill
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_file = f.name
    
    try:
        skill = LongTermMemorySkill({
            'fallback_file': temp_file,
            'use_webagents_storage': False
        })
        
        # Save multiple memories
        await skill.save_memory("User prefers TypeScript", "preference", 7, ["typescript", "preference"])
        await skill.save_memory("Project uses React framework", "project", 6, ["react", "framework"])
        await skill.save_memory("Must support Python 3.8+", "fact", 9, ["python", "requirement"])
        
        # List all memories
        result = await skill.list_memories()
        result_data = json.loads(result)
        
        # Check result structure
        assert "total_memories" in result_data
        assert "filtered_count" in result_data
        assert "memories" in result_data
        assert "categories" in result_data
        assert "storage_location" in result_data
        
        # Check values
        assert result_data["total_memories"] == 3
        assert result_data["filtered_count"] == 3
        assert len(result_data["memories"]) == 3
        assert len(result_data["categories"]) == 3
        assert result_data["storage_location"] == "local_file"
        
        # Test filtering by category
        result_filtered = await skill.list_memories(category="preference")
        filtered_data = json.loads(result_filtered)
        assert filtered_data["filtered_count"] == 1
        assert filtered_data["memories"][0]["category"] == "preference"
        
        # Test filtering by importance
        result_important = await skill.list_memories(min_importance=8)
        important_data = json.loads(result_important)
        assert important_data["filtered_count"] == 1
        assert important_data["memories"][0]["importance"] >= 8
        
        print("✅ List memories test passed")
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


@pytest.mark.asyncio
async def test_search_memories():
    """Test searching memories by content"""
    from webagents.agents.skills.core.long_term_memory import LongTermMemorySkill
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_file = f.name
    
    try:
        skill = LongTermMemorySkill({
            'fallback_file': temp_file,
            'use_webagents_storage': False
        })
        
        # Save test memories
        await skill.save_memory("User prefers pytest for unit testing", "preference", 8, ["pytest", "testing"])
        await skill.save_memory("Project uses React and TypeScript", "project", 7, ["react", "typescript"])
        await skill.save_memory("Testing should be comprehensive", "fact", 6, ["testing", "quality"])
        
        # Search for testing-related memories
        result = await skill.search_memories("testing")
        result_data = json.loads(result)
        
        # Check result structure
        assert "query" in result_data
        assert "total_matches" in result_data
        assert "memories" in result_data
        
        # Check values
        assert result_data["query"] == "testing"
        assert result_data["total_matches"] == 2  # Two memories contain "testing"
        assert len(result_data["memories"]) == 2
        
        # Verify search scoring
        memories = result_data["memories"]
        assert all("score" in memory for memory in memories)
        
        # Search for specific technology
        react_result = await skill.search_memories("React")
        react_data = json.loads(react_result)
        assert react_data["total_matches"] == 1
        
        print("✅ Search memories test passed")
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


@pytest.mark.asyncio
async def test_extract_key_memories():
    """Test automatic memory extraction from conversation"""
    from webagents.agents.skills.core.long_term_memory import LongTermMemorySkill
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_file = f.name
    
    try:
        skill = LongTermMemorySkill({
            'fallback_file': temp_file,
            'use_webagents_storage': False
        })
        
        # Test conversation context with extractable information
        context = """
        I prefer using pytest for all my testing. My current project is using React and TypeScript.
        The requirement is that we must support Python 3.8 and above. I usually organize my tests
        in a dedicated tests/ directory.
        """
        
        # Extract memories
        result = await skill.extract_key_memories(context)
        result_data = json.loads(result)
        
        # Check result structure
        assert "extracted_count" in result_data
        assert "memories" in result_data
        assert "status" in result_data
        
        # Should extract at least some memories
        assert result_data["extracted_count"] > 0
        assert len(result_data["memories"]) > 0
        assert result_data["status"] == "success"
        
        # Check that memories were actually stored
        assert len(skill.memories) == result_data["extracted_count"]
        
        # Verify memory categories
        categories = [m["category"] for m in result_data["memories"]]
        assert any(cat in ["preference", "project", "fact"] for cat in categories)
        
        print("✅ Extract key memories test passed")
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


@pytest.mark.asyncio
async def test_webagents_storage_integration():
    """Test integration with webagents storage skill"""
    from webagents.agents.skills.core.long_term_memory import LongTermMemorySkill
    
    # Mock storage skill
    mock_storage = AsyncMock()
    mock_storage.store_json_data = AsyncMock(return_value='{"success": true}')
    mock_storage.retrieve_json_data = AsyncMock(return_value='{"success": false, "error": "File not found"}')
    
    try:
        skill = LongTermMemorySkill({
            'agent_name': 'test_agent',
            'use_webagents_storage': True
        })
        
        # Simulate storage skill being available
        skill.storage_skill = mock_storage
        
        # Save a memory
        result = await skill.save_memory("Test webagents storage", "fact", 7)
        result_data = json.loads(result)
        
        assert result_data["status"] == "saved"
        
        # Verify storage skill was called
        mock_storage.store_json_data.assert_called_once()
        call_args = mock_storage.store_json_data.call_args
        assert "test_agent_memory.json" in call_args[0][0]
        
        print("✅ WebAgents storage integration test passed")
    finally:
        pass


@pytest.mark.asyncio
async def test_memory_stats():
    """Test getting memory statistics"""
    from webagents.agents.skills.core.long_term_memory import LongTermMemorySkill
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_file = f.name
    
    try:
        skill = LongTermMemorySkill({
            'fallback_file': temp_file,
            'use_webagents_storage': False,
            'agent_name': 'test_agent'
        })
        
        # Initially no memories
        result = await skill.get_memory_stats()
        result_data = json.loads(result)
        assert result_data["total_memories"] == 0
        assert result_data["agent_name"] == "test_agent"
        assert result_data["storage_location"] == "local_file"
        
        # Add some memories
        await skill.save_memory("Preference 1", "preference", 8)
        await skill.save_memory("Preference 2", "preference", 7)
        await skill.save_memory("Project info", "project", 6)
        await skill.save_memory("Important fact", "fact", 9)
        
        # Get stats again
        result = await skill.get_memory_stats()
        result_data = json.loads(result)
        
        # Check structure
        assert "total_memories" in result_data
        assert "categories" in result_data
        assert "importance_distribution" in result_data
        assert "most_accessed" in result_data
        assert "storage_location" in result_data
        assert "agent_name" in result_data
        assert "max_memories" in result_data
        
        # Check values
        assert result_data["total_memories"] == 4
        assert result_data["categories"]["preference"] == 2
        assert result_data["categories"]["project"] == 1
        assert result_data["categories"]["fact"] == 1
        assert result_data["agent_name"] == "test_agent"
        
        print("✅ Memory stats test passed")
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


@pytest.mark.asyncio
async def test_skill_info():
    """Test getting skill information"""
    from webagents.agents.skills.core.long_term_memory import LongTermMemorySkill
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_file = f.name
    
    try:
        skill = LongTermMemorySkill({'fallback_file': temp_file})
        info = skill.get_skill_info()
        
        # Check required fields
        assert "name" in info
        assert "description" in info
        assert "version" in info
        assert "capabilities" in info
        assert "tools" in info
        assert "total_memories" in info
        assert "config" in info
        
        # Check specific values
        assert info["name"] == "LongTermMemorySkill"
        assert info["version"] == "2.0.0"
        assert len(info["tools"]) == 6  # All the tool methods
        assert info["total_memories"] == 0
        assert "WebAgents portal storage integration" in info["description"]
        
        print("✅ Skill info test passed")
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


if __name__ == "__main__":
    # Run tests directly
    import asyncio
    
    async def run_long_term_memory_tests():
        print("🧪 Running LongTermMemorySkill Tests...")
        
        tests = [
            test_long_term_memory_skill_import,
            test_long_term_memory_skill_initialization,
            test_save_memory,
            test_list_memories,
            test_search_memories,
            test_extract_key_memories,
            test_webagents_storage_integration,
            test_memory_stats,
            test_skill_info
        ]
        
        for test in tests:
            try:
                await test()
            except Exception as e:
                print(f"❌ {test.__name__} failed: {e}")
                continue
        
        print("🎯 LongTermMemorySkill Tests Complete!")
    
    asyncio.run(run_long_term_memory_tests()) 