"""
Test suite for WebAgentsStorageSkill - Portal content storage integration via RobutlerClient
"""

import pytest
import json
import sys
import os
import tempfile
from unittest.mock import AsyncMock, patch

# Add webagents to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.asyncio
async def test_webagents_storage_skill_import():
    """Test that WebAgentsStorageSkill can be imported successfully"""
    from webagents.agents.skills.robutler.storage import WebAgentsStorageSkill
    assert WebAgentsStorageSkill is not None
    print("✅ WebAgentsStorageSkill import successful")


@pytest.mark.asyncio
async def test_webagents_storage_skill_initialization():
    """Test WebAgentsStorageSkill initialization"""
    from webagents.agents.skills.robutler.storage import WebAgentsStorageSkill
    
    # Default initialization
    skill = WebAgentsStorageSkill({'agent_name': 'test_agent'})
    assert skill is not None
    assert skill.agent_name == 'test_agent'
    assert skill.client is not None
    assert skill.portal_url == 'http://localhost:3000'
    
    # Custom configuration
    config = {
        'agent_name': 'custom_agent',
        'portal_url': 'https://custom.portal.com',
        'api_key': 'custom_key'
    }
    skill_custom = WebAgentsStorageSkill(config)
    assert skill_custom.agent_name == 'custom_agent'
    assert skill_custom.portal_url == 'https://custom.portal.com'
    assert skill_custom.api_key == 'custom_key'
    
    # Cleanup
    await skill.cleanup()
    await skill_custom.cleanup()
    
    print("✅ WebAgentsStorageSkill initialization test passed")


@pytest.mark.asyncio
async def test_store_json_data():
    """Test storing JSON data via RobutlerClient"""
    from webagents.agents.skills.robutler.storage import WebAgentsStorageSkill
    from robutler.api.types import ApiResponse
    
    skill = WebAgentsStorageSkill({'agent_name': 'test_agent'})
    
    # Mock the client upload_content method
    mock_response = ApiResponse(
        success=True,
        data={
            'file': {
                'id': 'test_file_id',
                'fileName': 'test_data.json',
                'url': 'https://example.com/test_data.json',
                'size': 100
            }
        },
        status_code=200
    )
    
    skill.client.upload_content = AsyncMock(return_value=mock_response)
    
    try:
        # Test storing data
        test_data = {"key": "value", "number": 42}
        result = await skill.store_json_data(
            filename="test_data",
            data=test_data,
            description="Test data storage"
        )
        
        result_data = json.loads(result)
        
        # Check result structure
        assert result_data["success"] == True
        assert "file_id" in result_data
        assert "filename" in result_data
        assert "url" in result_data
        assert "size" in result_data
        
        # Verify client was called correctly
        skill.client.upload_content.assert_called_once()
        call_args = skill.client.upload_content.call_args
        assert call_args[1]['filename'] == 'test_data.json'
        assert call_args[1]['content_type'] == 'application/json'
        assert call_args[1]['visibility'] == 'private'
        
        print("✅ Store JSON data test passed")
    finally:
        await skill.cleanup()


@pytest.mark.asyncio
async def test_retrieve_json_data():
    """Test retrieving JSON data via RobutlerClient"""
    from webagents.agents.skills.robutler.storage import WebAgentsStorageSkill
    from robutler.api.types import ApiResponse
    
    skill = WebAgentsStorageSkill({'agent_name': 'test_agent'})
    
    # Mock the client get_content method
    test_data = {"key": "value", "number": 42}
    mock_response = ApiResponse(
        success=True,
        data={
            'filename': 'test_data.json',
            'content': test_data,
            'metadata': {
                'size': 100,
                'uploadedAt': '2024-01-01T00:00:00Z',
                'description': 'Test data'
            }
        },
        status_code=200
    )
    
    skill.client.get_content = AsyncMock(return_value=mock_response)
    
    try:
        # Test retrieving data
        result = await skill.retrieve_json_data("test_data.json")
        result_data = json.loads(result)
        
        # Check result structure
        assert result_data["success"] == True
        assert "filename" in result_data
        assert "data" in result_data
        assert "metadata" in result_data
        assert result_data["data"] == test_data
        
        # Verify client was called correctly
        skill.client.get_content.assert_called_once_with('test_data.json')
        
        print("✅ Retrieve JSON data test passed")
    finally:
        await skill.cleanup()


@pytest.mark.asyncio
async def test_list_agent_files():
    """Test listing agent files via RobutlerClient"""
    from webagents.agents.skills.robutler.storage import WebAgentsStorageSkill
    from robutler.api.types import ApiResponse
    
    skill = WebAgentsStorageSkill({'agent_name': 'test_agent'})
    
    # Mock the client list_content method
    mock_response = ApiResponse(
        success=True,
        data={
            'files': [
                {
                    'fileName': 'test_agent_memory.json',
                    'size': 500,
                    'uploadedAt': '2024-01-01T00:00:00Z',
                    'description': 'Agent memory data',
                    'tags': ['agent_data', 'test_agent']
                },
                {
                    'fileName': 'other_file.json',
                    'size': 200,
                    'uploadedAt': '2024-01-01T00:00:00Z',
                    'description': 'Other data',
                    'tags': ['other']
                }
            ]
        },
        status_code=200
    )
    
    skill.client.list_content = AsyncMock(return_value=mock_response)
    
    try:
        # Test listing files
        result = await skill.list_agent_files()
        result_data = json.loads(result)
        
        # Check result structure
        assert result_data["success"] == True
        assert "agent_name" in result_data
        assert "total_files" in result_data
        assert "files" in result_data
        assert result_data["agent_name"] == "test_agent"
        assert result_data["total_files"] == 1  # Only one file has agent tags
        
        # Verify client was called correctly
        skill.client.list_content.assert_called_once()
        
        print("✅ List agent files test passed")
    finally:
        await skill.cleanup()


@pytest.mark.asyncio
async def test_delete_file():
    """Test deleting files via RobutlerClient"""
    from webagents.agents.skills.robutler.storage import WebAgentsStorageSkill
    from robutler.api.types import ApiResponse
    
    skill = WebAgentsStorageSkill({'agent_name': 'test_agent'})
    
    # Mock the client delete_content method
    mock_response = ApiResponse(
        success=True,
        data={},
        status_code=200
    )
    
    skill.client.delete_content = AsyncMock(return_value=mock_response)
    
    try:
        # Test deleting file
        result = await skill.delete_file("test_data.json")
        result_data = json.loads(result)
        
        # Check result structure
        assert result_data["success"] == True
        assert "message" in result_data
        assert "test_data.json" in result_data["message"]
        
        # Verify client was called correctly
        skill.client.delete_content.assert_called_once_with('test_data.json')
        
        print("✅ Delete file test passed")
    finally:
        await skill.cleanup()


@pytest.mark.asyncio
async def test_get_storage_stats():
    """Test getting storage statistics via RobutlerClient"""
    from webagents.agents.skills.robutler.storage import WebAgentsStorageSkill
    from robutler.api.types import ApiResponse
    
    skill = WebAgentsStorageSkill({'agent_name': 'test_agent'})
    
    # Mock the client list_content method
    mock_response = ApiResponse(
        success=True,
        data={
            'files': [
                {
                    'fileName': 'test_agent_memory.json',
                    'size': 500,
                    'tags': ['agent_data', 'test_agent']
                },
                {
                    'fileName': 'test_agent_config.json',
                    'size': 300,
                    'tags': ['agent_data', 'test_agent']
                },
                {
                    'fileName': 'other_file.json',
                    'size': 200,
                    'tags': ['other']
                }
            ]
        },
        status_code=200
    )
    
    skill.client.list_content = AsyncMock(return_value=mock_response)
    
    try:
        # Test getting stats
        result = await skill.get_storage_stats()
        result_data = json.loads(result)
        
        # Check result structure
        assert "agent_name" in result_data
        assert "total_files" in result_data
        assert "total_size_bytes" in result_data
        assert "total_size_mb" in result_data
        assert "portal_url" in result_data
        assert "storage_location" in result_data
        
        # Check values
        assert result_data["agent_name"] == "test_agent"
        assert result_data["total_files"] == 2  # Two files have agent tags
        assert result_data["total_size_bytes"] == 800  # 500 + 300
        
        print("✅ Get storage stats test passed")
    finally:
        await skill.cleanup()


@pytest.mark.asyncio
async def test_skill_info():
    """Test getting skill information"""
    from webagents.agents.skills.robutler.storage import WebAgentsStorageSkill
    
    skill = WebAgentsStorageSkill({'agent_name': 'test_agent'})
    
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
        assert info["name"] == "WebAgentsStorageSkill"
        assert info["version"] == "2.0.0"
        assert len(info["tools"]) == 6
        assert "RobutlerClient" in info["description"]
        assert info["config"]["client_type"] == "RobutlerClient"
        
        print("✅ Skill info test passed")
    finally:
        await skill.cleanup()


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in storage operations"""
    from webagents.agents.skills.robutler.storage import WebAgentsStorageSkill
    from robutler.api.types import ApiResponse
    
    skill = WebAgentsStorageSkill({'agent_name': 'test_agent'})
    
    # Mock client method to return error
    error_response = ApiResponse(
        success=False,
        error='API Error',
        message='Something went wrong',
        status_code=500
    )
    
    skill.client.get_content = AsyncMock(return_value=error_response)
    
    try:
        # Test error handling
        result = await skill.retrieve_json_data("nonexistent.json")
        result_data = json.loads(result)
        
        # Check error response
        assert result_data["success"] == False
        assert "error" in result_data
        
        print("✅ Error handling test passed")
    finally:
        await skill.cleanup()


if __name__ == "__main__":
    # Run tests directly
    import asyncio
    
    async def run_storage_tests():
        print("🧪 Running WebAgentsStorageSkill Tests...")
        
        tests = [
            test_webagents_storage_skill_import,
            test_webagents_storage_skill_initialization,
            test_store_json_data,
            test_retrieve_json_data,
            test_list_agent_files,
            test_delete_file,
            test_get_storage_stats,
            test_skill_info,
            test_error_handling
        ]
        
        for test in tests:
            try:
                await test()
            except Exception as e:
                print(f"❌ {test.__name__} failed: {e}")
                continue
        
        print("🎯 WebAgentsStorageSkill Tests Complete!")
    
    asyncio.run(run_storage_tests()) 