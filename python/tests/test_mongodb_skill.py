"""
Test suite for minimalistic MongoDB skill
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from webagents.agents.skills.ecosystem.mongodb.skill import MongoDBSkill

# Check if MongoDB dependencies are available
try:
    import pymongo
    from bson import ObjectId
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    # Create dummy ObjectId for testing when pymongo is not available
    class ObjectId:
        def __init__(self, *args, **kwargs):
            self._id = "507f1f77bcf86cd799439011"
        
        def __str__(self):
            return self._id


@pytest.fixture
def mongodb_skill():
    """Create MongoDBSkill instance for testing"""
    with patch('webagents.agents.skills.ecosystem.mongodb.skill.MONGODB_AVAILABLE', True):
        skill = MongoDBSkill()
        
        # Create proper mock agent
        class MockAgent:
            def __init__(self):
                self.name = 'test-agent'
                self.skills = {
                    'kv': MockKVSkill(),
                    'auth': MockAuthSkill()
                }
                self._mongodb_configs = {}
        
        skill.agent = MockAgent()
        skill.logger = MagicMock()
        return skill


class MockKVSkill:
    """Mock KV skill"""
    def __init__(self):
        self.storage = {}
    
    async def kv_set(self, key, value, namespace=None):
        full_key = f"{namespace}:{key}" if namespace else key
        self.storage[full_key] = value
        return "✅ Saved"
    
    async def kv_get(self, key, namespace=None):
        full_key = f"{namespace}:{key}" if namespace else key
        return self.storage.get(full_key, "")


class MockAuthSkill:
    """Mock auth skill"""
    pass


@pytest.fixture
def mock_context():
    """Mock context with user authentication"""
    context = MagicMock()
    context.auth = MagicMock()
    context.auth.user_id = 'test_user_123'
    context.auth.authenticated = True
    return context


@pytest.fixture
def mock_atlas_config():
    """Mock MongoDB Atlas configuration"""
    return {
        'connection_string': 'mongodb+srv://user:password@cluster.mongodb.net/mydb'
    }


@pytest.fixture
def mock_local_config():
    """Mock local MongoDB configuration"""
    return {
        'host': 'localhost',
        'port': 27017,
        'database': 'testdb'
    }


@pytest.fixture
def mock_auth_config():
    """Mock MongoDB configuration with authentication"""
    return {
        'host': 'localhost',
        'port': 27017,
        'username': 'testuser',
        'password': 'testpass',
        'database': 'testdb'
    }


class TestMongoDBSkill:
    """Test cases for minimalistic MongoDB skill"""

    def test_skill_initialization(self, mongodb_skill):
        """Test skill initializes correctly"""
        assert 'test-agent' == mongodb_skill.agent.name
        assert mongodb_skill.get_dependencies() == ['auth', 'kv']

    def test_skill_initialization_without_dependencies(self):
        """Test skill fails gracefully without dependencies installed"""
        with patch('webagents.agents.skills.ecosystem.mongodb.skill.MONGODB_AVAILABLE', False):
            with pytest.raises(ImportError, match="MongoDB dependencies not installed"):
                MongoDBSkill()

    def test_prompt_mentions_tools(self, mongodb_skill):
        """Test prompt mentions all tools"""
        prompt = mongodb_skill.mongodb_prompt()
        assert 'mongodb_setup(' in prompt
        assert 'mongodb_query(' in prompt
        assert 'mongodb_aggregate(' in prompt
        assert 'mongodb_status(' in prompt
        assert 'Minimalistic MongoDB integration' in prompt

    def test_has_4_tools(self, mongodb_skill):
        """Test skill has exactly 4 tools"""
        mongodb_tools = [m for m in dir(mongodb_skill) if m.startswith('mongodb_') and not m.startswith('mongodb_prompt')]
        assert len(mongodb_tools) == 4
        assert 'mongodb_setup' in mongodb_tools
        assert 'mongodb_query' in mongodb_tools
        assert 'mongodb_aggregate' in mongodb_tools
        assert 'mongodb_status' in mongodb_tools

    @pytest.mark.asyncio
    async def test_save_load_config_kv(self, mongodb_skill, mock_atlas_config):
        """Test saving and loading configuration via KV skill"""
        user_id = 'test_user_123'
        
        # Test save
        success = await mongodb_skill._save_mongodb_config(user_id, mock_atlas_config)
        assert success == True
        
        # Test load
        loaded_config = await mongodb_skill._load_mongodb_config(user_id)
        assert loaded_config['connection_string'] == 'mongodb+srv://user:password@cluster.mongodb.net/mydb'
        assert 'created_at' in loaded_config

    @pytest.mark.asyncio
    async def test_save_load_config_memory_fallback(self, mongodb_skill, mock_atlas_config):
        """Test memory fallback when KV skill unavailable"""
        # Remove KV skill to test fallback
        mongodb_skill.agent.skills['kv'] = None
        
        user_id = 'test_user_123'
        
        # Test save (should use memory fallback)
        success = await mongodb_skill._save_mongodb_config(user_id, mock_atlas_config)
        assert success == True
        
        # Test load (should use memory fallback)
        loaded_config = await mongodb_skill._load_mongodb_config(user_id)
        assert loaded_config['connection_string'] == 'mongodb+srv://user:password@cluster.mongodb.net/mydb'

    @pytest.mark.asyncio
    async def test_mongodb_setup_atlas_success(self, mongodb_skill, mock_context, mock_atlas_config):
        """Test successful MongoDB Atlas setup"""
        with patch('webagents.agents.skills.ecosystem.mongodb.skill.get_context', return_value=mock_context):
            with patch.object(mongodb_skill, '_create_mongodb_client') as mock_create_client:
                mock_client = MagicMock()
                mock_client.admin.command.return_value = {'ok': 1}
                mock_create_client.return_value = mock_client
                
                result = await mongodb_skill.mongodb_setup(mock_atlas_config)
                
                assert '✅ Atlas configuration saved successfully!' in result
                assert 'MongoDB' in result

    @pytest.mark.asyncio
    async def test_mongodb_setup_local_success(self, mongodb_skill, mock_context, mock_local_config):
        """Test successful local MongoDB setup"""
        with patch('webagents.agents.skills.ecosystem.mongodb.skill.get_context', return_value=mock_context):
            with patch.object(mongodb_skill, '_create_mongodb_client') as mock_create_client:
                mock_client = MagicMock()
                mock_client.admin.command.return_value = {'ok': 1}
                mock_create_client.return_value = mock_client
                
                result = await mongodb_skill.mongodb_setup(mock_local_config)
                
                assert '✅ MongoDB configuration saved successfully!' in result

    @pytest.mark.asyncio
    async def test_mongodb_setup_no_auth(self, mongodb_skill, mock_atlas_config):
        """Test setup without authentication"""
        with patch('webagents.agents.skills.ecosystem.mongodb.skill.get_context', return_value=None):
            result = await mongodb_skill.mongodb_setup(mock_atlas_config)
            
            assert '❌ Authentication required' in result

    @pytest.mark.asyncio
    async def test_mongodb_setup_empty_config(self, mongodb_skill, mock_context):
        """Test setup with empty configuration"""
        with patch('webagents.agents.skills.ecosystem.mongodb.skill.get_context', return_value=mock_context):
            result = await mongodb_skill.mongodb_setup({})
            
            assert '❌ Configuration is required' in result

    @pytest.mark.asyncio
    async def test_mongodb_setup_invalid_config(self, mongodb_skill, mock_context):
        """Test setup with invalid configuration"""
        invalid_config = {'invalid': 'config'}  # Missing required fields
        
        with patch('webagents.agents.skills.ecosystem.mongodb.skill.get_context', return_value=mock_context):
            result = await mongodb_skill.mongodb_setup(invalid_config)
            
            assert '❌ MongoDB configuration requires' in result

    @pytest.mark.asyncio
    async def test_mongodb_setup_connection_failure(self, mongodb_skill, mock_context, mock_atlas_config):
        """Test setup with connection failure"""
        with patch('webagents.agents.skills.ecosystem.mongodb.skill.get_context', return_value=mock_context):
            with patch.object(mongodb_skill, '_create_mongodb_client') as mock_create_client:
                mock_client = MagicMock()
                mock_client.admin.command.side_effect = Exception("Connection failed")
                mock_create_client.return_value = mock_client
                
                result = await mongodb_skill.mongodb_setup(mock_atlas_config)
                
                assert '❌ MongoDB connection error' in result

    @pytest.mark.asyncio
    async def test_mongodb_query_no_config(self, mongodb_skill, mock_context):
        """Test query without configuration"""
        with patch('webagents.agents.skills.ecosystem.mongodb.skill.get_context', return_value=mock_context):
            result = await mongodb_skill.mongodb_query('testdb', 'users', 'find')
            
            assert '❌ MongoDB not configured' in result
            assert 'mongodb_setup()' in result

    @pytest.mark.asyncio
    async def test_mongodb_query_invalid_operation(self, mongodb_skill, mock_context):
        """Test query with invalid operation"""
        with patch('webagents.agents.skills.ecosystem.mongodb.skill.get_context', return_value=mock_context):
            result = await mongodb_skill.mongodb_query('testdb', 'users', 'invalid_op')
            
            assert '❌ Invalid operation' in result
            assert 'find, find_one, insert_one' in result

    @pytest.mark.asyncio
    async def test_mongodb_query_find_success(self, mongodb_skill, mock_context, mock_atlas_config):
        """Test successful find operation"""
        with patch('webagents.agents.skills.ecosystem.mongodb.skill.get_context', return_value=mock_context):
            await mongodb_skill._save_mongodb_config('test_user_123', mock_atlas_config)
            
            with patch.object(mongodb_skill, '_create_mongodb_client') as mock_create_client:
                mock_client = MagicMock()
                mock_db = MagicMock()
                mock_collection = MagicMock()
                
                # Mock find result
                mock_collection.find.return_value = [
                    {'_id': ObjectId(), 'name': 'John Doe', 'email': 'john@example.com'},
                    {'_id': ObjectId(), 'name': 'Jane Smith', 'email': 'jane@example.com'}
                ]
                
                mock_db.__getitem__.return_value = mock_collection
                mock_client.__getitem__.return_value = mock_db
                mock_create_client.return_value = mock_client
                
                result = await mongodb_skill.mongodb_query('testdb', 'users', 'find', {'active': True})
                
                assert '✅ Find operation successful!' in result
                assert 'John Doe' in result
                assert 'Jane Smith' in result

    @pytest.mark.asyncio
    async def test_mongodb_query_insert_one_success(self, mongodb_skill, mock_context, mock_atlas_config):
        """Test successful insert_one operation"""
        with patch('webagents.agents.skills.ecosystem.mongodb.skill.get_context', return_value=mock_context):
            await mongodb_skill._save_mongodb_config('test_user_123', mock_atlas_config)
            
            with patch.object(mongodb_skill, '_create_mongodb_client') as mock_create_client:
                mock_client = MagicMock()
                mock_db = MagicMock()
                mock_collection = MagicMock()
                
                # Mock insert result
                mock_result = MagicMock()
                mock_result.inserted_id = ObjectId()
                mock_collection.insert_one.return_value = mock_result
                
                mock_db.__getitem__.return_value = mock_collection
                mock_client.__getitem__.return_value = mock_db
                mock_create_client.return_value = mock_client
                
                result = await mongodb_skill.mongodb_query(
                    'testdb', 'users', 'insert_one', 
                    data={'name': 'New User', 'email': 'new@example.com'}
                )
                
                assert '✅ Insert one operation successful!' in result
                assert 'Inserted document ID:' in result

    @pytest.mark.asyncio
    async def test_mongodb_query_insert_one_no_data(self, mongodb_skill, mock_context, mock_atlas_config):
        """Test insert_one operation without data"""
        with patch('webagents.agents.skills.ecosystem.mongodb.skill.get_context', return_value=mock_context):
            await mongodb_skill._save_mongodb_config('test_user_123', mock_atlas_config)
            
            with patch.object(mongodb_skill, '_create_mongodb_client') as mock_create_client:
                mock_client = MagicMock()
                mock_create_client.return_value = mock_client
                
                result = await mongodb_skill.mongodb_query('testdb', 'users', 'insert_one')
                
                assert '❌ Data is required for insert_one operation' in result

    @pytest.mark.asyncio
    async def test_mongodb_query_update_one_success(self, mongodb_skill, mock_context, mock_atlas_config):
        """Test successful update_one operation"""
        with patch('webagents.agents.skills.ecosystem.mongodb.skill.get_context', return_value=mock_context):
            await mongodb_skill._save_mongodb_config('test_user_123', mock_atlas_config)
            
            with patch.object(mongodb_skill, '_create_mongodb_client') as mock_create_client:
                mock_client = MagicMock()
                mock_db = MagicMock()
                mock_collection = MagicMock()
                
                # Mock update result
                mock_result = MagicMock()
                mock_result.matched_count = 1
                mock_result.modified_count = 1
                mock_collection.update_one.return_value = mock_result
                
                mock_db.__getitem__.return_value = mock_collection
                mock_client.__getitem__.return_value = mock_db
                mock_create_client.return_value = mock_client
                
                result = await mongodb_skill.mongodb_query(
                    'testdb', 'users', 'update_one', 
                    query={'_id': 'user123'}, 
                    data={'status': 'active'}
                )
                
                assert '✅ Update one operation successful!' in result
                assert 'Matched: 1, Modified: 1' in result

    @pytest.mark.asyncio
    async def test_mongodb_query_delete_one_no_query(self, mongodb_skill, mock_context, mock_atlas_config):
        """Test delete_one operation without query"""
        with patch('webagents.agents.skills.ecosystem.mongodb.skill.get_context', return_value=mock_context):
            await mongodb_skill._save_mongodb_config('test_user_123', mock_atlas_config)
            
            with patch.object(mongodb_skill, '_create_mongodb_client') as mock_create_client:
                mock_client = MagicMock()
                mock_create_client.return_value = mock_client
                
                result = await mongodb_skill.mongodb_query('testdb', 'users', 'delete_one')
                
                assert '❌ Query filter is required for delete_one operation' in result

    @pytest.mark.asyncio
    async def test_mongodb_aggregate_success(self, mongodb_skill, mock_context, mock_atlas_config):
        """Test successful aggregation pipeline"""
        with patch('webagents.agents.skills.ecosystem.mongodb.skill.get_context', return_value=mock_context):
            await mongodb_skill._save_mongodb_config('test_user_123', mock_atlas_config)
            
            with patch.object(mongodb_skill, '_create_mongodb_client') as mock_create_client:
                mock_client = MagicMock()
                mock_db = MagicMock()
                mock_collection = MagicMock()
                
                # Mock aggregation result
                mock_collection.aggregate.return_value = [
                    {'_id': 'active', 'count': 25},
                    {'_id': 'inactive', 'count': 5}
                ]
                
                mock_db.__getitem__.return_value = mock_collection
                mock_client.__getitem__.return_value = mock_db
                mock_create_client.return_value = mock_client
                
                pipeline = [
                    {'$group': {'_id': '$status', 'count': {'$sum': 1}}}
                ]
                
                result = await mongodb_skill.mongodb_aggregate('testdb', 'users', pipeline)
                
                assert '✅ Aggregation pipeline executed successfully!' in result
                assert 'active' in result
                assert 'inactive' in result

    @pytest.mark.asyncio
    async def test_mongodb_aggregate_invalid_pipeline(self, mongodb_skill, mock_context):
        """Test aggregation with invalid pipeline"""
        with patch('webagents.agents.skills.ecosystem.mongodb.skill.get_context', return_value=mock_context):
            result = await mongodb_skill.mongodb_aggregate('testdb', 'users', 'not_a_list')
            
            assert '❌ Pipeline must be a list of aggregation stages' in result

    @pytest.mark.asyncio
    async def test_mongodb_status_with_config(self, mongodb_skill, mock_context, mock_atlas_config):
        """Test status check when config exists"""
        with patch('webagents.agents.skills.ecosystem.mongodb.skill.get_context', return_value=mock_context):
            await mongodb_skill._save_mongodb_config('test_user_123', mock_atlas_config)
            
            with patch.object(mongodb_skill, '_create_mongodb_client') as mock_create_client:
                mock_client = MagicMock()
                mock_client.admin.command.return_value = {'ok': 1}
                mock_client.server_info.return_value = {'version': '5.0.0'}
                mock_create_client.return_value = mock_client
                
                result = await mongodb_skill.mongodb_status()
                
                assert '📋 MongoDB Status:' in result
                assert '✅ Configuration: Active' in result
                assert '🟢 Connection: Active' in result
                assert 'MongoDB Atlas' in result
                assert '5.0.0' in result

    @pytest.mark.asyncio
    async def test_mongodb_status_local_config(self, mongodb_skill, mock_context):
        """Test status check with local MongoDB config"""
        local_config = {'connection_string': 'mongodb://localhost:27017/testdb'}
        
        with patch('webagents.agents.skills.ecosystem.mongodb.skill.get_context', return_value=mock_context):
            await mongodb_skill._save_mongodb_config('test_user_123', local_config)
            
            with patch.object(mongodb_skill, '_create_mongodb_client') as mock_create_client:
                mock_client = MagicMock()
                mock_client.admin.command.return_value = {'ok': 1}
                mock_client.server_info.return_value = {'version': '4.4.0'}
                mock_create_client.return_value = mock_client
                
                result = await mongodb_skill.mongodb_status()
                
                assert 'Local MongoDB' in result

    @pytest.mark.asyncio
    async def test_mongodb_status_no_config(self, mongodb_skill, mock_context):
        """Test status check when no config exists"""
        with patch('webagents.agents.skills.ecosystem.mongodb.skill.get_context', return_value=mock_context):
            result = await mongodb_skill.mongodb_status()
            
            assert '❌ No configuration found' in result
            assert 'mongodb_setup()' in result

    @pytest.mark.asyncio
    async def test_mongodb_status_connection_error(self, mongodb_skill, mock_context, mock_atlas_config):
        """Test status check with connection error"""
        with patch('webagents.agents.skills.ecosystem.mongodb.skill.get_context', return_value=mock_context):
            await mongodb_skill._save_mongodb_config('test_user_123', mock_atlas_config)
            
            with patch.object(mongodb_skill, '_create_mongodb_client') as mock_create_client:
                mock_create_client.side_effect = Exception("Connection failed")
                
                result = await mongodb_skill.mongodb_status()
                
                assert '🟡 Connection: Warning' in result

    @pytest.mark.asyncio
    async def test_tools_require_authentication(self, mongodb_skill):
        """Test that all tools require authentication"""
        unauthenticated_context = MagicMock()
        unauthenticated_context.auth = None
        
        with patch('webagents.agents.skills.ecosystem.mongodb.skill.get_context', return_value=unauthenticated_context):
            # Test all tools return authentication required
            result = await mongodb_skill.mongodb_setup({'connection_string': 'mongodb://localhost'})
            assert '❌ Authentication required' in result
            
            result = await mongodb_skill.mongodb_query('db', 'collection', 'find')
            assert '❌ Authentication required' in result
            
            result = await mongodb_skill.mongodb_aggregate('db', 'collection', [])
            assert '❌ Authentication required' in result
            
            result = await mongodb_skill.mongodb_status()
            assert '❌ Authentication required' in result

    def test_create_mongodb_client_connection_string(self, mongodb_skill):
        """Test MongoDB client creation with connection string"""
        config = {'connection_string': 'mongodb://localhost:27017/testdb'}
        
        with patch('webagents.agents.skills.ecosystem.mongodb.skill.MongoClient') as mock_mongo_client:
            mock_client = MagicMock()
            mock_mongo_client.return_value = mock_client
            
            client = mongodb_skill._create_mongodb_client(config)
            
            assert client == mock_client
            mock_mongo_client.assert_called_once_with('mongodb://localhost:27017/testdb')

    def test_create_mongodb_client_individual_params(self, mongodb_skill):
        """Test MongoDB client creation with individual parameters"""
        config = {
            'host': 'localhost',
            'port': 27017,
            'username': 'testuser',
            'password': 'testpass',
            'database': 'testdb'
        }
        
        with patch('webagents.agents.skills.ecosystem.mongodb.skill.MongoClient') as mock_mongo_client:
            mock_client = MagicMock()
            mock_mongo_client.return_value = mock_client
            
            client = mongodb_skill._create_mongodb_client(config)
            
            assert client == mock_client
            expected_connection_string = 'mongodb://testuser:testpass@localhost:27017/testdb'
            mock_mongo_client.assert_called_once_with(expected_connection_string)

    def test_serialize_mongodb_result(self, mongodb_skill):
        """Test MongoDB result serialization"""
        # Test ObjectId serialization (using mock ObjectId that has __str__ method)
        obj_id = ObjectId()
        result = mongodb_skill._serialize_mongodb_result(obj_id)
        # Our mock ObjectId should be converted to string
        assert result == "507f1f77bcf86cd799439011"
        
        # Test nested document serialization
        doc = {
            '_id': ObjectId(),
            'name': 'Test User',
            'nested': {
                'id': ObjectId(),
                'value': 42
            }
        }
        result = mongodb_skill._serialize_mongodb_result(doc)
        assert result['_id'] == "507f1f77bcf86cd799439011"
        assert result['nested']['id'] == "507f1f77bcf86cd799439011"
        assert result['name'] == 'Test User'
        assert result['nested']['value'] == 42


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
