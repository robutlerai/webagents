"""
Test suite for minimalistic Supabase/PostgreSQL skill
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from webagents.agents.skills.ecosystem.database.skill import SupabaseSkill


# Check if Supabase dependencies are available
try:
    import supabase
    import psycopg2
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False


@pytest.fixture
def supabase_skill():
    """Create SupabaseSkill instance for testing"""
    with patch('webagents.agents.skills.ecosystem.database.skill.SUPABASE_AVAILABLE', True):
        skill = SupabaseSkill()
        
        # Create proper mock agent
        class MockAgent:
            def __init__(self):
                self.name = 'test-agent'
                self.skills = {
                    'kv': MockKVSkill(),
                    'auth': MockAuthSkill()
                }
                self._supabase_configs = {}
        
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
def mock_supabase_config():
    """Mock Supabase configuration"""
    return {
        'type': 'supabase',
        'supabase_url': 'https://test.supabase.co',
        'supabase_key': 'test_api_key'
    }


@pytest.fixture
def mock_postgres_config():
    """Mock PostgreSQL configuration"""
    return {
        'type': 'postgresql',
        'postgres_url': 'postgresql://user:password@localhost:5432/testdb'
    }


@pytest.fixture
def mock_postgres_config_params():
    """Mock PostgreSQL configuration with individual parameters"""
    return {
        'type': 'postgresql',
        'host': 'localhost',
        'port': 5432,
        'database': 'testdb',
        'user': 'testuser',
        'password': 'testpass'
    }


class TestSupabaseSkill:
    """Test cases for minimalistic Supabase skill"""

    def test_skill_initialization(self, supabase_skill):
        """Test skill initializes correctly"""
        assert 'test-agent' == supabase_skill.agent.name
        assert supabase_skill.get_dependencies() == ['auth', 'kv']

    def test_skill_initialization_without_dependencies(self):
        """Test skill fails gracefully without dependencies installed"""
        with patch('webagents.agents.skills.ecosystem.database.skill.SUPABASE_AVAILABLE', False):
            with pytest.raises(ImportError, match="Supabase dependencies not installed"):
                SupabaseSkill()

    def test_prompt_mentions_tools(self, supabase_skill):
        """Test prompt mentions all tools"""
        prompt = supabase_skill.supabase_prompt()
        assert 'supabase_setup(' in prompt
        assert 'supabase_query(' in prompt
        assert 'supabase_table_ops(' in prompt
        assert 'supabase_status(' in prompt
        assert 'Minimalistic Supabase/PostgreSQL integration' in prompt

    def test_has_4_tools(self, supabase_skill):
        """Test skill has exactly 4 tools"""
        supabase_tools = [m for m in dir(supabase_skill) if m.startswith('supabase_') and not m.startswith('supabase_prompt')]
        assert len(supabase_tools) == 4
        assert 'supabase_setup' in supabase_tools
        assert 'supabase_query' in supabase_tools
        assert 'supabase_table_ops' in supabase_tools
        assert 'supabase_status' in supabase_tools

    @pytest.mark.asyncio
    async def test_save_load_config_kv(self, supabase_skill, mock_supabase_config):
        """Test saving and loading configuration via KV skill"""
        user_id = 'test_user_123'
        
        # Test save
        success = await supabase_skill._save_db_config(user_id, mock_supabase_config)
        assert success == True
        
        # Test load
        loaded_config = await supabase_skill._load_db_config(user_id)
        assert loaded_config['type'] == 'supabase'
        assert loaded_config['supabase_url'] == 'https://test.supabase.co'
        assert 'created_at' in loaded_config

    @pytest.mark.asyncio
    async def test_save_load_config_memory_fallback(self, supabase_skill, mock_supabase_config):
        """Test memory fallback when KV skill unavailable"""
        # Remove KV skill to test fallback
        supabase_skill.agent.skills['kv'] = None
        
        user_id = 'test_user_123'
        
        # Test save (should use memory fallback)
        success = await supabase_skill._save_db_config(user_id, mock_supabase_config)
        assert success == True
        
        # Test load (should use memory fallback)
        loaded_config = await supabase_skill._load_db_config(user_id)
        assert loaded_config['type'] == 'supabase'

    @pytest.mark.asyncio
    async def test_supabase_setup_success(self, supabase_skill, mock_context, mock_supabase_config):
        """Test successful Supabase setup"""
        with patch('webagents.agents.skills.ecosystem.database.skill.get_context', return_value=mock_context):
            with patch.object(supabase_skill, '_create_supabase_client') as mock_create_client:
                mock_client = MagicMock()
                mock_client.table.return_value.select.return_value.limit.return_value.execute.return_value = MagicMock()
                mock_create_client.return_value = mock_client
                
                result = await supabase_skill.supabase_setup(mock_supabase_config)
                
                assert '✅ Supabase configuration saved successfully!' in result
                assert 'supabase' in result

    @pytest.mark.asyncio
    async def test_postgres_setup_success(self, supabase_skill, mock_context, mock_postgres_config):
        """Test successful PostgreSQL setup"""
        with patch('webagents.agents.skills.ecosystem.database.skill.get_context', return_value=mock_context):
            with patch.object(supabase_skill, '_create_postgres_connection') as mock_create_conn:
                mock_conn = MagicMock()
                mock_cursor = MagicMock()
                mock_conn.cursor.return_value = mock_cursor
                mock_create_conn.return_value = mock_conn
                
                result = await supabase_skill.supabase_setup(mock_postgres_config)
                
                assert '✅ Postgresql configuration saved successfully!' in result
                assert 'postgresql' in result

    @pytest.mark.asyncio
    async def test_supabase_setup_no_auth(self, supabase_skill, mock_supabase_config):
        """Test setup without authentication"""
        with patch('webagents.agents.skills.ecosystem.database.skill.get_context', return_value=None):
            result = await supabase_skill.supabase_setup(mock_supabase_config)
            
            assert '❌ Authentication required' in result

    @pytest.mark.asyncio
    async def test_supabase_setup_empty_config(self, supabase_skill, mock_context):
        """Test setup with empty configuration"""
        with patch('webagents.agents.skills.ecosystem.database.skill.get_context', return_value=mock_context):
            result = await supabase_skill.supabase_setup({})
            
            assert '❌ Configuration is required' in result

    @pytest.mark.asyncio
    async def test_supabase_setup_invalid_config(self, supabase_skill, mock_context):
        """Test setup with invalid Supabase configuration"""
        invalid_config = {'type': 'supabase', 'supabase_url': 'https://test.supabase.co'}  # Missing key
        
        with patch('webagents.agents.skills.ecosystem.database.skill.get_context', return_value=mock_context):
            result = await supabase_skill.supabase_setup(invalid_config)
            
            assert '❌ Supabase configuration requires' in result

    @pytest.mark.asyncio
    async def test_postgres_setup_invalid_config(self, supabase_skill, mock_context):
        """Test setup with invalid PostgreSQL configuration"""
        invalid_config = {'type': 'postgresql', 'host': 'localhost'}  # Missing required fields
        
        with patch('webagents.agents.skills.ecosystem.database.skill.get_context', return_value=mock_context):
            result = await supabase_skill.supabase_setup(invalid_config)
            
            assert '❌ PostgreSQL configuration requires' in result

    @pytest.mark.asyncio
    async def test_supabase_query_no_config(self, supabase_skill, mock_context):
        """Test query without configuration"""
        with patch('webagents.agents.skills.ecosystem.database.skill.get_context', return_value=mock_context):
            result = await supabase_skill.supabase_query("SELECT * FROM users")
            
            assert '❌ Database not configured' in result
            assert 'supabase_setup()' in result

    @pytest.mark.asyncio
    async def test_postgres_query_success(self, supabase_skill, mock_context, mock_postgres_config):
        """Test successful PostgreSQL query execution"""
        with patch('webagents.agents.skills.ecosystem.database.skill.get_context', return_value=mock_context):
            await supabase_skill._save_db_config('test_user_123', mock_postgres_config)
            
            with patch.object(supabase_skill, '_create_postgres_connection') as mock_create_conn:
                mock_conn = MagicMock()
                mock_cursor = MagicMock()
                mock_cursor.fetchall.return_value = [{'id': 1, 'name': 'Test User'}]
                mock_cursor.rowcount = 1
                mock_conn.cursor.return_value = mock_cursor
                mock_create_conn.return_value = mock_conn
                
                result = await supabase_skill.supabase_query("SELECT * FROM users")
                
                assert '✅ Query executed successfully!' in result
                assert 'Test User' in result

    @pytest.mark.asyncio
    async def test_supabase_table_ops_select(self, supabase_skill, mock_context, mock_supabase_config):
        """Test Supabase table select operation"""
        with patch('webagents.agents.skills.ecosystem.database.skill.get_context', return_value=mock_context):
            await supabase_skill._save_db_config('test_user_123', mock_supabase_config)
            
            with patch.object(supabase_skill, '_create_supabase_client') as mock_create_client:
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.data = [{'id': 1, 'name': 'Test User'}]
                
                mock_query = MagicMock()
                mock_query.execute.return_value = mock_response
                mock_query.eq.return_value = mock_query
                mock_client.table.return_value.select.return_value = mock_query
                mock_create_client.return_value = mock_client
                
                result = await supabase_skill.supabase_table_ops('select', 'users', filters={'active': True})
                
                assert '✅ Select operation successful!' in result
                assert 'Test User' in result

    @pytest.mark.asyncio
    async def test_supabase_table_ops_insert(self, supabase_skill, mock_context, mock_supabase_config):
        """Test Supabase table insert operation"""
        with patch('webagents.agents.skills.ecosystem.database.skill.get_context', return_value=mock_context):
            await supabase_skill._save_db_config('test_user_123', mock_supabase_config)
            
            with patch.object(supabase_skill, '_create_supabase_client') as mock_create_client:
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.data = [{'id': 2, 'name': 'New User'}]
                
                mock_client.table.return_value.insert.return_value.execute.return_value = mock_response
                mock_create_client.return_value = mock_client
                
                result = await supabase_skill.supabase_table_ops('insert', 'users', data={'name': 'New User'})
                
                assert '✅ Insert operation successful!' in result
                assert 'New User' in result

    @pytest.mark.asyncio
    async def test_supabase_table_ops_invalid_operation(self, supabase_skill, mock_context):
        """Test table operations with invalid operation"""
        with patch('webagents.agents.skills.ecosystem.database.skill.get_context', return_value=mock_context):
            result = await supabase_skill.supabase_table_ops('invalid_op', 'users')
            
            assert '❌ Invalid operation' in result
            assert 'select, insert, update, delete' in result

    @pytest.mark.asyncio
    async def test_supabase_table_ops_missing_data(self, supabase_skill, mock_context, mock_supabase_config):
        """Test insert operation without required data"""
        with patch('webagents.agents.skills.ecosystem.database.skill.get_context', return_value=mock_context):
            await supabase_skill._save_db_config('test_user_123', mock_supabase_config)
            
            with patch.object(supabase_skill, '_create_supabase_client') as mock_create_client:
                mock_client = MagicMock()
                mock_create_client.return_value = mock_client
                
                result = await supabase_skill.supabase_table_ops('insert', 'users')
                
                assert '❌ Data is required for insert operation' in result

    @pytest.mark.asyncio
    async def test_supabase_table_ops_missing_filters(self, supabase_skill, mock_context, mock_supabase_config):
        """Test update operation without required filters"""
        with patch('webagents.agents.skills.ecosystem.database.skill.get_context', return_value=mock_context):
            await supabase_skill._save_db_config('test_user_123', mock_supabase_config)
            
            with patch.object(supabase_skill, '_create_supabase_client') as mock_create_client:
                mock_client = MagicMock()
                mock_create_client.return_value = mock_client
                
                result = await supabase_skill.supabase_table_ops('update', 'users', data={'name': 'Updated'})
                
                assert '❌ Filters are required for update operation' in result

    @pytest.mark.asyncio
    async def test_postgres_table_ops_success(self, supabase_skill, mock_context, mock_postgres_config):
        """Test PostgreSQL table operations"""
        with patch('webagents.agents.skills.ecosystem.database.skill.get_context', return_value=mock_context):
            await supabase_skill._save_db_config('test_user_123', mock_postgres_config)
            
            with patch.object(supabase_skill, '_create_postgres_connection') as mock_create_conn:
                mock_conn = MagicMock()
                mock_cursor = MagicMock()
                mock_cursor.fetchall.return_value = [{'id': 1, 'name': 'Test User'}]
                mock_cursor.rowcount = 1
                mock_conn.cursor.return_value = mock_cursor
                mock_create_conn.return_value = mock_conn
                
                result = await supabase_skill.supabase_table_ops('select', 'users')
                
                assert '✅ Select operation successful!' in result

    @pytest.mark.asyncio
    async def test_supabase_status_with_config(self, supabase_skill, mock_context, mock_supabase_config):
        """Test status check when config exists"""
        with patch('webagents.agents.skills.ecosystem.database.skill.get_context', return_value=mock_context):
            await supabase_skill._save_db_config('test_user_123', mock_supabase_config)
            
            with patch.object(supabase_skill, '_create_supabase_client') as mock_create_client:
                mock_client = MagicMock()
                mock_client.table.return_value.select.return_value.limit.return_value.execute.return_value = MagicMock()
                mock_create_client.return_value = mock_client
                
                result = await supabase_skill.supabase_status()
                
                assert '📋 Database Status:' in result
                assert '✅ Configuration: Active' in result
                assert '🟢 Connection: Active' in result

    @pytest.mark.asyncio
    async def test_supabase_status_no_config(self, supabase_skill, mock_context):
        """Test status check when no config exists"""
        with patch('webagents.agents.skills.ecosystem.database.skill.get_context', return_value=mock_context):
            result = await supabase_skill.supabase_status()
            
            assert '❌ No configuration found' in result
            assert 'supabase_setup()' in result

    @pytest.mark.asyncio
    async def test_supabase_status_connection_error(self, supabase_skill, mock_context, mock_supabase_config):
        """Test status check with connection error"""
        with patch('webagents.agents.skills.ecosystem.database.skill.get_context', return_value=mock_context):
            await supabase_skill._save_db_config('test_user_123', mock_supabase_config)
            
            with patch.object(supabase_skill, '_create_supabase_client') as mock_create_client:
                mock_create_client.side_effect = Exception("Connection failed")
                
                result = await supabase_skill.supabase_status()
                
                assert '🟡 Connection: Warning' in result

    @pytest.mark.asyncio
    async def test_tools_require_authentication(self, supabase_skill):
        """Test that all tools require authentication"""
        unauthenticated_context = MagicMock()
        unauthenticated_context.auth = None
        
        with patch('webagents.agents.skills.ecosystem.database.skill.get_context', return_value=unauthenticated_context):
            # Test all tools return authentication required
            result = await supabase_skill.supabase_setup({'type': 'supabase'})
            assert '❌ Authentication required' in result
            
            result = await supabase_skill.supabase_query("SELECT 1")
            assert '❌ Authentication required' in result
            
            result = await supabase_skill.supabase_table_ops('select', 'users')
            assert '❌ Authentication required' in result
            
            result = await supabase_skill.supabase_status()
            assert '❌ Authentication required' in result

    def test_create_supabase_client(self, supabase_skill):
        """Test Supabase client creation"""
        config = {
            'supabase_url': 'https://test.supabase.co',
            'supabase_key': 'test_key'
        }
        
        with patch('webagents.agents.skills.ecosystem.database.skill.create_client') as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client
            
            client = supabase_skill._create_supabase_client(config)
            
            assert client == mock_client
            mock_create.assert_called_once_with('https://test.supabase.co', 'test_key')

    def test_create_postgres_connection_url(self, supabase_skill):
        """Test PostgreSQL connection creation with URL"""
        config = {'postgres_url': 'postgresql://user:pass@localhost:5432/db'}
        
        with patch('webagents.agents.skills.ecosystem.database.skill.psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            
            conn = supabase_skill._create_postgres_connection(config)
            
            assert conn == mock_conn

    def test_create_postgres_connection_params(self, supabase_skill):
        """Test PostgreSQL connection creation with individual parameters"""
        config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'testdb',
            'user': 'testuser',
            'password': 'testpass'
        }
        
        with patch('webagents.agents.skills.ecosystem.database.skill.psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            
            conn = supabase_skill._create_postgres_connection(config)
            
            assert conn == mock_conn


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
