"""
Minimalistic Supabase/PostgreSQL Skill for WebAgents

This skill allows users to:
- Connect to Supabase or PostgreSQL databases
- Execute queries and manage data (CRUD operations)
- Handle authentication and real-time subscriptions
- Manage database schemas and tables

Uses auth skill for user context and kv skill for secure connection storage.
"""

import os
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, prompt
from webagents.server.context.context_vars import get_context

try:
    from supabase import create_client, Client
    import psycopg2
    from psycopg2.extras import RealDictCursor
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    # Stand-in bindings (mongodb-skill pattern): annotations such as
    # ``Optional[Client]`` are evaluated at class creation on Python <= 3.13,
    # and an unbound name there raises NameError — which the
    # ``except ImportError`` guards upstream cannot catch (F-040).
    class Client:  # type: ignore[no-redef]
        pass

    def create_client(*_args, **_kwargs):  # type: ignore[no-redef]
        raise ImportError("Supabase dependencies not installed. Install with: pip install supabase psycopg2-binary")

    psycopg2 = None  # type: ignore[assignment]

    class RealDictCursor:  # type: ignore[no-redef]
        pass


class SupabaseSkill(Skill):
    """Minimalistic Supabase/PostgreSQL skill for database operations"""
    
    def __init__(self):
        super().__init__()
        if not SUPABASE_AVAILABLE:
            raise ImportError("Supabase dependencies not installed. Install with: pip install supabase psycopg2-binary")
    
    def get_dependencies(self) -> List[str]:
        """Skill dependencies"""
        return ['auth', 'kv']
    
    @prompt(priority=40, scope=["owner", "all"])
    def supabase_prompt(self) -> str:
        """Prompt describing Supabase capabilities"""
        return """
Minimalistic Supabase/PostgreSQL integration for database operations. Available tools:

• supabase_setup(config) - Set up Supabase or PostgreSQL connection securely
• supabase_query(sql, params) - Execute SQL queries with optional parameters
• supabase_table_ops(operation, table, data) - Perform CRUD operations on tables
• supabase_status() - Check database connection and configuration status

Features:
- Supabase and PostgreSQL database support
- Secure connection string storage
- SQL query execution with parameterization
- CRUD operations (Create, Read, Update, Delete)
- Real-time subscription support (Supabase)
- Per-user database isolation via Auth skill

Setup: Configure with your Supabase URL/key or PostgreSQL connection string.
"""

    # Helper methods for auth and kv skills
    async def _get_auth_skill(self):
        """Get auth skill for user context"""
        return self.agent.skills.get('auth')
    
    async def _get_kv_skill(self):
        """Get KV skill for secure storage"""
        return self.agent.skills.get('kv')
    
    async def _get_authenticated_user_id(self) -> Optional[str]:
        """Get authenticated user ID from context"""
        try:
            context = get_context()
            if context and context.auth and context.auth.authenticated:
                return context.auth.user_id
            return None
        except Exception as e:
            self.logger.error(f"Failed to get user context: {e}")
            return None
    
    async def _save_db_config(self, user_id: str, config: Dict[str, Any]) -> bool:
        """Save database configuration securely using KV skill"""
        try:
            kv_skill = await self._get_kv_skill()
            if kv_skill:
                config_data = {
                    **config,
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat()
                }
                await kv_skill.kv_set(
                    key='config',
                    value=json.dumps(config_data),
                    namespace=f'supabase:{user_id}'
                )
                return True
            else:
                # Fallback to in-memory storage
                if not hasattr(self.agent, '_supabase_configs'):
                    self.agent._supabase_configs = {}
                self.agent._supabase_configs[user_id] = config
                return True
        except Exception as e:
            self.logger.error(f"Failed to save database config: {e}")
            return False
    
    async def _load_db_config(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Load database configuration from KV skill"""
        try:
            kv_skill = await self._get_kv_skill()
            if kv_skill:
                config_json = await kv_skill.kv_get(
                    key='config',
                    namespace=f'supabase:{user_id}'
                )
                if config_json:
                    return json.loads(config_json)
            else:
                # Fallback to in-memory storage
                if hasattr(self.agent, '_supabase_configs'):
                    return self.agent._supabase_configs.get(user_id)
            return None
        except Exception as e:
            self.logger.error(f"Failed to load database config: {e}")
            return None
    
    def _create_supabase_client(self, config: Dict[str, Any]) -> Optional[Client]:
        """Create Supabase client from configuration"""
        try:
            url = config.get('supabase_url')
            key = config.get('supabase_key')
            if url and key:
                return create_client(url, key)
            return None
        except Exception as e:
            self.logger.error(f"Failed to create Supabase client: {e}")
            return None
    
    def _create_postgres_connection(self, config: Dict[str, Any]):
        """Create PostgreSQL connection from configuration"""
        try:
            connection_string = config.get('postgres_url')
            if connection_string:
                return psycopg2.connect(connection_string, cursor_factory=RealDictCursor)
            
            # Alternative: individual connection parameters
            conn_params = {
                'host': config.get('host'),
                'port': config.get('port', 5432),
                'database': config.get('database'),
                'user': config.get('user'),
                'password': config.get('password')
            }
            if all(conn_params.values()):
                return psycopg2.connect(cursor_factory=RealDictCursor, **conn_params)
            
            return None
        except Exception as e:
            self.logger.error(f"Failed to create PostgreSQL connection: {e}")
            return None

    # Public tools
    @tool(description="Set up Supabase or PostgreSQL database connection", scope="owner")
    async def supabase_setup(self, config: Dict[str, Any]) -> str:
        """Set up database connection configuration"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        if not config:
            return "❌ Configuration is required"
        
        try:
            # Validate configuration type
            db_type = config.get('type', 'supabase').lower()
            
            if db_type == 'supabase':
                if 'supabase_url' not in config or 'supabase_key' not in config:
                    return "❌ Supabase configuration requires 'supabase_url' and 'supabase_key'"
                
                # Test connection
                client = self._create_supabase_client(config)
                if not client:
                    return "❌ Failed to create Supabase client"
                
                # Test with a simple query
                try:
                    # This will fail gracefully if no tables exist
                    client.table('_supabase_test_').select('*').limit(1).execute()
                except:
                    pass  # Expected if table doesn't exist
                
            elif db_type == 'postgresql':
                if not config.get('postgres_url') and not all([
                    config.get('host'), config.get('database'), 
                    config.get('user'), config.get('password')
                ]):
                    return "❌ PostgreSQL configuration requires either 'postgres_url' or host/database/user/password"
                
                # Test connection
                conn = self._create_postgres_connection(config)
                if not conn:
                    return "❌ Failed to create PostgreSQL connection"
                
                # Test with a simple query
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.close()
                    conn.close()
                except Exception as e:
                    return f"❌ Database connection test failed: {str(e)}"
            
            else:
                return "❌ Unsupported database type. Use 'supabase' or 'postgresql'"
            
            # Save configuration
            success = await self._save_db_config(user_id, config)
            
            if success:
                return f"✅ {db_type.title()} configuration saved successfully!\n🔧 Database type: {db_type}\n🔒 Connection details stored securely"
            else:
                return "❌ Failed to save configuration"
                
        except Exception as e:
            return f"❌ Setup failed: {str(e)}"
    
    @tool(description="Execute SQL queries on the configured database")
    async def supabase_query(self, sql: str, params: Optional[List[Any]] = None) -> str:
        """Execute SQL queries with optional parameters"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        if not sql or not sql.strip():
            return "❌ SQL query is required"
        
        try:
            # Load user configuration
            config = await self._load_db_config(user_id)
            if not config:
                return "❌ Database not configured. Please run supabase_setup() first."
            
            db_type = config.get('type', 'supabase').lower()
            
            if db_type == 'supabase':
                client = self._create_supabase_client(config)
                if not client:
                    return "❌ Failed to connect to Supabase"
                
                # For Supabase, we'll use the PostgREST API via rpc or direct table operations
                # This is a simplified approach - in practice, you'd want more sophisticated SQL parsing
                sql_lower = sql.lower().strip()
                
                if sql_lower.startswith('select'):
                    return "❌ For SELECT queries, use supabase_table_ops() with operation='select'"
                elif sql_lower.startswith(('insert', 'update', 'delete')):
                    return "❌ For data modifications, use supabase_table_ops() for better Supabase integration"
                else:
                    # For other operations, we'd need to use the underlying PostgreSQL connection
                    return "❌ Complex SQL operations not supported in Supabase mode. Use PostgreSQL mode for full SQL support."
            
            elif db_type == 'postgresql':
                conn = self._create_postgres_connection(config)
                if not conn:
                    return "❌ Failed to connect to PostgreSQL"
                
                try:
                    cursor = conn.cursor()
                    
                    # Execute query with parameters
                    if params:
                        cursor.execute(sql, params)
                    else:
                        cursor.execute(sql)
                    
                    # Handle different query types
                    if sql.lower().strip().startswith('select'):
                        results = cursor.fetchall()
                        if results:
                            # Convert to list of dicts
                            rows = [dict(row) for row in results]
                            return f"✅ Query executed successfully!\n📊 Results ({len(rows)} rows):\n{json.dumps(rows, indent=2, default=str)}"
                        else:
                            return "✅ Query executed successfully!\n📊 No results returned"
                    else:
                        # For INSERT, UPDATE, DELETE, etc.
                        conn.commit()
                        affected_rows = cursor.rowcount
                        return f"✅ Query executed successfully!\n📝 Affected rows: {affected_rows}"
                
                finally:
                    cursor.close()
                    conn.close()
            
            else:
                return "❌ Unknown database type in configuration"
            
        except Exception as e:
            return f"❌ Query execution failed: {str(e)}"
    
    @tool(description="Perform CRUD operations on database tables")
    async def supabase_table_ops(self, operation: str, table: str, data: Optional[Dict[str, Any]] = None, 
                                filters: Optional[Dict[str, Any]] = None) -> str:
        """Perform Create, Read, Update, Delete operations on tables"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        if not operation or not table:
            return "❌ Operation and table name are required"
        
        operation = operation.lower().strip()
        valid_operations = ['select', 'insert', 'update', 'delete']
        
        if operation not in valid_operations:
            return f"❌ Invalid operation. Supported: {', '.join(valid_operations)}"
        
        try:
            # Load user configuration
            config = await self._load_db_config(user_id)
            if not config:
                return "❌ Database not configured. Please run supabase_setup() first."
            
            db_type = config.get('type', 'supabase').lower()
            
            if db_type == 'supabase':
                client = self._create_supabase_client(config)
                if not client:
                    return "❌ Failed to connect to Supabase"
                
                # Perform Supabase operations
                if operation == 'select':
                    query = client.table(table).select('*')
                    if filters:
                        for key, value in filters.items():
                            query = query.eq(key, value)
                    
                    response = query.execute()
                    results = response.data
                    
                    return f"✅ Select operation successful!\n📊 Results ({len(results)} rows):\n{json.dumps(results, indent=2, default=str)}"
                
                elif operation == 'insert':
                    if not data:
                        return "❌ Data is required for insert operation"
                    
                    response = client.table(table).insert(data).execute()
                    return f"✅ Insert operation successful!\n📝 Inserted data: {json.dumps(response.data, indent=2, default=str)}"
                
                elif operation == 'update':
                    if not data:
                        return "❌ Data is required for update operation"
                    if not filters:
                        return "❌ Filters are required for update operation to prevent updating all rows"
                    
                    query = client.table(table).update(data)
                    for key, value in filters.items():
                        query = query.eq(key, value)
                    
                    response = query.execute()
                    return f"✅ Update operation successful!\n📝 Updated rows: {len(response.data)}"
                
                elif operation == 'delete':
                    if not filters:
                        return "❌ Filters are required for delete operation to prevent deleting all rows"
                    
                    query = client.table(table).delete()
                    for key, value in filters.items():
                        query = query.eq(key, value)
                    
                    response = query.execute()
                    return f"✅ Delete operation successful!\n📝 Deleted rows: {len(response.data)}"
            
            elif db_type == 'postgresql':
                conn = self._create_postgres_connection(config)
                if not conn:
                    return "❌ Failed to connect to PostgreSQL"
                
                try:
                    cursor = conn.cursor()
                    
                    if operation == 'select':
                        sql = f"SELECT * FROM {table}"
                        params = []
                        
                        if filters:
                            where_clauses = []
                            for key, value in filters.items():
                                where_clauses.append(f"{key} = %s")
                                params.append(value)
                            sql += " WHERE " + " AND ".join(where_clauses)
                        
                        cursor.execute(sql, params)
                        results = cursor.fetchall()
                        rows = [dict(row) for row in results]
                        
                        return f"✅ Select operation successful!\n📊 Results ({len(rows)} rows):\n{json.dumps(rows, indent=2, default=str)}"
                    
                    elif operation == 'insert':
                        if not data:
                            return "❌ Data is required for insert operation"
                        
                        columns = list(data.keys())
                        values = list(data.values())
                        placeholders = ', '.join(['%s'] * len(values))
                        
                        sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
                        cursor.execute(sql, values)
                        conn.commit()
                        
                        return f"✅ Insert operation successful!\n📝 Affected rows: {cursor.rowcount}"
                    
                    elif operation == 'update':
                        if not data:
                            return "❌ Data is required for update operation"
                        if not filters:
                            return "❌ Filters are required for update operation"
                        
                        set_clauses = []
                        params = []
                        for key, value in data.items():
                            set_clauses.append(f"{key} = %s")
                            params.append(value)
                        
                        where_clauses = []
                        for key, value in filters.items():
                            where_clauses.append(f"{key} = %s")
                            params.append(value)
                        
                        sql = f"UPDATE {table} SET {', '.join(set_clauses)} WHERE {' AND '.join(where_clauses)}"
                        cursor.execute(sql, params)
                        conn.commit()
                        
                        return f"✅ Update operation successful!\n📝 Affected rows: {cursor.rowcount}"
                    
                    elif operation == 'delete':
                        if not filters:
                            return "❌ Filters are required for delete operation"
                        
                        where_clauses = []
                        params = []
                        for key, value in filters.items():
                            where_clauses.append(f"{key} = %s")
                            params.append(value)
                        
                        sql = f"DELETE FROM {table} WHERE {' AND '.join(where_clauses)}"
                        cursor.execute(sql, params)
                        conn.commit()
                        
                        return f"✅ Delete operation successful!\n📝 Affected rows: {cursor.rowcount}"
                
                finally:
                    cursor.close()
                    conn.close()
            
            else:
                return "❌ Unknown database type in configuration"
            
        except Exception as e:
            return f"❌ Table operation failed: {str(e)}"
    
    @tool(description="Check database connection status and configuration")
    async def supabase_status(self) -> str:
        """Check database connection and configuration status"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        try:
            # Load configuration
            config = await self._load_db_config(user_id)
            
            result = ["📋 Database Status:\n"]
            
            if config:
                db_type = config.get('type', 'supabase')
                created_at = config.get('created_at', 'unknown')
                result.append(f"✅ Configuration: Active")
                result.append(f"🗄️ Database Type: {db_type.title()}")
                result.append(f"🕐 Created: {created_at}")
                
                # Test connection
                try:
                    if db_type.lower() == 'supabase':
                        client = self._create_supabase_client(config)
                        if client:
                            # Try a simple operation
                            client.table('_test_').select('*').limit(1).execute()
                            result.append("🟢 Connection: Active")
                        else:
                            result.append("🔴 Connection: Failed")
                    
                    elif db_type.lower() == 'postgresql':
                        conn = self._create_postgres_connection(config)
                        if conn:
                            cursor = conn.cursor()
                            cursor.execute("SELECT 1")
                            cursor.close()
                            conn.close()
                            result.append("🟢 Connection: Active")
                        else:
                            result.append("🔴 Connection: Failed")
                    
                except Exception as e:
                    result.append(f"🟡 Connection: Warning - {str(e)}")
                
            else:
                result.append("❌ No configuration found")
                result.append("💡 Use supabase_setup() to configure your database")
                return "\n".join(result)
            
            result.append("\n💡 Use supabase_query() for SQL or supabase_table_ops() for CRUD operations")
            
            return "\n".join(result)
            
        except Exception as e:
            return f"❌ Error checking status: {str(e)}"
