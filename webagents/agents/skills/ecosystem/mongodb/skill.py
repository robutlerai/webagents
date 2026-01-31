"""
Minimalistic MongoDB Skill for WebAgents

This skill allows users to:
- Connect to MongoDB databases (local, Atlas, or self-hosted)
- Perform CRUD operations on collections
- Execute aggregation pipelines and queries
- Manage database schemas and indexes

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
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, OperationFailure, ConfigurationError
    from bson import ObjectId
    from bson.errors import InvalidId
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    # Create dummy classes for type hints when pymongo is not available
    class MongoClient:
        pass
    class ObjectId:
        pass
    class ConnectionFailure(Exception):
        pass


class MongoDBSkill(Skill):
    """Minimalistic MongoDB skill for document database operations"""
    
    def __init__(self):
        super().__init__()
        if not MONGODB_AVAILABLE:
            raise ImportError("MongoDB dependencies not installed. Install with: pip install pymongo")
    
    def get_dependencies(self) -> List[str]:
        """Skill dependencies"""
        return ['auth', 'kv']
    
    @prompt(priority=40, scope=["owner", "all"])
    def mongodb_prompt(self) -> str:
        """Prompt describing MongoDB capabilities"""
        return """
Minimalistic MongoDB integration for document database operations. Available tools:

• mongodb_setup(config) - Set up MongoDB connection securely (Atlas, local, or self-hosted)
• mongodb_query(database, collection, operation, query, data) - Execute database operations
• mongodb_aggregate(database, collection, pipeline) - Run aggregation pipelines
• mongodb_status() - Check database connection and configuration status

Features:
- MongoDB Atlas, local, and self-hosted support
- Document CRUD operations (Create, Read, Update, Delete)
- Aggregation pipeline execution
- Index management and optimization
- Secure connection string storage
- Per-user database isolation via Auth skill

Setup: Configure with your MongoDB connection string or Atlas credentials.
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
    
    async def _save_mongodb_config(self, user_id: str, config: Dict[str, Any]) -> bool:
        """Save MongoDB configuration securely using KV skill"""
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
                    namespace=f'mongodb:{user_id}'
                )
                return True
            else:
                # Fallback to in-memory storage
                if not hasattr(self.agent, '_mongodb_configs'):
                    self.agent._mongodb_configs = {}
                self.agent._mongodb_configs[user_id] = config
                return True
        except Exception as e:
            self.logger.error(f"Failed to save MongoDB config: {e}")
            return False
    
    async def _load_mongodb_config(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Load MongoDB configuration from KV skill"""
        try:
            kv_skill = await self._get_kv_skill()
            if kv_skill:
                config_json = await kv_skill.kv_get(
                    key='config',
                    namespace=f'mongodb:{user_id}'
                )
                if config_json:
                    return json.loads(config_json)
            else:
                # Fallback to in-memory storage
                if hasattr(self.agent, '_mongodb_configs'):
                    return self.agent._mongodb_configs.get(user_id)
            return None
        except Exception as e:
            self.logger.error(f"Failed to load MongoDB config: {e}")
            return None
    
    def _create_mongodb_client(self, config: Dict[str, Any]) -> Optional[MongoClient]:
        """Create MongoDB client from configuration"""
        try:
            connection_string = config.get('connection_string')
            if connection_string:
                # Use connection string (Atlas, local, or custom)
                return MongoClient(connection_string)
            
            # Alternative: individual connection parameters
            host = config.get('host', 'localhost')
            port = config.get('port', 27017)
            username = config.get('username')
            password = config.get('password')
            database = config.get('database')
            
            if username and password:
                if database:
                    connection_string = f"mongodb://{username}:{password}@{host}:{port}/{database}"
                else:
                    connection_string = f"mongodb://{username}:{password}@{host}:{port}"
            else:
                connection_string = f"mongodb://{host}:{port}"
            
            return MongoClient(connection_string)
            
        except Exception as e:
            self.logger.error(f"Failed to create MongoDB client: {e}")
            return None
    
    def _serialize_mongodb_result(self, data: Any) -> Any:
        """Serialize MongoDB results for JSON output"""
        if isinstance(data, ObjectId) or (hasattr(data, '__class__') and data.__class__.__name__ == 'ObjectId'):
            return str(data)
        elif isinstance(data, dict):
            return {k: self._serialize_mongodb_result(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._serialize_mongodb_result(item) for item in data]
        elif isinstance(data, datetime):
            return data.isoformat()
        else:
            return data

    # Public tools
    @tool(description="Set up MongoDB database connection", scope="owner")
    async def mongodb_setup(self, config: Dict[str, Any]) -> str:
        """Set up MongoDB connection configuration"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        if not config:
            return "❌ Configuration is required"
        
        try:
            # Validate configuration
            if not config.get('connection_string') and not all([
                config.get('host'), config.get('port')
            ]):
                return "❌ MongoDB configuration requires either 'connection_string' or 'host' and 'port'"
            
            # Test connection
            client = self._create_mongodb_client(config)
            if not client:
                return "❌ Failed to create MongoDB client"
            
            # Test connection with ping
            try:
                client.admin.command('ping')
                client.close()
            except ConnectionFailure:
                return "❌ MongoDB connection test failed"
            except Exception as e:
                return f"❌ MongoDB connection error: {str(e)}"
            
            # Save configuration
            success = await self._save_mongodb_config(user_id, config)
            
            if success:
                connection_type = "Atlas" if "mongodb.net" in config.get('connection_string', '') else "MongoDB"
                return f"✅ {connection_type} configuration saved successfully!\n🔧 Database type: MongoDB\n🔒 Connection details stored securely"
            else:
                return "❌ Failed to save configuration"
                
        except Exception as e:
            return f"❌ Setup failed: {str(e)}"
    
    @tool(description="Execute MongoDB operations on collections")
    async def mongodb_query(self, database: str, collection: str, operation: str, 
                          query: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None) -> str:
        """Execute MongoDB operations (find, insert, update, delete)"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        if not database or not collection or not operation:
            return "❌ Database, collection, and operation are required"
        
        operation = operation.lower().strip()
        valid_operations = ['find', 'find_one', 'insert_one', 'insert_many', 'update_one', 'update_many', 'delete_one', 'delete_many', 'count']
        
        if operation not in valid_operations:
            return f"❌ Invalid operation. Supported: {', '.join(valid_operations)}"
        
        try:
            # Load user configuration
            config = await self._load_mongodb_config(user_id)
            if not config:
                return "❌ MongoDB not configured. Please run mongodb_setup() first."
            
            # Create client and get collection
            client = self._create_mongodb_client(config)
            if not client:
                return "❌ Failed to connect to MongoDB"
            
            try:
                db = client[database]
                coll = db[collection]
                
                # Execute operation
                if operation == 'find':
                    query = query or {}
                    results = list(coll.find(query))
                    serialized_results = self._serialize_mongodb_result(results)
                    return f"✅ Find operation successful!\n📊 Results ({len(results)} documents):\n{json.dumps(serialized_results, indent=2, default=str)}"
                
                elif operation == 'find_one':
                    query = query or {}
                    result = coll.find_one(query)
                    if result:
                        serialized_result = self._serialize_mongodb_result(result)
                        return f"✅ Find one operation successful!\n📊 Result:\n{json.dumps(serialized_result, indent=2, default=str)}"
                    else:
                        return "✅ Find one operation successful!\n📊 No document found"
                
                elif operation == 'insert_one':
                    if not data:
                        return "❌ Data is required for insert_one operation"
                    
                    result = coll.insert_one(data)
                    return f"✅ Insert one operation successful!\n📝 Inserted document ID: {result.inserted_id}"
                
                elif operation == 'insert_many':
                    if not data or not isinstance(data.get('documents'), list):
                        return "❌ Data with 'documents' array is required for insert_many operation"
                    
                    result = coll.insert_many(data['documents'])
                    return f"✅ Insert many operation successful!\n📝 Inserted {len(result.inserted_ids)} documents"
                
                elif operation == 'update_one':
                    if not query:
                        return "❌ Query filter is required for update_one operation"
                    if not data:
                        return "❌ Update data is required for update_one operation"
                    
                    result = coll.update_one(query, {'$set': data})
                    return f"✅ Update one operation successful!\n📝 Matched: {result.matched_count}, Modified: {result.modified_count}"
                
                elif operation == 'update_many':
                    if not query:
                        return "❌ Query filter is required for update_many operation"
                    if not data:
                        return "❌ Update data is required for update_many operation"
                    
                    result = coll.update_many(query, {'$set': data})
                    return f"✅ Update many operation successful!\n📝 Matched: {result.matched_count}, Modified: {result.modified_count}"
                
                elif operation == 'delete_one':
                    if not query:
                        return "❌ Query filter is required for delete_one operation"
                    
                    result = coll.delete_one(query)
                    return f"✅ Delete one operation successful!\n📝 Deleted: {result.deleted_count} document"
                
                elif operation == 'delete_many':
                    if not query:
                        return "❌ Query filter is required for delete_many operation"
                    
                    result = coll.delete_many(query)
                    return f"✅ Delete many operation successful!\n📝 Deleted: {result.deleted_count} documents"
                
                elif operation == 'count':
                    query = query or {}
                    count = coll.count_documents(query)
                    return f"✅ Count operation successful!\n📊 Document count: {count}"
                
            finally:
                client.close()
            
        except Exception as e:
            return f"❌ MongoDB operation failed: {str(e)}"
    
    @tool(description="Execute MongoDB aggregation pipelines")
    async def mongodb_aggregate(self, database: str, collection: str, pipeline: List[Dict[str, Any]]) -> str:
        """Execute MongoDB aggregation pipeline"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        if not database or not collection or not pipeline:
            return "❌ Database, collection, and pipeline are required"
        
        if not isinstance(pipeline, list):
            return "❌ Pipeline must be a list of aggregation stages"
        
        try:
            # Load user configuration
            config = await self._load_mongodb_config(user_id)
            if not config:
                return "❌ MongoDB not configured. Please run mongodb_setup() first."
            
            # Create client and get collection
            client = self._create_mongodb_client(config)
            if not client:
                return "❌ Failed to connect to MongoDB"
            
            try:
                db = client[database]
                coll = db[collection]
                
                # Execute aggregation pipeline
                results = list(coll.aggregate(pipeline))
                serialized_results = self._serialize_mongodb_result(results)
                
                return f"✅ Aggregation pipeline executed successfully!\n📊 Results ({len(results)} documents):\n{json.dumps(serialized_results, indent=2, default=str)}"
                
            finally:
                client.close()
            
        except Exception as e:
            return f"❌ Aggregation pipeline failed: {str(e)}"
    
    @tool(description="Check MongoDB connection status and configuration")
    async def mongodb_status(self) -> str:
        """Check MongoDB connection and configuration status"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        try:
            # Load configuration
            config = await self._load_mongodb_config(user_id)
            
            result = ["📋 MongoDB Status:\n"]
            
            if config:
                connection_string = config.get('connection_string', '')
                created_at = config.get('created_at', 'unknown')
                
                # Determine connection type
                if 'mongodb.net' in connection_string:
                    db_type = "MongoDB Atlas"
                elif 'localhost' in connection_string or '127.0.0.1' in connection_string:
                    db_type = "Local MongoDB"
                else:
                    db_type = "MongoDB"
                
                result.append(f"✅ Configuration: Active")
                result.append(f"🗄️ Database Type: {db_type}")
                result.append(f"🕐 Created: {created_at}")
                
                # Test connection
                try:
                    client = self._create_mongodb_client(config)
                    if client:
                        client.admin.command('ping')
                        
                        # Get server info
                        server_info = client.server_info()
                        version = server_info.get('version', 'unknown')
                        
                        client.close()
                        result.append("🟢 Connection: Active")
                        result.append(f"📦 MongoDB Version: {version}")
                    else:
                        result.append("🔴 Connection: Failed")
                
                except Exception as e:
                    result.append(f"🟡 Connection: Warning - {str(e)}")
                
            else:
                result.append("❌ No configuration found")
                result.append("💡 Use mongodb_setup() to configure your MongoDB connection")
                return "\n".join(result)
            
            result.append("\n💡 Use mongodb_query() for CRUD operations or mongodb_aggregate() for pipelines")
            
            return "\n".join(result)
            
        except Exception as e:
            return f"❌ Error checking status: {str(e)}"
