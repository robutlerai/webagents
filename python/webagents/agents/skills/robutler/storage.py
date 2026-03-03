"""
WebAgentsStorageSkill - Portal Content Integration

Provides integration with the WebAgents portal's content storage system
for persistent file storage tied to user accounts.
"""

import httpx
import json
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from ..base import Skill
from ...tools.decorators import tool


class WebAgentsStorageSkill(Skill):
    """
    WebAgents portal storage integration skill.
    
    Features:
    - Store/retrieve files in user's private content area
    - Integration with portal authentication
    - Support for JSON data storage
    - User-scoped file management
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.portal_url = config.get('portal_url', 'http://localhost:3000') if config else 'http://localhost:3000'
        self.api_key = config.get('api_key') if config else None
        self.agent_name = config.get('agent_name', 'default_agent') if config else 'default_agent'

    async def initialize(self, agent):
        """Initialize with agent reference"""
        await super().initialize(agent)
        self.agent = agent
        
        # Use api_key as priority, fallback to agent's API key
        self.api_key = self.api_key or getattr(agent, 'api_key', None)
        
        # Extract agent name from agent reference if available
        if hasattr(agent, 'name'):
            self.agent_name = agent.name

    @tool
    async def store_json_data(
        self,
        filename: str,
        data: Dict[str, Any],
        description: Optional[str] = None
    ) -> str:
        """
        Store JSON data in user's private content area.
        
        Args:
            filename: Name of the file (will add .json if not present)
            data: JSON-serializable data to store
            description: Optional description of the file
            
        Returns:
            JSON string with storage result
        """
        try:
            # Ensure filename has .json extension
            if not filename.endswith('.json'):
                filename = f"{filename}.json"
            
            # Convert data to JSON string
            json_content = json.dumps(data, indent=2)
            json_bytes = json_content.encode('utf-8')
            
            # Prepare form data for upload
            files = {
                'file': (filename, json_bytes, 'application/json')
            }
            
            form_data = {
                'visibility': 'private',
                'description': description or f"JSON data storage for {self.agent_name}",
                'tags': json.dumps(['agent_data', self.agent_name])
            }
            
            # Upload to portal
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.portal_url}/api/content",
                    files=files,
                    data=form_data,
                    headers={'Authorization': f'Bearer {self.api_key}'}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return json.dumps({
                        "success": True,
                        "file_id": result['file']['id'],
                        "filename": result['file']['fileName'],
                        "url": result['file']['url'],
                        "size": result['file']['size']
                    }, indent=2)
                else:
                    return json.dumps({
                        "success": False,
                        "error": f"Upload failed: {response.status_code} - {response.text}"
                    })
                    
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Failed to store JSON data: {str(e)}"
            })

    @tool
    async def retrieve_json_data(self, filename: str) -> str:
        """
        Retrieve JSON data from user's private content area.
        
        Args:
            filename: Name of the file to retrieve
            
        Returns:
            JSON string with file content or error
        """
        try:
            # Ensure filename has .json extension
            if not filename.endswith('.json'):
                filename = f"{filename}.json"
            
            async with httpx.AsyncClient() as client:
                # First, list agent's files to find the file
                response = await client.get(
                    f"{self.portal_url}/api/content/agent",
                    headers={'Authorization': f'Bearer {self.api_key}'},
                    params={}  # Use explicit access via content_access table
                )
                
                if response.status_code != 200:
                    return json.dumps({
                        "success": False,
                        "error": f"Failed to list content: {response.status_code}"
                    })
                
                files_data = response.json()
                target_file = None
                
                # Find the target file
                for file_info in files_data.get('content', []):  # Agent API uses 'content' not 'files'
                    if file_info['fileName'] == filename or file_info['originalFileName'] == filename:
                        target_file = file_info
                        break
                
                if not target_file:
                    return json.dumps({
                        "success": False,
                        "error": f"File '{filename}' not found",
                        "available_files": [f['fileName'] for f in files_data.get('content', [])]
                    })
                
                # Retrieve file content
                file_response = await client.get(
                    target_file['url'],
                    headers={'Authorization': f'Bearer {self.api_key}'}
                )
                
                if file_response.status_code == 200:
                    # Parse JSON content
                    content_data = file_response.json()
                    return json.dumps({
                        "success": True,
                        "filename": target_file['fileName'],
                        "data": content_data,
                        "metadata": {
                            "size": target_file['size'],
                            "uploaded_at": target_file['uploadedAt'],
                            "description": target_file.get('description')
                        }
                    }, indent=2)
                else:
                    return json.dumps({
                        "success": False,
                        "error": f"Failed to retrieve file content: {file_response.status_code}"
                    })
                    
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Failed to retrieve JSON data: {str(e)}"
            })

    @tool
    async def update_json_data(
        self,
        filename: str,
        data: Dict[str, Any],
        description: Optional[str] = None
    ) -> str:
        """
        Update existing JSON data in user's content area.
        
        Args:
            filename: Name of the file to update
            data: New JSON data
            description: Optional new description
            
        Returns:
            JSON string with update result
        """
        try:
            # First delete the old file, then upload new one
            delete_result = await self.delete_file(filename)
            delete_data = json.loads(delete_result)
            
            if not delete_data.get("success"):
                # File might not exist, that's okay for update operation
                pass
            
            # Upload new version
            return await self.store_json_data(filename, data, description)
            
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Failed to update JSON data: {str(e)}"
            })

    @tool
    async def delete_file(self, filename: str) -> str:
        """
        Delete a file from user's content area.
        
        Args:
            filename: Name of the file to delete
            
        Returns:
            JSON string with deletion result
        """
        try:
            # Ensure filename has .json extension
            if not filename.endswith('.json'):
                filename = f"{filename}.json"
            
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.portal_url}/api/content",
                    headers={'Authorization': f'Bearer {self.api_key}'},
                    params={'fileName': filename}
                )
                
                if response.status_code == 200:
                    return json.dumps({
                        "success": True,
                        "message": f"File '{filename}' deleted successfully"
                    })
                else:
                    result = response.json() if response.headers.get('content-type', '').startswith('application/json') else {"error": response.text}
                    return json.dumps({
                        "success": False,
                        "error": result.get('error', f"Delete failed: {response.status_code}")
                    })
                    
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Failed to delete file: {str(e)}"
            })

    @tool
    async def list_agent_files(self) -> str:
        """
        List all files associated with this agent.
        
        Returns:
            JSON string with file list
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.portal_url}/api/content/agent",
                    headers={'Authorization': f'Bearer {self.api_key}'},
                    params={}
                )
                
                if response.status_code == 200:
                    files_data = response.json()
                    
                    # Agent content API already filters to agent-accessible files
                    agent_files = []
                    for file_info in files_data.get('content', []):  # Note: agent API uses 'content' not 'files'
                        agent_files.append({
                            "filename": file_info['fileName'],
                            "size": file_info['size'],
                            "uploaded_at": file_info['uploadedAt'],
                            "description": file_info.get('description'),
                            "tags": file_info.get('tags', [])
                        })
                    
                    return json.dumps({
                        "success": True,
                        "agent_name": self.agent_name,
                        "total_files": len(agent_files),
                        "files": agent_files
                    }, indent=2)
                else:
                    return json.dumps({
                        "success": False,
                        "error": f"Failed to list files: {response.status_code}"
                    })
                    
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Failed to list agent files: {str(e)}"
            })

    @tool
    async def get_storage_stats(self) -> str:
        """
        Get storage statistics for this agent.
        
        Returns:
            JSON string with storage statistics
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.portal_url}/api/content/agent",
                    headers={'Authorization': f'Bearer {self.api_key}'},
                    params={}
                )
                
                if response.status_code == 200:
                    files_data = response.json()
                    
                    # Calculate stats for this agent
                    agent_files = files_data.get('content', [])  # Agent API already filters to agent-accessible files
                    total_size = 0
                    
                    for file_info in agent_files:
                        total_size += file_info.get('size', 0)
                    
                    return json.dumps({
                        "agent_name": self.agent_name,
                        "total_files": len(agent_files),
                        "total_size_bytes": total_size,
                        "total_size_mb": round(total_size / (1024 * 1024), 2),
                        "portal_url": self.portal_url,
                        "storage_location": "webagents_portal_content"
                    }, indent=2)
                else:
                    return json.dumps({
                        "success": False,
                        "error": f"Failed to get storage stats: {response.status_code}"
                    })
                    
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Failed to get storage stats: {str(e)}"
            })

    def get_skill_info(self) -> Dict[str, Any]:
        """Get comprehensive skill information"""
        return {
            "name": "WebAgentsStorageSkill",
            "description": "Portal content storage integration for persistent data",
            "version": "1.0.0",
            "capabilities": [
                "Store JSON data in user's private content area",
                "Retrieve data from portal storage",
                "Update and delete stored files",
                "Agent-scoped file management",
                "Integration with portal authentication"
            ],
            "tools": [
                "store_json_data",
                "retrieve_json_data",
                "update_json_data",
                "delete_file",
                "list_agent_files",
                "get_storage_stats"
            ],
            "config": {
                "portal_url": self.portal_url,
                "agent_name": self.agent_name,
                "api_key_configured": bool(self.api_key)
            }
        } 