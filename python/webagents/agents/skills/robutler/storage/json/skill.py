"""
RobutlerJSONSkill - JSON Data Storage for Long-term Memory

Provides JSON data storage capabilities for agent long-term memory and persistent data.
"""

import json
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

import aiohttp

from ....base import Skill
from webagents.agents.tools.decorators import tool
from robutler.api.client import RobutlerClient

UPLOAD_PATH = "/api/content/upload"


class RobutlerJSONSkill(Skill):
    """
    WebAgents portal JSON storage skill for long-term memory using RobutlerClient.
    
    Features:
    - Store/retrieve JSON data for long-term memory
    - Update and manage JSON data files
    - Context-based scope access controls
    - Agent identity determined by API key
    
    Scope Configurations (from auth context):
    - 'owner' scope: Full access to all JSON operations
    - 'all' scope: No access (restricted)
    
    Tool Scope Restrictions:
    - store_json_data: owner scope only
    - retrieve_json_data: owner scope only
    - update_json_data: owner scope only
    - delete_json_file: owner scope only
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        cfg = config or {}
        self.portal_url = (
            cfg.get('portal_url')
            or os.getenv('ROBUTLER_INTERNAL_API_URL')
            or os.getenv('ROBUTLER_API_URL')
            or 'http://localhost:3000'
        )
        # No 'rok_testapikey' placeholder: a fake key produced a silently
        # unauthenticated client whose first visible failure was never the
        # real one (F-041 class).
        self.api_key = cfg.get('api_key') or os.getenv('WEBAGENTS_API_KEY')

        # RobutlerClient still serves the read/update/delete paths.
        self.client = RobutlerClient(
            api_key=self.api_key,
            base_url=self.portal_url
        )

    async def initialize(self, agent_reference):
        """Initialize with agent reference"""
        await super().initialize(agent_reference)
        self.agent = agent_reference
        
    async def cleanup(self):
        """Cleanup method to close client session"""
        if self.client:
            await self.client.close()

    def _get_scope_from_context(self) -> str:
        """
        Get scope from authentication context.
        
        Returns:
            'owner' or 'all' based on auth context
        """
        try:
            from webagents.server.context.context_vars import get_context
            context = get_context()
            if context:
                # Check auth scope from context
                auth_scope = context.get('auth_scope') or context.get('scope')
                if auth_scope == 'owner':
                    return 'owner'
            return 'all'  # Default to 'all' scope
        except Exception:
            return 'all'  # Fallback to 'all' scope

    def _get_storage_visibility(self) -> str:
        """
        Get the appropriate visibility for storing content based on scope.
        
        Returns:
            - 'private' for owner scope (private agent storage)
            - 'public' for all scope (public agent storage)
        """
        scope = self._get_scope_from_context()
        if scope == 'owner':
            return 'private'  # Store privately for owner
        else:  # 'all' scope
            return 'public'  # Store publicly for general access
    
    def _get_access_visibility(self) -> str:
        """
        Get the appropriate visibility for accessing content based on scope.
        
        Returns:
            - 'both' for owner scope (access to public and private content)
            - 'public' for all scope (access to only public content)
        """
        scope = self._get_scope_from_context()
        if scope == 'owner':
            return 'both'  # Access to both public and private content
        else:  # 'all' scope
            return 'public'  # Access to only public content

    @tool(scope="owner")
    async def store_json_data(
        self,
        filename: str,
        data: Dict[str, Any],
        description: Optional[str] = None
    ) -> str:
        """
        Store JSON data for long-term memory.
        
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
            
            # Convert data to JSON string and bytes
            json_content = json.dumps(data, indent=2)
            json_bytes = json_content.encode('utf-8')
            
            # Direct multipart POST to /api/content/upload. The client's
            # upload_content posts to /api/content, which answers 405, and
            # its `response.error` is a CONSTANT, so the old
            # `error or message` expression collapsed every failure to
            # "Upload failed: Upload failed" (F-041).
            url = f"{self.portal_url}{UPLOAD_PATH}"
            form = aiohttp.FormData()
            form.add_field('file', json_bytes, filename=filename, content_type='application/json')
            form.add_field('visibility', self._get_storage_visibility())
            headers = {"Authorization": f"Bearer {self.api_key}"}
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=form, headers=headers) as response:
                    body = await response.text()
                    if response.status != 200:
                        return json.dumps({
                            "success": False,
                            "error": f"Upload failed: HTTP {response.status} POST {UPLOAD_PATH} - {body[:300]}"
                        })
                    file_info = json.loads(body) if body else {}

            return json.dumps({
                "success": True,
                "file_id": file_info.get('id'),
                "filename": file_info.get('displayName'),
                "url": file_info.get('url'),
                "size": file_info.get('size')
            }, indent=2)
                    
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Failed to store JSON data: {str(e)}"
            })

    @tool(scope="owner")
    async def retrieve_json_data(self, filename: str) -> str:
        """
        Retrieve JSON data from long-term memory.
        
        Args:
            filename: Name of the file to retrieve
            
        Returns:
            JSON string with file content or error
        """
        try:
            # Ensure filename has .json extension
            if not filename.endswith('.json'):
                filename = f"{filename}.json"
            
            # Get content using RobutlerClient
            response = await self.client.get_content(filename)
            
            if response.success and response.data:
                return json.dumps({
                    "success": True,
                    "filename": response.data.get('filename'),
                    "data": response.data.get('content'),
                    "metadata": response.data.get('metadata', {})
                }, indent=2)
            else:
                # Get available files for better error message
                list_response = await self.client.list_content(visibility=self._get_access_visibility())
                available_files = []
                if list_response.success and list_response.data:
                    files = list_response.data.get('files', [])
                    for file_info in files:
                        if 'json_data' in file_info.get('tags', []):
                            available_files.append(file_info.get('fileName', ''))
                
                return json.dumps({
                    "success": False,
                    "error": response.error or response.message or "File not found",
                    "available_json_files": available_files
                })
                    
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Failed to retrieve JSON data: {str(e)}"
            })

    @tool(scope="owner")
    async def update_json_data(
        self,
        filename: str,
        data: Dict[str, Any],
        description: Optional[str] = None
    ) -> str:
        """
        Update existing JSON data in long-term memory.
        
        Args:
            filename: Name of the file to update
            data: New JSON data
            description: Optional new description
            
        Returns:
            JSON string with update result
        """
        try:
            # Ensure filename has .json extension
            if not filename.endswith('.json'):
                filename = f"{filename}.json"
            
            # Convert data to JSON string and bytes
            json_content = json.dumps(data, indent=2)
            json_bytes = json_content.encode('utf-8')
            
            # Update using RobutlerClient
            response = await self.client.update_content(
                filename=filename,
                content_data=json_bytes,
                content_type='application/json',
                description=description or "Updated JSON data for long-term memory",
                tags=['json_data', 'memory'],
                visibility=self._get_storage_visibility()
            )
            
            if response.success and response.data:
                file_info = response.data.get('file', {})
                return json.dumps({
                    "success": True,
                    "file_id": file_info.get('id'),
                    "filename": file_info.get('fileName'),
                    "url": file_info.get('url'),
                    "size": file_info.get('size')
                }, indent=2)
            else:
                return json.dumps({
                    "success": False,
                    "error": f"Update failed: {response.error or response.message}"
                })
            
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Failed to update JSON data: {str(e)}"
            })

    @tool(scope="owner")
    async def delete_json_file(self, filename: str) -> str:
        """
        Delete a JSON file from long-term memory.
        
        Args:
            filename: Name of the file to delete
            
        Returns:
            JSON string with deletion result
        """
        try:
            # Ensure filename has .json extension
            if not filename.endswith('.json'):
                filename = f"{filename}.json"
            
            # Delete using RobutlerClient
            response = await self.client.delete_content(filename)
            
            if response.success:
                return json.dumps({
                    "success": True,
                    "message": f"JSON file '{filename}' deleted successfully"
                })
            else:
                return json.dumps({
                    "success": False,
                    "error": response.error or response.message or f"Delete failed"
                })
                    
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Failed to delete JSON file: {str(e)}"
            })

    def get_skill_info(self) -> Dict[str, Any]:
        """Skill information; `tools` is derived from the live registry by the
        base class so the advertised and registered surfaces cannot drift."""
        info = super().get_skill_info()
        info.update({
            "name": "RobutlerJSONSkill",
            "description": "JSON data storage for long-term memory",
            "capabilities": [
                "Store JSON data for long-term memory (owner scope only)",
                "Retrieve JSON data from memory (owner scope only)",
                "Update and manage JSON files (owner scope only)",
                "Delete JSON files (owner scope only)",
                "Agent identity determined by API key",
                "Integration with portal authentication via RobutlerClient",
                "Owner scope: Full access to JSON operations",
                "All scope: No access (restricted)"
            ],
            "config": {
                "portal_url": self.portal_url,
                "api_key_configured": bool(self.api_key),
                "client_type": "RobutlerClient",
                "scope": self._get_scope_from_context(),
                "storage_visibility": self._get_storage_visibility(),
                "access_visibility": self._get_access_visibility()
            }
        })
        return info 