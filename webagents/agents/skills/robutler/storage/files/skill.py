"""
RobutlerFilesSkill - File Management with Harmonized API
Uses the new harmonized content API for cleaner and more efficient operations.
"""

import json
import os
import base64
import aiohttp
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from ....base import Skill
from webagents.agents.tools.decorators import tool
from robutler.api.client import RobutlerClient
from webagents.agents.skills.robutler.payments import pricing, PricingInfo

class RobutlerFilesSkill(Skill):
    """
    WebAgents portal file management skill using harmonized API.
    
    Features:
    - Download and store files from URLs
    - Store files from base64 data
    - List files with agent-based access
    - Agent access is automatically handled by the API
    
    Uses the new /api/content/agent endpoints for agent operations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.portal_url = config.get('portal_url', 'http://localhost:3000') if config else 'http://localhost:3000'
        # Base URL used by the chat frontend to serve public content
        self.chat_base_url = (config.get('chat_base_url') if config else None) or os.getenv('ROBUTLER_CHAT_URL', 'http://localhost:3001')
        self.api_key = config.get('api_key', os.getenv('WEBAGENTS_API_KEY', 'rok_testapikey')) if config else os.getenv('WEBAGENTS_API_KEY', 'rok_testapikey')
        
        # Initialize RobutlerClient
        self.client = RobutlerClient(
            api_key=self.api_key,
            base_url=self.portal_url
        )

    async def initialize(self, agent_reference):
        """Initialize with agent reference"""
        await super().initialize(agent_reference)
        self.agent = agent_reference
        
        # Check if agent has its own API key
        if hasattr(agent_reference, 'api_key') and agent_reference.api_key:
            self.agent_api_key = agent_reference.api_key
            
            # Debug logging for agent API key
            agent_key_prefix = self.agent_api_key[:20] + "..." if len(self.agent_api_key) > 20 else self.agent_api_key
            print(f"🔑 Storage skill using agent API key: {agent_key_prefix}")
            
            # Create a separate client for agent operations
            self.agent_client = RobutlerClient(
                api_key=self.agent_api_key,
                base_url=self.portal_url
            )
        else:
            # Fall back to user API key
            self.agent_api_key = self.api_key
            self.agent_client = self.client
            
            # Debug logging for fallback
            user_key_prefix = self.api_key[:20] + "..." if len(self.api_key) > 20 else self.api_key
            print(f"🔑 Storage skill using user API key (fallback): {user_key_prefix}")
        
    async def cleanup(self):
        """Cleanup method to close client sessions"""
        if self.client:
            await self.client.close()
        
        # Close agent client if it's different from the main client
        if hasattr(self, 'agent_client') and self.agent_client != self.client:
            await self.agent_client.close()

    def _get_agent_name_from_context(self) -> str:
        """
        Get the current agent name from context.
        
        Returns:
            Agent name (e.g., 'van-gogh') or empty string if not found
        """
        try:
            from webagents.server.context.context_vars import get_agent_name
            
            # Use the utility function
            agent_name = get_agent_name()
            if agent_name:
                return agent_name
                
            # Fallback: try to get from agent instance
            if hasattr(self, 'agent') and hasattr(self.agent, 'name'):
                return self.agent.name or ''
                        
            return ''  # Default to empty string
        except Exception:
            return ''  # Fallback to empty string

    def _rewrite_public_url(self, url: Optional[str]) -> Optional[str]:
        """Rewrite portal public content URLs to chat base URL.
        Examples:
          http://localhost:3000/api/content/public/.. -> http://localhost:3001/api/content/public/..
          /api/content/public/... stays relative and gets chat base prefixed when rendered client-side
        """
        if not url:
            return url
        try:
            if url.startswith('/api/content/public'):
                # Already relative; prefix with chat base for clarity
                return f"{self.chat_base_url}{url}"
            portal_prefix = f"{self.portal_url}/api/content/public"
            if url.startswith(portal_prefix):
                return url.replace(self.portal_url, self.chat_base_url, 1)
        except Exception:
            return url
        return url

    # @tool(scope="owner")
    async def store_file_from_url(
        self,
        url: str,
        filename: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        visibility: str = "private"
    ) -> str:
        """
        A tool for downloading and storing a file from a URL. Never use this tool for files that you already own, e.g. URLs returned by list_files.
        
        Args:
            url: URL to download file from
            filename: Optional custom filename (auto-detected if not provided)
            description: Optional description of the file
            tags: Optional list of tags for the file
            visibility: File visibility - "public", "private", or "shared" (default: "private")
            
        Returns:
            JSON string with storage result
        """
        try:
            # Download file from URL
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        return json.dumps({
                            "success": False,
                            "error": f"Failed to download file: HTTP {response.status}"
                        })
                    
                    content_data = await response.read()
                    content_type = response.headers.get('content-type', 'application/octet-stream')
                    
                    # Auto-detect filename if not provided
                    if not filename:
                        filename = url.split('/')[-1] or 'downloaded_file'
                        # Remove query parameters
                        filename = filename.split('?')[0]
            
            # Get agent name for filename prefixing
            agent_name = self._get_agent_name_from_context()
            
            # Prefix filename with agent name if available
            if agent_name and not filename.startswith(f"{agent_name}_"):
                filename = f"{agent_name}_{filename}"
            
            # Store file using RobutlerClient with new API
            response = await self.client.upload_content(
                filename=filename,
                content_data=content_data,
                content_type=content_type,
                visibility=visibility,
                description=description or f"File downloaded from {url} by {agent_name or 'agent'}",
                tags=tags
            )
            
            if response.success and response.data:
                return json.dumps({
                    "success": True,
                    "id": response.data.get('id'),
                    "filename": response.data.get('fileName'),
                    "url": self._rewrite_public_url(response.data.get('url')),
                    "size": response.data.get('size'),
                    "content_type": content_type,
                    "visibility": visibility,
                    "source_url": url
                }, indent=2)
            else:
                return json.dumps({
                    "success": False,
                    "error": f"Upload failed: {response.error or response.message}"
                })
                    
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Failed to store file from URL: {str(e)}"
            })

    # @tool(scope="owner")
    async def store_file_from_base64(
        self,
        filename: str,
        base64_data: str,
        content_type: str = "application/octet-stream",
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        visibility: str = "private"
    ) -> str:
        """
        A tool for storing a file from base64 encoded data.
        
        Args:
            filename: Name of the file
            base64_data: Base64 encoded file content
            content_type: MIME type of the file
            description: Optional description of the file
            tags: Optional list of tags for the file
            visibility: File visibility - "public", "private", or "shared" (default: "private")
            
        Returns:
            JSON string with storage result
        """
        try:
            # Decode base64 data
            content_data = base64.b64decode(base64_data)
            
            # Get agent name for filename prefixing
            agent_name = self._get_agent_name_from_context()
            
            # Prefix filename with agent name if available
            if agent_name and not filename.startswith(f"{agent_name}_"):
                filename = f"{agent_name}_{filename}"
            
            # Store file using RobutlerClient with new API
            response = await self.client.upload_content(
                filename=filename,
                content_data=content_data,
                content_type=content_type,
                visibility=visibility,
                description=description or f"File uploaded from base64 data by {agent_name or 'agent'}",
                tags=tags
            )
            
            if response.success and response.data:
                return json.dumps({
                    "success": True,
                    "id": response.data.get('id'),
                    "filename": response.data.get('fileName'),
                    "url": self._rewrite_public_url(response.data.get('url')),
                    "size": response.data.get('size'),
                    "content_type": content_type,
                    "visibility": visibility
                }, indent=2)
            else:
                return json.dumps({
                    "success": False,
                    "error": f"Upload failed: {response.error or response.message}"
                })
                    
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Failed to store file from base64: {str(e)}"
            })

    @tool
    @pricing(credits_per_call=0.005)
    async def list_files(
        self,
        scope: Optional[str] = None
    ) -> str:
        """
        List files accessible by the current agent with scope-based filtering.
        
        The behavior depends on who is calling:
        - Agent owner calling "show all files" (scope=None): Returns all private + public agent files
        - Agent owner calling "show public files" (scope="public"): Returns only public agent files  
        - Agent owner calling "show private files" (scope="private"): Returns only private agent files
        - Non-owner calling: Always returns only public agent files regardless of scope
        
        Args:
            scope: Optional scope filter - "public", "private", or None (all files for owner)
            
        Returns:
            JSON string with file list based on scope and ownership
        """
        try:
            from webagents.server.context.context_vars import get_context
            from webagents.utils.logging import get_logger
            logger = get_logger('webagents_files')
            
            # Get context for agent information and auth
            context = get_context()
            if not context:
                return json.dumps({
                    "success": False,
                    "error": "Agent context not available"
                })
            
            # Debug context attributes
            print(f"🔍 DEBUG: Context available, type: {type(context)}")
            print(f"🔍 DEBUG: Context attributes: {[attr for attr in dir(context) if not attr.startswith('_')]}")
            if hasattr(context, 'custom_data'):
                print(f"🔍 DEBUG: Context custom_data keys: {list(context.custom_data.keys())}")

            agent_name = self._get_agent_name_from_context()
            
            # Determine if current user is the actual owner of the agent
            # SECURITY: Only actual owners should see private content, not just ADMINs
            is_owner = False
            try:
                print(f"🔍 DEBUG: Determining actual ownership...")
                
                # Check if we have auth context with user info
                current_user_id = None
                if context.auth and hasattr(context.auth, 'user_id'):
                    current_user_id = context.auth.user_id
                    print(f"🔍 DEBUG: Current user ID from auth: {current_user_id}")
                
                # Get agent owner ID
                agent_owner_id = None
                if hasattr(self.agent, 'owner_user_id'):
                    agent_owner_id = self.agent.owner_user_id
                    print(f"🔍 DEBUG: Agent owner ID: {agent_owner_id}")
                elif hasattr(self.agent, 'userId'):
                    agent_owner_id = self.agent.userId
                    print(f"🔍 DEBUG: Agent userId: {agent_owner_id}")
                
                # Check if current user is the actual owner
                if current_user_id and agent_owner_id:
                    is_owner = current_user_id == agent_owner_id
                    print(f"🔍 DEBUG: Ownership check: {current_user_id} == {agent_owner_id} = {is_owner}")
                else:
                    print(f"🔍 DEBUG: Missing user ID or agent owner ID, defaulting to non-owner")
                    is_owner = False
                
                print(f"🔍 DEBUG: Final isOwner determination: {is_owner}")
            except Exception as e:
                print(f"🔍 DEBUG: Error determining ownership: {e}")
                is_owner = False

            # Build URL with query parameters
            url = f"{self.portal_url}/api/content/agent"
            params = []
            
            # Add isOwner parameter for security filtering
            params.append(f"isOwner={str(is_owner).lower()}")
            
            # Add scope parameter for filtering based on ownership and visibility
            if scope:
                params.append(f"scope={scope}")
            
            if params:
                url += "?" + "&".join(params)

            # Make request to new agent content endpoint
            api_key_prefix = self.agent_api_key[:20] + "..." if len(self.agent_api_key) > 20 else self.agent_api_key
            print(f"🔍 DEBUG: Calling /api/content/agent using API key: {api_key_prefix}")
            print(f"🔍 DEBUG: Final URL: {url}")
            print(f"🔍 DEBUG: isOwner parameter being sent: {is_owner}")
            
            # Make request with agent API key
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.agent_api_key}",
                    "Content-Type": "application/json"
                }
                
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Agent content API error: {response.status} - {error_text}")
                        return json.dumps({
                            "success": False,
                            "error": f"Failed to list files: HTTP {response.status}"
                        })
                    
                    data = await response.json()
                    logger.debug(f"API Response status: {response.status}")
                    logger.debug(f"API Response content count: {len(data.get('content', []))}")
                    logger.debug(f"API Response scope info: {data.get('scope', {})}")
                    
                    # Log each file for debugging
                    for item in data.get('content', []):
                        logger.debug(f"File: {item.get('fileName')} - Visibility: {item.get('visibility')}")
                    
                    # Extract relevant fields from response
                    files = []
                    for item in data.get('content', []):
                        files.append({
                            "id": item.get('id'),
                            "filename": item.get('fileName'),
                            "original_filename": item.get('originalFileName'),
                            "size": item.get('size'),
                            "uploaded_at": item.get('uploadedAt'),
                            "description": item.get('description'),
                            "content_type": item.get('contentType'),
                            "url": self._rewrite_public_url(item.get('url')),
                            "visibility": item.get('visibility'),
                            "tags": item.get('tags', [])
                        })
                    
                    return json.dumps({
                        "success": True,
                        "agent_name": data.get('agent', {}).get('name', agent_name),
                        "total_files": len(files),
                        "files": files
                    }, indent=2)
                    
        except Exception as e:
            logger.error(f"Error in list_files: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to list files: {str(e)}"
            })

    def get_skill_info(self) -> Dict[str, Any]:
        """Get comprehensive skill information"""
        return {
            "name": "RobutlerFilesSkill",
            "description": "File management using harmonized content API",
            "version": "1.2.0",
            "capabilities": [
                "Download and store files from URLs (owner scope only)",
                "Store files from base64 data (owner scope only)",
                "List agent-accessible files using new API",
                "Automatic agent name prefixing for uploaded files",
                "Integration with harmonized content API",
                "Simplified agent access management"
            ],
            "tools": [
                "store_file_from_url",
                "store_file_from_base64",
                "list_files"
            ],
            "config": {
                "portal_url": self.portal_url,
                "api_key_configured": bool(self.api_key),
                "api_version": "harmonized"
            }
        }
