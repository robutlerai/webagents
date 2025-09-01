"""
Minimalistic n8n Skill for WebAgents

This skill allows users to:
- Set up n8n API key securely (via auth/kv skills)
- Execute n8n workflows 
- List available workflows
- Get workflow execution status

Uses auth skill for user context and kv skill for secure API key storage.
"""

import os
import httpx
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, prompt
from webagents.server.context.context_vars import get_context


class N8nSkill(Skill):
    """Minimalistic n8n skill for workflow automation"""
    
    def __init__(self):
        super().__init__()
        self.default_n8n_url = os.getenv('N8N_BASE_URL', 'http://localhost:5678')
        
    def get_dependencies(self) -> List[str]:
        """Skill dependencies"""
        return ['auth', 'kv']
    
    @prompt(priority=40, scope=["owner", "all"])
    def n8n_prompt(self) -> str:
        """Prompt describing n8n capabilities"""
        return """
Minimalistic n8n integration for workflow automation. Available tools:

• n8n_setup(api_key, base_url) - Set up n8n API credentials securely
• n8n_execute(workflow_id, data) - Execute a specific workflow with optional input data  
• n8n_list_workflows() - List all available workflows in your n8n instance
• n8n_status(execution_id) - Check the status of a workflow execution

Features:
- Secure API key storage via KV skill
- Per-user credential isolation via Auth skill  
- Execute workflows with custom input data
- Monitor workflow execution status
- List and discover available workflows

Setup: First run n8n_setup() with your n8n API key and instance URL.
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
    
    async def _save_n8n_credentials(self, user_id: str, api_key: str, base_url: str) -> bool:
        """Save n8n credentials securely using KV skill"""
        try:
            kv_skill = await self._get_kv_skill()
            if kv_skill:
                credentials = {
                    'api_key': api_key,
                    'base_url': base_url,
                    'created_at': datetime.now().isoformat()
                }
                await kv_skill.kv_set(
                    key='credentials',
                    value=json.dumps(credentials),
                    namespace=f'n8n:{user_id}'
                )
                return True
            else:
                # Fallback to in-memory storage
                if not hasattr(self.agent, '_n8n_credentials'):
                    self.agent._n8n_credentials = {}
                self.agent._n8n_credentials[user_id] = {
                    'api_key': api_key,
                    'base_url': base_url
                }
                return True
        except Exception as e:
            self.logger.error(f"Failed to save n8n credentials: {e}")
            return False
    
    async def _load_n8n_credentials(self, user_id: str) -> Optional[Dict[str, str]]:
        """Load n8n credentials from KV skill"""
        try:
            kv_skill = await self._get_kv_skill()
            if kv_skill:
                credentials_json = await kv_skill.kv_get(
                    key='credentials',
                    namespace=f'n8n:{user_id}'
                )
                if credentials_json:
                    return json.loads(credentials_json)
            else:
                # Fallback to in-memory storage
                if hasattr(self.agent, '_n8n_credentials'):
                    return self.agent._n8n_credentials.get(user_id)
            return None
        except Exception as e:
            self.logger.error(f"Failed to load n8n credentials: {e}")
            return None
    
    async def _make_n8n_request(self, method: str, endpoint: str, data: Optional[Dict] = None, user_id: str = None) -> Dict[str, Any]:
        """Make authenticated request to n8n API"""
        if not user_id:
            user_id = await self._get_authenticated_user_id()
            if not user_id:
                raise Exception("Authentication required")
        
        credentials = await self._load_n8n_credentials(user_id)
        if not credentials:
            raise Exception("n8n credentials not found. Please run n8n_setup() first.")
        
        api_key = credentials['api_key']
        base_url = credentials['base_url'].rstrip('/')
        
        headers = {
            'X-N8N-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
        
        url = f"{base_url}/api/v1{endpoint}"
        
        async with httpx.AsyncClient() as client:
            if method.upper() == 'GET':
                response = await client.get(url, headers=headers)
            elif method.upper() == 'POST':
                response = await client.post(url, headers=headers, json=data)
            elif method.upper() == 'PUT':
                response = await client.put(url, headers=headers, json=data)
            elif method.upper() == 'DELETE':
                response = await client.delete(url, headers=headers)
            else:
                raise Exception(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json() if response.content else {}

    # Public tools
    @tool(description="Set up n8n API credentials securely. Get your API key from n8n Settings > n8n API.", scope="owner")
    async def n8n_setup(self, api_key: str, base_url: str = None) -> str:
        """Set up n8n API credentials for secure access"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        if not api_key or not api_key.strip():
            return "❌ API key is required. Generate one from n8n Settings > n8n API."
        
        # Use provided base_url or default
        n8n_url = base_url or self.default_n8n_url
        if not n8n_url.startswith(('http://', 'https://')):
            n8n_url = f"https://{n8n_url}"
        
        try:
            # Test the API key by making a simple request
            await self._make_n8n_request('GET', '/workflows', user_id=user_id)
            
            # If test succeeds, save credentials
            success = await self._save_n8n_credentials(user_id, api_key.strip(), n8n_url)
            
            if success:
                return f"✅ n8n credentials saved successfully!\n🌐 Base URL: {n8n_url}\n🔑 API key configured"
            else:
                return "❌ Failed to save credentials"
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return "❌ Invalid API key. Please check your n8n API key."
            elif e.response.status_code == 404:
                return f"❌ n8n instance not found at {n8n_url}. Please check the URL."
            else:
                return f"❌ API test failed: HTTP {e.response.status_code}"
        except Exception as e:
            return f"❌ Setup failed: {str(e)}"
    
    @tool(description="Execute an n8n workflow with optional input data")
    async def n8n_execute(self, workflow_id: str, data: Dict[str, Any] = None) -> str:
        """Execute an n8n workflow with optional input data"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        if not workflow_id or not workflow_id.strip():
            return "❌ Workflow ID is required"
        
        try:
            # Prepare execution data
            execution_data = {}
            if data:
                execution_data = data
            
            # Execute the workflow
            response = await self._make_n8n_request(
                'POST', 
                f'/workflows/{workflow_id.strip()}/execute',
                execution_data,
                user_id
            )
            
            execution_id = response.get('id', 'unknown')
            status = response.get('status', 'unknown')
            
            return f"✅ Workflow executed successfully!\n📋 Execution ID: {execution_id}\n📊 Status: {status}"
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return "❌ Authentication failed. Please run n8n_setup() again."
            elif e.response.status_code == 404:
                return f"❌ Workflow '{workflow_id}' not found"
            else:
                return f"❌ Execution failed: HTTP {e.response.status_code}"
        except Exception as e:
            return f"❌ Error executing workflow: {str(e)}"
    
    @tool(description="List all available workflows in your n8n instance")
    async def n8n_list_workflows(self) -> str:
        """List all available workflows"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        try:
            response = await self._make_n8n_request('GET', '/workflows', user_id=user_id)
            
            workflows = response.get('data', [])
            
            if not workflows:
                return "📭 No workflows found in your n8n instance"
            
            result = ["📋 Available n8n Workflows:\n"]
            
            for workflow in workflows:
                workflow_id = workflow.get('id', 'unknown')
                name = workflow.get('name', 'Unnamed')
                active = workflow.get('active', False)
                status_icon = "🟢" if active else "🔴"
                
                result.append(f"{status_icon} **{name}** (ID: {workflow_id})")
                
                # Add tags if available
                tags = workflow.get('tags', [])
                if tags:
                    tag_names = [tag.get('name', 'Unknown') for tag in tags]
                    result.append(f"   🏷️ Tags: {', '.join(tag_names)}")
                
                result.append("")  # Empty line for spacing
            
            result.append("💡 Use n8n_execute(workflow_id, data) to run a workflow")
            
            return "\n".join(result)
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return "❌ Authentication failed. Please run n8n_setup() again."
            else:
                return f"❌ Failed to list workflows: HTTP {e.response.status_code}"
        except Exception as e:
            return f"❌ Error listing workflows: {str(e)}"
    
    @tool(description="Check the status of a workflow execution")
    async def n8n_status(self, execution_id: str) -> str:
        """Check the status of a workflow execution"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        if not execution_id or not execution_id.strip():
            return "❌ Execution ID is required"
        
        try:
            response = await self._make_n8n_request(
                'GET', 
                f'/executions/{execution_id.strip()}',
                user_id=user_id
            )
            
            status = response.get('status', 'unknown')
            workflow_id = response.get('workflowId', 'unknown')
            start_time = response.get('startedAt', 'unknown')
            end_time = response.get('stoppedAt', 'running')
            
            # Status icons
            status_icons = {
                'success': '✅',
                'error': '❌', 
                'running': '🔄',
                'waiting': '⏳',
                'canceled': '🚫'
            }
            
            status_icon = status_icons.get(status.lower(), '❓')
            
            result = [
                f"📊 Execution Status Report",
                f"🆔 Execution ID: {execution_id}",
                f"🔧 Workflow ID: {workflow_id}",
                f"{status_icon} Status: {status}",
                f"🕐 Started: {start_time}",
                f"🕑 Finished: {end_time}"
            ]
            
            # Add error details if execution failed
            if status.lower() == 'error' and 'data' in response:
                error_data = response.get('data', {})
                if 'resultData' in error_data:
                    result.append(f"❌ Error details available in execution data")
            
            return "\n".join(result)
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return "❌ Authentication failed. Please run n8n_setup() again."
            elif e.response.status_code == 404:
                return f"❌ Execution '{execution_id}' not found"
            else:
                return f"❌ Status check failed: HTTP {e.response.status_code}"
        except Exception as e:
            return f"❌ Error checking status: {str(e)}"
