"""
Minimalistic Zapier Skill for WebAgents

This skill allows users to:
- Set up Zapier API key securely (via auth/kv skills)
- Trigger Zapier workflows (Zaps)
- List available Zaps
- Get Zap execution status

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


class ZapierSkill(Skill):
    """Minimalistic Zapier skill for workflow automation"""
    
    def __init__(self):
        super().__init__()
        self.api_base = "https://api.zapier.com/v1"
        
    def get_dependencies(self) -> List[str]:
        """Skill dependencies"""
        return ['auth', 'kv']
    
    @prompt(priority=40, scope=["owner", "all"])
    def zapier_prompt(self) -> str:
        """Prompt describing Zapier capabilities"""
        return """
Minimalistic Zapier integration for workflow automation. Available tools:

• zapier_setup(api_key) - Set up Zapier API credentials securely
• zapier_trigger(zap_id, data) - Trigger a specific Zap with optional input data  
• zapier_list_zaps() - List all available Zaps in your Zapier account
• zapier_status(task_id) - Check the status of a Zap execution

Features:
- Secure API key storage via KV skill
- Per-user credential isolation via Auth skill  
- Trigger Zaps with custom input data
- Monitor Zap execution status
- List and discover available Zaps

Setup: First run zapier_setup() with your Zapier API key from your account settings.
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
    
    async def _save_zapier_credentials(self, user_id: str, api_key: str) -> bool:
        """Save Zapier credentials securely using KV skill"""
        try:
            kv_skill = await self._get_kv_skill()
            if kv_skill:
                credentials = {
                    'api_key': api_key,
                    'created_at': datetime.now().isoformat()
                }
                await kv_skill.kv_set(
                    key='credentials',
                    value=json.dumps(credentials),
                    namespace=f'zapier:{user_id}'
                )
                return True
            else:
                # Fallback to in-memory storage
                if not hasattr(self.agent, '_zapier_credentials'):
                    self.agent._zapier_credentials = {}
                self.agent._zapier_credentials[user_id] = {
                    'api_key': api_key
                }
                return True
        except Exception as e:
            self.logger.error(f"Failed to save Zapier credentials: {e}")
            return False
    
    async def _load_zapier_credentials(self, user_id: str) -> Optional[Dict[str, str]]:
        """Load Zapier credentials from KV skill"""
        try:
            kv_skill = await self._get_kv_skill()
            if kv_skill:
                credentials_json = await kv_skill.kv_get(
                    key='credentials',
                    namespace=f'zapier:{user_id}'
                )
                if credentials_json:
                    return json.loads(credentials_json)
            else:
                # Fallback to in-memory storage
                if hasattr(self.agent, '_zapier_credentials'):
                    return self.agent._zapier_credentials.get(user_id)
            return None
        except Exception as e:
            self.logger.error(f"Failed to load Zapier credentials: {e}")
            return None
    
    async def _make_zapier_request(self, method: str, endpoint: str, data: Optional[Dict] = None, user_id: str = None) -> Dict[str, Any]:
        """Make authenticated request to Zapier API"""
        if not user_id:
            user_id = await self._get_authenticated_user_id()
            if not user_id:
                raise Exception("Authentication required")
        
        credentials = await self._load_zapier_credentials(user_id)
        if not credentials:
            raise Exception("Zapier credentials not found. Please run zapier_setup() first.")
        
        api_key = credentials['api_key']
        
        headers = {
            'X-API-Key': api_key,
            'Content-Type': 'application/json'
        }
        
        url = f"{self.api_base}{endpoint}"
        
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
    @tool(description="Set up Zapier API credentials securely. Get your API key from Zapier account settings.", scope="owner")
    async def zapier_setup(self, api_key: str) -> str:
        """Set up Zapier API credentials for secure access"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        if not api_key or not api_key.strip():
            return "❌ API key is required. Get one from your Zapier account settings."
        
        try:
            # Test the API key by making a simple request
            test_response = await self._make_zapier_request('GET', '/zaps', user_id=user_id)
            
            # If test succeeds, save credentials
            success = await self._save_zapier_credentials(user_id, api_key.strip())
            
            if success:
                zap_count = len(test_response.get('data', []))
                return f"✅ Zapier credentials saved successfully!\n🔑 API key configured\n📊 Found {zap_count} Zaps in your account"
            else:
                return "❌ Failed to save credentials"
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return "❌ Invalid API key. Please check your Zapier API key."
            elif e.response.status_code == 403:
                return "❌ API key doesn't have required permissions."
            else:
                return f"❌ API test failed: HTTP {e.response.status_code}"
        except Exception as e:
            return f"❌ Setup failed: {str(e)}"
    
    @tool(description="Trigger a Zapier Zap with optional input data")
    async def zapier_trigger(self, zap_id: str, data: Dict[str, Any] = None) -> str:
        """Trigger a Zapier Zap with optional input data"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        if not zap_id or not zap_id.strip():
            return "❌ Zap ID is required"
        
        try:
            # Prepare trigger data
            trigger_data = data or {}
            
            # Trigger the Zap
            response = await self._make_zapier_request(
                'POST', 
                f'/zaps/{zap_id.strip()}/trigger',
                trigger_data,
                user_id
            )
            
            task_id = response.get('id', 'unknown')
            status = response.get('status', 'triggered')
            
            return f"✅ Zap triggered successfully!\n📋 Task ID: {task_id}\n📊 Status: {status}"
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return "❌ Authentication failed. Please run zapier_setup() again."
            elif e.response.status_code == 404:
                return f"❌ Zap '{zap_id}' not found"
            elif e.response.status_code == 403:
                return "❌ Permission denied. Check if Zap is enabled and you have access."
            else:
                return f"❌ Trigger failed: HTTP {e.response.status_code}"
        except Exception as e:
            return f"❌ Error triggering Zap: {str(e)}"
    
    @tool(description="List all available Zaps in your Zapier account")
    async def zapier_list_zaps(self) -> str:
        """List all available Zaps"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        try:
            response = await self._make_zapier_request('GET', '/zaps', user_id=user_id)
            
            zaps = response.get('data', [])
            
            if not zaps:
                return "📭 No Zaps found in your Zapier account"
            
            result = ["📋 Available Zapier Zaps:\n"]
            
            for zap in zaps:
                zap_id = zap.get('id', 'unknown')
                title = zap.get('title', 'Untitled Zap')
                status = zap.get('status', 'unknown')
                
                # Status icons
                status_icon = "🟢" if status == 'on' else "🔴" if status == 'off' else "⚠️"
                
                result.append(f"{status_icon} **{title}** (ID: {zap_id})")
                result.append(f"   📊 Status: {status}")
                
                # Add trigger and action info if available
                steps = zap.get('steps', [])
                if steps:
                    trigger_app = steps[0].get('app', {}).get('title', 'Unknown') if steps else 'Unknown'
                    result.append(f"   🔗 Trigger: {trigger_app}")
                
                result.append("")  # Empty line for spacing
            
            result.append("💡 Use zapier_trigger(zap_id, data) to trigger a Zap")
            
            return "\n".join(result)
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return "❌ Authentication failed. Please run zapier_setup() again."
            else:
                return f"❌ Failed to list Zaps: HTTP {e.response.status_code}"
        except Exception as e:
            return f"❌ Error listing Zaps: {str(e)}"
    
    @tool(description="Check the status of a Zap execution")
    async def zapier_status(self, task_id: str) -> str:
        """Check the status of a Zap execution"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        if not task_id or not task_id.strip():
            return "❌ Task ID is required"
        
        try:
            response = await self._make_zapier_request(
                'GET', 
                f'/tasks/{task_id.strip()}',
                user_id=user_id
            )
            
            status = response.get('status', 'unknown')
            zap_id = response.get('zap_id', 'unknown')
            created_at = response.get('created_at', 'unknown')
            updated_at = response.get('updated_at', 'running')
            
            # Status icons
            status_icons = {
                'success': '✅',
                'error': '❌', 
                'running': '🔄',
                'waiting': '⏳',
                'cancelled': '🚫',
                'throttled': '🐌'
            }
            
            status_icon = status_icons.get(status.lower(), '❓')
            
            result = [
                f"📊 Zap Execution Status Report",
                f"🆔 Task ID: {task_id}",
                f"🔧 Zap ID: {zap_id}",
                f"{status_icon} Status: {status}",
                f"🕐 Created: {created_at}",
                f"🕑 Updated: {updated_at}"
            ]
            
            # Add error details if execution failed
            if status.lower() == 'error' and 'error' in response:
                error_msg = response.get('error', 'Unknown error')
                result.append(f"❌ Error: {error_msg}")
            
            return "\n".join(result)
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return "❌ Authentication failed. Please run zapier_setup() again."
            elif e.response.status_code == 404:
                return f"❌ Task '{task_id}' not found"
            else:
                return f"❌ Status check failed: HTTP {e.response.status_code}"
        except Exception as e:
            return f"❌ Error checking status: {str(e)}"
