"""
Replicate Skill for WebAgents

This skill allows users to:
- Set up Replicate API key securely (via auth/kv skills)
- List available models on Replicate
- Run predictions with models
- Get prediction status and results
- Cancel running predictions

Uses auth skill for user context and kv skill for secure API key storage.
"""

import os
import httpx
import json
import asyncio
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, prompt
from webagents.server.context.context_vars import get_context


class ReplicateSkill(Skill):
    """Replicate skill for running machine learning models"""
    
    def __init__(self):
        super().__init__()
        self.api_base = "https://api.replicate.com/v1"
        
    def get_dependencies(self) -> List[str]:
        """Skill dependencies"""
        return ['auth', 'kv']
    
    @prompt(priority=40, scope=["owner", "all"])
    def replicate_prompt(self) -> str:
        """Prompt describing Replicate capabilities"""
        return """
Replicate integration for running machine learning models. Available tools:

• replicate_setup(api_token) - Set up Replicate API credentials securely
• replicate_list_models(owner) - List models from a specific owner or popular models
• replicate_run_prediction(model, input_data) - Run a prediction with a model
• replicate_get_prediction(prediction_id) - Get prediction status and results
• replicate_cancel_prediction(prediction_id) - Cancel a running prediction
• replicate_get_model_info(model) - Get detailed information about a model

Features:
- Secure API token storage via KV skill
- Per-user credential isolation via Auth skill  
- Run any public model on Replicate
- Monitor prediction progress and results
- Handle both sync and async predictions
- Support for text, image, audio, and video models

Setup: First run replicate_setup() with your Replicate API token from your account settings.
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
            if hasattr(self, 'logger'):
                self.logger.error(f"Failed to get user context: {e}")
            return None
    
    async def _save_replicate_credentials(self, user_id: str, api_token: str) -> bool:
        """Save Replicate credentials securely using KV skill"""
        try:
            kv_skill = await self._get_kv_skill()
            if kv_skill:
                credentials = {
                    'api_token': api_token,
                    'created_at': datetime.now().isoformat()
                }
                try:
                    await kv_skill.kv_set(
                        key='credentials',
                        value=json.dumps(credentials),
                        namespace=f'replicate:{user_id}'
                    )
                    return True
                except Exception as kv_error:
                    if hasattr(self, 'logger'):
                        self.logger.error(f"KV storage failed, falling back to memory: {kv_error}")
                    # Fall through to memory storage
            
            # Fallback to in-memory storage (either no KV skill or KV failed)
            if not hasattr(self.agent, '_replicate_credentials'):
                self.agent._replicate_credentials = {}
            self.agent._replicate_credentials[user_id] = {
                'api_token': api_token,
                'created_at': datetime.now().isoformat()
            }
            return True
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Failed to save Replicate credentials: {e}")
            return False
    
    async def _load_replicate_credentials(self, user_id: str) -> Optional[Dict[str, str]]:
        """Load Replicate credentials from KV skill"""
        try:
            kv_skill = await self._get_kv_skill()
            if kv_skill:
                try:
                    credentials_json = await kv_skill.kv_get(
                        key='credentials',
                        namespace=f'replicate:{user_id}'
                    )
                    if credentials_json:
                        return json.loads(credentials_json)
                except Exception as kv_error:
                    if hasattr(self, 'logger'):
                        self.logger.error(f"KV retrieval failed, falling back to memory: {kv_error}")
                    # Fall through to memory storage
            
            # Fallback to in-memory storage (either no KV skill or KV failed)
            if hasattr(self.agent, '_replicate_credentials'):
                return self.agent._replicate_credentials.get(user_id)
            return None
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Failed to load Replicate credentials: {e}")
            return None
    
    async def _make_replicate_request(self, method: str, endpoint: str, data: Optional[Dict] = None, user_id: str = None) -> Dict[str, Any]:
        """Make authenticated request to Replicate API"""
        if not user_id:
            user_id = await self._get_authenticated_user_id()
            if not user_id:
                raise Exception("Authentication required")
        
        credentials = await self._load_replicate_credentials(user_id)
        if not credentials:
            raise Exception("Replicate credentials not found. Please run replicate_setup() first.")
        
        api_token = credentials['api_token']
        
        headers = {
            'Authorization': f'Token {api_token}',
            'Content-Type': 'application/json'
        }
        
        url = f"{self.api_base}{endpoint}"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            if method.upper() == 'GET':
                response = await client.get(url, headers=headers)
            elif method.upper() == 'POST':
                response = await client.post(url, headers=headers, json=data)
            elif method.upper() == 'DELETE':
                response = await client.delete(url, headers=headers)
            else:
                raise Exception(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json() if response.content else {}

    # Public tools
    @tool(description="Set up Replicate API token securely. Get your token from Replicate account settings.", scope="owner")
    async def replicate_setup(self, api_token: str) -> str:
        """Set up Replicate API credentials for secure access"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        if not api_token or not api_token.strip():
            return "❌ API token is required. Get one from your Replicate account settings."
        
        try:
            # Test the API token by making a simple request
            test_response = await self._make_replicate_request('GET', '/models', user_id=user_id)
            
            # If test succeeds, save credentials
            success = await self._save_replicate_credentials(user_id, api_token.strip())
            
            if success:
                return f"✅ Replicate credentials saved successfully!\n🔑 API token configured\n📊 Connection to Replicate API verified"
            else:
                return "❌ Failed to save credentials"
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return "❌ Invalid API token. Please check your Replicate API token."
            elif e.response.status_code == 403:
                return "❌ API token doesn't have required permissions."
            else:
                return f"❌ API test failed: HTTP {e.response.status_code}"
        except Exception as e:
            return f"❌ Setup failed: {str(e)}"
    
    @tool(description="List models from a specific owner or popular models")
    async def replicate_list_models(self, owner: str = None) -> str:
        """List available models on Replicate"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        try:
            # Build endpoint
            if owner:
                endpoint = f"/models?owner={owner}"
            else:
                endpoint = "/models"
            
            response = await self._make_replicate_request('GET', endpoint, user_id=user_id)
            
            models = response.get('results', [])
            
            if not models:
                return f"📭 No models found{f' for owner {owner}' if owner else ''}"
            
            result = [f"🤖 Available Models{f' from {owner}' if owner else ''}:\n"]
            
            for model in models[:10]:  # Limit to first 10 models
                name = model.get('name', 'Unknown')
                full_name = f"{model.get('owner', 'unknown')}/{name}"
                description = model.get('description', 'No description')
                visibility = model.get('visibility', 'unknown')
                
                # Visibility icon
                vis_icon = "🔒" if visibility == 'private' else "🌍"
                
                result.append(f"{vis_icon} **{full_name}**")
                result.append(f"   📝 {description[:100]}{'...' if len(description) > 100 else ''}")
                result.append("")  # Empty line for spacing
            
            if len(models) > 10:
                result.append(f"... and {len(models) - 10} more models")
                result.append("")
            
            result.append("💡 Use replicate_run_prediction(model, input_data) to run a model")
            
            return "\n".join(result)
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return "❌ Authentication failed. Please run replicate_setup() again."
            else:
                return f"❌ Failed to list models: HTTP {e.response.status_code}"
        except Exception as e:
            return f"❌ Error listing models: {str(e)}"
    
    @tool(description="Get detailed information about a specific model")
    async def replicate_get_model_info(self, model: str) -> str:
        """Get detailed information about a model"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        if not model or not model.strip():
            return "❌ Model name is required (format: owner/model-name)"
        
        try:
            # Clean model name
            model_name = model.strip()
            if '/' not in model_name:
                return "❌ Model name must be in format: owner/model-name"
            
            response = await self._make_replicate_request('GET', f'/models/{model_name}', user_id=user_id)
            
            name = response.get('name', 'Unknown')
            owner = response.get('owner', 'unknown')
            description = response.get('description', 'No description')
            visibility = response.get('visibility', 'unknown')
            github_url = response.get('github_url', '')
            paper_url = response.get('paper_url', '')
            license_url = response.get('license_url', '')
            
            # Get latest version info
            latest_version = response.get('latest_version', {})
            version_id = latest_version.get('id', 'unknown')
            created_at = latest_version.get('created_at', 'unknown')
            
            # Schema info
            schema = latest_version.get('openapi_schema', {})
            input_schema = schema.get('components', {}).get('schemas', {}).get('Input', {}).get('properties', {})
            
            vis_icon = "🔒" if visibility == 'private' else "🌍"
            
            result = [
                f"🤖 Model Information: {owner}/{name}",
                f"{vis_icon} Visibility: {visibility}",
                f"📝 Description: {description}",
                f"🆔 Latest Version: {version_id}",
                f"📅 Created: {created_at}",
                ""
            ]
            
            if github_url:
                result.append(f"🔗 GitHub: {github_url}")
            if paper_url:
                result.append(f"📄 Paper: {paper_url}")
            if license_url:
                result.append(f"📜 License: {license_url}")
            
            if input_schema:
                result.append("\n📋 Input Parameters:")
                for param_name, param_info in input_schema.items():
                    param_type = param_info.get('type', 'unknown')
                    param_desc = param_info.get('description', 'No description')
                    required = param_name in schema.get('components', {}).get('schemas', {}).get('Input', {}).get('required', [])
                    req_icon = "⚠️" if required else "📝"
                    result.append(f"  {req_icon} {param_name} ({param_type}): {param_desc}")
            
            result.append("\n💡 Use replicate_run_prediction() to run this model")
            
            return "\n".join(result)
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return "❌ Authentication failed. Please run replicate_setup() again."
            elif e.response.status_code == 404:
                return f"❌ Model '{model}' not found"
            else:
                return f"❌ Failed to get model info: HTTP {e.response.status_code}"
        except Exception as e:
            return f"❌ Error getting model info: {str(e)}"
    
    @tool(description="Run a prediction with a Replicate model")
    async def replicate_run_prediction(self, model: str, input_data: Dict[str, Any]) -> str:
        """Run a prediction with a model"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        if not model or not model.strip():
            return "❌ Model name is required (format: owner/model-name)"
        
        if not input_data:
            return "❌ Input data is required"
        
        try:
            # Clean model name
            model_name = model.strip()
            if '/' not in model_name:
                return "❌ Model name must be in format: owner/model-name"
            
            # Prepare prediction data
            prediction_data = {
                "version": f"{model_name}:latest",
                "input": input_data
            }
            
            # Create prediction
            response = await self._make_replicate_request(
                'POST', 
                '/predictions',
                prediction_data,
                user_id
            )
            
            prediction_id = response.get('id', 'unknown')
            status = response.get('status', 'starting')
            
            result = [
                f"🚀 Prediction started successfully!",
                f"🆔 Prediction ID: {prediction_id}",
                f"🤖 Model: {model_name}",
                f"📊 Status: {status}"
            ]
            
            # If prediction completed immediately, show results
            if status == 'succeeded':
                output = response.get('output')
                if output:
                    result.append(f"✅ Output: {str(output)[:200]}{'...' if len(str(output)) > 200 else ''}")
            elif status == 'failed':
                error = response.get('error', 'Unknown error')
                result.append(f"❌ Error: {error}")
            else:
                result.append("⏳ Use replicate_get_prediction() to check status and get results")
            
            return "\n".join(result)
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return "❌ Authentication failed. Please run replicate_setup() again."
            elif e.response.status_code == 404:
                return f"❌ Model '{model}' not found"
            elif e.response.status_code == 422:
                return "❌ Invalid input data. Check model requirements with replicate_get_model_info()"
            else:
                return f"❌ Prediction failed: HTTP {e.response.status_code}"
        except Exception as e:
            return f"❌ Error running prediction: {str(e)}"
    
    @tool(description="Get prediction status and results")
    async def replicate_get_prediction(self, prediction_id: str) -> str:
        """Get prediction status and results"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        if not prediction_id or not prediction_id.strip():
            return "❌ Prediction ID is required"
        
        try:
            response = await self._make_replicate_request(
                'GET', 
                f'/predictions/{prediction_id.strip()}',
                user_id=user_id
            )
            
            status = response.get('status', 'unknown')
            model = response.get('model', 'unknown')
            created_at = response.get('created_at', 'unknown')
            started_at = response.get('started_at', '')
            completed_at = response.get('completed_at', '')
            
            # Status icons
            status_icons = {
                'starting': '🔄',
                'processing': '⚙️', 
                'succeeded': '✅',
                'failed': '❌',
                'canceled': '🚫'
            }
            
            status_icon = status_icons.get(status, '❓')
            
            result = [
                f"📊 Prediction Status Report",
                f"🆔 Prediction ID: {prediction_id}",
                f"🤖 Model: {model}",
                f"{status_icon} Status: {status}",
                f"🕐 Created: {created_at}"
            ]
            
            if started_at:
                result.append(f"▶️ Started: {started_at}")
            if completed_at:
                result.append(f"🏁 Completed: {completed_at}")
            
            # Add results if succeeded
            if status == 'succeeded':
                output = response.get('output')
                if output:
                    output_str = str(output)
                    if len(output_str) > 500:
                        result.append(f"📋 Output: {output_str[:500]}...")
                        result.append("💡 Output truncated. Full results available via API.")
                    else:
                        result.append(f"📋 Output: {output_str}")
            
            # Add error details if failed
            elif status == 'failed':
                error = response.get('error', 'Unknown error')
                logs = response.get('logs', '')
                result.append(f"❌ Error: {error}")
                if logs:
                    result.append(f"📝 Logs: {logs[-200:]}...")  # Show last 200 chars of logs
            
            return "\n".join(result)
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return "❌ Authentication failed. Please run replicate_setup() again."
            elif e.response.status_code == 404:
                return f"❌ Prediction '{prediction_id}' not found"
            else:
                return f"❌ Status check failed: HTTP {e.response.status_code}"
        except Exception as e:
            return f"❌ Error checking prediction: {str(e)}"
    
    @tool(description="Cancel a running prediction")
    async def replicate_cancel_prediction(self, prediction_id: str) -> str:
        """Cancel a running prediction"""
        user_id = await self._get_authenticated_user_id()
        if not user_id:
            return "❌ Authentication required"
        
        if not prediction_id or not prediction_id.strip():
            return "❌ Prediction ID is required"
        
        try:
            response = await self._make_replicate_request(
                'POST', 
                f'/predictions/{prediction_id.strip()}/cancel',
                user_id=user_id
            )
            
            status = response.get('status', 'unknown')
            
            if status == 'canceled':
                return f"✅ Prediction {prediction_id} canceled successfully"
            else:
                return f"⚠️ Prediction status: {status} (may not be cancelable)"
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return "❌ Authentication failed. Please run replicate_setup() again."
            elif e.response.status_code == 404:
                return f"❌ Prediction '{prediction_id}' not found"
            elif e.response.status_code == 422:
                return f"❌ Cannot cancel prediction (may be completed or already canceled)"
            else:
                return f"❌ Cancel failed: HTTP {e.response.status_code}"
        except Exception as e:
            return f"❌ Error canceling prediction: {str(e)}"

