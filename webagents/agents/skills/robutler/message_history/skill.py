"""
MessageHistorySkill - Save agent conversation messages to WebAgents Portal

This skill enables agents to:
- Automatically save conversation messages on connection finalization
- Create and manage conversation threads
- Retrieve conversation history for context
- Support agent-to-agent and user-to-agent conversations
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, hook
from webagents.utils.logging import get_logger, log_skill_event


class MessageHistorySkill(Skill):
    """
    Skill for automatically saving agent conversation history in WebAgents Portal
    
    Features:
    - Automatic message saving on connection finalization
    - Create and manage conversation threads
    - Support for agent-to-agent and user-to-agent conversations
    - Retrieve conversation history for context
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all")
        
        # Configuration
        self.portal_url = config.get('portal_url') if config else None
        self.api_key = config.get('api_key') if config else None
        
        # Runtime state
        self.current_conversation_id = None
        self.agent_id = None
        self.api_client = None
        self.session_messages = []
        
    async def initialize(self, agent: 'BaseAgent') -> None:
        """Initialize the message history skill"""
        from webagents.utils.logging import get_logger, log_skill_event
        
        self.agent = agent
        self.agent_id = getattr(agent, 'id', None) or getattr(agent, 'name', 'unknown')
        self.logger = get_logger('skill.message_history', agent.name)
        
        # Initialize API client
        try:
            from robutler.api.client import RobutlerClient
            
            # Get API key from agent config, skill config, or environment
            api_key = (
                self.api_key or 
                getattr(agent, 'api_key', None) or
                os.getenv('WEBAGENTS_API_KEY')
            )
            
            if not api_key:
                self.logger.warning("No API key available for MessageHistorySkill")
                return
                
            self.api_client = RobutlerClient(
                api_key=api_key,
                base_url=self.portal_url or os.getenv('ROBUTLER_API_URL', 'https://webagents.ai')
            )
            
            log_skill_event(agent.name, 'message_history', 'initialized', {
                'agent_id': self.agent_id
            })
            
        except ImportError:
            self.logger.warning("RobutlerClient not available - MessageHistorySkill disabled")
        except Exception as e:
            self.logger.error(f"Failed to initialize MessageHistorySkill: {e}")

    @tool(description="Create a new conversation thread for organizing messages")
    async def create_conversation(
        self, 
        title: str,
        other_party_id: str,
        other_party_type: str = "user",
        visibility: str = "private"
    ) -> str:
        """Create a new conversation thread"""
        if not self.api_client:
            return "❌ Error: API client not initialized"
            
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {self.api_client.api_key}',
                    'Content-Type': 'application/json'
                }
                
                payload = {
                    'operation': 'createChat',
                    'agentId': self.agent_id,
                    'title': title,
                    'visibility': visibility,
                    'otherPartyId': other_party_id,
                    'otherPartyType': other_party_type
                }
                
                async with session.post(
                    f"{self.api_client.base_url}/api/chat",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        conversation_id = data.get('id')
                        self.current_conversation_id = conversation_id
                        return f"✅ Created conversation: {conversation_id}"
                    else:
                        error_text = await response.text()
                        return f"❌ Failed to create conversation: {response.status} - {error_text}"
                        
        except Exception as e:
            self.logger.error(f"Error creating conversation: {e}")
            return f"❌ Error creating conversation: {str(e)}"

    @tool(description="Retrieve conversation history for context")
    async def get_conversation_history(
        self,
        conversation_id: str = None,
        limit: int = 50
    ) -> str:
        """Retrieve conversation history"""
        if not self.api_client:
            return "❌ Error: API client not initialized"
            
        try:
            chat_id = conversation_id or self.current_conversation_id
            if not chat_id:
                return "❌ Error: No conversation ID provided and no current conversation"
                
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {self.api_client.api_key}',
                    'Content-Type': 'application/json'
                }
                
                params = {
                    'operation': 'getMessagesByChatId',
                    'chatId': chat_id
                }
                
                async with session.get(
                    f"{self.api_client.base_url}/api/messages",
                    headers=headers,
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        messages = data.get('messages', [])
                        
                        # Format messages for context
                        formatted_messages = []
                        for msg in messages[-limit:]:  # Get last N messages
                            sender_info = f"{msg.get('senderType', 'unknown')}:{msg.get('senderId', 'unknown')}"
                            content = msg.get('parts', '')
                            if isinstance(content, list):
                                content = ' '.join(str(part) for part in content)
                            formatted_messages.append(f"[{sender_info}] {content}")
                        
                        return "\n".join(formatted_messages)
                    else:
                        error_text = await response.text()
                        return f"❌ Failed to get conversation history: {response.status} - {error_text}"
                        
        except Exception as e:
            self.logger.error(f"Error getting conversation history: {e}")
            return f"❌ Error getting conversation history: {str(e)}"

    @tool(description="Set the current active conversation")
    async def set_current_conversation(self, conversation_id: str) -> str:
        """Set the current active conversation"""
        self.current_conversation_id = conversation_id
        return f"✅ Set current conversation to: {conversation_id}"

    @hook("on_message")
    async def capture_session_messages(self, context):
        """Capture messages during the session for later saving"""
        try:
            # Extract message information from context
            messages = getattr(context, 'messages', [])
            if messages:
                # Store the full message exchange for this session
                self.session_messages = messages.copy()
                self.logger.debug(f"Captured {len(messages)} messages for session")
                
        except Exception as e:
            self.logger.warning(f"Failed to capture session messages: {e}")
            
        return context

    @hook("finalize_connection")
    async def save_session_messages(self, context):
        """Automatically save all session messages when connection finalizes"""
        if not self.api_client or not self.session_messages or not self.current_conversation_id:
            return context
            
        try:
            # Prepare messages for saving
            messages_to_save = []
            
            for i, msg in enumerate(self.session_messages):
                # Determine sender info based on message role
                if msg.get('role') == 'user':
                    sender_id = "user_session"  # Default user ID - could be enhanced
                    sender_type = "user"
                elif msg.get('role') == 'assistant':
                    sender_id = self.agent_id
                    sender_type = "agent"
                else:
                    continue  # Skip system messages
                
                message_data = {
                    'id': f"msg_{self.current_conversation_id}_{i}_{hash(str(msg))}",
                    'chatId': self.current_conversation_id,
                    'role': msg.get('role'),
                    'parts': msg.get('content', ''),
                    'attachments': json.dumps([]),
                    'senderId': sender_id,
                    'senderType': sender_type
                }
                messages_to_save.append(message_data)
            
            if messages_to_save:
                # Save messages via API
                import aiohttp
                
                async with aiohttp.ClientSession() as session:
                    headers = {
                        'Authorization': f'Bearer {self.api_client.api_key}',
                        'Content-Type': 'application/json'
                    }
                    
                    payload = {
                        'operation': 'saveMessages',
                        'messages': messages_to_save
                    }
                    
                    async with session.post(
                        f"{self.api_client.base_url}/api/messages",
                        headers=headers,
                        json=payload
                    ) as response:
                        if response.status == 200:
                            self.logger.info(f"Successfully saved {len(messages_to_save)} messages to conversation {self.current_conversation_id}")
                        else:
                            error_text = await response.text()
                            self.logger.error(f"Failed to save messages: {response.status} - {error_text}")
            
            # Clear session messages after saving
            self.session_messages = []
                    
        except Exception as e:
            self.logger.error(f"Error saving session messages: {e}")
            
        return context 