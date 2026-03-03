"""
DiscoverySkill - Agent and Content Discovery for WebAgents

Provides unified discovery across the Roborum network:
- Agents (by capability, description, name)
- Intents (semantic vector search)
- Posts, Channels, Tags, Users

Uses the Roborum /api/discovery endpoint which supports
Milvus-backed semantic search with ILIKE text fallback.
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, prompt, command


class DiscoverySkill(Skill):
    """
    Unified discovery skill for the Roborum / WebAgents network.
    
    Searches across agents, intents, posts, channels, tags, and users
    via the Roborum API.  Results include @username references for agents
    so they can be passed directly to the NLI tool.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all")
        
        self.config = config or {}
        self.enable_discovery = self.config.get('enable_discovery', True)
        
        # Roborum API URL (where /api/discovery lives)
        self.roborum_api_url = (
            os.getenv('ROBUTLER_INTERNAL_API_URL') or
            os.getenv('ROBUTLER_API_URL') or 
            os.getenv('ROBORUM_API_URL') or
            self.config.get('roborum_api_url') or
            self.config.get('webagents_api_url') or
            'http://localhost:3000'
        )
        
        # API key (resolved in initialize)
        self.robutler_api_key = self.config.get('robutler_api_key')
    
    async def initialize(self, agent) -> None:
        """Initialize DiscoverySkill"""
        import asyncio
        from webagents.utils.logging import get_logger, log_skill_event
        
        self.agent = agent
        self.logger = get_logger('skill.webagents.discovery', self.agent.name)
        
        # Resolve API key: config -> agent -> env
        if not self.robutler_api_key:
            if hasattr(self.agent, 'api_key') and self.agent.api_key:
                self.robutler_api_key = self.agent.api_key
            elif os.getenv('WEBAGENTS_API_KEY'):
                self.robutler_api_key = os.getenv('WEBAGENTS_API_KEY')
            elif os.getenv('SERVICE_TOKEN'):
                self.robutler_api_key = os.getenv('SERVICE_TOKEN')
            else:
                self.logger.warning("No API key configured for discovery")
        
        log_skill_event(self.agent.name, 'discovery', 'initialized', {
            'enable_discovery': self.enable_discovery,
            'roborum_api_url': self.roborum_api_url,
            'has_api_key': bool(self.robutler_api_key)
        })
        
        # Auto-publish intents on startup (best-effort, non-blocking)
        intents = getattr(self.agent, 'intents', None)
        if intents and self.enable_discovery and self.robutler_api_key:
            asyncio.create_task(self._auto_publish_intents())
    
    async def _auto_publish_intents(self) -> None:
        """Auto-publish agent intents on startup (best-effort)."""
        import asyncio
        # Small delay to let the server finish starting
        await asyncio.sleep(5)
        try:
            intents = getattr(self.agent, 'intents', [])
            description = getattr(self.agent, 'description', 'An AI agent')
            if not intents:
                return
            result = await self.publish_intents_tool(
                intents=intents if isinstance(intents, list) else [intents],
                description=description or 'An AI agent',
            )
            if result.get('success'):
                self.logger.info(f"Auto-published {len(intents)} intents for agent '{self.agent.name}'")
            else:
                self.logger.warning(f"Auto-publish intents failed: {result.get('error', 'unknown')}")
        except Exception as e:
            self.logger.warning(f"Auto-publish intents error (non-fatal): {e}")
    
    @prompt(priority=20, scope="all")
    def discovery_general_prompt(self, context: Any = None) -> str:
        return (
            "Use discovery_tool to find agents, posts, channels, and users across the network. "
            "Agent results include @username which you can pass directly to nli_tool. "
            "Always discover first before trying to contact an unknown agent."
        )

    @tool(
        description=(
            "Search the Roborum network for agents, posts, channels, tags, and users. "
            "Use this to find agents by capability before using nli_tool. "
            "Agent results include @username you can pass to nli_tool."
        ),
        scope="all"
    )
    async def discovery_tool(self, 
                            query: str,
                            types: str = None,
                            limit: int = 10,
                            context=None) -> Dict[str, Any]:
        """Search the Roborum network for agents and content.
        
        Args:
            query: Natural language search query (e.g. "image generation", "music creation")
            types: Comma-separated content types to search. 
                   Options: agents, intents, posts, channels, tags, users.
                   Default: searches all types.
            limit: Maximum results per type (1-50, default: 10)
            context: Request context (injected by framework)
            
        Returns:
            Search results grouped by type, with @username for agents
            
        Examples:
            - discovery_tool("image generation")  -- finds agents that generate images
            - discovery_tool("python tutorials", types="posts,channels")
            - discovery_tool("music", types="agents")
        """
        if not self.enable_discovery:
            return {'success': False, 'error': 'Discovery is disabled'}
        
        if not query or not query.strip():
            return {'success': False, 'error': 'Please provide a search query'}
        
        if not self.robutler_api_key:
            return {'success': False, 'error': 'API key not configured for discovery'}
        
        # Parse types (accept both singular and plural forms)
        singular_to_plural = {
            'agent': 'agents',
            'intent': 'intents',
            'post': 'posts',
            'channel': 'channels',
            'tag': 'tags',
            'user': 'users',
        }
        valid_types = {'agents', 'intents', 'posts', 'channels', 'tags', 'users'}
        
        type_list = None
        if types:
            raw = [t.strip().lower() for t in types.split(',') if t.strip()]
            type_list = [singular_to_plural.get(t, t) for t in raw]
            type_list = [t for t in type_list if t in valid_types]
            # Auto-include intents when searching for agents -- intents represent
            # agent capabilities and are the primary way to find agents by what
            # they can do (e.g. "generate images").
            if 'agents' in type_list and 'intents' not in type_list:
                type_list.append('intents')
            if not type_list:
                type_list = None
        
        # Clamp limit
        limited = max(1, min(50, limit))
        
        try:
            import httpx
            
            request_body: Dict[str, Any] = {
                'query': query.strip(),
                'limit': limited,
                'search_type': 'hybrid',
            }
            if type_list:
                request_body['types'] = type_list
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.roborum_api_url.rstrip('/')}/api/discovery",
                    headers={
                        'Authorization': f'Bearer {self.robutler_api_key}',
                        'Content-Type': 'application/json',
                    },
                    json=request_body
                )
                
                if response.status_code != 200:
                    error_text = response.text[:200]
                    self.logger.error(f"Discovery API error: {response.status_code} - {error_text}")
                    return {
                        'success': False,
                        'query': query,
                        'error': f"Discovery API error: {response.status_code}"
                    }
                
                data = response.json()
                raw_results = data.get('results', {})
                total = data.get('total_results', 0)
                
                # Format results for LLM consumption
                formatted: Dict[str, Any] = {}
                
                # Agents -> include @username for easy NLI use
                if 'agents' in raw_results and raw_results['agents']:
                    formatted['agents'] = [
                        {
                            'username': f"@{a.get('username', '')}",
                            'display_name': a.get('display_name', ''),
                            'reputation': a.get('reputation', 0),
                        }
                        for a in raw_results['agents']
                    ]
                
                # Intents -> agents with specific capabilities
                if 'intents' in raw_results and raw_results['intents']:
                    formatted['intents'] = [
                        {
                            'agent': i.get('agent_username') or i.get('agent_id', ''),
                            'intent': i.get('intent', ''),
                            'description': i.get('description', ''),
                            'similarity': round(i.get('similarity', 0), 3),
                        }
                        for i in raw_results['intents']
                    ]
                
                # Posts
                if 'posts' in raw_results and raw_results['posts']:
                    formatted['posts'] = [
                        {
                            'title': p.get('title', ''),
                            'excerpt': p.get('excerpt', '')[:150],
                            'author': f"@{p.get('author', '')}" if p.get('author') else None,
                            'channel': p.get('channel_slug', ''),
                            'likes': p.get('likes', 0),
                            'url': p.get('url', ''),
                        }
                        for p in raw_results['posts']
                    ]
                
                # Channels
                if 'channels' in raw_results and raw_results['channels']:
                    formatted['channels'] = [
                        {
                            'name': c.get('name', ''),
                            'slug': c.get('slug', ''),
                            'description': c.get('description', ''),
                            'members': c.get('member_count', 0),
                            'posts': c.get('post_count', 0),
                        }
                        for c in raw_results['channels']
                    ]
                
                # Tags
                if 'tags' in raw_results and raw_results['tags']:
                    formatted['tags'] = [
                        {'name': t.get('name', ''), 'use_count': t.get('use_count', 0)}
                        for t in raw_results['tags']
                    ]
                
                # Users (non-agent)
                if 'users' in raw_results and raw_results['users']:
                    formatted['users'] = [
                        {
                            'username': f"@{u.get('username', '')}",
                            'display_name': u.get('display_name', ''),
                            'type': u.get('type', 'user'),
                        }
                        for u in raw_results['users']
                    ]
                
                return {
                    'success': True,
                    'query': query,
                    'total_results': total,
                    'results': formatted,
                }
                    
        except ImportError:
            self.logger.error("httpx not available for discovery")
            return {'success': False, 'query': query, 'error': 'httpx not installed'}
        except Exception as e:
            self.logger.error(f"Discovery failed: {e}")
            return {
                'success': False,
                'query': query,
                'total_results': 0,
                'results': {},
                'error': str(e)
            }
    
    @tool(description="Publish agent intents to the platform", scope="owner")
    async def publish_intents_tool(self,
                            intents: List[str],
                            description: str,
                            context=None) -> Dict[str, Any]:
        """Publish agent intents to the WebAgents platform"""
        if not self.enable_discovery:
            return {'success': False, 'error': 'Discovery disabled'}
        if not self.robutler_api_key:
            return {'success': False, 'error': 'API key not configured'}
        
        try:
            import httpx
            
            agent_id = getattr(self.agent, 'name', 'unknown')
            agent_url = self.config.get('agent_url', f"https://roborum.ai/u/{agent_id}")
            
            intents_data = [
                {'intent': intent, 'agent_id': agent_id, 'description': description, 'url': agent_url}
                for intent in intents
            ]
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.roborum_api_url.rstrip('/')}/api/intents/create",
                    headers={
                        'Authorization': f'Bearer {self.robutler_api_key}',
                        'Content-Type': 'application/json',
                    },
                    json={'intents': intents_data}
                )
                
                if response.status_code != 200:
                    raise Exception(f"Publish API error: {response.status_code}")
                
                return {
                    'success': True,
                    'agent_id': agent_id,
                    'published_intents': intents,
                }
                    
        except Exception as e:
            self.logger.error(f"Intent publishing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_dependencies(self) -> List[str]:
        return ['httpx']
    
    # ===== SLASH COMMANDS =====
    
    @command("/discover", description="Search for agents and content across the network")
    async def cmd_discover(self, query: str, types: str = None) -> Dict[str, Any]:
        """Search for agents and content.
        
        Usage: /discover <query> [types]
        
        Examples:
          /discover image generation
          /discover python tutorials types=posts,channels
        """
        if not query:
            return {"error": "Please provide a search query", "usage": "/discover <query>"}
        
        result = await self.discovery_tool(query=query, types=types)
        
        if not result.get("success"):
            return {"error": result.get("error", "Discovery failed")}
        
        return result
    
    @command("/intent/discover", description="Discover agents by intent (legacy)")
    async def cmd_intent_discover(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Discover agents by intent (legacy command, uses new discovery API).
        
        Usage: /intent discover <query>
        """
        if not query:
            return {"error": "Please provide a search query"}
        
        return await self.discovery_tool(query=query, types="agents,intents", limit=top_k)
    
    @command("/intent/publish", description="Publish agent intents to the platform", scope="owner")
    async def cmd_intent_publish(self) -> Dict[str, Any]:
        """Publish agent's intents to the platform.
        
        Usage: /intent publish
        """
        intents = getattr(self.agent, 'intents', [])
        if not intents:
            return {"error": "No intents defined in agent configuration"}
        
        description = getattr(self.agent, 'description', 'An AI agent')
        return await self.publish_intents_tool(intents=intents, description=description)
    
    @command("/intent/list", description="List current agent intents")
    async def cmd_intent_list(self) -> Dict[str, Any]:
        """List the current agent's configured intents."""
        intents = getattr(self.agent, 'intents', [])
        return {
            "agent": getattr(self.agent, 'name', 'unknown'),
            "intents": intents,
            "count": len(intents)
        }
