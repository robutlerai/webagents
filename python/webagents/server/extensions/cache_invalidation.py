"""
Cache Invalidation Utilities

Provides Redis Pub/Sub based cache invalidation for distributed deployments.
Extensions can use RedisCacheInvalidator to subscribe to invalidation events.
"""

import asyncio
import json
import logging
import os
import threading
from typing import Callable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .interface import AgentSource

logger = logging.getLogger("webagents.cache")


class RedisCacheInvalidator:
    """Redis Pub/Sub based cache invalidation for distributed deployments.
    
    Subscribes to Redis channels and calls invalidate() on registered AgentSources
    when cache invalidation messages are received.
    
    Usage:
        invalidator = RedisCacheInvalidator()
        invalidator.register_source(my_agent_source)
        await invalidator.start()  # Or start_background()
    
    Redis Message Format:
        Channel: agent:* or agent.updated:*
        Payload: JSON with 'agentId' or 'scope.agentId' field
        
    Compatible with robutler-portal's cache invalidation messages.
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        channels: Optional[List[str]] = None,
    ):
        """Initialize Redis cache invalidator.
        
        Args:
            redis_url: Redis connection URL (default: REDIS_URL env var or redis://localhost:6379)
            channels: Redis channels to subscribe to (default: ['agent:*', 'agent.updated:*'])
        """
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.channels = channels or ['agent:*', 'agent.updated:*']
        
        self._sources: List["AgentSource"] = []
        self._task: Optional[asyncio.Task] = None
        self._started = False
        self._stop_event = asyncio.Event()
    
    def register_source(self, source: "AgentSource") -> None:
        """Register an AgentSource to receive invalidation events."""
        if source not in self._sources:
            self._sources.append(source)
            logger.debug(f"Registered source for cache invalidation: {source.get_source_type()}")
    
    def unregister_source(self, source: "AgentSource") -> None:
        """Unregister an AgentSource from invalidation events."""
        if source in self._sources:
            self._sources.remove(source)
    
    async def start(self) -> None:
        """Start the Redis subscriber (blocking)."""
        await self._subscribe_loop()
    
    def start_background(self) -> None:
        """Start the Redis subscriber in a background task."""
        if self._started and self._task and not self._task.done():
            return
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._task = loop.create_task(self._subscribe_loop())
                self._started = True
                logger.info(f"Started Redis cache invalidation subscriber: {self.redis_url}")
            else:
                # Fallback: start in a dedicated thread
                self._start_in_thread()
        except RuntimeError:
            # No event loop - start in thread
            self._start_in_thread()
    
    def _start_in_thread(self) -> None:
        """Start subscriber in a dedicated daemon thread."""
        def run():
            try:
                asyncio.run(self._subscribe_loop())
            except Exception as e:
                logger.warning(f"Redis subscriber thread terminated: {e}")
        
        thread = threading.Thread(target=run, name="redis-cache-invalidator", daemon=True)
        thread.start()
        self._started = True
        logger.info(f"Started Redis cache invalidation subscriber in thread: {self.redis_url}")
    
    async def stop(self) -> None:
        """Stop the Redis subscriber."""
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._started = False
    
    async def _subscribe_loop(self) -> None:
        """Main subscription loop with reconnection handling."""
        try:
            import redis.asyncio as aioredis
        except ImportError:
            logger.warning("Redis client not available (pip install redis), skipping cache invalidation")
            return
        
        while not self._stop_event.is_set():
            client = None
            try:
                logger.info(f"Connecting to Redis for cache invalidation: {self.redis_url}")
                client = aioredis.from_url(self.redis_url)
                pubsub = client.pubsub()
                
                # Subscribe to channels (pattern subscribe for wildcards)
                await pubsub.psubscribe(*self.channels)
                logger.info(f"Subscribed to Redis channels: {self.channels}")
                
                async for message in pubsub.listen():
                    if self._stop_event.is_set():
                        break
                    
                    try:
                        await self._handle_message(message)
                    except Exception as e:
                        logger.debug(f"Error handling message: {e}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Redis subscription error, will retry: {e}")
            finally:
                if client:
                    try:
                        await client.close()
                    except Exception:
                        pass
            
            # Backoff before reconnect
            if not self._stop_event.is_set():
                await asyncio.sleep(2)
    
    async def _handle_message(self, message: dict) -> None:
        """Handle a Redis pub/sub message."""
        if not message or message.get('type') not in ('message', 'pmessage'):
            return
        
        raw = message.get('data')
        channel = message.get('channel') or message.get('pattern')
        
        # Decode bytes
        if isinstance(channel, (bytes, bytearray)):
            channel = channel.decode('utf-8', errors='ignore')
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode('utf-8', errors='ignore')
        
        # Extract agent ID from message
        agent_id = self._extract_agent_id(raw)
        
        if agent_id:
            logger.info(f"Cache invalidation event ({channel}): {agent_id}")
            self._invalidate_agent(agent_id)
    
    def _extract_agent_id(self, raw: str) -> Optional[str]:
        """Extract agent ID from message payload."""
        try:
            payload = json.loads(raw)
            # Support both 'scope.agentId' and 'agentId' formats
            return (
                (payload.get('scope') or {}).get('agentId')
                or payload.get('agentId')
                or payload.get('agent_id')
                or payload.get('name')
            )
        except (json.JSONDecodeError, TypeError):
            # Plain string might be agent ID directly
            if isinstance(raw, str) and len(raw) >= 3 and '-' in raw:
                return raw
        return None
    
    def _invalidate_agent(self, agent_id: str) -> None:
        """Invalidate agent cache in all registered sources."""
        for source in self._sources:
            try:
                if source.invalidate(agent_id):
                    logger.debug(f"Invalidated '{agent_id}' in {source.get_source_type()}")
            except Exception as e:
                logger.warning(f"Failed to invalidate '{agent_id}' in {source.get_source_type()}: {e}")
    
    def invalidate_all(self) -> int:
        """Manually invalidate all agents in all sources."""
        total = 0
        for source in self._sources:
            try:
                total += source.invalidate_all()
            except Exception as e:
                logger.warning(f"Failed to invalidate_all in {source.get_source_type()}: {e}")
        return total
