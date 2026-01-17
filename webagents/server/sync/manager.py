"""
Metadata Sync Manager

Sync local and remote agent metadata for published agents.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from ...utils.logging import get_logger


class MetadataSyncManager:
    """Sync local and remote agent metadata for published agents"""
    
    def __init__(self, local_store, remote_client):
        """Initialize sync manager
        
        Args:
            local_store: Local metadata store (JSONMetadataStore or LiteSQLMetadataStore)
            remote_client: Remote API client for publishing agents
        """
        self.local_store = local_store
        self.remote_client = remote_client
        self.logger = get_logger('metadata_sync')
    
    async def sync_agent(self, agent_name: str):
        """Sync agent metadata (bidirectional)
        
        Args:
            agent_name: Agent name to sync
        
        Strategy:
        - If agent only exists locally, push to remote (publish)
        - If agent only exists remotely, pull to local (download)
        - If agent exists both places, compare timestamps and merge newer
        """
        local_meta = self.local_store.get_agent(agent_name)
        
        try:
            remote_meta = await self.remote_client.get_agent(agent_name)
        except Exception as e:
            self.logger.warning(f"Failed to fetch remote metadata for {agent_name}: {e}")
            remote_meta = None
        
        # Compare timestamps
        if not local_meta and remote_meta:
            # Pull from remote
            self.local_store.register_agent(agent_name, remote_meta)
            self.logger.info(f"📥 Pulled agent metadata from remote: {agent_name}")
        elif local_meta and not remote_meta:
            # Push to remote (publish)
            await self.remote_client.publish_agent(agent_name, local_meta)
            self.logger.info(f"📤 Published agent metadata to remote: {agent_name}")
        elif local_meta and remote_meta:
            # Merge based on timestamps
            local_updated = local_meta.get("updated_at", "")
            remote_updated = remote_meta.get("updated_at", "")
            
            if local_updated > remote_updated:
                # Local is newer, push to remote
                await self.remote_client.update_agent(agent_name, local_meta)
                self.logger.info(f"📤 Updated remote agent metadata: {agent_name}")
            elif remote_updated > local_updated:
                # Remote is newer, pull to local
                self.local_store.register_agent(agent_name, remote_meta)
                self.logger.info(f"📥 Updated local agent metadata: {agent_name}")
            else:
                # Already in sync
                self.logger.debug(f"✅ Agent metadata in sync: {agent_name}")
    
    async def sync_all(self):
        """Sync all published agents
        
        Syncs all agents marked as "published" in local metadata.
        """
        local_agents = self.local_store.list_agents()
        
        for agent_meta in local_agents:
            if agent_meta.get("published"):
                try:
                    await self.sync_agent(agent_meta["name"])
                except Exception as e:
                    self.logger.error(f"Failed to sync agent {agent_meta['name']}: {e}")
        
        self.logger.info(f"Sync completed for {len(local_agents)} agents")
    
    async def mark_as_published(self, agent_name: str):
        """Mark an agent as published
        
        Args:
            agent_name: Agent name to mark
        """
        agent_meta = self.local_store.get_agent(agent_name)
        if agent_meta:
            agent_meta["published"] = True
            self.local_store.register_agent(agent_name, agent_meta)
            self.logger.info(f"📢 Marked agent as published: {agent_name}")
    
    async def unpublish(self, agent_name: str):
        """Unpublish an agent
        
        Args:
            agent_name: Agent name to unpublish
        """
        agent_meta = self.local_store.get_agent(agent_name)
        if agent_meta:
            agent_meta["published"] = False
            self.local_store.register_agent(agent_name, agent_meta)
            
            # Optionally delete from remote
            try:
                await self.remote_client.delete_agent(agent_name)
                self.logger.info(f"🗑️  Unpublished and deleted from remote: {agent_name}")
            except Exception as e:
                self.logger.warning(f"Failed to delete from remote: {e}")
