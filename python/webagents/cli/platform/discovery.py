"""
Agent Discovery

Local and global agent discovery by intent.
"""

from typing import Optional, List, Dict
from dataclasses import dataclass


@dataclass
class DiscoveryResult:
    """A discovery result."""
    name: str
    namespace: str
    description: str
    intents: List[str]
    score: float
    source: str  # local or remote
    matching_intents: List[str] = None


class LocalDiscovery:
    """Local agent discovery using intent matching."""
    
    def __init__(self, registry=None):
        """Initialize local discovery.
        
        Args:
            registry: LocalRegistry or DaemonRegistry
        """
        self.registry = registry
    
    def discover(
        self,
        intent: str,
        limit: int = 10,
    ) -> List[DiscoveryResult]:
        """Discover agents matching intent.
        
        Args:
            intent: What you want to accomplish
            limit: Max results
            
        Returns:
            List of matching agents
        """
        if not self.registry:
            return []
        
        results = []
        intent_lower = intent.lower()
        
        for agent in self.registry.list_agents():
            # Simple keyword matching for now
            # TODO: Use embeddings for semantic matching
            score = self._calculate_score(intent_lower, agent)
            
            if score > 0:
                matching = [i for i in agent.intents if self._matches(intent_lower, i)]
                results.append(DiscoveryResult(
                    name=agent.name,
                    namespace=agent.namespace,
                    description=getattr(agent, 'description', ''),
                    intents=agent.intents,
                    score=score,
                    source="local",
                    matching_intents=matching,
                ))
        
        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
    
    def _calculate_score(self, intent: str, agent) -> float:
        """Calculate match score for agent.
        
        Args:
            intent: Search intent (lowercased)
            agent: Agent to score
            
        Returns:
            Score between 0 and 1
        """
        max_score = 0.0
        
        # Check intents
        for agent_intent in agent.intents:
            score = self._similarity(intent, agent_intent.lower())
            max_score = max(max_score, score)
        
        # Check description
        description = getattr(agent, 'description', '')
        if description:
            desc_score = self._similarity(intent, description.lower()) * 0.5
            max_score = max(max_score, desc_score)
        
        # Check name
        name_score = self._similarity(intent, agent.name.lower()) * 0.3
        max_score = max(max_score, name_score)
        
        return max_score
    
    def _similarity(self, query: str, text: str) -> float:
        """Calculate simple similarity score.
        
        Args:
            query: Query string
            text: Text to match against
            
        Returns:
            Score between 0 and 1
        """
        query_words = set(query.split())
        text_words = set(text.split())
        
        if not query_words or not text_words:
            return 0.0
        
        # Jaccard-like similarity
        intersection = len(query_words & text_words)
        union = len(query_words | text_words)
        
        return intersection / union if union > 0 else 0.0
    
    def _matches(self, query: str, intent: str) -> bool:
        """Check if intent matches query.
        
        Args:
            query: Query string (lowercased)
            intent: Intent to check
            
        Returns:
            True if matches
        """
        return self._similarity(query, intent.lower()) > 0.3


class GlobalDiscovery:
    """Global agent discovery via platform API."""
    
    def __init__(self, api=None):
        """Initialize global discovery.
        
        Args:
            api: RobutlerAPI instance
        """
        self.api = api
    
    async def discover(
        self,
        intent: str,
        namespace: str = None,
        limit: int = 10,
    ) -> List[DiscoveryResult]:
        """Discover agents globally.
        
        Args:
            intent: What you want to accomplish
            namespace: Filter by namespace
            limit: Max results
            
        Returns:
            List of matching agents
        """
        if not self.api:
            from .api import RobutlerAPI
            self.api = RobutlerAPI()
        
        async with self.api as api:
            results = await api.discover(intent, namespace=namespace, limit=limit)
        
        return [
            DiscoveryResult(
                name=r.get("name"),
                namespace=r.get("namespace"),
                description=r.get("description", ""),
                intents=r.get("intents", []),
                score=r.get("score", 0),
                source="remote",
                matching_intents=r.get("matching_intents", []),
            )
            for r in results
        ]


class UnifiedDiscovery:
    """Unified local and global discovery."""
    
    def __init__(self):
        """Initialize unified discovery."""
        from ..state.registry import LocalRegistry
        
        self.local = LocalDiscovery(LocalRegistry())
        self.global_ = GlobalDiscovery()
    
    async def discover(
        self,
        intent: str,
        local_only: bool = False,
        namespace: str = None,
        limit: int = 10,
    ) -> List[DiscoveryResult]:
        """Discover agents.
        
        Args:
            intent: What you want to accomplish
            local_only: Only search locally
            namespace: Filter by namespace
            limit: Max results
            
        Returns:
            Combined results
        """
        results = []
        
        # Local discovery
        local_results = self.local.discover(intent, limit=limit)
        results.extend(local_results)
        
        # Global discovery
        if not local_only:
            try:
                from .auth import is_authenticated
                if is_authenticated():
                    global_results = await self.global_.discover(
                        intent, namespace=namespace, limit=limit
                    )
                    results.extend(global_results)
            except Exception:
                pass
        
        # Deduplicate by name
        seen = set()
        unique = []
        for r in sorted(results, key=lambda x: x.score, reverse=True):
            if r.name not in seen:
                seen.add(r.name)
                unique.append(r)
        
        return unique[:limit]
