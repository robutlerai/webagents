"""
Marketplace Client - Plugin Discovery and Search

Provides plugin marketplace discovery from claudemarketplaces.com API
with fuzzy search using rapidfuzz and GitHub star ranking.
"""

import asyncio
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Constants
MARKETPLACE_API = "https://claudemarketplaces.com/api/marketplaces"
GITHUB_API = "https://api.github.com"
CACHE_DIR = Path.home() / ".webagents" / "plugin_cache"
REFRESH_INTERVAL = 6 * 3600  # 6 hours in seconds


class MarketplaceClient:
    """Plugin marketplace discovery with fuzzy search.
    
    Features:
    - Fetches plugin registries from claudemarketplaces.com
    - Caches marketplace index to disk
    - Fuzzy search with rapidfuzz
    - GitHub star ranking boost
    - Periodic background refresh
    """
    
    def __init__(self, github_token: Optional[str] = None):
        """Initialize marketplace client.
        
        Args:
            github_token: Optional GitHub token for higher API rate limits
        """
        self._index: Dict[str, Dict[str, Any]] = {}
        self._last_refresh: float = 0
        self._github_token = github_token or os.environ.get("GITHUB_TOKEN")
        self._cache_file = CACHE_DIR / "marketplace_index.json"
        self._lock = asyncio.Lock()
    
    async def refresh_index(self) -> None:
        """Fetch marketplaces and build plugin index.
        
        Fetches all marketplace registries, collects plugins,
        fetches GitHub stars, and caches to disk.
        """
        async with self._lock:
            try:
                import httpx
            except ImportError:
                logger.error("httpx is required for marketplace. Install with: pip install httpx")
                return
            
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                # 1. Get marketplace list
                try:
                    resp = await client.get(MARKETPLACE_API)
                    resp.raise_for_status()
                    marketplaces = resp.json()
                except httpx.HTTPError as e:
                    logger.error(f"Failed to fetch marketplace list: {e}")
                    return
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from marketplace API: {e}")
                    return
                
                # Handle both list and dict responses
                if isinstance(marketplaces, dict):
                    marketplaces = marketplaces.get("marketplaces", [])
                
                # 2. For each marketplace, fetch plugin registry
                new_index: Dict[str, Dict[str, Any]] = {}
                
                for mp in marketplaces:
                    plugins = await self._fetch_marketplace_plugins(client, mp)
                    for plugin in plugins:
                        # Fetch GitHub stars (with rate limiting consideration)
                        stars = await self._get_github_stars(client, plugin.get("repo") or plugin.get("repository"))
                        plugin["stars"] = stars
                        new_index[plugin["name"]] = plugin
                
                self._index = new_index
                self._last_refresh = time.time()
                
                # Cache to disk
                self._save_cache()
                
                logger.info(f"Marketplace index refreshed: {len(self._index)} plugins")
    
    async def _fetch_marketplace_plugins(
        self, 
        client: "httpx.AsyncClient", 
        marketplace: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Fetch plugins from a marketplace registry.
        
        Args:
            client: HTTP client
            marketplace: Marketplace metadata with URL
            
        Returns:
            List of plugin metadata dicts
        """
        url = marketplace.get("url") or marketplace.get("registry_url")
        if not url:
            return []
        
        # Try multiple registry file patterns
        registry_patterns = [
            f"{url.rstrip('/')}/raw/main/plugins.json",
            f"{url.rstrip('/')}/plugins.json",
            f"{url.rstrip('/')}/-/raw/main/registry.json",
        ]
        
        # Handle GitHub URLs
        if "github.com" in url:
            # Convert github.com URL to raw.githubusercontent.com
            parts = url.rstrip("/").split("/")
            if len(parts) >= 5:
                owner, repo = parts[-2], parts[-1]
                registry_patterns.insert(0, f"https://raw.githubusercontent.com/{owner}/{repo}/main/plugins.json")
                registry_patterns.insert(1, f"https://raw.githubusercontent.com/{owner}/{repo}/main/registry.json")
        
        for registry_url in registry_patterns:
            try:
                resp = await client.get(registry_url, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    
                    # Handle both direct list and {plugins: [...]} format
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict):
                        return data.get("plugins", [])
            except Exception as e:
                logger.debug(f"Failed to fetch registry from {registry_url}: {e}")
        
        return []
    
    async def _get_github_stars(
        self, 
        client: "httpx.AsyncClient", 
        repo_url: Optional[str]
    ) -> int:
        """Get GitHub star count for a repository.
        
        Args:
            client: HTTP client
            repo_url: GitHub repository URL
            
        Returns:
            Star count or 0 if unavailable
        """
        if not repo_url or "github.com" not in repo_url:
            return 0
        
        # Parse owner/repo from URL
        parts = repo_url.rstrip("/").split("/")
        if len(parts) < 2:
            return 0
        
        owner, repo = parts[-2], parts[-1]
        if repo.endswith(".git"):
            repo = repo[:-4]
        
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self._github_token:
            headers["Authorization"] = f"token {self._github_token}"
        
        try:
            resp = await client.get(
                f"{GITHUB_API}/repos/{owner}/{repo}",
                headers=headers,
                timeout=10
            )
            if resp.status_code == 200:
                return resp.json().get("stargazers_count", 0)
        except Exception as e:
            logger.debug(f"Failed to get stars for {repo_url}: {e}")
        
        return 0
    
    def _save_cache(self) -> None:
        """Save index to disk cache."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self._cache_file.write_text(json.dumps({
                "timestamp": self._last_refresh,
                "plugins": self._index,
            }, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save marketplace cache: {e}")
    
    def load_cached_index(self) -> bool:
        """Load index from disk cache.
        
        Returns:
            True if cache loaded successfully
        """
        if not self._cache_file.exists():
            return False
        
        try:
            data = json.loads(self._cache_file.read_text())
            self._index = data.get("plugins", {})
            self._last_refresh = data.get("timestamp", 0)
            logger.info(f"Loaded {len(self._index)} plugins from cache")
            return True
        except Exception as e:
            logger.warning(f"Failed to load marketplace cache: {e}")
            return False
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fuzzy search plugins, ranked by match score * stars.
        
        Uses rapidfuzz for fuzzy matching and boosts results
        by GitHub star count (log scale).
        
        Args:
            query: Search query string
            limit: Maximum results to return
            
        Returns:
            List of matching plugins sorted by relevance
        """
        if not self._index:
            return []
        
        try:
            from rapidfuzz import fuzz, process as rfuzz_process
        except ImportError:
            logger.warning("rapidfuzz not installed, falling back to simple search")
            return self._simple_search(query, limit)
        
        # Build search corpus: combine name, description, keywords
        choices = {}
        for name, plugin in self._index.items():
            searchable = f"{name} {plugin.get('description', '')} {' '.join(plugin.get('keywords', []))}"
            choices[name] = searchable
        
        # Fuzzy match
        matches = rfuzz_process.extract(
            query, 
            choices, 
            scorer=fuzz.WRatio, 
            limit=limit * 2  # Get extra for re-ranking
        )
        
        # Re-rank by stars (log scale boost)
        results = []
        for name, score, _ in matches:
            plugin = self._index[name].copy()
            plugin["_match_score"] = score
            
            # Boost by stars using log scale to avoid domination
            stars = plugin.get("stars", 0)
            star_boost = math.log10(stars + 1) * 10 if stars > 0 else 0
            plugin["_rank"] = score + star_boost
            
            results.append(plugin)
        
        # Sort by combined rank
        results.sort(key=lambda p: p["_rank"], reverse=True)
        
        return results[:limit]
    
    def _simple_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Simple substring search fallback.
        
        Used when rapidfuzz is not available.
        """
        query_lower = query.lower()
        results = []
        
        for name, plugin in self._index.items():
            searchable = f"{name} {plugin.get('description', '')}".lower()
            if query_lower in searchable:
                plugin_copy = plugin.copy()
                plugin_copy["_match_score"] = 100 if query_lower in name.lower() else 50
                results.append(plugin_copy)
        
        # Sort by stars as tiebreaker
        results.sort(key=lambda p: (p.get("_match_score", 0), p.get("stars", 0)), reverse=True)
        
        return results[:limit]
    
    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """Get plugin by exact name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin metadata or None
        """
        return self._index.get(name)
    
    def get_completions(self) -> List[str]:
        """Return all plugin names for command completions.
        
        Returns:
            List of plugin names
        """
        return list(self._index.keys())
    
    def needs_refresh(self) -> bool:
        """Check if index should be refreshed.
        
        Returns:
            True if refresh interval has elapsed
        """
        return time.time() - self._last_refresh > REFRESH_INTERVAL
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get marketplace index statistics.
        
        Returns:
            Dict with index stats
        """
        return {
            "total_plugins": len(self._index),
            "last_refresh": self._last_refresh,
            "cache_age_seconds": time.time() - self._last_refresh if self._last_refresh else None,
            "needs_refresh": self.needs_refresh(),
        }
