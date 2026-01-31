"""
Web Search Skill

Provides tools for searching the web using various search engines.
Supports DuckDuckGo (no API key required), Google Custom Search, and Bing.
"""

import re
import html
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
from urllib.parse import urlencode, unquote

import httpx

from ...base import Skill
from ....tools.decorators import tool


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str
    source: Literal['duckduckgo', 'google', 'bing'] = 'duckduckgo'


@dataclass
class SearchResponse:
    """Search response containing results."""
    query: str
    results: List[SearchResult] = field(default_factory=list)
    total_results: Optional[int] = None
    error: Optional[str] = None


class WebSearchSkill(Skill):
    """
    Web Search Skill for searching the internet.
    
    Supports multiple search engines:
    - DuckDuckGo: Free, no API key required (default)
    - Google: Requires Google Custom Search API key and CSE ID
    - Bing: Requires Bing Search API key
    
    Example:
        ```python
        search_skill = WebSearchSkill()
        agent.add_skill(search_skill)
        
        # Or with API keys for premium search
        search_skill = WebSearchSkill(
            google_api_key="...",
            google_cse_id="...",
            bing_api_key="..."
        )
        ```
    """
    
    def __init__(
        self,
        default_engine: Literal['duckduckgo', 'google', 'bing'] = 'duckduckgo',
        max_results: int = 10,
        timeout: int = 10,
        google_api_key: Optional[str] = None,
        google_cse_id: Optional[str] = None,
        bing_api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.default_engine = default_engine
        self.max_results = max_results
        self.timeout = timeout
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        self.bing_api_key = bing_api_key
        self._client: Optional[httpx.AsyncClient] = None
    
    async def initialize(self, agent) -> None:
        """Initialize the skill and register tools."""
        await super().initialize(agent)
        
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; WebAgents/1.0)'}
        )
        
        # Register tools
        self.register_tool(self.web_search)
        self.register_tool(self.search_duckduckgo)
        self.register_tool(self.search_google)
        self.register_tool(self.search_bing)
        self.register_tool(self.fetch_page)
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self._client:
            await self._client.aclose()
    
    @tool(description="Search the web for information")
    async def web_search(
        self,
        query: str,
        max_results: Optional[int] = None,
        engine: Optional[Literal['duckduckgo', 'google', 'bing']] = None
    ) -> Dict[str, Any]:
        """
        Search the web using the configured search engine.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            engine: Search engine to use (overrides default)
            
        Returns:
            Dictionary with search results
        """
        search_engine = engine or self.default_engine
        limit = max_results or self.max_results
        
        if search_engine == 'google':
            response = await self._search_google(query, limit)
        elif search_engine == 'bing':
            response = await self._search_bing(query, limit)
        else:
            response = await self._search_duckduckgo(query, limit)
        
        return {
            'query': response.query,
            'results': [
                {
                    'title': r.title,
                    'url': r.url,
                    'snippet': r.snippet,
                    'source': r.source
                }
                for r in response.results
            ],
            'total_results': response.total_results,
            'error': response.error
        }
    
    @tool(description="Search using DuckDuckGo (no API key required)")
    async def search_duckduckgo(
        self,
        query: str,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """Search using DuckDuckGo HTML search."""
        response = await self._search_duckduckgo(query, max_results)
        return {
            'query': response.query,
            'results': [
                {'title': r.title, 'url': r.url, 'snippet': r.snippet}
                for r in response.results
            ],
            'error': response.error
        }
    
    @tool(description="Search using Google Custom Search (requires API key)")
    async def search_google(
        self,
        query: str,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """Search using Google Custom Search API."""
        response = await self._search_google(query, max_results)
        return {
            'query': response.query,
            'results': [
                {'title': r.title, 'url': r.url, 'snippet': r.snippet}
                for r in response.results
            ],
            'total_results': response.total_results,
            'error': response.error
        }
    
    @tool(description="Search using Bing Search API (requires API key)")
    async def search_bing(
        self,
        query: str,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """Search using Bing Search API."""
        response = await self._search_bing(query, max_results)
        return {
            'query': response.query,
            'results': [
                {'title': r.title, 'url': r.url, 'snippet': r.snippet}
                for r in response.results
            ],
            'total_results': response.total_results,
            'error': response.error
        }
    
    @tool(description="Fetch and extract text content from a web page")
    async def fetch_page(
        self,
        url: str,
        max_length: int = 5000
    ) -> Dict[str, Any]:
        """
        Fetch a web page and extract its text content.
        
        Args:
            url: URL to fetch
            max_length: Maximum content length to return
            
        Returns:
            Dictionary with title and content
        """
        try:
            response = await self._client.get(url, follow_redirects=True)
            response.raise_for_status()
            
            html_content = response.text
            
            # Extract title
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', html_content, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else ''
            
            # Extract text content
            content = html_content
            # Remove scripts, styles, etc.
            content = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', content, flags=re.IGNORECASE)
            content = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', content, flags=re.IGNORECASE)
            content = re.sub(r'<noscript[^>]*>[\s\S]*?</noscript>', '', content, flags=re.IGNORECASE)
            content = re.sub(r'<!--[\s\S]*?-->', '', content)
            # Remove tags
            content = re.sub(r'<[^>]+>', ' ', content)
            # Decode entities
            content = html.unescape(content)
            # Clean whitespace
            content = re.sub(r'\s+', ' ', content).strip()
            
            # Truncate if needed
            if len(content) > max_length:
                content = content[:max_length] + '...'
            
            return {'title': title, 'content': content, 'url': url}
            
        except Exception as e:
            return {'title': '', 'content': '', 'url': url, 'error': str(e)}
    
    # Internal search implementations
    
    async def _search_duckduckgo(self, query: str, max_results: int) -> SearchResponse:
        """Search using DuckDuckGo HTML interface."""
        try:
            url = f"https://html.duckduckgo.com/html/?{urlencode({'q': query})}"
            response = await self._client.get(url)
            response.raise_for_status()
            
            html_content = response.text
            results = self._parse_duckduckgo_results(html_content, max_results)
            
            return SearchResponse(
                query=query,
                results=[SearchResult(**r, source='duckduckgo') for r in results]
            )
            
        except Exception as e:
            return SearchResponse(query=query, error=str(e))
    
    async def _search_google(self, query: str, max_results: int) -> SearchResponse:
        """Search using Google Custom Search API."""
        if not self.google_api_key or not self.google_cse_id:
            return SearchResponse(
                query=query,
                error="Google API key and CSE ID required"
            )
        
        try:
            params = {
                'key': self.google_api_key,
                'cx': self.google_cse_id,
                'q': query,
                'num': min(max_results, 10)
            }
            url = f"https://www.googleapis.com/customsearch/v1?{urlencode(params)}"
            
            response = await self._client.get(url)
            response.raise_for_status()
            
            data = response.json()
            results = [
                SearchResult(
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    snippet=item.get('snippet', ''),
                    source='google'
                )
                for item in data.get('items', [])
            ]
            
            total = int(data.get('searchInformation', {}).get('totalResults', 0))
            
            return SearchResponse(query=query, results=results, total_results=total)
            
        except Exception as e:
            return SearchResponse(query=query, error=str(e))
    
    async def _search_bing(self, query: str, max_results: int) -> SearchResponse:
        """Search using Bing Search API."""
        if not self.bing_api_key:
            return SearchResponse(
                query=query,
                error="Bing API key required"
            )
        
        try:
            params = {'q': query, 'count': max_results}
            url = f"https://api.bing.microsoft.com/v7.0/search?{urlencode(params)}"
            
            headers = {'Ocp-Apim-Subscription-Key': self.bing_api_key}
            response = await self._client.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            web_pages = data.get('webPages', {})
            
            results = [
                SearchResult(
                    title=item.get('name', ''),
                    url=item.get('url', ''),
                    snippet=item.get('snippet', ''),
                    source='bing'
                )
                for item in web_pages.get('value', [])
            ]
            
            total = web_pages.get('totalEstimatedMatches')
            
            return SearchResponse(query=query, results=results, total_results=total)
            
        except Exception as e:
            return SearchResponse(query=query, error=str(e))
    
    def _parse_duckduckgo_results(self, html_content: str, max_results: int) -> List[Dict[str, str]]:
        """Parse DuckDuckGo HTML results."""
        results = []
        
        # Pattern for result links and snippets
        pattern = r'<a class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>[\s\S]*?<a class="result__snippet"[^>]*>([^<]*(?:<[^>]+>[^<]*)*)</a>'
        
        for match in re.finditer(pattern, html_content, re.IGNORECASE):
            if len(results) >= max_results:
                break
                
            url = self._decode_duckduckgo_url(match.group(1))
            title = match.group(2).strip()
            snippet = re.sub(r'<[^>]+>', '', match.group(3)).strip()
            
            if url and title and 'duckduckgo.com' not in url:
                results.append({
                    'title': title,
                    'url': url,
                    'snippet': snippet
                })
        
        # Fallback: simpler parsing
        if not results:
            simple_pattern = r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]*)"[^>]*>([^<]+)</a>'
            for match in re.finditer(simple_pattern, html_content, re.IGNORECASE):
                if len(results) >= max_results:
                    break
                    
                url = self._decode_duckduckgo_url(match.group(1))
                title = match.group(2).strip()
                
                if url and title and 'duckduckgo.com' not in url:
                    results.append({
                        'title': title,
                        'url': url,
                        'snippet': ''
                    })
        
        return results
    
    def _decode_duckduckgo_url(self, url: str) -> str:
        """Decode DuckDuckGo redirect URL."""
        if '//duckduckgo.com/l/?' in url:
            match = re.search(r'uddg=([^&]+)', url)
            if match:
                return unquote(match.group(1))
        return url
