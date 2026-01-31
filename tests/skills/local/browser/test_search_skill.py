"""
WebSearchSkill Unit Tests
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from webagents.agents.skills.local.browser.search import WebSearchSkill, SearchResult, SearchResponse


class TestWebSearchSkillInit:
    """Test WebSearchSkill initialization."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        skill = WebSearchSkill()
        
        assert skill.default_engine == 'duckduckgo'
        assert skill.max_results == 10
        assert skill.timeout == 10
        assert skill.google_api_key is None
        assert skill.bing_api_key is None
    
    def test_custom_configuration(self):
        """Test custom configuration."""
        skill = WebSearchSkill(
            default_engine='google',
            max_results=20,
            timeout=5,
            google_api_key='test-key',
            google_cse_id='test-cse',
            bing_api_key='bing-key',
        )
        
        assert skill.default_engine == 'google'
        assert skill.max_results == 20
        assert skill.timeout == 5
        assert skill.google_api_key == 'test-key'
        assert skill.google_cse_id == 'test-cse'
        assert skill.bing_api_key == 'bing-key'


class TestDuckDuckGoSearch:
    """Test DuckDuckGo search functionality."""
    
    @pytest.fixture
    def skill(self):
        """Create a skill instance with mock client."""
        skill = WebSearchSkill()
        skill._client = AsyncMock(spec=httpx.AsyncClient)
        return skill
    
    @pytest.mark.asyncio
    async def test_successful_search(self, skill):
        """Test successful DuckDuckGo search."""
        mock_html = '''
        <html>
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage1">Example Page 1</a>
            <a class="result__snippet">This is the first result snippet</a>
        </html>
        '''
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = mock_html
        mock_response.raise_for_status = MagicMock()
        
        skill._client.get = AsyncMock(return_value=mock_response)
        
        result = await skill._search_duckduckgo('test query', 10)
        
        assert result.query == 'test query'
        assert result.error is None
    
    @pytest.mark.asyncio
    async def test_search_error_handling(self, skill):
        """Test error handling in DuckDuckGo search."""
        skill._client.get = AsyncMock(side_effect=Exception('Network error'))
        
        result = await skill._search_duckduckgo('test query', 10)
        
        assert result.query == 'test query'
        assert result.error == 'Network error'
        assert result.results == []
    
    def test_url_decoding(self):
        """Test DuckDuckGo URL decoding."""
        skill = WebSearchSkill()
        
        # Test encoded URL
        encoded = '//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpath'
        decoded = skill._decode_duckduckgo_url(encoded)
        assert decoded == 'https://example.com/path'
        
        # Test direct URL
        direct = 'https://example.com/direct'
        assert skill._decode_duckduckgo_url(direct) == direct


class TestGoogleSearch:
    """Test Google Custom Search functionality."""
    
    @pytest.fixture
    def skill(self):
        """Create a skill instance with Google credentials."""
        skill = WebSearchSkill(
            google_api_key='test-api-key',
            google_cse_id='test-cse-id',
        )
        skill._client = AsyncMock(spec=httpx.AsyncClient)
        return skill
    
    @pytest.mark.asyncio
    async def test_missing_credentials(self):
        """Test error when credentials are missing."""
        skill = WebSearchSkill()
        skill._client = AsyncMock(spec=httpx.AsyncClient)
        
        result = await skill._search_google('test query', 10)
        
        assert result.error == 'Google API key and CSE ID required'
    
    @pytest.mark.asyncio
    async def test_successful_search(self, skill):
        """Test successful Google search."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'items': [
                {'title': 'Result 1', 'link': 'https://example.com/1', 'snippet': 'Snippet 1'},
                {'title': 'Result 2', 'link': 'https://example.com/2', 'snippet': 'Snippet 2'},
            ],
            'searchInformation': {'totalResults': '1000'},
        }
        mock_response.raise_for_status = MagicMock()
        
        skill._client.get = AsyncMock(return_value=mock_response)
        
        result = await skill._search_google('test query', 10)
        
        assert result.query == 'test query'
        assert len(result.results) == 2
        assert result.results[0].title == 'Result 1'
        assert result.results[0].source == 'google'
        assert result.total_results == 1000


class TestBingSearch:
    """Test Bing Search functionality."""
    
    @pytest.fixture
    def skill(self):
        """Create a skill instance with Bing credentials."""
        skill = WebSearchSkill(bing_api_key='test-bing-key')
        skill._client = AsyncMock(spec=httpx.AsyncClient)
        return skill
    
    @pytest.mark.asyncio
    async def test_missing_api_key(self):
        """Test error when API key is missing."""
        skill = WebSearchSkill()
        skill._client = AsyncMock(spec=httpx.AsyncClient)
        
        result = await skill._search_bing('test query', 10)
        
        assert result.error == 'Bing API key required'
    
    @pytest.mark.asyncio
    async def test_successful_search(self, skill):
        """Test successful Bing search."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'webPages': {
                'value': [
                    {'name': 'Bing Result', 'url': 'https://example.com', 'snippet': 'Snippet'},
                ],
                'totalEstimatedMatches': 500,
            },
        }
        mock_response.raise_for_status = MagicMock()
        
        skill._client.get = AsyncMock(return_value=mock_response)
        
        result = await skill._search_bing('test query', 10)
        
        assert len(result.results) == 1
        assert result.results[0].title == 'Bing Result'
        assert result.results[0].source == 'bing'
        assert result.total_results == 500


class TestFetchPage:
    """Test page fetching functionality."""
    
    @pytest.fixture
    def skill(self):
        """Create a skill instance with mock client."""
        skill = WebSearchSkill()
        skill._client = AsyncMock(spec=httpx.AsyncClient)
        return skill
    
    @pytest.mark.asyncio
    async def test_content_extraction(self, skill):
        """Test HTML content extraction."""
        mock_html = '''
        <html>
            <head><title>Test Page</title></head>
            <body>
                <script>console.log('hidden');</script>
                <style>.hidden { display: none; }</style>
                <p>Main content here.</p>
                <noscript>No JS</noscript>
            </body>
        </html>
        '''
        
        mock_response = MagicMock()
        mock_response.text = mock_html
        mock_response.raise_for_status = MagicMock()
        
        skill._client.get = AsyncMock(return_value=mock_response)
        
        result = await skill.fetch_page('https://example.com', 5000)
        
        assert result['title'] == 'Test Page'
        assert 'Main content here' in result['content']
        assert 'console.log' not in result['content']
        assert 'No JS' not in result['content']
        assert result['url'] == 'https://example.com'
    
    @pytest.mark.asyncio
    async def test_content_truncation(self, skill):
        """Test content truncation."""
        long_content = 'a' * 10000
        mock_html = f'<html><body>{long_content}</body></html>'
        
        mock_response = MagicMock()
        mock_response.text = mock_html
        mock_response.raise_for_status = MagicMock()
        
        skill._client.get = AsyncMock(return_value=mock_response)
        
        result = await skill.fetch_page('https://example.com', 100)
        
        assert len(result['content']) <= 103  # 100 + '...'
        assert result['content'].endswith('...')
    
    @pytest.mark.asyncio
    async def test_error_handling(self, skill):
        """Test fetch error handling."""
        skill._client.get = AsyncMock(side_effect=Exception('Connection refused'))
        
        result = await skill.fetch_page('https://example.com', 5000)
        
        assert result['error'] == 'Connection refused'
        assert result['title'] == ''
        assert result['content'] == ''


class TestUnifiedSearch:
    """Test unified search interface."""
    
    @pytest.mark.asyncio
    async def test_default_engine_selection(self):
        """Test that default engine is used."""
        skill = WebSearchSkill(default_engine='duckduckgo')
        skill._client = AsyncMock(spec=httpx.AsyncClient)
        
        # Mock DuckDuckGo response
        mock_response = MagicMock()
        mock_response.text = '<html></html>'
        mock_response.raise_for_status = MagicMock()
        skill._client.get = AsyncMock(return_value=mock_response)
        
        result = await skill.web_search('test', max_results=5)
        
        assert result['query'] == 'test'
        # Verify DuckDuckGo was called
        call_url = skill._client.get.call_args[0][0]
        assert 'duckduckgo.com' in call_url


class TestDataClasses:
    """Test data classes."""
    
    def test_search_result(self):
        """Test SearchResult dataclass."""
        result = SearchResult(
            title='Test Title',
            url='https://example.com',
            snippet='Test snippet',
            source='google',
        )
        
        assert result.title == 'Test Title'
        assert result.url == 'https://example.com'
        assert result.snippet == 'Test snippet'
        assert result.source == 'google'
    
    def test_search_response(self):
        """Test SearchResponse dataclass."""
        response = SearchResponse(
            query='test query',
            results=[
                SearchResult('Title', 'https://example.com', 'Snippet', 'duckduckgo'),
            ],
            total_results=100,
        )
        
        assert response.query == 'test query'
        assert len(response.results) == 1
        assert response.total_results == 100
        assert response.error is None
