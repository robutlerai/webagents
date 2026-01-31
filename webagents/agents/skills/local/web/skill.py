"""
Web Fetch Skill

Local tool to fetch and process web content, matching Gemini CLI specification.
"""

import re
import asyncio
import httpx
import trafilatura
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from ...base import Skill
from webagents.agents.tools.decorators import tool

class WebSkill(Skill):
    """Web operations matching Gemini CLI specs"""
    
    def __init__(self, session=None, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.session = session
        
        # Configure client with contact info in User-Agent as per Wikimedia policy
        # https://meta.wikimedia.org/wiki/User-Agent_policy
        self.client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=30.0,
            headers={
                "User-Agent": "WebAgentsBot/1.0 (https://github.com/robutler/webagents; contact@robutler.ai) python-httpx/0.24.0"
            }
        )

    @tool
    async def web_fetch(self, prompt: str) -> str:
        """Summarize, compare, or extract information from web pages.
        
        The web_fetch tool processes content from one or more URLs (up to 20) embedded
        in a prompt. web_fetch takes a natural language prompt and returns the 
        fetched content for you to process.
        
        Args:
            prompt: A comprehensive prompt that includes the URL(s) (up to 20) to fetch 
                   and specific instructions on how to process their content.
                   Example: "Summarize https://example.com/article and extract key points from https://another.com/data"
                   The prompt must contain at least one URL starting with http:// or https://.
            
        Returns:
            The content fetched from the URLs formatted for processing.
        """
        # Extract URLs
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*'
        urls = re.findall(url_pattern, prompt)
        
        if not urls:
            return "Error: No URLs found in prompt. The prompt must contain at least one URL starting with http:// or https://."
        
        if len(urls) > 20:
             return "Error: Too many URLs. Maximum 20 URLs allowed."
             
        # Ask for confirmation if session is available
        if self.session and hasattr(self.session, 'confirm'):
            confirmed = await self.session.confirm(f"Fetch content from {len(urls)} URL(s)?\n" + "\n".join([f"- {u}" for u in urls]))
            if not confirmed:
                return "Error: User declined the request to fetch URLs."
        
        # Fetch contents
        results = []
        for url in urls:
            try:
                response = await self.client.get(url)
                response.raise_for_status()
                
                # Use trafilatura for intelligent content extraction and Markdown conversion
                content = trafilatura.extract(
                    response.text, 
                    output_format='markdown',
                    include_links=True,
                    include_images=False,
                    include_tables=True
                )
                
                if not content:
                    # Fallback to simple cleaning if trafilatura fails to find main content
                    content = response.text
                    content = re.sub(r'<(script|style).*?>.*?</\1>', '', content, flags=re.DOTALL | re.IGNORECASE)
                    content = re.sub(r'<[^>]+>', ' ', content)
                    content = re.sub(r'\s+', ' ', content).strip()
                
                # Truncate if too long (rough limit for context)
                if len(content) > 50000:
                    content = content[:50000] + "... [Content Truncated]"
                
                results.append(f"--- SOURCE: {url} ---\n{content}\n")
                
            except Exception as e:
                results.append(f"--- SOURCE: {url} ---\nError fetching URL: {str(e)}\n")
        
        combined_content = "\n".join(results)
        
        # Return the content to the agent
        return (
            f"I have fetched the content from the URLs. Please process the following information "
            f"based on the original prompt: \"{prompt}\"\n\n{combined_content}"
        )

    def __del__(self):
        # Close the client when the skill is destroyed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.client.aclose())
            else:
                asyncio.run(self.client.aclose())
        except:
            pass
