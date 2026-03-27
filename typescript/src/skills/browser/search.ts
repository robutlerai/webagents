/**
 * Web Search Skill
 * 
 * Provides tools for searching the web using various search engines.
 * Useful for agents that need to gather information from the internet.
 */

import { Skill } from '../../core/skill';
import { tool } from '../../core/decorators';

/**
 * Search result item
 */
export interface SearchResult {
  /** Result title */
  title: string;
  /** Result URL */
  url: string;
  /** Result snippet/description */
  snippet: string;
  /** Source search engine */
  source: 'duckduckgo' | 'google' | 'bing';
}

/**
 * Search response
 */
export interface SearchResponse {
  /** Search query */
  query: string;
  /** Search results */
  results: SearchResult[];
  /** Total results found (if available) */
  totalResults?: number;
  /** Error message if search failed */
  error?: string;
}

/**
 * Web Search Skill configuration
 */
export interface WebSearchConfig {
  /** Default search engine */
  defaultEngine?: 'duckduckgo' | 'google' | 'bing';
  /** Maximum results to return */
  maxResults?: number;
  /** Request timeout in ms */
  timeout?: number;
  /** Google API key (for Google Custom Search) */
  googleApiKey?: string;
  /** Google Custom Search Engine ID */
  googleCseId?: string;
  /** Bing API key */
  bingApiKey?: string;
}

/**
 * Web Search Skill
 * 
 * Provides web search capabilities using DuckDuckGo (no API key required),
 * Google Custom Search, or Bing Search APIs.
 * 
 * @example
 * ```typescript
 * const search = new WebSearchSkill();
 * 
 * // Search using DuckDuckGo (default, no API key needed)
 * const results = await search.search('TypeScript tutorial');
 * 
 * // Search with specific engine
 * const googleResults = await search.searchGoogle('React hooks');
 * ```
 */
export class WebSearchSkill extends Skill {
  private searchConfig: Required<WebSearchConfig>;

  /** Unique skill identifier */
  readonly id: string = 'web-search';
  
  /** Skill description */
  readonly description: string = 'Search the web using DuckDuckGo, Google, or Bing';

  constructor(config: WebSearchConfig = {}) {
    super({ name: 'Web Search' });
    this.searchConfig = {
      defaultEngine: config.defaultEngine || 'duckduckgo',
      maxResults: config.maxResults || 10,
      timeout: config.timeout || 10000,
      googleApiKey: config.googleApiKey || '',
      googleCseId: config.googleCseId || '',
      bingApiKey: config.bingApiKey || '',
    };
  }

  // ============================================================================
  // Search Tools
  // ============================================================================

  /**
   * Search the web using the default search engine
   */
  @tool({
    name: 'web_search',
    description: 'Search the web for information',
    parameters: {
      query: {
        type: 'string',
        description: 'Search query',
      },
      maxResults: {
        type: 'number',
        description: 'Maximum number of results (default: 10)',
      },
    },
  })
  async search(query: string, maxResults?: number): Promise<SearchResponse> {
    const limit = maxResults || this.searchConfig.maxResults;
    
    switch (this.searchConfig.defaultEngine) {
      case 'google':
        return this.searchGoogle(query, limit);
      case 'bing':
        return this.searchBing(query, limit);
      default:
        return this.searchDuckDuckGo(query, limit);
    }
  }

  /**
   * Search using DuckDuckGo (no API key required)
   */
  @tool({
    name: 'search_duckduckgo',
    description: 'Search the web using DuckDuckGo (no API key required)',
    parameters: {
      query: {
        type: 'string',
        description: 'Search query',
      },
      maxResults: {
        type: 'number',
        description: 'Maximum number of results (default: 10)',
      },
    },
  })
  async searchDuckDuckGo(query: string, maxResults: number = 10): Promise<SearchResponse> {
    try {
      // DuckDuckGo HTML search (works without API)
      const encodedQuery = encodeURIComponent(query);
      const url = `https://html.duckduckgo.com/html/?q=${encodedQuery}`;
      
      const response = await fetch(url, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (compatible; WebAgents/1.0)',
        },
        signal: AbortSignal.timeout(this.searchConfig.timeout),
      });

      if (!response.ok) {
        return { query, results: [], error: `HTTP ${response.status}` };
      }

      const html = await response.text();
      const results = this.parseDuckDuckGoResults(html, maxResults);
      
      return {
        query,
        results: results.map(r => ({ ...r, source: 'duckduckgo' as const })),
      };
    } catch (error) {
      return { query, results: [], error: (error as Error).message };
    }
  }

  /**
   * Search using Google Custom Search API
   */
  @tool({
    name: 'search_google',
    description: 'Search the web using Google Custom Search (requires API key)',
    parameters: {
      query: {
        type: 'string',
        description: 'Search query',
      },
      maxResults: {
        type: 'number',
        description: 'Maximum number of results (default: 10)',
      },
    },
  })
  async searchGoogle(query: string, maxResults: number = 10): Promise<SearchResponse> {
    if (!this.searchConfig.googleApiKey || !this.searchConfig.googleCseId) {
      return { query, results: [], error: 'Google API key and CSE ID required' };
    }

    try {
      const encodedQuery = encodeURIComponent(query);
      const url = `https://www.googleapis.com/customsearch/v1?key=${this.searchConfig.googleApiKey}&cx=${this.searchConfig.googleCseId}&q=${encodedQuery}&num=${Math.min(maxResults, 10)}`;
      
      const response = await fetch(url, {
        signal: AbortSignal.timeout(this.searchConfig.timeout),
      });

      if (!response.ok) {
        return { query, results: [], error: `HTTP ${response.status}` };
      }

      const data = await response.json();
      const results: SearchResult[] = (data.items || []).map((item: { title: string; link: string; snippet: string }) => ({
        title: item.title,
        url: item.link,
        snippet: item.snippet,
        source: 'google' as const,
      }));

      return {
        query,
        results,
        totalResults: parseInt(data.searchInformation?.totalResults || '0', 10),
      };
    } catch (error) {
      return { query, results: [], error: (error as Error).message };
    }
  }

  /**
   * Search using Bing Search API
   */
  @tool({
    name: 'search_bing',
    description: 'Search the web using Bing Search API (requires API key)',
    parameters: {
      query: {
        type: 'string',
        description: 'Search query',
      },
      maxResults: {
        type: 'number',
        description: 'Maximum number of results (default: 10)',
      },
    },
  })
  async searchBing(query: string, maxResults: number = 10): Promise<SearchResponse> {
    if (!this.searchConfig.bingApiKey) {
      return { query, results: [], error: 'Bing API key required' };
    }

    try {
      const encodedQuery = encodeURIComponent(query);
      const url = `https://api.bing.microsoft.com/v7.0/search?q=${encodedQuery}&count=${maxResults}`;
      
      const response = await fetch(url, {
        headers: {
          'Ocp-Apim-Subscription-Key': this.searchConfig.bingApiKey,
        },
        signal: AbortSignal.timeout(this.searchConfig.timeout),
      });

      if (!response.ok) {
        return { query, results: [], error: `HTTP ${response.status}` };
      }

      const data = await response.json();
      const results: SearchResult[] = (data.webPages?.value || []).map((item: { name: string; url: string; snippet: string }) => ({
        title: item.name,
        url: item.url,
        snippet: item.snippet,
        source: 'bing' as const,
      }));

      return {
        query,
        results,
        totalResults: data.webPages?.totalEstimatedMatches,
      };
    } catch (error) {
      return { query, results: [], error: (error as Error).message };
    }
  }

  /**
   * Fetch and extract text content from a URL
   */
  @tool({
    name: 'fetch_page',
    description: 'Fetch a web page and extract its text content',
    parameters: {
      url: {
        type: 'string',
        description: 'URL to fetch',
      },
      maxLength: {
        type: 'number',
        description: 'Maximum content length (default: 5000)',
      },
    },
  })
  async fetchPage(url: string, maxLength: number = 5000): Promise<{ title: string; content: string; url: string; error?: string }> {
    try {
      const response = await fetch(url, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (compatible; WebAgents/1.0)',
        },
        signal: AbortSignal.timeout(this.searchConfig.timeout),
      });

      if (!response.ok) {
        return { title: '', content: '', url, error: `HTTP ${response.status}` };
      }

      const html = await response.text();
      
      // Extract title
      const titleMatch = html.match(/<title[^>]*>([^<]+)<\/title>/i);
      const title = titleMatch ? titleMatch[1].trim() : '';
      
      // Extract text content (basic extraction)
      let content = html
        // Remove scripts, styles, etc.
        .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '')
        .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '')
        .replace(/<noscript[^>]*>[\s\S]*?<\/noscript>/gi, '')
        .replace(/<!--[\s\S]*?-->/g, '')
        // Remove tags
        .replace(/<[^>]+>/g, ' ')
        // Decode entities
        .replace(/&nbsp;/g, ' ')
        .replace(/&amp;/g, '&')
        .replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>')
        .replace(/&quot;/g, '"')
        // Clean whitespace
        .replace(/\s+/g, ' ')
        .trim();
      
      // Truncate if needed
      if (content.length > maxLength) {
        content = content.slice(0, maxLength) + '...';
      }

      return { title, content, url };
    } catch (error) {
      return { title: '', content: '', url, error: (error as Error).message };
    }
  }

  // ============================================================================
  // Internal Methods
  // ============================================================================

  /**
   * Parse DuckDuckGo HTML results
   */
  private parseDuckDuckGoResults(html: string, maxResults: number): Omit<SearchResult, 'source'>[] {
    const results: Omit<SearchResult, 'source'>[] = [];
    
    // Match result blocks
    const resultRegex = /<a class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)<\/a>[\s\S]*?<a class="result__snippet"[^>]*>([^<]*(?:<[^>]+>[^<]*)*)<\/a>/gi;
    
    let match;
    while ((match = resultRegex.exec(html)) !== null && results.length < maxResults) {
      const url = this.decodeDuckDuckGoUrl(match[1]);
      const title = match[2].trim();
      const snippet = match[3].replace(/<[^>]+>/g, '').trim();
      
      if (url && title) {
        results.push({ title, url, snippet });
      }
    }
    
    // Fallback: simpler parsing
    if (results.length === 0) {
      const simpleRegex = /<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]*)"[^>]*>([^<]+)<\/a>/gi;
      while ((match = simpleRegex.exec(html)) !== null && results.length < maxResults) {
        const url = this.decodeDuckDuckGoUrl(match[1]);
        const title = match[2].trim();
        if (url && title && !url.includes('duckduckgo.com')) {
          results.push({ title, url, snippet: '' });
        }
      }
    }

    return results;
  }

  /**
   * Decode DuckDuckGo redirect URL
   */
  private decodeDuckDuckGoUrl(url: string): string {
    if (url.includes('//duckduckgo.com/l/?')) {
      const match = url.match(/uddg=([^&]+)/);
      if (match) {
        return decodeURIComponent(match[1]);
      }
    }
    return url;
  }
}
