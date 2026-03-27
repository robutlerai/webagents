/**
 * RAG Skill (Retrieval-Augmented Generation)
 *
 * Vector embedding + semantic search for agent knowledge bases.
 * Supports multiple backends:
 * - In-memory (development, small datasets)
 * - Portal API (production, Milvus-backed)
 *
 * Agents can ingest documents, chunk them, embed, and search
 * semantically to augment their responses.
 */

import { Skill } from '../../core/skill';
import { tool } from '../../core/decorators';
import type { Context } from '../../core/types';

export interface RAGConfig {
  name?: string;
  enabled?: boolean;
  /** Backend: 'memory' (default) or 'portal' */
  backend?: 'memory' | 'portal';
  /** Portal API URL for 'portal' backend */
  portalUrl?: string;
  /** API key */
  apiKey?: string;
  /** Collection name (default: 'default') */
  collection?: string;
  /** Chunk size in characters (default: 1000) */
  chunkSize?: number;
  /** Chunk overlap in characters (default: 200) */
  chunkOverlap?: number;
  /** Max results for search (default: 5) */
  topK?: number;
}

interface VectorEntry {
  id: string;
  text: string;
  vector: number[];
  metadata: Record<string, unknown>;
}

export class RAGSkill extends Skill {
  private store = new Map<string, VectorEntry>();
  private backend: 'memory' | 'portal';
  private portalUrl?: string;
  private apiKey?: string;
  private collection: string;
  private chunkSize: number;
  private chunkOverlap: number;
  private topK: number;

  constructor(config: RAGConfig = {}) {
    super({ ...config, name: config.name || 'rag' });
    this.backend = config.backend ?? 'memory';
    this.portalUrl = config.portalUrl ?? process.env.PORTAL_URL;
    this.apiKey = config.apiKey ?? process.env.PLATFORM_SERVICE_KEY;
    this.collection = config.collection ?? 'default';
    this.chunkSize = config.chunkSize ?? 1000;
    this.chunkOverlap = config.chunkOverlap ?? 200;
    this.topK = config.topK ?? 5;
  }

  private chunkText(text: string): string[] {
    const chunks: string[] = [];
    let start = 0;
    while (start < text.length) {
      const end = Math.min(start + this.chunkSize, text.length);
      chunks.push(text.slice(start, end));
      start += this.chunkSize - this.chunkOverlap;
      if (start >= text.length) break;
    }
    return chunks;
  }

  /**
   * Simple TF-based embedding for the in-memory backend.
   * In production, use the portal backend which calls a real embedding model.
   */
  private simpleEmbed(text: string): number[] {
    const words = text.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/);
    const vocab = new Map<string, number>();
    for (const w of words) {
      vocab.set(w, (vocab.get(w) ?? 0) + 1);
    }
    const sorted = [...vocab.entries()].sort((a, b) => b[1] - a[1]).slice(0, 128);
    const vec = new Array(128).fill(0);
    sorted.forEach(([_, count], i) => {
      vec[i] = count / words.length;
    });
    return vec;
  }

  private cosineSim(a: number[], b: number[]): number {
    let dot = 0, magA = 0, magB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      magA += a[i] * a[i];
      magB += b[i] * b[i];
    }
    const denom = Math.sqrt(magA) * Math.sqrt(magB);
    return denom === 0 ? 0 : dot / denom;
  }

  @tool({
    name: 'rag_ingest',
    description: 'Ingest a document into the knowledge base. Text is chunked, embedded, and stored for semantic search.',
    parameters: {
      type: 'object',
      properties: {
        text: { type: 'string', description: 'Document text to ingest' },
        source: { type: 'string', description: 'Source identifier (e.g. filename, URL)' },
        metadata: { type: 'object', description: 'Optional metadata to attach to chunks' },
      },
      required: ['text'],
    },
  })
  async ragIngest(
    params: { text: string; source?: string; metadata?: Record<string, unknown> },
    _context: Context,
  ): Promise<{ chunks: number }> {
    if (this.backend === 'portal' && this.portalUrl) {
      const res = await fetch(`${this.portalUrl}/api/rag/ingest`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {}),
        },
        body: JSON.stringify({
          text: params.text,
          source: params.source,
          metadata: params.metadata,
          collection: this.collection,
          chunkSize: this.chunkSize,
          chunkOverlap: this.chunkOverlap,
        }),
      });
      const body = await res.json() as { chunks?: number };
      return { chunks: body.chunks ?? 0 };
    }

    const chunks = this.chunkText(params.text);
    for (let i = 0; i < chunks.length; i++) {
      const id = `${params.source ?? 'doc'}_chunk_${i}_${Date.now()}`;
      this.store.set(id, {
        id,
        text: chunks[i],
        vector: this.simpleEmbed(chunks[i]),
        metadata: {
          ...params.metadata,
          source: params.source,
          chunkIndex: i,
          totalChunks: chunks.length,
        },
      });
    }
    return { chunks: chunks.length };
  }

  @tool({
    name: 'rag_search',
    description: 'Semantic search over ingested documents. Returns the most relevant text chunks.',
    parameters: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Search query' },
        top_k: { type: 'number', description: 'Number of results (default 5)' },
        filter: { type: 'object', description: 'Metadata filter' },
      },
      required: ['query'],
    },
  })
  async ragSearch(
    params: { query: string; top_k?: number; filter?: Record<string, unknown> },
    _context: Context,
  ): Promise<Array<{ text: string; score: number; metadata: Record<string, unknown> }>> {
    const topK = params.top_k ?? this.topK;

    if (this.backend === 'portal' && this.portalUrl) {
      const res = await fetch(`${this.portalUrl}/api/rag/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {}),
        },
        body: JSON.stringify({
          query: params.query,
          topK,
          filter: params.filter,
          collection: this.collection,
        }),
      });
      return res.json() as Promise<Array<{ text: string; score: number; metadata: Record<string, unknown> }>>;
    }

    const queryVec = this.simpleEmbed(params.query);
    const scored = [...this.store.values()]
      .map((entry) => ({
        text: entry.text,
        score: this.cosineSim(queryVec, entry.vector),
        metadata: entry.metadata,
      }))
      .filter((r) => {
        if (!params.filter) return true;
        return Object.entries(params.filter).every(
          ([k, v]) => r.metadata[k] === v,
        );
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);

    return scored;
  }

  @tool({
    name: 'rag_delete',
    description: 'Delete documents from the knowledge base by source.',
    parameters: {
      type: 'object',
      properties: {
        source: { type: 'string', description: 'Source identifier to delete' },
      },
      required: ['source'],
    },
  })
  async ragDelete(params: { source: string }, _context: Context): Promise<{ deleted: number }> {
    if (this.backend === 'portal' && this.portalUrl) {
      const res = await fetch(`${this.portalUrl}/api/rag/delete`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {}),
        },
        body: JSON.stringify({ source: params.source, collection: this.collection }),
      });
      return res.json() as Promise<{ deleted: number }>;
    }

    let deleted = 0;
    for (const [id, entry] of this.store) {
      if (entry.metadata.source === params.source) {
        this.store.delete(id);
        deleted++;
      }
    }
    return { deleted };
  }

  @tool({
    name: 'rag_stats',
    description: 'Get statistics about the knowledge base.',
    parameters: { type: 'object', properties: {} },
  })
  async ragStats(
    _params: Record<string, unknown>,
    _context: Context,
  ): Promise<{ totalChunks: number; sources: string[] }> {
    if (this.backend === 'portal' && this.portalUrl) {
      const res = await fetch(`${this.portalUrl}/api/rag/stats?collection=${this.collection}`, {
        headers: this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {},
      });
      return res.json() as Promise<{ totalChunks: number; sources: string[] }>;
    }

    const sources = new Set<string>();
    for (const entry of this.store.values()) {
      if (entry.metadata.source) sources.add(entry.metadata.source as string);
    }
    return { totalChunks: this.store.size, sources: [...sources] };
  }

  override async cleanup(): Promise<void> {
    this.store.clear();
  }
}
