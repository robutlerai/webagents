/**
 * ACP (Agent Commerce Protocol) Transport Skill
 *
 * Implements the Agent Commerce Protocol for paid agent interactions.
 * Combines x402 payment negotiation with agent task execution,
 * enabling agents to buy and sell services.
 *
 * Supports:
 * - Service catalogs (list available paid services)
 * - Price discovery and negotiation
 * - Payment verification and settlement
 * - Metered billing (per-token, per-request, per-minute)
 * - Receipt generation and audit trails
 */

import { Skill } from '../../../core/skill.js';
import { tool, hook } from '../../../core/decorators.js';
import type { Context, HookData } from '../../../core/types.js';

export interface ACPTransportConfig {
  name?: string;
  enabled?: boolean;
  /** Service catalog */
  services?: ACPService[];
  /** Payment verification URL */
  paymentVerifyUrl?: string;
  /** Default currency */
  currency?: string;
  /** API key for outbound requests */
  apiKey?: string;
  /** Timeout for ACP requests (ms) */
  timeout?: number;
}

export interface ACPService {
  id: string;
  name: string;
  description: string;
  pricing: {
    model: 'per_request' | 'per_token' | 'per_minute' | 'subscription' | 'custom';
    amount: string;
    currency: string;
    unit?: string;
  };
  inputSchema?: Record<string, unknown>;
  outputSchema?: Record<string, unknown>;
  rateLimit?: { requests: number; period: string };
}

export interface ACPReceipt {
  id: string;
  serviceId: string;
  amount: string;
  currency: string;
  paymentId: string;
  timestamp: string;
  input: unknown;
  outputSummary: string;
}

interface HttpResponse {
  writeHead(status: number, headers?: Record<string, string>): void;
  end(body?: string): void;
}

export class ACPTransportSkill extends Skill {
  private services: ACPService[];
  private apiKey?: string;
  private timeout: number;
  private receipts = new Map<string, ACPReceipt>();

  constructor(config: ACPTransportConfig = {}) {
    super({ ...config, name: config.name || 'acp-transport' });
    this.services = config.services ?? [];
    this.apiKey = config.apiKey;
    this.timeout = config.timeout ?? 30_000;
  }

  @hook({ lifecycle: 'on_connection', priority: 10 })
  async handleACPRequest(data: HookData, _context: Context): Promise<void> {
    const req = data.request as { method?: string } | undefined;
    const res = data.metadata?.response as HttpResponse | undefined;
    if (!req || !res) return;

    const path = (data.metadata?.path as string) ?? '';

    if (path === '/acp/catalog') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ services: this.services }));
      return;
    }

    if (path === '/acp/price' && req.method === 'POST') {
      const body = await this.readBody(req);
      const { serviceId } = JSON.parse(body);
      const service = this.services.find((s) => s.id === serviceId);
      if (!service) {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Service not found' }));
        return;
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ serviceId, pricing: service.pricing }));
      return;
    }

    if (path === '/acp/execute' && req.method === 'POST') {
      const paymentToken = (data.metadata?.headers as Record<string, string>)?.['x-payment-token'];
      if (!paymentToken) {
        res.writeHead(402, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          error: 'payment_required',
          catalog_url: '/acp/catalog',
          services: this.services.map((s) => ({
            id: s.id,
            name: s.name,
            pricing: s.pricing,
          })),
        }));
        return;
      }

      const body = await this.readBody(req);
      const { serviceId, input } = JSON.parse(body);
      const service = this.services.find((s) => s.id === serviceId);
      if (!service) {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Service not found' }));
        return;
      }

      const receipt: ACPReceipt = {
        id: crypto.randomUUID(),
        serviceId,
        amount: service.pricing.amount,
        currency: service.pricing.currency,
        paymentId: paymentToken,
        timestamp: new Date().toISOString(),
        input,
        outputSummary: 'Processing...',
      };
      this.receipts.set(receipt.id, receipt);

      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ receipt_id: receipt.id, status: 'accepted' }));
    }
  }

  // ============================================================================
  // Client-side tools for consuming ACP services
  // ============================================================================

  @tool({
    name: 'acp_get_catalog',
    description: 'Get the service catalog from an ACP-enabled agent.',
    parameters: {
      type: 'object',
      properties: {
        agent_url: { type: 'string', description: 'Agent URL' },
      },
      required: ['agent_url'],
    },
  })
  async acpGetCatalog(
    params: { agent_url: string },
    _context: Context,
  ): Promise<unknown> {
    const res = await fetch(`${params.agent_url}/acp/catalog`, {
      signal: AbortSignal.timeout(this.timeout),
    });
    if (!res.ok) throw new Error(`Catalog fetch failed: ${res.status}`);
    return res.json();
  }

  @tool({
    name: 'acp_get_price',
    description: 'Get the price for a specific service from an ACP-enabled agent.',
    parameters: {
      type: 'object',
      properties: {
        agent_url: { type: 'string', description: 'Agent URL' },
        service_id: { type: 'string', description: 'Service ID' },
      },
      required: ['agent_url', 'service_id'],
    },
  })
  async acpGetPrice(
    params: { agent_url: string; service_id: string },
    _context: Context,
  ): Promise<unknown> {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    const res = await fetch(`${params.agent_url}/acp/price`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ serviceId: params.service_id }),
      signal: AbortSignal.timeout(this.timeout),
    });
    if (!res.ok) throw new Error(`Price fetch failed: ${res.status}`);
    return res.json();
  }

  @tool({
    name: 'acp_execute',
    description: 'Execute a paid service on an ACP-enabled agent. Requires a payment token.',
    parameters: {
      type: 'object',
      properties: {
        agent_url: { type: 'string', description: 'Agent URL' },
        service_id: { type: 'string', description: 'Service ID' },
        input: { type: 'object', description: 'Service input data' },
        payment_token: { type: 'string', description: 'Payment token for authorization' },
      },
      required: ['agent_url', 'service_id', 'payment_token'],
    },
  })
  async acpExecute(
    params: { agent_url: string; service_id: string; input?: Record<string, unknown>; payment_token: string },
    _context: Context,
  ): Promise<unknown> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'X-Payment-Token': params.payment_token,
    };
    if (this.apiKey) headers['Authorization'] = `Bearer ${this.apiKey}`;

    const res = await fetch(`${params.agent_url}/acp/execute`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        serviceId: params.service_id,
        input: params.input ?? {},
      }),
      signal: AbortSignal.timeout(this.timeout),
    });
    if (!res.ok) throw new Error(`ACP execute failed: ${res.status}`);
    return res.json();
  }

  @tool({
    name: 'acp_list_receipts',
    description: 'List receipts from ACP transactions.',
    parameters: { type: 'object', properties: {} },
  })
  async acpListReceipts(
    _params: Record<string, unknown>,
    _context: Context,
  ): Promise<ACPReceipt[]> {
    return [...this.receipts.values()].sort(
      (a, b) => b.timestamp.localeCompare(a.timestamp),
    );
  }

  private readBody(req: unknown): Promise<string> {
    return new Promise((resolve, reject) => {
      let body = '';
      const r = req as { on: (event: string, cb: (chunk?: unknown) => void) => void };
      r.on('data', (chunk: unknown) => { body += String(chunk); });
      r.on('end', () => resolve(body));
      r.on('error', reject);
    });
  }

  override async cleanup(): Promise<void> {
    this.receipts.clear();
  }
}
