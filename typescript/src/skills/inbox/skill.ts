/**
 * The agent's own mailbox, for an SDK agent with no MCP connection
 * (portal build plan M2-03, the third of three doors).
 *
 * The platform gives an agent two ways to read and answer its inbox: the MCP
 * tools `inbox_read` and `inbox_reply`, and the HTTP pair this skill wraps.
 * An agent served from this SDK holds an AOAuth bearer and speaks HTTP to the
 * platform already; asking it to also open an MCP connection to read its own
 * queue would be a second credential and a second failure mode for the same
 * two calls.
 *
 * THE IDENTITY RULE, and it is the whole reason this file is careful. The
 * runtime compares scopes against `context.auth.user_id`, which for a
 * platform-hosted agent is its OWNER, not the agent. An inbox read keyed on
 * that identity would hand the agent its owner's conversations. So this skill
 * never derives an identity: the platform routes take the agent from the
 * BEARER, and where a caller must name one it names `agentId` from config.
 *
 * WHAT IS DELIBERATELY ABSENT: accepting or declining a first contact. Under
 * the platform's D-01 the acceptor is a policy the platform evaluates, never
 * a tool the agent holds, because an agent that can accept its own first
 * contacts is an agent that can be talked into accepting them.
 */

import { Skill } from '../../core/skill';
import { tool } from '../../core/decorators';
import type { Context } from '../../core/types';

export interface InboxSkillConfig {
  /** The platform base URL. Defaults to ROBUTLER_API_URL, then ROBUTLER_INTERNAL_API_URL. */
  portalApiUrl?: string;
  /**
   * The agent's platform bearer. Defaults to WEBAGENTS_AGENT_TOKEN, which is
   * what `registerWithPlatform` persists and what the heartbeat already uses.
   */
  token?: string;
  /**
   * The agent's platform id or handle, for the read door's path segment. When
   * absent the skill reads `/api/agents/me/inbox`, which the platform resolves
   * from the bearer.
   */
  agentId?: string;
  /** Request timeout, ms. */
  timeoutMs?: number;
}

interface InboxTurn {
  turnId: string;
  chatId: string;
  state: string;
  transport: string;
  dueAt: string;
  from: string | null;
  messages: Array<{ id: string; content: string; createdAt: string; from: string | null }>;
}

function envVar(name: string): string | undefined {
  return typeof process !== 'undefined' ? process.env?.[name] : undefined;
}

export class InboxSkill extends Skill {
  private portalApiUrl: string;
  private token?: string;
  private agentRef: string;
  private timeoutMs: number;

  constructor(config: InboxSkillConfig = {}) {
    super();
    this.portalApiUrl = (
      config.portalApiUrl
      || envVar('ROBUTLER_API_URL')
      || envVar('ROBUTLER_INTERNAL_API_URL')
      || 'http://localhost:3000'
    ).replace(/\/+$/, '');
    this.token = config.token || envVar('WEBAGENTS_AGENT_TOKEN');
    this.agentRef = config.agentId || 'me';
    this.timeoutMs = config.timeoutMs ?? 15_000;
  }

  private async call(path: string, init: RequestInit = {}): Promise<{ ok: boolean; status: number; body: unknown }> {
    if (!this.token) {
      return { ok: false, status: 0, body: { error: 'no_token' } };
    }
    const res = await fetch(`${this.portalApiUrl}${path}`, {
      ...init,
      headers: {
        authorization: `Bearer ${this.token}`,
        'content-type': 'application/json',
        ...(init.headers ?? {}),
      },
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    const body = await res.json().catch(() => null);
    return { ok: res.ok, status: res.status, body };
  }

  @tool({
    name: 'inbox_read',
    description:
      'Read the turns waiting for you: who wrote, what they said, and when each is due. '
      + 'Also reports how many first-time requests your owner has not decided yet. '
      + 'You cannot accept or decline those yourself; your owner does.',
    parameters: {
      type: 'object',
      properties: {
        limit: { type: 'number', description: 'Max turns to return (1-50). Default 50.' },
      },
    },
  })
  async inboxRead(params: { limit?: number }, _context: Context): Promise<string> {
    const q = params.limit ? `?limit=${Math.min(Math.max(1, Math.floor(params.limit)), 50)}` : '';
    const r = await this.call(`/api/agents/${encodeURIComponent(this.agentRef)}/inbox${q}`);
    if (!r.ok) {
      if (r.status === 0) return 'No platform token is configured, so there is no inbox to read.';
      return `Could not read the inbox (HTTP ${r.status}).`;
    }
    const out = r.body as { turns?: InboxTurn[]; pendingRequests?: number };
    const turns = out.turns ?? [];
    if (turns.length === 0) {
      return out.pendingRequests
        ? `Nothing waiting. ${out.pendingRequests} first-time request(s) are with your owner.`
        : 'Nothing waiting.';
    }
    // Rendered rather than dumped: the model reads this, and a JSON blob of
    // twenty messages per turn crowds out the conversation it belongs to.
    const lines = turns.map((t) => {
      const last = t.messages[t.messages.length - 1];
      return `- turn ${t.turnId} (${t.state}, ${t.transport}, due ${t.dueAt}) from ${t.from ?? 'unknown'}: `
        + `${(last?.content ?? '').slice(0, 400)}`;
    });
    const tail = out.pendingRequests
      ? `\n${out.pendingRequests} first-time request(s) are with your owner.`
      : '';
    return `${turns.length} waiting:\n${lines.join('\n')}${tail}`;
  }

  @tool({
    name: 'inbox_reply',
    description:
      'Answer one turn from your inbox. The reply is posted as you, into the conversation the turn belongs to.',
    parameters: {
      type: 'object',
      properties: {
        turn_id: { type: 'string', description: 'Turn id, from inbox_read.' },
        content: { type: 'string', description: 'What to say.' },
      },
      required: ['turn_id', 'content'],
    },
  })
  async inboxReply(params: { turn_id: string; content: string }, _context: Context): Promise<string> {
    const r = await this.call(`/api/turns/${encodeURIComponent(params.turn_id)}/reply`, {
      method: 'POST',
      body: JSON.stringify({ content: params.content }),
    });
    if (r.ok) return `Replied to turn ${params.turn_id}.`;
    if (r.status === 0) return 'No platform token is configured, so the reply was not sent.';
    // 409 is the one refusal worth distinguishing to the model: the turn is
    // real and yours, it is simply finished, and retrying will never work.
    if (r.status === 409) return `Turn ${params.turn_id} is already finished; nothing was sent.`;
    if (r.status === 404) return `Turn ${params.turn_id} was not found, or it is not yours.`;
    return `Could not reply to turn ${params.turn_id} (HTTP ${r.status}).`;
  }
}
