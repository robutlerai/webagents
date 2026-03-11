/**
 * Social Skills
 *
 * Platform social capabilities for agents:
 * - Chats: Manage chat sessions and message history
 * - Notifications: Send notifications to users with optional CTA buttons
 * - Publish: Publish content (posts, updates) to agent feeds
 * - PortalConnect: Connect to the portal for real-time features
 * - PortalWS: WebSocket connection management for real-time updates
 */

import { Skill } from '../../core/skill.js';
import { tool } from '../../core/decorators.js';
import type { Context } from '../../core/types.js';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

export interface SocialConfig {
  name?: string;
  enabled?: boolean;
  /** Portal API base URL */
  portalUrl?: string;
  /** API key */
  apiKey?: string;
  /** Agent ID */
  agentId?: string;
  /** Owner user ID (for notifications) */
  ownerId?: string;
}

async function portalFetch(
  baseUrl: string,
  path: string,
  apiKey: string | undefined,
  init?: RequestInit,
): Promise<Response> {
  const url = `${baseUrl.replace(/\/$/, '')}${path}`;
  const headers: Record<string, string> = {
    ...(init?.headers as Record<string, string> ?? {}),
  };
  if (apiKey) headers['Authorization'] = `Bearer ${apiKey}`;
  return fetch(url, { ...init, headers });
}

// ---------------------------------------------------------------------------
// Chats Skill
// ---------------------------------------------------------------------------

export class ChatsSkill extends Skill {
  private portalUrl: string;
  private apiKey?: string;
  private agentId?: string;

  constructor(config: SocialConfig = {}) {
    super({ ...config, name: config.name || 'chats' });
    this.portalUrl = config.portalUrl ?? process.env.PORTAL_URL ?? 'https://robutler.ai';
    this.apiKey = config.apiKey ?? process.env.PLATFORM_SERVICE_KEY;
    this.agentId = config.agentId;
  }

  @tool({
    name: 'chat_create',
    description: 'Create a new chat session.',
    parameters: {
      type: 'object',
      properties: {
        title: { type: 'string', description: 'Chat title' },
        participant_ids: { type: 'array', items: { type: 'string' }, description: 'User/agent IDs to include' },
        metadata: { type: 'object', description: 'Optional metadata' },
      },
      required: ['title'],
    },
  })
  async chatCreate(
    params: { title: string; participant_ids?: string[]; metadata?: Record<string, unknown> },
    context: Context,
  ): Promise<unknown> {
    const agentId = this.agentId ?? context.auth?.agentId;
    const res = await portalFetch(this.portalUrl, '/api/chats', this.apiKey, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: params.title,
        participantIds: params.participant_ids,
        metadata: params.metadata,
        agentId,
      }),
    });
    if (!res.ok) return { error: `Chat create failed: ${res.status}` };
    return res.json();
  }

  @tool({
    name: 'chat_list',
    description: 'List chat sessions for the current agent.',
    parameters: {
      type: 'object',
      properties: {
        limit: { type: 'number', description: 'Max results (default 50)' },
        offset: { type: 'number', description: 'Offset for pagination' },
      },
    },
  })
  async chatList(
    params: { limit?: number; offset?: number },
    context: Context,
  ): Promise<unknown> {
    const agentId = this.agentId ?? context.auth?.agentId;
    const qs = new URLSearchParams();
    if (agentId) qs.set('agentId', agentId);
    if (params.limit) qs.set('limit', String(params.limit));
    if (params.offset) qs.set('offset', String(params.offset));
    const res = await portalFetch(this.portalUrl, `/api/chats?${qs}`, this.apiKey);
    if (!res.ok) return { error: `Chat list failed: ${res.status}` };
    return res.json();
  }

  @tool({
    name: 'chat_get_messages',
    description: 'Get messages from a chat session.',
    parameters: {
      type: 'object',
      properties: {
        chat_id: { type: 'string', description: 'Chat ID' },
        limit: { type: 'number', description: 'Max messages to return' },
        before: { type: 'string', description: 'Cursor for pagination (message ID)' },
      },
      required: ['chat_id'],
    },
  })
  async chatGetMessages(
    params: { chat_id: string; limit?: number; before?: string },
    _context: Context,
  ): Promise<unknown> {
    const qs = new URLSearchParams();
    if (params.limit) qs.set('limit', String(params.limit));
    if (params.before) qs.set('before', params.before);
    const res = await portalFetch(
      this.portalUrl,
      `/api/chats/${params.chat_id}/messages?${qs}`,
      this.apiKey,
    );
    if (!res.ok) return { error: `Get messages failed: ${res.status}` };
    return res.json();
  }

  @tool({
    name: 'chat_send_message',
    description: 'Send a message to a chat session.',
    parameters: {
      type: 'object',
      properties: {
        chat_id: { type: 'string', description: 'Chat ID' },
        content: { type: 'string', description: 'Message content' },
        role: { type: 'string', enum: ['assistant', 'system'], description: 'Message role' },
      },
      required: ['chat_id', 'content'],
    },
  })
  async chatSendMessage(
    params: { chat_id: string; content: string; role?: string },
    context: Context,
  ): Promise<unknown> {
    const agentId = this.agentId ?? context.auth?.agentId;
    const res = await portalFetch(
      this.portalUrl,
      `/api/chats/${params.chat_id}/messages`,
      this.apiKey,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: params.content,
          role: params.role ?? 'assistant',
          agentId,
        }),
      },
    );
    if (!res.ok) return { error: `Send message failed: ${res.status}` };
    return res.json();
  }
}

// ---------------------------------------------------------------------------
// Notifications Skill (consolidated, fixed wiring, CTA support)
// ---------------------------------------------------------------------------

export class NotificationsSkill extends Skill {
  private portalUrl: string;
  private apiKey?: string;
  private agentId?: string;
  private ownerId?: string;

  constructor(config: SocialConfig = {}) {
    super({ ...config, name: config.name || 'notifications' });
    this.portalUrl = config.portalUrl ?? process.env.PORTAL_URL ?? 'https://robutler.ai';
    this.apiKey = config.apiKey ?? process.env.PLATFORM_SERVICE_KEY;
    this.agentId = config.agentId;
    this.ownerId = config.ownerId;
  }

  @tool({
    name: 'notify',
    description:
      'Send a notification to your owner (the user who created you), or check the ' +
      'status of a previously sent notification. Use this to alert the user about ' +
      'completed tasks, important events, or when you need their attention.\n\n' +
      'Optionally include action buttons (CTAs) for the user to approve, deny, view ' +
      'a link, or dismiss. All CTA clicks are tracked and queryable.\n\n' +
      'Notifications appear as push notifications and in the user\'s notification center.\n\n' +
      'Examples:\n' +
      '- Simple alert: action="send", title="Task complete", message="Image generated successfully"\n' +
      '- With approval: action="send", title="Deploy to prod?", message="v2.1.0 ready", ' +
      'actions=[{label:"Approve", action:"approve"}, {label:"Deny", action:"deny"}]\n' +
      '- With link: action="send", title="Report ready", message="Weekly report is ready", ' +
      'actions=[{label:"View report", action:"view", url:"/reports/weekly"}]\n' +
      '- Check status: action="status", notification_id="abc-123" → {actionTaken:"approve", ...}',
    parameters: {
      type: 'object',
      properties: {
        action: {
          type: 'string',
          enum: ['send', 'status'],
          description: 'Action to perform: "send" to create a notification, "status" to check a previous one (default: "send")',
        },
        title: { type: 'string', description: 'Notification title (required for send)' },
        message: { type: 'string', description: 'Notification body text (required for send)' },
        actions: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              label: { type: 'string', description: 'Button text' },
              action: { type: 'string', enum: ['approve', 'deny', 'view', 'dismiss'], description: 'Action type' },
              url: { type: 'string', description: 'Redirect URL (for "view" action)' },
            },
            required: ['label', 'action'],
          },
          description: 'CTA buttons (max 3, optional)',
        },
        notification_id: { type: 'string', description: 'Notification ID (required for status)' },
      },
    },
  })
  async notify(
    params: {
      action?: string;
      title?: string;
      message?: string;
      actions?: Array<{ label: string; action: string; url?: string }>;
      notification_id?: string;
    },
    context: Context,
  ): Promise<unknown> {
    const op = params.action ?? 'send';

    if (op === 'status') {
      if (!params.notification_id) return { error: 'notification_id is required for status check' };
      return this._checkStatus(params.notification_id);
    }

    if (!params.title) return { error: 'title is required' };
    if (!params.message) return { error: 'message is required' };

    const ownerId = this.ownerId ?? context.auth?.user_id;
    if (!ownerId) return { error: 'Cannot determine owner user ID' };

    const data: Record<string, unknown> = {
      agentName: this.agentId,
      priority: 'normal',
    };
    if (params.actions?.length) {
      data.actions = params.actions.slice(0, 3);
    }

    const res = await portalFetch(this.portalUrl, '/api/notifications/send', this.apiKey, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: params.title,
        body: params.message,
        type: 'agent_update',
        priority: 'normal',
        userIds: [ownerId],
        requireInteraction: !!(params.actions?.length),
        data,
      }),
    });

    if (!res.ok) {
      const errText = await res.text().catch(() => '');
      return { error: `Notification failed: ${res.status} ${errText.slice(0, 200)}` };
    }
    const result = await res.json();
    return {
      success: true,
      notification_id: result.notificationId || result.id,
      sent: result.sent,
    };
  }

  private async _checkStatus(notificationId: string): Promise<unknown> {
    const res = await portalFetch(
      this.portalUrl,
      `/api/notifications/${encodeURIComponent(notificationId)}`,
      this.apiKey,
    );
    if (!res.ok) {
      if (res.status === 404) return { error: 'Notification not found' };
      return { error: `Status check failed: ${res.status}` };
    }
    const data = await res.json();
    return {
      notification_id: notificationId,
      actionTaken: data.actionTaken ?? null,
      actionAt: data.updatedAt ?? null,
      isRead: data.isRead ?? false,
    };
  }
}

// ---------------------------------------------------------------------------
// Publish Skill
// ---------------------------------------------------------------------------

export class PublishSkill extends Skill {
  private portalUrl: string;
  private apiKey?: string;
  private agentId?: string;

  constructor(config: SocialConfig = {}) {
    super({ ...config, name: config.name || 'publish' });
    this.portalUrl = config.portalUrl ?? process.env.PORTAL_URL ?? 'https://robutler.ai';
    this.apiKey = config.apiKey ?? process.env.PLATFORM_SERVICE_KEY;
    this.agentId = config.agentId;
  }

  @tool({
    name: 'publish_post',
    description: 'Publish a post to the agent feed.',
    parameters: {
      type: 'object',
      properties: {
        content: { type: 'string', description: 'Post content (markdown supported)' },
        title: { type: 'string', description: 'Optional title' },
        tags: { type: 'array', items: { type: 'string' }, description: 'Tags' },
        visibility: { type: 'string', enum: ['public', 'followers', 'private'], description: 'Visibility' },
      },
      required: ['content'],
    },
  })
  async publishPost(
    params: { content: string; title?: string; tags?: string[]; visibility?: string },
    context: Context,
  ): Promise<unknown> {
    const agentId = this.agentId ?? context.auth?.agentId;
    const res = await portalFetch(this.portalUrl, '/api/feed/posts', this.apiKey, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        content: params.content,
        title: params.title,
        tags: params.tags,
        visibility: params.visibility ?? 'public',
        agentId,
      }),
    });
    if (!res.ok) return { error: `Publish failed: ${res.status}` };
    return res.json();
  }

  @tool({
    name: 'publish_update',
    description: 'Publish a status update for the agent.',
    parameters: {
      type: 'object',
      properties: {
        status: { type: 'string', description: 'Status message' },
        type: { type: 'string', enum: ['online', 'busy', 'maintenance', 'offline'], description: 'Status type' },
      },
      required: ['status'],
    },
  })
  async publishUpdate(
    params: { status: string; type?: string },
    context: Context,
  ): Promise<unknown> {
    const agentId = this.agentId ?? context.auth?.agentId;
    const res = await portalFetch(this.portalUrl, '/api/feed/status', this.apiKey, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        status: params.status,
        type: params.type ?? 'online',
        agentId,
      }),
    });
    if (!res.ok) return { error: `Status update failed: ${res.status}` };
    return res.json();
  }
}

// ---------------------------------------------------------------------------
// Portal Connect Skill
// ---------------------------------------------------------------------------

export class PortalConnectSkill extends Skill {
  private portalUrl: string;
  private apiKey?: string;
  private agentId?: string;

  constructor(config: SocialConfig = {}) {
    super({ ...config, name: config.name || 'portal-connect' });
    this.portalUrl = config.portalUrl ?? process.env.PORTAL_URL ?? 'https://robutler.ai';
    this.apiKey = config.apiKey ?? process.env.PLATFORM_SERVICE_KEY;
    this.agentId = config.agentId;
  }

  @tool({
    name: 'portal_register',
    description: 'Register this agent with the portal, making it discoverable.',
    parameters: {
      type: 'object',
      properties: {
        display_name: { type: 'string', description: 'Display name' },
        description: { type: 'string', description: 'Agent description' },
        capabilities: { type: 'array', items: { type: 'string' }, description: 'Capability tags' },
        endpoint: { type: 'string', description: 'Agent HTTP endpoint URL' },
      },
      required: ['display_name'],
    },
  })
  async portalRegister(
    params: { display_name: string; description?: string; capabilities?: string[]; endpoint?: string },
    context: Context,
  ): Promise<unknown> {
    const agentId = this.agentId ?? context.auth?.agentId;
    const res = await portalFetch(this.portalUrl, '/api/auth/agent/register', this.apiKey, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        agentId,
        displayName: params.display_name,
        description: params.description,
        capabilities: params.capabilities,
        endpoint: params.endpoint,
      }),
    });
    if (!res.ok) return { error: `Registration failed: ${res.status}` };
    return res.json();
  }

  @tool({
    name: 'portal_heartbeat',
    description: 'Send a heartbeat to the portal to maintain agent registration.',
    parameters: {
      type: 'object',
      properties: {
        status: { type: 'string', enum: ['online', 'busy', 'offline'], description: 'Current status' },
        load: { type: 'number', description: 'Current load (0-100)' },
      },
    },
  })
  async portalHeartbeat(
    params: { status?: string; load?: number },
    context: Context,
  ): Promise<string> {
    const agentId = this.agentId ?? context.auth?.agentId;
    const res = await portalFetch(this.portalUrl, '/api/agents/heartbeat', this.apiKey, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        agentId,
        status: params.status ?? 'online',
        load: params.load,
      }),
    });
    if (!res.ok) return `Heartbeat failed: ${res.status}`;
    return 'OK';
  }

  @tool({
    name: 'portal_deregister',
    description: 'Remove this agent from the portal registry.',
    parameters: { type: 'object', properties: {} },
  })
  async portalDeregister(_params: Record<string, unknown>, context: Context): Promise<string> {
    const agentId = this.agentId ?? context.auth?.agentId;
    const res = await portalFetch(
      this.portalUrl,
      `/api/agents/${agentId}/deregister`,
      this.apiKey,
      { method: 'POST' },
    );
    if (!res.ok) return `Deregister failed: ${res.status}`;
    return 'OK';
  }
}

// ---------------------------------------------------------------------------
// Portal WebSocket Skill
// ---------------------------------------------------------------------------

export class PortalWSSkill extends Skill {
  private portalUrl: string;
  private apiKey?: string;
  private agentId?: string;
  private ws: WebSocket | null = null;
  private messageHandlers = new Map<string, ((data: unknown) => void)[]>();

  constructor(config: SocialConfig = {}) {
    super({ ...config, name: config.name || 'portal-ws' });
    this.portalUrl = config.portalUrl ?? process.env.PORTAL_URL ?? 'https://robutler.ai';
    this.apiKey = config.apiKey ?? process.env.PLATFORM_SERVICE_KEY;
    this.agentId = config.agentId;
  }

  @tool({
    name: 'ws_connect',
    description: 'Open a WebSocket connection to the portal for real-time events.',
    parameters: { type: 'object', properties: {} },
  })
  async wsConnect(_params: Record<string, unknown>, context: Context): Promise<string> {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return 'Already connected';
    }

    const agentId = this.agentId ?? context.auth?.agentId;
    const wsUrl = this.portalUrl.replace(/^http/, 'ws');
    const qs = new URLSearchParams();
    if (agentId) qs.set('agentId', agentId);
    if (this.apiKey) qs.set('token', this.apiKey);

    this.ws = new WebSocket(`${wsUrl}/api/ws?${qs}`);

    return new Promise<string>((resolve) => {
      const timeout = setTimeout(() => resolve('Connection timeout'), 10_000);

      this.ws!.addEventListener('open', () => {
        clearTimeout(timeout);
        resolve('Connected');
      });

      this.ws!.addEventListener('message', (event) => {
        try {
          const data = JSON.parse(String(event.data));
          const type = data?.type as string;
          if (type) {
            const handlers = this.messageHandlers.get(type) ?? [];
            for (const h of handlers) h(data);
          }
        } catch {
          // Ignore non-JSON messages
        }
      });

      this.ws!.addEventListener('error', () => {
        clearTimeout(timeout);
        resolve('Connection error');
      });

      this.ws!.addEventListener('close', () => {
        this.ws = null;
      });
    });
  }

  @tool({
    name: 'ws_send',
    description: 'Send a message over the portal WebSocket.',
    parameters: {
      type: 'object',
      properties: {
        type: { type: 'string', description: 'Message type' },
        data: { type: 'object', description: 'Message payload' },
      },
      required: ['type', 'data'],
    },
  })
  async wsSend(
    params: { type: string; data: Record<string, unknown> },
    _context: Context,
  ): Promise<string> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return 'Not connected. Call ws_connect first.';
    }
    this.ws.send(JSON.stringify({ type: params.type, ...params.data }));
    return 'Sent';
  }

  @tool({
    name: 'ws_disconnect',
    description: 'Close the portal WebSocket connection.',
    parameters: { type: 'object', properties: {} },
  })
  async wsDisconnect(_params: Record<string, unknown>, _context: Context): Promise<string> {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    return 'Disconnected';
  }

  onMessage(type: string, handler: (data: unknown) => void): void {
    const handlers = this.messageHandlers.get(type) ?? [];
    handlers.push(handler);
    this.messageHandlers.set(type, handlers);
  }

  override async cleanup(): Promise<void> {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.messageHandlers.clear();
  }
}
