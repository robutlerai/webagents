import type { ExtensionConfig, AgentStatus, TaskRecord } from '../shared/types';
import { loadConfig, saveConfig } from '../shared/storage';
import { BaseAgent } from '../../core/agent';
import { BrowserControlSkill } from '../../skills/browser';
import type { ClientEvent, ServerEvent } from '../../uamp/events';
import { createInputTextEvent, createSessionCreateEvent } from '../../uamp/events';
import type { ContentItem } from '../../uamp/types';
import { ChromeBrowserControlAdapter } from './chrome-browser-adapter';
import { PortalChatCompletionsSkill } from './portal-chat-completions-skill';

export class ExtensionAgentRuntime {
  private config: ExtensionConfig | null = null;
  private ws: WebSocket | null = null;
  private connected = false;
  private registered = false;
  private startTime = 0;
  private taskCount = 0;
  private lastActivity: number | null = null;
  private tasks: TaskRecord[] = [];
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectDelay = 1000;
  private toolCallCount = 0;
  private toolCallWindowStart = 0;
  private agent: BaseAgent | null = null;
  private pendingConnect: {
    resolve: () => void;
    reject: (err: Error) => void;
    timeout: ReturnType<typeof setTimeout>;
  } | null = null;

  async initialize(): Promise<void> {
    this.config = await loadConfig();
    this.startTime = Date.now();
    await this.rebuildAgent();
  }

  getStatus(): AgentStatus {
    return {
      connected: this.connected,
      registered: this.registered,
      agentName: this.config?.agentName ?? null,
      uptime: this.startTime > 0 ? Date.now() - this.startTime : 0,
      taskCount: this.taskCount,
      lastActivity: this.lastActivity,
      llmMode: this.config?.llmMode ?? 'cloud',
      localModelLoaded: false,
    };
  }

  getTasks(): TaskRecord[] {
    return this.tasks.slice(-50);
  }

  async updateConfig(partial: Partial<ExtensionConfig>): Promise<void> {
    this.config = await saveConfig(partial);
    await this.rebuildAgent();
  }

  async loginWithRobutler(): Promise<void> {
    if (!this.config) this.config = await loadConfig();
    const portalUrl = this.config?.portalUrl || 'https://robutler.ai';
    const state = crypto.randomUUID();
    const startUrl = `${portalUrl.replace(/\/+$/, '')}/api/browser-extension/auth/start?state=${encodeURIComponent(state)}`;

    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        chrome.tabs.onUpdated.removeListener(listener);
        reject(new Error('Login timed out. Try again from the extension popup.'));
      }, 5 * 60_000);

      const listener = (tabId: number, changeInfo: { url?: string }) => {
        if (!changeInfo.url) return;
        let url: URL;
        try {
          url = new URL(changeInfo.url);
        } catch {
          return;
        }
        const code = url.searchParams.get('browser_extension_code');
        const returnedState = url.searchParams.get('browser_extension_state');
        if (!code || returnedState !== state) return;

        clearTimeout(timeout);
        chrome.tabs.onUpdated.removeListener(listener);
        this.exchangeLoginCode(code)
          .then(() => {
            chrome.tabs.remove(tabId).catch(() => {});
            resolve();
          })
          .catch(reject);
      };

      chrome.tabs.onUpdated.addListener(listener);
      chrome.tabs.create({ url: startUrl }).catch((err) => {
        clearTimeout(timeout);
        chrome.tabs.onUpdated.removeListener(listener);
        reject(err);
      });
    });
  }

  async logout(): Promise<void> {
    this.disconnect();
    await this.updateConfig({
      sessionToken: null,
      extensionToken: null,
      agentToken: null,
      username: null,
      agentName: null,
      agentId: null,
    });
  }

  async connect(): Promise<void> {
    if (!this.config?.agentToken && !this.config?.sessionToken) {
      throw new Error('Not logged in. Use Login with Robutler first.');
    }
    if (this.connected) return;

    await this.register();
    await this.connectWebSocket();
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close(1000, 'User disconnected');
      this.ws = null;
    }
    this.connected = false;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.pendingConnect) {
      clearTimeout(this.pendingConnect.timeout);
      this.pendingConnect.reject(new Error('Disconnected before session was created.'));
      this.pendingConnect = null;
    }
  }

  private async register(): Promise<void> {
    if (!this.config?.sessionToken || !this.config.username) return;
    if (this.config.agentName && this.config.agentToken) return;

    const agentName = `${this.config.username}-browser`;
    const resp = await fetch(`${this.config.portalUrl}/api/agents`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.config.sessionToken}`,
      },
      body: JSON.stringify({
        name: 'browser',
        description: `Browser agent for ${this.config.username}`,
        instructions: this.browserAgentInstructions(),
        namespaced: true,
        acceptFrom: [`@${this.config.username}`],
        talkTo: ['nobody'],
        enabledTools: {},
        skills: {
          browserExtension: {
            requireApproval: this.config.requireApproval,
            maxToolCallsPerMinute: this.config.maxToolCallsPerMinute,
            blockedDomains: this.config.blockedDomains,
            allowJavascriptEvaluation: this.config.allowJavascriptEvaluation,
          },
        },
      }),
    });

    if (!resp.ok && resp.status !== 409) {
      throw new Error(`Failed to register: ${resp.status} ${resp.statusText}`);
    }

    if (resp.ok) {
      const data = await resp.json() as {
        agent?: { id?: string; username?: string };
        rawApiKey?: string;
      };
      await this.updateConfig({
        agentId: data.agent?.id ?? null,
        agentName: data.agent?.username ?? agentName,
        agentToken: data.rawApiKey ?? this.config.sessionToken,
      });
    } else {
      await this.updateConfig({ agentName });
    }
    this.registered = true;
  }

  private async exchangeLoginCode(code: string): Promise<void> {
    if (!this.config) this.config = await loadConfig();
    const portalUrl = this.config?.portalUrl || 'https://robutler.ai';
    const resp = await fetch(`${portalUrl.replace(/\/+$/, '')}/api/browser-extension/auth/exchange`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code }),
    });
    if (!resp.ok) {
      throw new Error(`Login exchange failed: ${resp.status} ${resp.statusText}`);
    }
    const data = await resp.json() as {
      user: { username: string };
      agent: { id: string; username: string };
      extensionToken: string;
      agentToken: string;
    };
    await this.updateConfig({
      username: data.user.username,
      agentId: data.agent.id,
      agentName: data.agent.username,
      extensionToken: data.extensionToken,
      agentToken: data.agentToken,
      sessionToken: null,
    });
  }

  private async syncPortalSettings(): Promise<boolean> {
    const token = this.config?.extensionToken ?? this.config?.sessionToken;
    const agentName = this.config?.agentName;
    if (!token || !agentName || !this.config?.portalUrl) return false;

    try {
      const resp = await fetch(`${this.config.portalUrl.replace(/\/+$/, '')}/api/agents/${encodeURIComponent(agentName)}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!resp.ok) return false;
      const data = await resp.json() as {
        config?: {
          skills?: Record<string, unknown>;
          acceptFrom?: unknown;
          talkTo?: unknown;
          enabledTools?: unknown;
        };
      };
      const browserConfig = data.config?.skills?.browserExtension as {
        requireApproval?: ExtensionConfig['requireApproval'];
        maxToolCallsPerMinute?: number;
        blockedDomains?: string[];
        allowJavascriptEvaluation?: boolean;
      } | undefined;
      if (!browserConfig) return false;
      const next = {
        requireApproval: browserConfig.requireApproval ?? this.config.requireApproval,
        maxToolCallsPerMinute: browserConfig.maxToolCallsPerMinute ?? this.config.maxToolCallsPerMinute,
        blockedDomains: Array.isArray(browserConfig.blockedDomains)
          ? browserConfig.blockedDomains
          : this.config.blockedDomains,
        allowJavascriptEvaluation: browserConfig.allowJavascriptEvaluation === true,
      };
      const changed = next.requireApproval !== this.config.requireApproval
        || next.maxToolCallsPerMinute !== this.config.maxToolCallsPerMinute
        || next.blockedDomains.join('\n') !== this.config.blockedDomains.join('\n')
        || next.allowJavascriptEvaluation !== this.config.allowJavascriptEvaluation;
      this.config = await saveConfig({
        requireApproval: next.requireApproval,
        maxToolCallsPerMinute: next.maxToolCallsPerMinute,
        blockedDomains: next.blockedDomains,
        allowJavascriptEvaluation: next.allowJavascriptEvaluation,
      });
      return changed;
    } catch (err) {
      console.warn('[robutler] Failed to sync browser extension settings:', (err as Error).message);
      return false;
    }
  }

  private connectWebSocket(): Promise<void> {
    const token = this.config?.agentToken ?? this.config?.sessionToken;
    if (!token || !this.config?.agentName) {
      return Promise.reject(new Error('Missing agent token or agent name.'));
    }

    const wsUrl = this.config.portalUrl.replace(/^http/, 'ws').replace(/\/+$/, '');
    this.ws = new WebSocket(
      `${wsUrl}/ws?token=${encodeURIComponent(token)}`,
    );

    const connectPromise = new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pendingConnect = null;
        this.ws?.close(4000, 'session.create timeout');
        reject(new Error('Timed out waiting for Robutler session.'));
      }, 15_000);
      this.pendingConnect = { resolve, reject, timeout };
    });

    this.ws.onopen = () => {
      console.debug('[robutler] websocket open, creating agent session', this.config?.agentName);
      this.ws!.send(
        JSON.stringify({
          type: 'session.create',
          agent: this.config!.agentName,
          token,
          uamp_version: '1.0',
          event_id: crypto.randomUUID(),
          session: {
            modalities: ['text'],
            instructions: this.browserAgentInstructions(),
          },
          client_capabilities: this.agent?.getCapabilities(),
        }),
      );
    };

    this.ws.onmessage = (event) => {
      try {
        this.handleUampMessage(JSON.parse(event.data as string));
      } catch {
        console.warn('[robutler] Failed to parse UAMP message:', event.data);
      }
    };

    this.ws.onclose = (event) => {
      this.connected = false;
      this.registered = false;
      if (this.pendingConnect) {
        clearTimeout(this.pendingConnect.timeout);
        this.pendingConnect.reject(new Error(`WebSocket closed before session was created (${event.code || 'unknown'}).`));
        this.pendingConnect = null;
      }
      if (event.code !== 1000) {
        this.scheduleReconnect();
      }
    };

    this.ws.onerror = () => {
      this.connected = false;
      if (this.pendingConnect) {
        clearTimeout(this.pendingConnect.timeout);
        this.pendingConnect.reject(new Error('WebSocket connection failed.'));
        this.pendingConnect = null;
      }
    };

    return connectPromise;
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connectWebSocket().catch((err) => {
        console.warn('[robutler] reconnect failed:', (err as Error).message);
      });
      this.reconnectDelay = Math.min(this.reconnectDelay * 2, 30_000);
    }, this.reconnectDelay);
  }

  private async handleUampMessage(msg: Record<string, unknown>): Promise<void> {
    const type = msg.type as string;

    if (type === 'session.created') {
      this.connected = true;
      this.registered = true;
      this.reconnectDelay = 1000;
      if (this.pendingConnect) {
        clearTimeout(this.pendingConnect.timeout);
        this.pendingConnect.resolve();
        this.pendingConnect = null;
      }
      return;
    }

    if (type === 'session.error') {
      const message = (msg.error as { message?: string } | undefined)?.message ?? 'Session creation failed.';
      if (this.pendingConnect) {
        clearTimeout(this.pendingConnect.timeout);
        this.pendingConnect.reject(new Error(message));
        this.pendingConnect = null;
      }
      this.connected = false;
      this.registered = false;
      return;
    }

    if (type === 'input.text' || type === 'conversation.item.create') {
      const content =
        typeof msg.text === 'string'
          ? msg.text
          : typeof msg.content === 'string'
            ? msg.content
            : '';
      if (!content) return;

      const taskId = (msg.event_id as string) ?? crypto.randomUUID();
      const source = (msg.source as string) ?? 'unknown';

      const task: TaskRecord = {
        id: taskId,
        source,
        instruction: content,
        status: 'running',
        startedAt: Date.now(),
        completedAt: null,
        result: null,
      };
      this.tasks.push(task);
      this.taskCount++;
      this.lastActivity = Date.now();

      try {
        const result = await this.runWebAgentsLoop(msg, content);
        task.status = 'completed';
        task.completedAt = Date.now();
        task.result = result;
      } catch (err) {
        task.status = 'failed';
        task.completedAt = Date.now();
        task.result = String(err);

        this.ws?.send(
          JSON.stringify({
            type: 'response.done',
            event_id: taskId,
            text: `Error: ${err}`,
            error: true,
          }),
        );
      }
    }
  }

  async sendChat(userMessage: string): Promise<string> {
    return this.runWebAgentsLoop(createInputTextEvent(userMessage) as unknown as Record<string, unknown>, userMessage);
  }

  private async runWebAgentsLoop(rawEvent: Record<string, unknown>, instruction: string): Promise<string> {
    const configChanged = await this.syncPortalSettings();
    if (configChanged) await this.rebuildAgent();
    if (!this.agent) await this.rebuildAgent();
    if (!this.agent) throw new Error('Agent runtime is not initialized');

    const session = createSessionCreateEvent({
      modalities: ['text'],
      instructions: this.browserAgentInstructions(),
    });
    const input = {
      ...createInputTextEvent(instruction),
      ...rawEvent,
      type: 'input.text',
      text: instruction,
    } as ClientEvent & { session_id?: string };
    const sid = typeof rawEvent.session_id === 'string' ? rawEvent.session_id : undefined;
    if (sid) {
      (session as unknown as { session_id?: string }).session_id = sid;
      input.session_id = sid;
    }
    const paymentToken = typeof rawEvent.payment_token === 'string' ? rawEvent.payment_token : undefined;
    if (paymentToken) {
      (this.agent as unknown as { context: { payment?: { valid: boolean; token: string } } }).context.payment = {
        valid: true,
        token: paymentToken,
      };
    }

    let finalText = '';
    for await (const event of this.agent.processUAMP([session, input])) {
      const withSession = sid ? { ...event, session_id: sid } : event;
      finalText += this.extractText(withSession);
      this.ws?.send(JSON.stringify(withSession));
    }
    return finalText || 'Task completed.';
  }

  private checkRateLimit(): boolean {
    if (!this.config) return false;
    const now = Date.now();
    if (now - this.toolCallWindowStart > 60_000) {
      this.toolCallCount = 0;
      this.toolCallWindowStart = now;
    }
    return this.toolCallCount < this.config.maxToolCallsPerMinute;
  }

  private requiresApproval(toolName: string): boolean {
    if (!this.config) return true;
    if (this.config.requireApproval === 'never') return false;
    if (this.config.requireApproval === 'always') return true;
    // 'sensitive' — only for dangerous operations
    return [
      'browser_evaluate',
      'browser_navigate',
      'browser_open_tab',
      'browser_close_tab',
      'browser_take_screenshot',
      'execute_script',
      'navigate',
      'screenshot',
    ].includes(toolName);
  }

  private async requestApproval(
    toolName: string,
    args: Record<string, unknown>,
  ): Promise<boolean> {
    return new Promise((resolve) => {
      chrome.notifications.create(
        {
          type: 'basic',
          iconUrl: chrome.runtime.getURL('icons/icon-128.png'),
          title: 'Robutler: Action Approval Required',
          message: `Allow ${toolName}(${JSON.stringify(args).slice(0, 100)})?`,
          buttons: [{ title: 'Allow' }, { title: 'Deny' }],
          requireInteraction: true,
        },
        (notificationId) => {
          const handler = (
            id: string,
            buttonIndex: number,
          ) => {
            if (id === notificationId) {
              chrome.notifications.onButtonClicked.removeListener(handler);
              resolve(buttonIndex === 0);
            }
          };
          chrome.notifications.onButtonClicked.addListener(handler);

          // Auto-deny after 30 seconds
          setTimeout(() => {
            chrome.notifications.onButtonClicked.removeListener(handler);
            chrome.notifications.clear(notificationId);
            resolve(false);
          }, 30_000);
        },
      );
    });
  }

  private async rebuildAgent(): Promise<void> {
    if (!this.config) return;
    const browserSkill = new BrowserControlSkill({
      adapter: new ChromeBrowserControlAdapter(),
      policy: {
        beforeTool: async (toolName, params) => {
          console.debug('[robutler] browser tool start', toolName, params);
          if (!this.checkRateLimit()) {
            throw new Error('Rate limit exceeded. Please wait before making more tool calls.');
          }
          if (toolName === 'browser_evaluate' && !this.config?.allowJavascriptEvaluation) {
            throw new Error('JavaScript evaluation is disabled in this browser agent settings.');
          }
          await this.assertDomainAllowed();
          if (this.requiresApproval(toolName)) {
            const approved = await this.requestApproval(toolName, params);
            if (!approved) throw new Error('User denied permission for this browser action.');
          }
        },
        afterTool: async () => {
          console.debug('[robutler] browser tool complete');
          this.toolCallCount++;
          this.lastActivity = Date.now();
        },
      },
    });

    this.agent = new BaseAgent({
      name: this.config.agentName ?? `${this.config.username ?? 'user'}-browser`,
      description: 'Robutler browser companion agent',
      instructions: this.browserAgentInstructions(),
      maxToolIterations: 20,
      skills: [
        new PortalChatCompletionsSkill({
          portalUrl: this.config.portalUrl,
          tokenProvider: () => this.config?.extensionToken ?? this.config?.sessionToken ?? null,
          modelProvider: () => this.config?.cloudModel || 'gpt-4o',
        }),
        browserSkill,
      ],
    });
    await this.agent.initialize();
  }

  private browserAgentInstructions(): string {
    return [
      'You are a Robutler browser companion agent running in the user\'s Chrome extension.',
      'Use browser_* tools to inspect and control the active browser tab.',
      'Prefer reading page state before taking actions.',
      'Ask for clarification when an action is destructive, ambiguous, or requires credentials.',
      'Summarize completed work concisely.',
    ].join(' ');
  }

  private async assertDomainAllowed(): Promise<void> {
    const blocked = this.config?.blockedDomains ?? [];
    if (blocked.length === 0) return;
    const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
    if (!tab?.url) return;
    const hostname = new URL(tab.url).hostname.toLowerCase();
    const matched = blocked.find((domain) => hostname === domain || hostname.endsWith(`.${domain}`));
    if (matched) throw new Error(`Browser actions are blocked on ${matched}.`);
  }

  private extractText(event: ServerEvent | Record<string, unknown>): string {
    if (event.type === 'response.delta') {
      const delta = (event as { delta?: { type?: string; text?: string } }).delta;
      return delta?.type === 'text' && delta.text ? delta.text : '';
    }
    if (event.type === 'response.done') {
      const output = (event as { response?: { output?: ContentItem[] } }).response?.output ?? [];
      return output
        .filter((item): item is { type: 'text'; text: string } => item.type === 'text')
        .map((item) => item.text)
        .join('');
    }
    return '';
  }
}
