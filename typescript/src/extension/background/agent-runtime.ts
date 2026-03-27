import type { ExtensionConfig, AgentStatus, TaskRecord } from '../shared/types';
import { loadConfig, saveConfig } from '../shared/storage';
import {
  BROWSER_TOOL_DEFINITIONS,
  executeBrowserTool,
  type BrowserToolName,
} from './browser-tools';

interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | null;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}

interface ToolCall {
  id: string;
  type: 'function';
  function: { name: string; arguments: string };
}

interface CompletionResponse {
  choices: Array<{
    message: ChatMessage;
    finish_reason: string;
  }>;
}

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

  async initialize(): Promise<void> {
    this.config = await loadConfig();
    this.startTime = Date.now();
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
  }

  async connect(): Promise<void> {
    if (!this.config?.sessionToken) {
      throw new Error('Not logged in. Set session token first.');
    }
    if (this.connected) return;

    await this.register();
    this.connectWebSocket();
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
  }

  private async register(): Promise<void> {
    if (!this.config?.sessionToken || !this.config.username) return;

    const agentName = `${this.config.username}-browser`;
    const resp = await fetch(`${this.config.portalUrl}/api/auth/agent/create`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.config.sessionToken}`,
      },
      body: JSON.stringify({
        username: agentName,
        description: `Browser agent for ${this.config.username}`,
        capabilities: ['browser-automation', 'dom-access', 'screenshots', 'navigation'],
        runtimeEngine: 'typescript',
        isDaemon: true,
      }),
    });

    if (resp.ok || resp.status === 409) {
      this.registered = true;
      await this.updateConfig({ agentName });
    } else {
      throw new Error(`Failed to register: ${resp.status} ${resp.statusText}`);
    }
  }

  private connectWebSocket(): void {
    if (!this.config?.sessionToken || !this.config.agentName) return;

    const wsUrl = this.config.portalUrl.replace(/^http/, 'ws');
    this.ws = new WebSocket(
      `${wsUrl}/ws?agent=${encodeURIComponent(this.config.agentName)}&token=${encodeURIComponent(this.config.sessionToken)}`,
    );

    this.ws.onopen = () => {
      this.connected = true;
      this.reconnectDelay = 1000;

      this.ws!.send(
        JSON.stringify({
          type: 'session.created',
          agent: this.config!.agentName,
          capabilities: BROWSER_TOOL_DEFINITIONS.map((t) => t.function.name),
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
      if (event.code !== 1000) {
        this.scheduleReconnect();
      }
    };

    this.ws.onerror = () => {
      this.connected = false;
    };
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connectWebSocket();
      this.reconnectDelay = Math.min(this.reconnectDelay * 2, 30_000);
    }, this.reconnectDelay);
  }

  private async handleUampMessage(msg: Record<string, unknown>): Promise<void> {
    const type = msg.type as string;

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
        const result = await this.runAgenticLoop(content);
        task.status = 'completed';
        task.completedAt = Date.now();
        task.result = result;

        this.ws?.send(
          JSON.stringify({
            type: 'response.done',
            event_id: taskId,
            text: result,
          }),
        );
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
    return this.runAgenticLoop(userMessage);
  }

  private async runAgenticLoop(instruction: string): Promise<string> {
    const messages: ChatMessage[] = [
      {
        role: 'system',
        content: [
          'You are a browser automation agent running locally in the user\'s browser extension.',
          'You have access to browser tools to interact with web pages.',
          'Execute the user\'s request by using the available tools.',
          'Be concise in your responses. When you complete a task, summarize what you did.',
        ].join(' '),
      },
      { role: 'user', content: instruction },
    ];

    const maxIterations = 15;

    for (let i = 0; i < maxIterations; i++) {
      const response = await this.callLLM(messages);
      const choice = response.choices[0];
      if (!choice) return 'No response from LLM';

      messages.push(choice.message);

      if (choice.finish_reason === 'stop' || !choice.message.tool_calls?.length) {
        return choice.message.content ?? 'Task completed.';
      }

      for (const toolCall of choice.message.tool_calls) {
        if (!this.checkRateLimit()) {
          const errorResult: ChatMessage = {
            role: 'tool',
            content: 'Rate limit exceeded. Please wait before making more tool calls.',
            tool_call_id: toolCall.id,
          };
          messages.push(errorResult);
          continue;
        }

        let args: Record<string, unknown>;
        try {
          args = JSON.parse(toolCall.function.arguments);
        } catch {
          messages.push({
            role: 'tool',
            content: JSON.stringify({ success: false, error: 'Invalid JSON in tool arguments' }),
            tool_call_id: toolCall.id,
          });
          continue;
        }
        const toolName = toolCall.function.name as BrowserToolName;

        const needsApproval = this.requiresApproval(toolName);
        if (needsApproval) {
          const approved = await this.requestApproval(toolName, args);
          if (!approved) {
            messages.push({
              role: 'tool',
              content: 'User denied permission for this action.',
              tool_call_id: toolCall.id,
            });
            continue;
          }
        }

        let result;
        try {
          result = await executeBrowserTool(toolName, args);
        } catch (err) {
          result = { success: false, error: String(err) };
        }
        this.toolCallCount++;
        this.lastActivity = Date.now();

        messages.push({
          role: 'tool',
          content: JSON.stringify(result),
          tool_call_id: toolCall.id,
        });
      }
    }

    return 'Reached maximum tool iterations. Partial results may be available.';
  }

  private async callLLM(messages: ChatMessage[]): Promise<CompletionResponse> {
    if (!this.config) throw new Error('Not initialized');

    const model = this.config.llmMode === 'local' ? this.config.localModel : this.config.cloudModel;
    const endpoint =
      this.config.llmMode === 'local'
        ? null
        : `${this.config.portalUrl}/api/llm/chat/completions`;

    if (!endpoint) {
      throw new Error('Local LLM not yet implemented in extension. Use cloud or hybrid mode.');
    }

    const resp = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.config.sessionToken}`,
      },
      body: JSON.stringify({
        model,
        messages,
        tools: BROWSER_TOOL_DEFINITIONS,
        tool_choice: 'auto',
      }),
    });

    if (!resp.ok) {
      throw new Error(`LLM call failed: ${resp.status} ${resp.statusText}`);
    }

    return resp.json() as Promise<CompletionResponse>;
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
    return toolName === 'execute_script' || toolName === 'navigate';
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
}
