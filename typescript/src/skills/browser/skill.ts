import { Skill } from '../../core/skill';
import { tool } from '../../core/decorators';
import type { Context, SkillConfig } from '../../core/types';

export interface BrowserToolResult {
  success: boolean;
  data?: unknown;
  error?: string;
}

export interface BrowserControlAdapter {
  listTabs(): Promise<BrowserToolResult>;
  openTab(url?: string, active?: boolean): Promise<BrowserToolResult>;
  closeTab(tabId: number): Promise<BrowserToolResult>;
  switchTab(tabId: number): Promise<BrowserToolResult>;
  readPage(tabId?: number): Promise<BrowserToolResult>;
  click(selector: string, tabId?: number): Promise<BrowserToolResult>;
  type(selector: string, text: string, tabId?: number): Promise<BrowserToolResult>;
  pressKey(key: string, tabId?: number): Promise<BrowserToolResult>;
  selectOption(selector: string, value: string, tabId?: number): Promise<BrowserToolResult>;
  screenshot(tabId?: number): Promise<BrowserToolResult>;
  navigate(url: string, tabId?: number): Promise<BrowserToolResult>;
  getPageInfo(tabId?: number): Promise<BrowserToolResult>;
  waitFor(selector?: string, timeoutMs?: number, tabId?: number): Promise<BrowserToolResult>;
  evaluate(code: string, tabId?: number): Promise<BrowserToolResult>;
}

export interface BrowserControlPolicy {
  beforeTool?: (toolName: string, params: Record<string, unknown>, context: Context) => Promise<void>;
  afterTool?: (
    toolName: string,
    params: Record<string, unknown>,
    result: BrowserToolResult,
    context: Context,
  ) => Promise<void>;
}

export interface BrowserControlSkillConfig extends SkillConfig {
  adapter: BrowserControlAdapter;
  policy?: BrowserControlPolicy;
}

export class BrowserControlSkill extends Skill {
  private readonly adapter: BrowserControlAdapter;
  private readonly policy?: BrowserControlPolicy;

  constructor(config: BrowserControlSkillConfig) {
    super({ ...config, name: config.name || 'browser-control' });
    this.adapter = config.adapter;
    this.policy = config.policy;
  }

  private async runTool(
    name: string,
    params: Record<string, unknown>,
    context: Context,
    fn: () => Promise<BrowserToolResult>,
  ): Promise<BrowserToolResult> {
    await this.policy?.beforeTool?.(name, params, context);
    const result = await fn();
    await this.policy?.afterTool?.(name, params, result, context);
    return result;
  }

  @tool({
    name: 'browser_tabs',
    description: 'List open browser tabs with IDs, URLs, titles, active state, and window IDs.',
    provides: 'browser.tabs',
    parameters: { type: 'object', properties: {}, required: [] },
  })
  async browserTabs(params: Record<string, unknown>, context: Context): Promise<BrowserToolResult> {
    return this.runTool('browser_tabs', params, context, () => this.adapter.listTabs());
  }

  @tool({
    name: 'browser_open_tab',
    description: 'Open a new browser tab, optionally at the given URL, and return its tab ID.',
    provides: 'browser.tabs',
    parameters: {
      type: 'object',
      properties: {
        url: { type: 'string', description: 'URL to open. Domains like google.com are accepted and normalized to https://google.com.' },
        active: { type: 'boolean', description: 'Whether to focus the new tab. Defaults to true.' },
      },
      required: [],
    },
  })
  async browserOpenTab(params: Record<string, unknown>, context: Context): Promise<BrowserToolResult> {
    return this.runTool('browser_open_tab', params, context, () =>
      this.adapter.openTab(typeof params.url === 'string' ? params.url : undefined, params.active !== false),
    );
  }

  @tool({
    name: 'browser_close_tab',
    description: 'Close a browser tab by ID.',
    provides: 'browser.tabs',
    parameters: {
      type: 'object',
      properties: {
        tab_id: { type: 'number', description: 'Browser tab ID to close.' },
      },
      required: ['tab_id'],
    },
  })
  async browserCloseTab(params: Record<string, unknown>, context: Context): Promise<BrowserToolResult> {
    return this.runTool('browser_close_tab', params, context, () =>
      this.adapter.closeTab(Number(params.tab_id)),
    );
  }

  @tool({
    name: 'browser_switch_tab',
    description: 'Focus an existing browser tab by ID.',
    provides: 'browser.tabs',
    parameters: {
      type: 'object',
      properties: {
        tab_id: { type: 'number', description: 'Browser tab ID to focus.' },
      },
      required: ['tab_id'],
    },
  })
  async browserSwitchTab(params: Record<string, unknown>, context: Context): Promise<BrowserToolResult> {
    return this.runTool('browser_switch_tab', params, context, () =>
      this.adapter.switchTab(Number(params.tab_id)),
    );
  }

  @tool({
    name: 'browser_read_page',
    description: 'Read the visible page text, HTML, URL, title, and metadata for a browser tab.',
    provides: 'browser.read',
    parameters: {
      type: 'object',
      properties: {
        tab_id: { type: 'number', description: 'Browser tab ID. Omit to use the active tab.' },
      },
      required: [],
    },
  })
  async browserReadPage(params: Record<string, unknown>, context: Context): Promise<BrowserToolResult> {
    return this.runTool('browser_read_page', params, context, () =>
      this.adapter.readPage(params.tab_id as number | undefined),
    );
  }

  @tool({
    name: 'browser_snapshot',
    description: 'Get a lightweight accessibility-oriented snapshot of the current tab content.',
    provides: 'browser.read',
    parameters: {
      type: 'object',
      properties: {
        tab_id: { type: 'number', description: 'Browser tab ID. Omit to use the active tab.' },
      },
      required: [],
    },
  })
  async browserSnapshot(params: Record<string, unknown>, context: Context): Promise<BrowserToolResult> {
    return this.runTool('browser_snapshot', params, context, () =>
      this.adapter.readPage(params.tab_id as number | undefined),
    );
  }

  @tool({
    name: 'browser_click',
    description: 'Click an element on a page using a CSS selector.',
    provides: 'browser.action',
    parameters: {
      type: 'object',
      properties: {
        selector: { type: 'string', description: 'CSS selector for the target element.' },
        tab_id: { type: 'number', description: 'Browser tab ID. Omit to use the active tab.' },
      },
      required: ['selector'],
    },
  })
  async browserClick(params: Record<string, unknown>, context: Context): Promise<BrowserToolResult> {
    return this.runTool('browser_click', params, context, () =>
      this.adapter.click(String(params.selector ?? ''), params.tab_id as number | undefined),
    );
  }

  @tool({
    name: 'browser_type',
    description: 'Type text into an input or textarea selected by CSS selector.',
    provides: 'browser.action',
    parameters: {
      type: 'object',
      properties: {
        selector: { type: 'string', description: 'CSS selector for the target input.' },
        text: { type: 'string', description: 'Text to enter.' },
        tab_id: { type: 'number', description: 'Browser tab ID. Omit to use the active tab.' },
      },
      required: ['selector', 'text'],
    },
  })
  async browserType(params: Record<string, unknown>, context: Context): Promise<BrowserToolResult> {
    return this.runTool('browser_type', params, context, () =>
      this.adapter.type(
        String(params.selector ?? ''),
        String(params.text ?? params.value ?? ''),
        params.tab_id as number | undefined,
      ),
    );
  }

  @tool({
    name: 'browser_press_key',
    description: 'Press a keyboard key in the current page context, such as Enter, Tab, Escape, ArrowDown.',
    provides: 'browser.action',
    parameters: {
      type: 'object',
      properties: {
        key: { type: 'string', description: 'Keyboard key to press.' },
        tab_id: { type: 'number', description: 'Browser tab ID. Omit to use the active tab.' },
      },
      required: ['key'],
    },
  })
  async browserPressKey(params: Record<string, unknown>, context: Context): Promise<BrowserToolResult> {
    return this.runTool('browser_press_key', params, context, () =>
      this.adapter.pressKey(String(params.key ?? ''), params.tab_id as number | undefined),
    );
  }

  @tool({
    name: 'browser_select_option',
    description: 'Select an option in a select element using a CSS selector and option value.',
    provides: 'browser.action',
    parameters: {
      type: 'object',
      properties: {
        selector: { type: 'string', description: 'CSS selector for the select element.' },
        value: { type: 'string', description: 'Option value to select.' },
        tab_id: { type: 'number', description: 'Browser tab ID. Omit to use the active tab.' },
      },
      required: ['selector', 'value'],
    },
  })
  async browserSelectOption(params: Record<string, unknown>, context: Context): Promise<BrowserToolResult> {
    return this.runTool('browser_select_option', params, context, () =>
      this.adapter.selectOption(
        String(params.selector ?? ''),
        String(params.value ?? ''),
        params.tab_id as number | undefined,
      ),
    );
  }

  @tool({
    name: 'browser_take_screenshot',
    description: 'Capture a screenshot of the visible area of a browser tab.',
    provides: 'browser.screenshot',
    parameters: {
      type: 'object',
      properties: {
        tab_id: { type: 'number', description: 'Browser tab ID. Omit to use the active tab.' },
      },
      required: [],
    },
  })
  async browserTakeScreenshot(params: Record<string, unknown>, context: Context): Promise<BrowserToolResult> {
    return this.runTool('browser_take_screenshot', params, context, () =>
      this.adapter.screenshot(params.tab_id as number | undefined),
    );
  }

  @tool({
    name: 'browser_navigate',
    description: 'Navigate a browser tab to a URL.',
    provides: 'browser.navigate',
    parameters: {
      type: 'object',
      properties: {
        url: { type: 'string', description: 'Absolute URL to navigate to.' },
        tab_id: { type: 'number', description: 'Browser tab ID. Omit to use the active tab.' },
      },
      required: ['url'],
    },
  })
  async browserNavigate(params: Record<string, unknown>, context: Context): Promise<BrowserToolResult> {
    return this.runTool('browser_navigate', params, context, () =>
      this.adapter.navigate(String(params.url ?? ''), params.tab_id as number | undefined),
    );
  }

  @tool({
    name: 'browser_wait_for',
    description: 'Wait for a selector to appear, or wait briefly for page state to settle.',
    provides: 'browser.wait',
    parameters: {
      type: 'object',
      properties: {
        selector: { type: 'string', description: 'Optional CSS selector to wait for.' },
        timeout_ms: { type: 'number', description: 'Maximum wait time in milliseconds.' },
        tab_id: { type: 'number', description: 'Browser tab ID. Omit to use the active tab.' },
      },
      required: [],
    },
  })
  async browserWaitFor(params: Record<string, unknown>, context: Context): Promise<BrowserToolResult> {
    return this.runTool('browser_wait_for', params, context, () =>
      this.adapter.waitFor(
        typeof params.selector === 'string' ? params.selector : undefined,
        params.timeout_ms as number | undefined,
        params.tab_id as number | undefined,
      ),
    );
  }

  @tool({
    name: 'browser_evaluate',
    description: 'Evaluate JavaScript in the page context and return the result. Use sparingly.',
    provides: 'browser.evaluate',
    parameters: {
      type: 'object',
      properties: {
        code: { type: 'string', description: 'JavaScript expression or function body to evaluate.' },
        tab_id: { type: 'number', description: 'Browser tab ID. Omit to use the active tab.' },
      },
      required: ['code'],
    },
  })
  async browserEvaluate(params: Record<string, unknown>, context: Context): Promise<BrowserToolResult> {
    return this.runTool('browser_evaluate', params, context, () =>
      this.adapter.evaluate(String(params.code ?? ''), params.tab_id as number | undefined),
    );
  }
}
