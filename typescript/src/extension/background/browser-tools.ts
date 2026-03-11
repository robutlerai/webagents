import type { ContentScriptMessage } from '../shared/types.js';

/**
 * Browser tools for the extension agent runtime.
 * Each tool communicates with content scripts via chrome.tabs.sendMessage.
 */

export interface BrowserToolResult {
  success: boolean;
  data?: unknown;
  error?: string;
}

async function sendToTab(
  tabId: number,
  message: ContentScriptMessage,
): Promise<ContentScriptMessage> {
  return new Promise((resolve, reject) => {
    chrome.tabs.sendMessage(tabId, message, (response) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
      } else if (!response) {
        reject(new Error('No response from content script (tab may not have the content script loaded)'));
      } else {
        resolve(response);
      }
    });
  });
}

async function getActiveTabId(): Promise<number> {
  const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  if (!tab?.id) throw new Error('No active tab found');
  return tab.id;
}

export async function listTabs(): Promise<BrowserToolResult> {
  const tabs = await chrome.tabs.query({});
  return {
    success: true,
    data: tabs.map((t) => ({
      id: t.id,
      url: t.url,
      title: t.title,
      active: t.active,
      windowId: t.windowId,
    })),
  };
}

export async function readPage(tabId?: number): Promise<BrowserToolResult> {
  const id = tabId ?? (await getActiveTabId());
  const result = await sendToTab(id, { type: 'READ_PAGE' });
  if (result.type === 'READ_PAGE_RESULT') {
    return { success: true, data: { text: result.text, html: result.html } };
  }
  return { success: false, error: 'Unexpected response' };
}

export async function clickElement(
  selector: string,
  tabId?: number,
): Promise<BrowserToolResult> {
  const id = tabId ?? (await getActiveTabId());
  const result = await sendToTab(id, { type: 'CLICK', selector });
  if (result.type === 'CLICK_RESULT') {
    return { success: result.success, error: result.error };
  }
  return { success: false, error: 'Unexpected response' };
}

export async function fillInput(
  selector: string,
  value: string,
  tabId?: number,
): Promise<BrowserToolResult> {
  const id = tabId ?? (await getActiveTabId());
  const result = await sendToTab(id, { type: 'FILL', selector, value });
  if (result.type === 'FILL_RESULT') {
    return { success: result.success, error: result.error };
  }
  return { success: false, error: 'Unexpected response' };
}

export async function screenshot(tabId?: number): Promise<BrowserToolResult> {
  const id = tabId ?? (await getActiveTabId());
  const tab = await chrome.tabs.get(id);
  if (!tab.windowId) return { success: false, error: 'No window for tab' };
  const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, { format: 'png' });
  return { success: true, data: { dataUrl } };
}

export async function navigate(url: string, tabId?: number): Promise<BrowserToolResult> {
  const id = tabId ?? (await getActiveTabId());
  await chrome.tabs.update(id, { url });
  return { success: true, data: { url } };
}

export async function getPageInfo(tabId?: number): Promise<BrowserToolResult> {
  const id = tabId ?? (await getActiveTabId());
  const result = await sendToTab(id, { type: 'GET_PAGE_INFO' });
  if (result.type === 'PAGE_INFO_RESULT') {
    return { success: true, data: { url: result.url, title: result.title, meta: result.meta } };
  }
  return { success: false, error: 'Unexpected response' };
}

export async function executeScript(
  code: string,
  tabId?: number,
): Promise<BrowserToolResult> {
  const id = tabId ?? (await getActiveTabId());
  const result = await sendToTab(id, { type: 'EXECUTE_SCRIPT', code });
  if (result.type === 'EXECUTE_SCRIPT_RESULT') {
    if (result.error) return { success: false, error: result.error };
    return { success: true, data: result.result };
  }
  return { success: false, error: 'Unexpected response' };
}

/**
 * Tool definitions for the agent runtime.
 * Each entry maps to a function the LLM can call.
 */
export const BROWSER_TOOL_DEFINITIONS = [
  {
    type: 'function' as const,
    function: {
      name: 'list_tabs',
      description: 'List all open browser tabs with their IDs, URLs, and titles',
      parameters: { type: 'object', properties: {}, required: [] },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'read_page',
      description: 'Read the text content and HTML of a browser tab',
      parameters: {
        type: 'object',
        properties: {
          tab_id: { type: 'number', description: 'Tab ID. Omit for active tab.' },
        },
        required: [],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'click',
      description: 'Click an element on the page using a CSS selector',
      parameters: {
        type: 'object',
        properties: {
          selector: { type: 'string', description: 'CSS selector of the element to click' },
          tab_id: { type: 'number', description: 'Tab ID. Omit for active tab.' },
        },
        required: ['selector'],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'fill',
      description: 'Fill an input field with a value using a CSS selector',
      parameters: {
        type: 'object',
        properties: {
          selector: { type: 'string', description: 'CSS selector of the input element' },
          value: { type: 'string', description: 'Value to fill' },
          tab_id: { type: 'number', description: 'Tab ID. Omit for active tab.' },
        },
        required: ['selector', 'value'],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'screenshot',
      description: 'Capture a screenshot of the visible area of a tab',
      parameters: {
        type: 'object',
        properties: {
          tab_id: { type: 'number', description: 'Tab ID. Omit for active tab.' },
        },
        required: [],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'navigate',
      description: 'Navigate a tab to a URL',
      parameters: {
        type: 'object',
        properties: {
          url: { type: 'string', description: 'URL to navigate to' },
          tab_id: { type: 'number', description: 'Tab ID. Omit for active tab.' },
        },
        required: ['url'],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'get_page_info',
      description: 'Get the URL, title, and meta tags of a tab',
      parameters: {
        type: 'object',
        properties: {
          tab_id: { type: 'number', description: 'Tab ID. Omit for active tab.' },
        },
        required: [],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'execute_script',
      description: 'Execute JavaScript code in the page context. Returns the result.',
      parameters: {
        type: 'object',
        properties: {
          code: { type: 'string', description: 'JavaScript code to execute' },
          tab_id: { type: 'number', description: 'Tab ID. Omit for active tab.' },
        },
        required: ['code'],
      },
    },
  },
];

export type BrowserToolName =
  | 'list_tabs'
  | 'read_page'
  | 'click'
  | 'fill'
  | 'screenshot'
  | 'navigate'
  | 'get_page_info'
  | 'execute_script';

export async function executeBrowserTool(
  name: BrowserToolName,
  args: Record<string, unknown>,
): Promise<BrowserToolResult> {
  switch (name) {
    case 'list_tabs':
      return listTabs();
    case 'read_page':
      return readPage(args.tab_id as number | undefined);
    case 'click':
      return clickElement(args.selector as string, args.tab_id as number | undefined);
    case 'fill':
      return fillInput(
        args.selector as string,
        args.value as string,
        args.tab_id as number | undefined,
      );
    case 'screenshot':
      return screenshot(args.tab_id as number | undefined);
    case 'navigate':
      return navigate(args.url as string, args.tab_id as number | undefined);
    case 'get_page_info':
      return getPageInfo(args.tab_id as number | undefined);
    case 'execute_script':
      return executeScript(args.code as string, args.tab_id as number | undefined);
    default:
      return { success: false, error: `Unknown tool: ${name}` };
  }
}
