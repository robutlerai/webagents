import type {
  BrowserControlAdapter,
  BrowserToolResult,
} from '../../skills/browser';
import type { ContentScriptMessage } from '../shared/types';

async function sendToTab(
  tabId: number,
  message: ContentScriptMessage,
): Promise<ContentScriptMessage> {
  return new Promise((resolve, reject) => {
    chrome.tabs.sendMessage(tabId, message, (response) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
      } else if (!response) {
        reject(new Error('No response from content script. Reload the tab and try again.'));
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

function coerceResult(
  success: boolean,
  data?: unknown,
  error?: string,
): BrowserToolResult {
  return error ? { success: false, error } : { success, data };
}

function normalizeUrl(raw: string): string {
  const trimmed = raw.trim();
  if (!trimmed) throw new Error('URL is required');
  if (/^https?:\/\//i.test(trimmed)) return trimmed;
  return `https://${trimmed}`;
}

export class ChromeBrowserControlAdapter implements BrowserControlAdapter {
  async listTabs(): Promise<BrowserToolResult> {
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

  async openTab(url?: string, active = true): Promise<BrowserToolResult> {
    const normalized = url?.trim() ? normalizeUrl(url) : undefined;
    const tab = await chrome.tabs.create({ ...(normalized ? { url: normalized } : {}), active });
    return {
      success: true,
      data: {
        id: tab.id,
        url: tab.url ?? normalized ?? null,
        title: tab.title,
        active: tab.active,
        windowId: tab.windowId,
      },
    };
  }

  async closeTab(tabId: number): Promise<BrowserToolResult> {
    if (!Number.isFinite(tabId)) return { success: false, error: 'tab_id is required' };
    await chrome.tabs.remove(tabId);
    return { success: true, data: { tabId } };
  }

  async switchTab(tabId: number): Promise<BrowserToolResult> {
    if (!Number.isFinite(tabId)) return { success: false, error: 'tab_id is required' };
    const tab = await chrome.tabs.update(tabId, { active: true });
    if (tab.windowId) await chrome.windows.update(tab.windowId, { focused: true });
    return {
      success: true,
      data: {
        id: tab.id,
        url: tab.url,
        title: tab.title,
        active: tab.active,
        windowId: tab.windowId,
      },
    };
  }

  async readPage(tabId?: number): Promise<BrowserToolResult> {
    const id = tabId ?? (await getActiveTabId());
    const [page, info] = await Promise.all([
      sendToTab(id, { type: 'READ_PAGE' }),
      sendToTab(id, { type: 'GET_PAGE_INFO' }),
    ]);
    if (page.type === 'READ_PAGE_RESULT' && info.type === 'PAGE_INFO_RESULT') {
      return {
        success: true,
        data: {
          text: page.text,
          html: page.html,
          url: info.url,
          title: info.title,
          meta: info.meta,
        },
      };
    }
    return { success: false, error: 'Unexpected page read response' };
  }

  async click(selector: string, tabId?: number): Promise<BrowserToolResult> {
    const id = tabId ?? (await getActiveTabId());
    const result = await sendToTab(id, { type: 'CLICK', selector });
    if (result.type === 'CLICK_RESULT') return coerceResult(result.success, undefined, result.error);
    return { success: false, error: 'Unexpected click response' };
  }

  async type(selector: string, text: string, tabId?: number): Promise<BrowserToolResult> {
    const id = tabId ?? (await getActiveTabId());
    const result = await sendToTab(id, { type: 'FILL', selector, value: text });
    if (result.type === 'FILL_RESULT') return coerceResult(result.success, undefined, result.error);
    return { success: false, error: 'Unexpected type response' };
  }

  async pressKey(key: string, tabId?: number): Promise<BrowserToolResult> {
    const id = tabId ?? (await getActiveTabId());
    const result = await sendToTab(id, { type: 'PRESS_KEY', key });
    if (result.type === 'PRESS_KEY_RESULT') return coerceResult(result.success, undefined, result.error);
    return { success: false, error: 'Unexpected key response' };
  }

  async selectOption(selector: string, value: string, tabId?: number): Promise<BrowserToolResult> {
    const id = tabId ?? (await getActiveTabId());
    const result = await sendToTab(id, { type: 'SELECT_OPTION', selector, value });
    if (result.type === 'SELECT_OPTION_RESULT') return coerceResult(result.success, undefined, result.error);
    return { success: false, error: 'Unexpected select response' };
  }

  async screenshot(tabId?: number): Promise<BrowserToolResult> {
    const id = tabId ?? (await getActiveTabId());
    const tab = await chrome.tabs.get(id);
    if (!tab.windowId) return { success: false, error: 'No window for tab' };
    const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, { format: 'png' });
    return { success: true, data: { dataUrl } };
  }

  async navigate(url: string, tabId?: number): Promise<BrowserToolResult> {
    const id = tabId ?? (await getActiveTabId());
    await chrome.tabs.update(id, { url });
    return { success: true, data: { url } };
  }

  async getPageInfo(tabId?: number): Promise<BrowserToolResult> {
    const id = tabId ?? (await getActiveTabId());
    const result = await sendToTab(id, { type: 'GET_PAGE_INFO' });
    if (result.type === 'PAGE_INFO_RESULT') {
      return { success: true, data: { url: result.url, title: result.title, meta: result.meta } };
    }
    return { success: false, error: 'Unexpected page info response' };
  }

  async waitFor(selector?: string, timeoutMs?: number, tabId?: number): Promise<BrowserToolResult> {
    const id = tabId ?? (await getActiveTabId());
    const result = await sendToTab(id, { type: 'WAIT_FOR', selector, timeoutMs });
    if (result.type === 'WAIT_FOR_RESULT') return coerceResult(result.success, undefined, result.error);
    return { success: false, error: 'Unexpected wait response' };
  }

  async evaluate(code: string, tabId?: number): Promise<BrowserToolResult> {
    const id = tabId ?? (await getActiveTabId());
    const result = await sendToTab(id, { type: 'EXECUTE_SCRIPT', code });
    if (result.type === 'EXECUTE_SCRIPT_RESULT') {
      if (result.error) return { success: false, error: result.error };
      return { success: true, data: result.result };
    }
    return { success: false, error: 'Unexpected evaluate response' };
  }
}
