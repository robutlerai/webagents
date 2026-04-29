import type { ContentScriptMessage } from '../shared/types';

chrome.runtime.onMessage.addListener(
  (
    message: ContentScriptMessage,
    _sender: chrome.runtime.MessageSender,
    sendResponse: (response: ContentScriptMessage) => void,
  ) => {
    handleMessage(message)
      .then(sendResponse)
      .catch((err) =>
        sendResponse({ type: 'EXECUTE_SCRIPT_RESULT', result: null, error: String(err) }),
      );
    return true; // async response
  },
);

async function handleMessage(msg: ContentScriptMessage): Promise<ContentScriptMessage> {
  switch (msg.type) {
    case 'READ_PAGE':
      return readPage();
    case 'CLICK':
      return clickElement(msg.selector);
    case 'FILL':
      return fillElement(msg.selector, msg.value);
    case 'PRESS_KEY':
      return pressKey(msg.key);
    case 'SELECT_OPTION':
      return selectOption(msg.selector, msg.value);
    case 'GET_PAGE_INFO':
      return getPageInfo();
    case 'WAIT_FOR':
      return waitFor(msg.selector, msg.timeoutMs);
    case 'EXECUTE_SCRIPT':
      return executeScript(msg.code);
    default:
      return { type: 'EXECUTE_SCRIPT_RESULT', result: null, error: `Unknown message type` };
  }
}

function isEditableElement(el: Element | null): el is HTMLInputElement | HTMLTextAreaElement {
  return !!el && (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement);
}

function readPage(): ContentScriptMessage {
  const text = document.body?.innerText ?? '';
  const html = document.documentElement?.outerHTML ?? '';
  const truncatedHtml = html.length > 100_000 ? html.slice(0, 100_000) + '...[truncated]' : html;
  return { type: 'READ_PAGE_RESULT', text, html: truncatedHtml };
}

function clickElement(selector: string): ContentScriptMessage {
  try {
    const el = document.querySelector(selector);
    if (!el) return { type: 'CLICK_RESULT', success: false, error: `Element not found: ${selector}` };
    (el as HTMLElement).click();
    return { type: 'CLICK_RESULT', success: true };
  } catch (err) {
    return { type: 'CLICK_RESULT', success: false, error: String(err) };
  }
}

function fillElement(selector: string, value: string): ContentScriptMessage {
  try {
    const el = document.querySelector(selector);
    if (!isEditableElement(el)) {
      return { type: 'FILL_RESULT', success: false, error: `Editable element not found: ${selector}` };
    }

    el.focus();
    el.value = value;

    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    return { type: 'FILL_RESULT', success: true };
  } catch (err) {
    return { type: 'FILL_RESULT', success: false, error: String(err) };
  }
}

function pressKey(key: string): ContentScriptMessage {
  try {
    const target = (document.activeElement || document.body) as HTMLElement;
    const eventInit: KeyboardEventInit = { key, bubbles: true, cancelable: true };
    target.dispatchEvent(new KeyboardEvent('keydown', eventInit));
    target.dispatchEvent(new KeyboardEvent('keyup', eventInit));
    return { type: 'PRESS_KEY_RESULT', success: true };
  } catch (err) {
    return { type: 'PRESS_KEY_RESULT', success: false, error: String(err) };
  }
}

function selectOption(selector: string, value: string): ContentScriptMessage {
  try {
    const el = document.querySelector(selector);
    if (!(el instanceof HTMLSelectElement)) {
      return { type: 'SELECT_OPTION_RESULT', success: false, error: `Select element not found: ${selector}` };
    }
    el.value = value;
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    return { type: 'SELECT_OPTION_RESULT', success: true };
  } catch (err) {
    return { type: 'SELECT_OPTION_RESULT', success: false, error: String(err) };
  }
}

function getPageInfo(): ContentScriptMessage {
  const meta: Record<string, string> = {};
  document.querySelectorAll('meta').forEach((m) => {
    const name = m.getAttribute('name') || m.getAttribute('property') || '';
    const content = m.getAttribute('content') || '';
    if (name && content) meta[name] = content;
  });
  return {
    type: 'PAGE_INFO_RESULT',
    url: window.location.href,
    title: document.title,
    meta,
  };
}

async function waitFor(selector?: string, timeoutMs = 5000): Promise<ContentScriptMessage> {
  const deadline = Date.now() + Math.max(250, Math.min(timeoutMs, 30_000));
  while (Date.now() < deadline) {
    if (!selector || document.querySelector(selector)) {
      return { type: 'WAIT_FOR_RESULT', success: true };
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  return { type: 'WAIT_FOR_RESULT', success: false, error: selector ? `Timed out waiting for ${selector}` : 'Timed out' };
}

function executeScript(code: string): ContentScriptMessage {
  try {
    const result = new Function(code)();
    return { type: 'EXECUTE_SCRIPT_RESULT', result: String(result) };
  } catch (err) {
    return { type: 'EXECUTE_SCRIPT_RESULT', result: null, error: String(err) };
  }
}
