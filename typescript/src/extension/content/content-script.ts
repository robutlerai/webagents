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
    case 'GET_PAGE_INFO':
      return getPageInfo();
    case 'EXECUTE_SCRIPT':
      return executeScript(msg.code);
    default:
      return { type: 'EXECUTE_SCRIPT_RESULT', result: null, error: `Unknown message type` };
  }
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
    const el = document.querySelector(selector) as HTMLInputElement | HTMLTextAreaElement | null;
    if (!el) return { type: 'FILL_RESULT', success: false, error: `Element not found: ${selector}` };

    el.focus();
    el.value = value;

    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    return { type: 'FILL_RESULT', success: true };
  } catch (err) {
    return { type: 'FILL_RESULT', success: false, error: String(err) };
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

function executeScript(code: string): ContentScriptMessage {
  try {
    const result = new Function(code)();
    return { type: 'EXECUTE_SCRIPT_RESULT', result: String(result) };
  } catch (err) {
    return { type: 'EXECUTE_SCRIPT_RESULT', result: null, error: String(err) };
  }
}
