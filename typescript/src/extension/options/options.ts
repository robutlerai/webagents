import type { BackgroundMessage, ExtensionConfig } from '../shared/types.js';

function $(id: string): HTMLInputElement | HTMLSelectElement {
  return document.getElementById(id) as HTMLInputElement | HTMLSelectElement;
}

function sendMessage(msg: BackgroundMessage): Promise<BackgroundMessage> {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(msg, resolve);
  });
}

function showToast(text: string): void {
  const el = document.getElementById('toast')!;
  el.textContent = text;
  el.classList.add('visible');
  setTimeout(() => el.classList.remove('visible'), 2000);
}

async function loadSettings(): Promise<void> {
  const resp = await sendMessage({ type: 'GET_CONFIG' });
  if (resp.type !== 'CONFIG_RESPONSE') return;
  const c: ExtensionConfig = resp.config;

  $('portalUrl').value = c.portalUrl;
  $('username').value = c.username ?? '';
  $('sessionToken').value = c.sessionToken ?? '';
  $('llmMode').value = c.llmMode;
  $('cloudModel').value = c.cloudModel;
  $('localModel').value = c.localModel;
  $('requireApproval').value = c.requireApproval;
  ($('maxToolCalls') as HTMLInputElement).value = String(c.maxToolCallsPerMinute);
  $('trustedAgents').value = c.trustedAgents.join(', ');

  const loginDot = document.getElementById('loginDot')!;
  const loginText = document.getElementById('loginText')!;
  if (c.sessionToken && c.username) {
    loginDot.classList.add('ok');
    loginText.textContent = `Logged in as @${c.username}`;
  } else {
    loginDot.classList.remove('ok');
    loginText.textContent = 'Not logged in';
  }
}

async function saveLogin(): Promise<void> {
  await sendMessage({
    type: 'SET_CONFIG',
    config: {
      portalUrl: $('portalUrl').value || 'https://robutler.ai',
      username: $('username').value || null,
      sessionToken: $('sessionToken').value || null,
    },
  });
  await loadSettings();
  showToast('Login saved');
}

async function saveAll(): Promise<void> {
  const trusted = $('trustedAgents').value
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);

  await sendMessage({
    type: 'SET_CONFIG',
    config: {
      portalUrl: $('portalUrl').value || 'https://robutler.ai',
      username: $('username').value || null,
      sessionToken: $('sessionToken').value || null,
      llmMode: $('llmMode').value as ExtensionConfig['llmMode'],
      cloudModel: $('cloudModel').value || 'gpt-4o',
      localModel: $('localModel').value || 'Llama-3.2-3B-Instruct-q4f16_1-MLC',
      requireApproval: $('requireApproval').value as ExtensionConfig['requireApproval'],
      maxToolCallsPerMinute: Number(($('maxToolCalls') as HTMLInputElement).value) || 30,
      trustedAgents: trusted,
    },
  });
  showToast('Settings saved');
}

document.getElementById('loginBtn')!.addEventListener('click', saveLogin);
document.getElementById('saveBtn')!.addEventListener('click', saveAll);

loadSettings();
