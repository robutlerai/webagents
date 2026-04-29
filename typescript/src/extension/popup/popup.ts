import type { AgentStatus, BackgroundMessage, TaskRecord } from '../shared/types';

function $(id: string): HTMLElement {
  return document.getElementById(id)!;
}

function formatUptime(ms: number): string {
  if (ms < 60_000) return `${Math.floor(ms / 1000)}s`;
  if (ms < 3_600_000) return `${Math.floor(ms / 60_000)}m`;
  return `${Math.floor(ms / 3_600_000)}h`;
}

function sendMessage(msg: BackgroundMessage): Promise<BackgroundMessage> {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(msg, resolve);
  });
}

async function refreshStatus(): Promise<void> {
  const resp = await sendMessage({ type: 'GET_STATUS' });
  if (!resp || resp.type !== 'STATUS_RESPONSE') return;
  const s: AgentStatus = resp.status;
  const configResp = await sendMessage({ type: 'GET_CONFIG' });
  const config = configResp.type === 'CONFIG_RESPONSE' ? configResp.config : null;
  const loggedIn = !!(config?.extensionToken || config?.sessionToken);

  $('statusDot').classList.toggle('connected', s.connected);
  $('agentName').textContent = s.agentName ? `@${s.agentName}` : '—';
  $('taskCount').textContent = String(s.taskCount);
  $('llmMode').textContent = s.llmMode;
  $('uptime').textContent = s.uptime > 0 ? formatUptime(s.uptime) : '—';
  $('connectionText').textContent = s.connected ? 'Connected to Robutler' : loggedIn ? 'Ready to connect' : 'Not logged in';
  $('accountName').textContent = config?.username ? `@${config.username}` : 'Not logged in';
  $('accountHint').textContent = loggedIn
    ? 'Use Robutler chat to talk to your browser agent.'
    : 'Log in to create and run your browser companion agent.';

  const btn = $('connectBtn') as HTMLButtonElement;
  if (s.connected) {
    btn.textContent = 'Disconnect';
    btn.className = 'btn-danger';
  } else {
    btn.textContent = 'Connect';
    btn.className = 'btn-primary';
  }
  btn.toggleAttribute('disabled', !loggedIn);
  $('loginBtn').classList.toggle('btn-hidden', loggedIn);
  $('logoutBtn').classList.toggle('btn-hidden', !loggedIn);
}

async function refreshTasks(): Promise<void> {
  const resp = await sendMessage({ type: 'GET_TASKS' });
  if (!resp || resp.type !== 'TASKS_RESPONSE') return;

  const list = $('taskList');
  if (resp.tasks.length === 0) {
    list.innerHTML = '<div class="empty">No tasks yet</div>';
    return;
  }

  list.innerHTML = resp.tasks
    .slice(-5)
    .reverse()
    .map(
      (t: TaskRecord) => `
      <div class="task">
        <div class="source">@${t.source}</div>
        <div class="instruction">${escapeHtml(t.instruction.slice(0, 80))}</div>
        <div class="status">${t.status} ${t.completedAt ? `(${Math.round((t.completedAt - t.startedAt) / 1000)}s)` : ''}</div>
      </div>
    `,
    )
    .join('');
}

function escapeHtml(s: string): string {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

$('loginBtn').addEventListener('click', async () => {
  await sendMessage({ type: 'LOGIN' });
  await refreshStatus();
});

$('logoutBtn').addEventListener('click', async () => {
  await sendMessage({ type: 'LOGOUT' });
  await refreshStatus();
  await refreshTasks();
});

$('connectBtn').addEventListener('click', async () => {
  const resp = await sendMessage({ type: 'GET_STATUS' });
  if (resp.type === 'STATUS_RESPONSE' && resp.status.connected) {
    await sendMessage({ type: 'DISCONNECT' });
  } else {
    const connectResp = await sendMessage({ type: 'CONNECT' });
    if (connectResp.type === 'ERROR') {
      $('connectionText').textContent = connectResp.error;
      return;
    }
  }
  await refreshStatus();
});

$('settingsBtn').addEventListener('click', () => {
  chrome.runtime.openOptionsPage();
});

$('portalLink').addEventListener('click', async (e) => {
  e.preventDefault();
  const resp = await sendMessage({ type: 'GET_CONFIG' });
  const url = resp.type === 'CONFIG_RESPONSE' ? resp.config.portalUrl : 'https://robutler.ai';
  chrome.tabs.create({ url });
});

refreshStatus();
refreshTasks();
setInterval(() => { refreshStatus(); refreshTasks(); }, 5000);
