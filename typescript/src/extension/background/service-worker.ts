import type { BackgroundMessage } from '../shared/types';
import { ExtensionAgentRuntime } from './agent-runtime';

const runtime = new ExtensionAgentRuntime();

runtime.initialize().catch(console.error);

chrome.runtime.onMessage.addListener(
  (
    message: BackgroundMessage,
    _sender: chrome.runtime.MessageSender,
    sendResponse: (response: BackgroundMessage) => void,
  ) => {
    handleMessage(message)
      .then(sendResponse)
      .catch((err) => sendResponse({ type: 'ERROR', error: String(err) }));
    return true; // async response
  },
);

async function handleMessage(msg: BackgroundMessage): Promise<BackgroundMessage> {
  switch (msg.type) {
    case 'GET_STATUS':
      return { type: 'STATUS_RESPONSE', status: runtime.getStatus() };

    case 'GET_CONFIG': {
      const { loadConfig } = await import('../shared/storage.js');
      return { type: 'CONFIG_RESPONSE', config: await loadConfig() };
    }

    case 'SET_CONFIG': {
      await runtime.updateConfig(msg.config);
      const { loadConfig: load } = await import('../shared/storage.js');
      return { type: 'CONFIG_RESPONSE', config: await load() };
    }

    case 'GET_TASKS':
      return { type: 'TASKS_RESPONSE', tasks: runtime.getTasks() };

    case 'LOGIN':
      await runtime.loginWithRobutler();
      return { type: 'STATUS_RESPONSE', status: runtime.getStatus() };

    case 'LOGOUT':
      await runtime.logout();
      return { type: 'STATUS_RESPONSE', status: runtime.getStatus() };

    case 'CONNECT':
      await runtime.connect();
      return { type: 'STATUS_RESPONSE', status: runtime.getStatus() };

    case 'DISCONNECT':
      runtime.disconnect();
      return { type: 'STATUS_RESPONSE', status: runtime.getStatus() };

    case 'SEND_CHAT': {
      const response = await runtime.sendChat(msg.message);
      return { type: 'CHAT_RESPONSE', response };
    }

    default:
      return { type: 'ERROR', error: 'Unknown message type' };
  }
}

chrome.alarms.create('heartbeat', { periodInMinutes: 1 });
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === 'heartbeat') {
    const status = runtime.getStatus();
    if (status.connected) {
      // Keep-alive — service workers can be terminated after 30s of inactivity
      console.debug('[robutler] heartbeat', status.agentName, status.taskCount);
    }
  }
});
