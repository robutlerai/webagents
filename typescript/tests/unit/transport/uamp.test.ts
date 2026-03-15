/**
 * UAMPTransportSkill Unit Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { UAMPTransportSkill } from '../../../src/skills/transport/uamp/skill.js';
import { BaseAgent } from '../../../src/core/agent.js';
import { Skill } from '../../../src/core/skill.js';
import { handoff } from '../../../src/core/decorators.js';
import type { Context } from '../../../src/core/types.js';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events.js';
import {
  createResponseDeltaEvent,
  createResponseDoneEvent,
} from '../../../src/uamp/events.js';

class EchoLLM extends Skill {
  @handoff({ name: 'echo-llm' })
  async *processUAMP(events: ClientEvent[], _c: Context): AsyncGenerator<ServerEvent> {
    const texts: string[] = [];
    for (const e of events) {
      if (e.type === 'input.text') {
        texts.push((e as { text: string }).text);
      }
    }
    const response = texts.join(' ');
    yield createResponseDeltaEvent('r1', { type: 'text', text: response });
    yield createResponseDoneEvent('r1', [{ type: 'text', text: response }]);
  }
}

function createMockWebSocket(): WebSocket & {
  sentMessages: string[];
  triggerMessage: (data: string) => void;
  triggerClose: () => void;
} {
  const sentMessages: string[] = [];
  let onmessageHandler: ((ev: MessageEvent) => void) | null = null;
  let oncloseHandler: (() => void) | null = null;

  const ws = {
    sentMessages,
    send(data: string) {
      sentMessages.push(data);
    },
    close() {},
    get onmessage() { return onmessageHandler; },
    set onmessage(fn: ((ev: MessageEvent) => void) | null) { onmessageHandler = fn; },
    get onclose() { return oncloseHandler; },
    set onclose(fn: (() => void) | null) { oncloseHandler = fn; },
    onerror: null as ((ev: Event) => void) | null,
    triggerMessage(data: string) {
      if (onmessageHandler) {
        onmessageHandler({ data } as MessageEvent);
      }
    },
    triggerClose() {
      if (oncloseHandler) {
        oncloseHandler();
      }
    },
  } as unknown as WebSocket & {
    sentMessages: string[];
    triggerMessage: (data: string) => void;
    triggerClose: () => void;
  };

  return ws;
}

describe('UAMPTransportSkill', () => {
  let skill: UAMPTransportSkill;

  beforeEach(() => {
    skill = new UAMPTransportSkill();
  });

  describe('wsEndpoints', () => {
    it('registers /uamp in wsEndpoints', () => {
      const endpoints = skill.wsEndpoints ?? [];
      const uamp = endpoints.find(e => e.path === '/uamp');
      expect(uamp).toBeDefined();
    });
  });

  describe('handleConnection lifecycle', () => {
    it('sends session.created and capabilities on session.create', async () => {
      const agent = new BaseAgent({
        name: 'uamp-test',
        skills: [new EchoLLM()],
      });
      agent.addSkill(skill);

      const ws = createMockWebSocket();
      const ctx = {} as Context;
      skill.handleConnection(ws, ctx);

      ws.triggerMessage(JSON.stringify({
        type: 'session.create',
        event_id: 'e1',
        uamp_version: '1.0',
        session: { modalities: ['text'] },
      }));

      // Wait for async processing
      await new Promise(r => setTimeout(r, 50));

      expect(ws.sentMessages.length).toBeGreaterThanOrEqual(2);
      const types = ws.sentMessages.map(m => JSON.parse(m).type);
      expect(types).toContain('session.created');
      expect(types).toContain('capabilities');
    });

    it('sends error when message received before session.create', async () => {
      const agent = new BaseAgent({
        name: 'uamp-test',
        skills: [new EchoLLM()],
      });
      agent.addSkill(skill);

      const ws = createMockWebSocket();
      skill.handleConnection(ws, {} as Context);

      ws.triggerMessage(JSON.stringify({
        type: 'input.text',
        event_id: 'e2',
        text: 'hello',
        role: 'user',
      }));

      await new Promise(r => setTimeout(r, 50));

      const lastMsg = JSON.parse(ws.sentMessages[ws.sentMessages.length - 1]);
      expect(lastMsg.type).toBe('response.error');
      expect(lastMsg.error.code).toBe('session_required');
    });

    it('sends error when no agent attached', async () => {
      const ws = createMockWebSocket();
      skill.handleConnection(ws, {} as Context);

      ws.triggerMessage(JSON.stringify({
        type: 'session.create',
        session: { modalities: ['text'] },
      }));
      await new Promise(r => setTimeout(r, 50));

      ws.triggerMessage(JSON.stringify({
        type: 'input.text',
        text: 'hello',
        role: 'user',
      }));
      await new Promise(r => setTimeout(r, 100));

      const allMsgs = ws.sentMessages.map(m => JSON.parse(m));
      const errorMsg = allMsgs.find(m => m.type === 'response.error');
      expect(errorMsg).toBeDefined();
      expect(errorMsg.error.code).toBe('no_agent');
    });
  });

  describe('cleanup', () => {
    it('clears sessions on cleanup', async () => {
      const agent = new BaseAgent({
        name: 'uamp-cleanup',
        skills: [new EchoLLM()],
      });
      agent.addSkill(skill);

      const ws = createMockWebSocket();
      skill.handleConnection(ws, {} as Context);

      ws.triggerMessage(JSON.stringify({
        type: 'session.create',
        session: { modalities: ['text'] },
      }));
      await new Promise(r => setTimeout(r, 50));

      await skill.cleanup();

      // After cleanup, sessions should be empty
      // Verify by closing the ws (should not error)
      expect(() => ws.triggerClose()).not.toThrow();
    });
  });
});
