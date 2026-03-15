/**
 * A2ATransportSkill Unit Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { A2ATransportSkill } from '../../../src/skills/transport/a2a/skill.js';
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

describe('A2ATransportSkill', () => {
  let skill: A2ATransportSkill;

  beforeEach(() => {
    skill = new A2ATransportSkill();
  });

  describe('httpEndpoints', () => {
    it('registers POST:/a2a in httpEndpoints', () => {
      const endpoints = skill.httpEndpoints ?? [];
      const a2a = endpoints.find(e => e.path === '/a2a' && e.method === 'POST');
      expect(a2a).toBeDefined();
    });

    it('registers GET:/.well-known/agent.json in httpEndpoints', () => {
      const endpoints = skill.httpEndpoints ?? [];
      const card = endpoints.find(e => e.path === '/.well-known/agent.json' && e.method === 'GET');
      expect(card).toBeDefined();
    });
  });

  describe('agent card endpoint', () => {
    it('returns 404 when no agent card configured and no agent set', async () => {
      const handler = skill.httpEndpoints?.find(e => e.path === '/.well-known/agent.json');
      const req = new Request('http://localhost/.well-known/agent.json');
      const ctx = {} as Context;
      const res = await handler!.handler(req, ctx);
      expect(res.status).toBe(404);
    });

    it('returns agent card after setAgent is called', async () => {
      const agent = new BaseAgent({
        name: 'test-agent',
        description: 'A test agent',
        skills: [new EchoLLM()],
      });
      agent.addSkill(skill);

      const handler = skill.httpEndpoints?.find(e => e.path === '/.well-known/agent.json');
      const req = new Request('http://localhost/.well-known/agent.json');
      const ctx = {} as Context;
      const res = await handler!.handler(req, ctx);
      expect(res.status).toBe(200);

      const card = await res.json();
      expect(card.name).toBe('test-agent');
      expect(card.description).toBe('A test agent');
    });
  });

  describe('tasks/send via /a2a', () => {
    it('creates and processes a task via processUAMP when agent is attached', async () => {
      const agent = new BaseAgent({
        name: 'a2a-agent',
        skills: [new EchoLLM()],
      });
      agent.addSkill(skill);

      const handler = skill.httpEndpoints?.find(e => e.path === '/a2a' && e.method === 'POST');
      const req = new Request('http://localhost/a2a', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jsonrpc: '2.0',
          id: 1,
          method: 'tasks/send',
          params: {
            message: {
              role: 'user',
              parts: [{ type: 'text', text: 'Hello from A2A' }],
            },
          },
        }),
      });
      const ctx = {} as Context;
      const res = await handler!.handler(req, ctx);
      expect(res.status).toBe(200);

      const body = await res.json();
      expect(body.jsonrpc).toBe('2.0');
      expect(body.result.status.state).toBe('completed');
      expect(body.result.artifacts).toBeDefined();
      expect(body.result.artifacts[0].parts[0].text).toContain('Hello from A2A');
    });

    it('returns failed task when no agent attached', async () => {
      const handler = skill.httpEndpoints?.find(e => e.path === '/a2a' && e.method === 'POST');
      const req = new Request('http://localhost/a2a', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jsonrpc: '2.0',
          id: 1,
          method: 'tasks/send',
          params: {
            message: {
              role: 'user',
              parts: [{ type: 'text', text: 'Hello' }],
            },
          },
        }),
      });
      const ctx = {} as Context;
      const res = await handler!.handler(req, ctx);
      const body = await res.json();
      expect(body.result.status.state).toBe('failed');
    });
  });

  describe('tasks/get and tasks/cancel', () => {
    it('can retrieve and cancel a task', async () => {
      const agent = new BaseAgent({
        name: 'cancel-agent',
        skills: [new EchoLLM()],
      });
      agent.addSkill(skill);

      const handler = skill.httpEndpoints?.find(e => e.path === '/a2a' && e.method === 'POST');
      const ctx = {} as Context;

      // Create a task
      const sendRes = await handler!.handler(
        new Request('http://localhost/a2a', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            jsonrpc: '2.0', id: 1,
            method: 'tasks/send',
            params: { id: 'task-123', message: { role: 'user', parts: [{ type: 'text', text: 'hi' }] } },
          }),
        }),
        ctx,
      );
      const sendBody = await sendRes.json();
      expect(sendBody.result.id).toBe('task-123');

      // Get the task
      const getRes = await handler!.handler(
        new Request('http://localhost/a2a', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            jsonrpc: '2.0', id: 2,
            method: 'tasks/get',
            params: { id: 'task-123' },
          }),
        }),
        ctx,
      );
      const getBody = await getRes.json();
      expect(getBody.result.id).toBe('task-123');

      // Cancel
      const cancelRes = await handler!.handler(
        new Request('http://localhost/a2a', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            jsonrpc: '2.0', id: 3,
            method: 'tasks/cancel',
            params: { id: 'task-123' },
          }),
        }),
        ctx,
      );
      const cancelBody = await cancelRes.json();
      expect(cancelBody.result.status.state).toBe('canceled');
    });
  });

  describe('unknown method', () => {
    it('returns -32601 for unknown JSON-RPC method', async () => {
      const handler = skill.httpEndpoints?.find(e => e.path === '/a2a' && e.method === 'POST');
      const req = new Request('http://localhost/a2a', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jsonrpc: '2.0', id: 1,
          method: 'tasks/unknown',
        }),
      });
      const res = await handler!.handler(req, {} as Context);
      const body = await res.json();
      expect(body.error.code).toBe(-32601);
    });
  });
});
