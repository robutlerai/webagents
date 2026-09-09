/**
 * Connected agents integration test.
 *
 * Validates the "connected agents" story from the docs:
 * two agents on a WebAgentsServer, communicating via UAMP and chat completions.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { BaseAgent } from '../../src/core/agent.js';
import { Skill } from '../../src/core/skill.js';
import { tool, handoff } from '../../src/core/decorators.js';
import { WebAgentsServer } from '../../src/server/multi.js';
import type { Context } from '../../src/core/types.js';
import type { ClientEvent, ServerEvent } from '../../src/uamp/events.js';
import {
  createSessionCreateEvent,
  createInputTextEvent,
  createResponseCreateEvent,
  createResponseDeltaEvent,
  createResponseDoneEvent,
} from '../../src/uamp/events.js';

/**
 * What every real caller of a billable route presents. The credential floor
 * (src/server/credential-floor.ts, installed on WebAgentsServer in
 * src/server/multi.ts) refuses an anonymous POST to `/agents/<name>/uamp`
 * before the route runs; these cases are about what the route does once a
 * caller is let in. The floor checks presence, not validity, and these agents
 * carry no AuthSkill, so a dummy bearer is the honest instrument. Same value
 * as the Python suite's AUTHED_HEADERS (python/tests/server/conftest.py).
 */
const AUTHED_HEADERS = {
  'Content-Type': 'application/json',
  Authorization: 'Bearer test-service-token',
};

// ---------------------------------------------------------------------------
// Skills
// ---------------------------------------------------------------------------

class EchoLLM extends Skill {
  @handoff({ name: 'echo-llm' })
  async *processUAMP(
    events: ClientEvent[],
    _ctx: Context,
  ): AsyncGenerator<ServerEvent> {
    const texts: string[] = [];
    for (const e of events) {
      if (e.type === 'input.text') {
        texts.push((e as { type: string; text: string }).text);
      }
    }
    const response = texts.join(' | ');
    yield createResponseDeltaEvent('r1', { type: 'text', text: response });
    yield createResponseDoneEvent(
      'r1',
      [{ type: 'text', text: response }],
      'completed',
      { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
    );
  }
}

class GreeterTool extends Skill {
  @tool({ description: 'Generate a greeting' })
  async greet(params: { name: string }, _ctx: Context) {
    return `Hello, ${params.name}!`;
  }
}

// ---------------------------------------------------------------------------
// Server setup
// ---------------------------------------------------------------------------

let server: WebAgentsServer;

beforeAll(async () => {
  server = new WebAgentsServer({ port: 0 });

  const agentA = new BaseAgent({
    name: 'echo',
    description: 'An echo agent that repeats input',
    skills: [new EchoLLM()],
  });

  const agentB = new BaseAgent({
    name: 'greeter',
    description: 'A greeter agent with a greeting tool',
    skills: [new EchoLLM(), new GreeterTool()],
  });

  await server.addAgent('echo', agentA);
  await server.addAgent('greeter', agentB);
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('Server routing', () => {
  it('echo agent health endpoint returns 200', async () => {
    const res = await server
      .getApp()
      .fetch(new Request('http://localhost/agents/echo/'));
    expect(res.status).toBe(200);
  });

  it('greeter agent health endpoint returns 200', async () => {
    const res = await server
      .getApp()
      .fetch(new Request('http://localhost/agents/greeter/'));
    expect(res.status).toBe(200);
  });

  it('unknown agent returns 404', async () => {
    const res = await server
      .getApp()
      .fetch(new Request('http://localhost/agents/nonexistent/'));
    expect(res.status).toBe(404);
  });
});

describe('UAMP communication', () => {
  it('echo agent echoes input via UAMP', async () => {
    const events = [
      createSessionCreateEvent({ modalities: ['text'] }),
      createInputTextEvent('hello from test'),
      createResponseCreateEvent(),
    ];

    const res = await server.getApp().fetch(
      new Request('http://localhost/agents/echo/uamp', {
        method: 'POST',
        headers: AUTHED_HEADERS,
        body: JSON.stringify(events),
      }),
    );

    expect(res.status).toBe(200);
    const body = await res.json();
    const types = body.map((e: { type: string }) => e.type);
    expect(types).toContain('response.delta');
    expect(types).toContain('response.done');

    const delta = body.find((e: { type: string }) => e.type === 'response.delta');
    expect(delta.delta.text).toBe('hello from test');
  });

  it('greeter agent echoes via UAMP', async () => {
    const events = [
      createSessionCreateEvent({ modalities: ['text'] }),
      createInputTextEvent('testing greeter'),
      createResponseCreateEvent(),
    ];

    const res = await server.getApp().fetch(
      new Request('http://localhost/agents/greeter/uamp', {
        method: 'POST',
        headers: AUTHED_HEADERS,
        body: JSON.stringify(events),
      }),
    );

    expect(res.status).toBe(200);
    const body = await res.json();
    const delta = body.find((e: { type: string }) => e.type === 'response.delta');
    expect(delta.delta.text).toBe('testing greeter');
  });

  // The floor witness for this file: with every UAMP case above credentialed,
  // this is what tells "the floor is installed" apart from "the floor is gone".
  it('refuses an anonymous UAMP POST with 401 while the public info GET stays open', async () => {
    const events = [
      createSessionCreateEvent({ modalities: ['text'] }),
      createInputTextEvent('anonymous'),
      createResponseCreateEvent(),
    ];

    const refused = await server.getApp().fetch(
      new Request('http://localhost/agents/echo/uamp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(events),
      }),
    );
    expect(refused.status).toBe(401);
    const body = await refused.json();
    expect(body.error.code).toBe('unauthorized');

    const info = await server
      .getApp()
      .fetch(new Request('http://localhost/agents/echo/info'));
    expect(info.status).toBe(200);
  });
});

describe('Agent info', () => {
  it('echo agent info includes name and description', async () => {
    const res = await server
      .getApp()
      .fetch(new Request('http://localhost/agents/echo/info'));
    expect(res.status).toBe(200);
    const info = await res.json();
    expect(info.name).toBe('echo');
    expect(info.description).toBe('An echo agent that repeats input');
  });

  it('greeter agent info includes tools', async () => {
    const res = await server
      .getApp()
      .fetch(new Request('http://localhost/agents/greeter/info'));
    expect(res.status).toBe(200);
    const info = await res.json();
    expect(info.name).toBe('greeter');
    const toolNames = (info.tools ?? []).map(
      (t: { function: { name: string } }) => t.function.name,
    );
    expect(toolNames).toContain('greet');
  });
});
