/**
 * A credentialed WebSocket upgrade gets past loading `ws` (2026-09-24).
 *
 * `server/node.ts` loaded `ws` with a bare `require`, which does not exist in
 * an ES module. It threw `ReferenceError`, the surrounding `catch` answered
 * "500 ws package not available", and so every UAMP WebSocket to a `serve()`d
 * agent failed although `ws` is a declared dependency. The billable-route
 * suite only ever sent ANONYMOUS upgrades, which are refused with 401 before
 * that line runs, so nothing reached it. Found by the onboarding docs audit.
 */

import { describe, expect, it } from 'vitest';

import { BaseAgent } from '../../../src/core/agent.js';
import { createAgentApp } from '../../../src/server/node.js';
import { UAMPTransportSkill } from '../../../src/skills/transport/uamp/skill.js';

describe('the WebSocket upgrade on a served agent', () => {
  it('loads ws instead of answering 500', async () => {
    const skill = new UAMPTransportSkill();
    const agent = new BaseAgent({ name: 'mini', instructions: 'x', skills: [skill] });
    await skill.initialize?.(agent as never);

    const { handleUpgrade } = createAgentApp(agent, { basePath: '/agents/mini', logging: false });

    const written: string[] = [];
    const socket = {
      write: (chunk: string) => { written.push(String(chunk)); return true; },
      destroy: () => {},
      on: () => socket,
      removeListener: () => socket,
      readable: true,
      writable: true,
    };
    const req = {
      url: '/agents/mini/uamp?token=any-credential',
      method: 'GET',
      headers: { host: 'agent.example.com', upgrade: 'websocket', connection: 'Upgrade' },
    };

    try {
      handleUpgrade(req as never, socket as never, Buffer.alloc(0));
    } catch {
      // `ws` may reject this hand-made socket further in; that is past the point under test.
    }

    // THE BUG: this is what every credentialed upgrade answered.
    expect(written.join('')).not.toContain('ws package not available');
    // And it was not refused as anonymous either: the credential was present.
    expect(written.join('')).not.toContain('401 Unauthorized');
  });
});

describe('no bare require in the ES module sources', () => {
  // Vitest's module runner PROVIDES a `require`, so the behavioural test above
  // passes against the broken line too; the failure only exists in the built
  // ES module under plain Node. So the rule itself is pinned, statically.
  it('src/server has no bare require() left', async () => {
    const fs = await import('node:fs');
    const path = await import('node:path');
    const dir = path.resolve(__dirname, '../../../src/server');
    const offenders: string[] = [];
    for (const file of fs.readdirSync(dir).filter((f) => f.endsWith('.ts'))) {
      fs.readFileSync(path.join(dir, file), 'utf-8').split('\n').forEach((line, i) => {
        const code = line.replace(/\/\/.*$/, '');
        if (/(^|[^\w.])require\(/.test(code)) offenders.push(`${file}:${i + 1}`);
      });
    }
    expect(offenders).toEqual([]);
  });
});
