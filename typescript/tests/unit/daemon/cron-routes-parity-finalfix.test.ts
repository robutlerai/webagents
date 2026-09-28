/**
 * The daemon lists the served agents' schedules at the two paths the shared
 * fixture names (`python/tests/fixtures/daemon/cron.json`, `routes`), which
 * the Python daemon serves too since 2026-09-27 (the final e2e re-run; it
 * served `/agents/cron` only). Read-only at both (S-273).
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { WebAgentsDaemon } from '../../../src/daemon/server';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROUTES = (JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/daemon/cron.json'), 'utf8')) as {
  routes: { list: string[]; body_key: string };
}).routes;

describe('the schedule listing is served at the fixture’s paths', () => {
  it('answers 200 with the schedules at each, and takes nothing at either', async () => {
    const daemon = new WebAgentsDaemon({ port: 0, watch: false, cron: false, healthChecks: false });
    for (const route of ROUTES.list) {
      const listed = await daemon.app.fetch(new Request(`http://localhost${route}`));
      expect(listed.status, route).toBe(200);
      const body = (await listed.json()) as Record<string, unknown>;
      expect(Array.isArray(body[ROUTES.body_key]), route).toBe(true);
      const add = await daemon.app.fetch(
        new Request(`http://localhost${route}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id: 'evil', cron: '* * * * *', agentName: 'reporter', task: 'x' }) }),
      );
      expect(add.status, `POST ${route}`).toBe(404);
    }
  });
});
