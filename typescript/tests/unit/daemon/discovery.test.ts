/**
 * Which agents `webagents daemon` serves, the same in both SDKs (2026-09-25).
 *
 * The tree is `python/tests/fixtures/daemon/discovery.json`, which the Python
 * suite builds too (`tests/cli/test_daemon_discovery.py`). This daemon served
 * nothing without `-w`, and with it only files touched after it started: the
 * watcher's first scan emitted nothing (`src/daemon/watcher.ts`). Pinned here:
 * the files it reads, the working directory's agents served with no folder
 * named, a change reloading the agent under its new name, a deleted file let
 * go, and two files declaring one name said out loud.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { WebAgentsDaemon } from '../../../src/daemon/server';
import { findAgentFiles } from '../../../src/daemon/watcher';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(
  fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/daemon/discovery.json'), 'utf8'),
) as {
  agents: { path: string; name: string }[];
  files: string[];
  discovered: string[];
  served: string[];
  not_served: string[];
};

const tempDir = tempDirs();
const cwd = process.cwd();
let root = '';
let daemon: WebAgentsDaemon | null = null;

function write(rel: string, text: string): void {
  const file = path.join(root, rel);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, text);
}

const agentText = (name: string) => `---\nname: ${name}\n---\nHelp.\n`;

beforeEach(() => {
  root = fs.realpathSync(tempDir('wa-daemon-'));
  for (const agent of FIXTURE.agents) write(agent.path, agentText(agent.name));
  for (const file of FIXTURE.files) write(file, 'Notes.\n');
  process.chdir(root);
});

afterEach(() => {
  daemon?.stop();
  daemon = null;
  process.chdir(cwd);
  vi.restoreAllMocks();
});

type Inside = {
  app: { fetch(request: Request): Promise<Response> };
  watcher: { rescan(): void };
};

async function started(config: ConstructorParameters<typeof WebAgentsDaemon>[0] = {}): Promise<Inside> {
  vi.spyOn(console, 'log').mockImplementation(() => {});
  daemon = new WebAgentsDaemon({ port: 0, cron: false, healthChecks: false, ...config });
  await daemon.discover();
  return daemon as unknown as Inside;
}

async function status(inside: Inside, name: string): Promise<number> {
  return (await inside.app.fetch(new Request(`http://localhost/agents/${name}`))).status;
}

async function settled(inside: Inside): Promise<void> {
  inside.watcher.rescan();
  await daemon!.discover();
}

describe('webagents daemon discovery', () => {
  it('reads the files the fixture says', () => {
    const found = [...findAgentFiles(root).keys()].map((file) => path.relative(root, file).split(path.sep).join('/'));
    expect(found.sort()).toEqual(FIXTURE.discovered);
  });

  it("without a folder named, serves the working directory's agents, and only those", async () => {
    const inside = await started();
    for (const name of FIXTURE.served) expect(await status(inside, name), name).toBe(200);
    for (const name of FIXTURE.not_served) expect(await status(inside, name), name).toBe(404);
  });

  it('-w names another folder', async () => {
    const elsewhere = tempDir('wa-daemon-elsewhere-');
    fs.writeFileSync(path.join(elsewhere, 'AGENT.md'), agentText('over-there'));
    const inside = await started({ watchDir: elsewhere });
    expect(await status(inside, 'over-there')).toBe(200);
    expect(await status(inside, 'main-agent')).toBe(404);
  });

  it('a changed file serves the agent under its new name, and a deleted one is let go', async () => {
    const inside = await started();
    // A different size, so the change is seen whatever the clock's resolution.
    write('AGENT-helper.md', agentText('assistant-renamed'));
    await settled(inside);
    expect(await status(inside, 'helper')).toBe(404);
    expect(await status(inside, 'assistant-renamed')).toBe(200);

    fs.unlinkSync(path.join(root, 'AGENT-helper.md'));
    await settled(inside);
    expect(await status(inside, 'assistant-renamed')).toBe(404);
    expect(await status(inside, 'main-agent')).toBe(200);
  });

  it('two files declaring one name: the later is served, and the daemon says so', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const inside = await started();
    write('team/AGENT-copy.md', agentText('main-agent'));
    await settled(inside);
    expect(await status(inside, 'main-agent')).toBe(200);
    expect(warn).toHaveBeenCalledWith(
      `agent 'main-agent' is declared by two files; ${path.join(root, 'team', 'AGENT-copy.md')} now replaces ${path.join(root, 'AGENT.md')}`,
    );
  });
});
