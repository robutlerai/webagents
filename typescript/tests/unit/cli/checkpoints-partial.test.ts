/**
 * The /undo snapshot in a large folder (2026-09-28), the same in both SDKs
 * (`python/tests/cli/test_checkpoint_partial.py` runs these cases too).
 *
 * The walk used to go on past its caps, listing every later path as skipped:
 * under ~/dev/portal (331,695 files) that was a 9 s walk before every message,
 * before the chat's spinner had started. It now stops at the cap and says the
 * snapshot is partial, and a restore of a partial snapshot removes nothing,
 * because it cannot tell a file made since from one past the cap. The chat
 * takes it once the spinner is drawing, through the yielding walk.
 */

import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';

import {
  UNDO_WORDS,
  latestCheckpoint,
  planRestore,
  restoreSnapshot,
  scanFolder,
  takeSnapshot,
  takeSnapshotYielding,
  type Manifest,
} from '../../../src/cli/checkpoints';
import { tempDirs } from '../../helpers/cli';

const tempDir = tempDirs();

function folderWith(names: string[]): { root: string; store: string } {
  const base = tempDir('cp-partial-');
  const root = path.join(base, 'project');
  for (const name of names) {
    const p = path.join(root, name);
    fs.mkdirSync(path.dirname(p), { recursive: true });
    fs.writeFileSync(p, `${name}\n`);
  }
  return { root, store: path.join(base, 'store') };
}

describe('a folder past the cap', () => {
  it('stops the walk at the cap and says the snapshot is partial', () => {
    const { root, store } = folderWith(['a.txt', 'b/c.txt', 'b/d.txt', 'e.txt', 'f.txt']);
    const manifest = takeSnapshot(root, 'before', { store, limits: { maxFiles: 3 } });
    expect(manifest.partial).toBe(true);
    // Walk order: names sorted at each level, directories entered in place.
    expect(Object.keys(manifest.files).sort()).toEqual(['a.txt', 'b/c.txt', 'b/d.txt']);
    expect(manifest.skipped).toEqual({});
  });

  it('a folder under the cap is not partial, and its manifest has no partial key', () => {
    const { root, store } = folderWith(['a.txt', 'b.txt']);
    const manifest = takeSnapshot(root, 'before', { store, limits: { maxFiles: 3 } });
    expect('partial' in manifest).toBe(false);
  });

  it('the byte cap stops it too', () => {
    const { root, store } = folderWith(['a.txt', 'b.txt', 'c.txt']);
    const state = scanFolder(root, store, undefined, { maxTotalBytes: 12 });
    expect(state.partial).toBe(true);
    expect(Object.keys(state.files)).toEqual(['a.txt', 'b.txt']);
  });

  it('a restore of a partial snapshot puts back what it recorded and removes nothing', () => {
    const { root, store } = folderWith(['a.txt', 'b.txt', 'c.txt', 'd.txt']);
    const target = takeSnapshot(root, 'before', { store, limits: { maxFiles: 2 } });
    fs.writeFileSync(path.join(root, 'a.txt'), 'changed by the agent\n');
    fs.writeFileSync(path.join(root, 'new.txt'), 'made since\n');
    const plan = planRestore(target, scanFolder(root, store));
    expect(plan.write).toEqual(['a.txt']);
    expect(plan.remove).toEqual([]);
    const result = restoreSnapshot(root, target.id, 'before /undo', { store });
    expect(result.removed).toEqual([]);
    expect(fs.readFileSync(path.join(root, 'a.txt'), 'utf8')).toBe('a.txt\n');
    // Past the cap when the snapshot was taken, or made since: either way, left.
    expect(fs.existsSync(path.join(root, 'new.txt'))).toBe(true);
    expect(fs.existsSync(path.join(root, 'd.txt'))).toBe(true);
  });

  it('the chats say why nothing is removed', () => {
    expect(UNDO_WORDS.partialNote).toBe(
      'This folder is larger than a snapshot holds (20000 files, 200 MB), so files made since it are left in place.',
    );
  });
});

describe('the newest snapshot', () => {
  function write(store: string, m: Partial<Manifest> & { id: string; created_at: string }): void {
    fs.mkdirSync(store, { recursive: true });
    const manifest = { version: 1, label: m.id, files: {}, links: {}, skipped: {}, ...m };
    fs.writeFileSync(path.join(store, `${m.id}.json`), JSON.stringify(manifest));
  }

  it('is found among several, breaking a same-second tie by the time inside it', () => {
    const store = path.join(tempDir('cp-partial-'), 'store');
    write(store, { id: 'cp_20260928T120000Z_ffffffff', created_at: '2026-09-28T12:00:00.100Z' });
    write(store, { id: 'cp_20260928T120001Z_00000001', created_at: '2026-09-28T12:00:01.900Z' });
    write(store, { id: 'cp_20260928T120001Z_ffffffff', created_at: '2026-09-28T12:00:01.200Z' });
    expect(latestCheckpoint(store)?.id).toBe('cp_20260928T120001Z_00000001');
  });

  it('skips a newest manifest that cannot be read', () => {
    const store = path.join(tempDir('cp-partial-'), 'store');
    write(store, { id: 'cp_20260928T120000Z_aaaaaaaa', created_at: '2026-09-28T12:00:00.000Z' });
    fs.writeFileSync(path.join(store, 'cp_20260928T120005Z_bbbbbbbb.json'), '{not json');
    expect(latestCheckpoint(store)?.id).toBe('cp_20260928T120000Z_aaaaaaaa');
  });

  it('is none in an empty store', () => {
    expect(latestCheckpoint(path.join(tempDir('cp-partial-'), 'missing'))).toBeUndefined();
  });
});

describe('the chat takes it without freezing', () => {
  it('lets the event loop run while it walks, and records what takeSnapshot records', async () => {
    const names = Array.from({ length: 3000 }, (_, i) => `d${Math.floor(i / 100)}/f${i}.txt`);
    const { root, store } = folderWith(names);
    let ticks = 0;
    const timer = setInterval(() => {
      ticks += 1;
    }, 1);
    let manifest: Manifest;
    try {
      manifest = await takeSnapshotYielding(root, 'before', { store });
    } finally {
      clearInterval(timer);
    }
    expect(ticks).toBeGreaterThan(0);
    expect(Object.keys(manifest.files)).toHaveLength(3000);
    const again = takeSnapshot(root, 'again', { store });
    expect(again.id).toBe(manifest.id); // Nothing changed: the same snapshot.
  });
});
