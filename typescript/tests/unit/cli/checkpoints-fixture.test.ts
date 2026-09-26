/**
 * Snapshots of an agent's folder for /undo and /rewind (2026-09-25), the same
 * in both SDKs: `python/tests/fixtures/checkpoints/checkpoints.json`, which
 * the Python suite runs too (`tests/cli/test_checkpoint_fixture.py`). What a
 * snapshot of a folder holds, what a restore does and leaves the folder as,
 * the words both chats say; and beside the fixture, what keeps a restore safe.
 */

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  KEEP,
  UNDO_OFF_HERE,
  UNDO_WORDS,
  checkpointsDir,
  isCheckpointId,
  listCheckpoints,
  newCheckpointId,
  planLines,
  planRestore,
  restoreSnapshot,
  restoredSentence,
  rewindHeader,
  scanFolder,
  snapshotsOffReason,
  takeSnapshot,
  turnLabel,
  type Manifest,
} from '../../../src/cli/checkpoints';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(
  readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/checkpoints/checkpoints.json')),
);
function readFileSync(file: string): string {
  return fs.readFileSync(file, 'utf8');
}

const tempDir = tempDirs();

interface Tree {
  files: { path: string; text: string; mode: number }[];
  links: { path: string; target: string }[];
  large: { path: string; size: number }[];
}

function build(root: string, tree: Tree): void {
  for (const f of tree.files) {
    const p = path.join(root, f.path);
    fs.mkdirSync(path.dirname(p), { recursive: true });
    fs.writeFileSync(p, f.text);
    fs.chmodSync(p, f.mode);
  }
  for (const l of tree.links) fs.symlinkSync(l.target, path.join(root, l.path));
  for (const b of tree.large) {
    const p = path.join(root, b.path);
    fs.mkdirSync(path.dirname(p), { recursive: true });
    fs.writeFileSync(p, '');
    fs.truncateSync(p, b.size);
  }
}

function change(root: string, changes: Record<string, unknown>[]): void {
  for (const c of changes) {
    if (typeof c.write === 'string') {
      const p = path.join(root, c.write);
      fs.mkdirSync(path.dirname(p), { recursive: true });
      fs.writeFileSync(p, c.text as string);
    } else if (typeof c.remove === 'string') {
      fs.unlinkSync(path.join(root, c.remove));
    } else if (typeof c.chmod === 'string') {
      fs.chmodSync(path.join(root, c.chmod), c.mode as number);
    } else if (typeof c.relink === 'string') {
      fs.unlinkSync(path.join(root, c.relink));
      fs.symlinkSync(c.target as string, path.join(root, c.relink));
    }
  }
}

const held = (m: Pick<Manifest, 'files' | 'links' | 'skipped'>) => ({
  files: Object.fromEntries(Object.entries(m.files).map(([k, v]) => [k, { sha256: v.sha256, size: v.size, mode: v.mode }])),
  links: m.links,
  skipped: m.skipped,
});

let root = '';
let store = '';

beforeEach(() => {
  const base = tempDir('wa-cp-');
  root = path.join(base, 'agent');
  store = path.join(base, 'store');
  fs.mkdirSync(root);
  build(root, FIXTURE.tree);
});

describe('snapshots (shared fixture)', () => {
  it('a snapshot holds what the fixture says', () => {
    const m = takeSnapshot(root, 'before "plan the launch"', { store });
    expect(held(m)).toEqual(FIXTURE.snapshot);
    expect(isCheckpointId(m.id)).toBe(true);
    expect(m.created_at).toMatch(/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$/);
  });

  it('a restore does what the fixture says and leaves the folder as it was', () => {
    const m = takeSnapshot(root, 'before', { store });
    change(root, FIXTURE.changes);

    const plan = planRestore(m, scanFolder(root, store, m));
    expect(plan).toEqual(FIXTURE.plan);

    const result = restoreSnapshot(root, m.id, 'before /undo', { store });
    expect({ written: result.written, removed: result.removed, kept: result.kept }).toEqual(FIXTURE.restored);
    expect(result.failed).toEqual([]);
    expect(held(scanFolder(root, store))).toEqual(FIXTURE.snapshot);
    expect(result.before.label).toBe('before /undo');
    expect(result.before.id).not.toBe(m.id);
    expect(fs.readFileSync(path.join(root, 'node_modules/pkg/index.js'), 'utf8')).toBe('changed\n');
  });

  it('nothing changed is not kept twice', () => {
    const first = takeSnapshot(root, 'one', { store });
    const second = takeSnapshot(root, 'two', { store });
    expect(second.id).toBe(first.id);
    expect(listCheckpoints(store)).toHaveLength(1);
  });

  it('only the newest are kept, with the objects they use', () => {
    const start = Date.UTC(2026, 8, 25, 12, 0, 0);
    for (let i = 0; i < KEEP + 2; i++) {
      fs.writeFileSync(path.join(root, 'notes/plan.md'), `version ${i}\n`);
      takeSnapshot(root, `v${i}`, { store, now: new Date(start + i * 1000) });
    }
    const kept = listCheckpoints(store);
    expect(kept).toHaveLength(KEEP);
    expect(kept[0].label).toBe(`v${KEEP + 1}`);
    expect(kept[kept.length - 1].label).toBe('v2');
    const used = new Set(kept.flatMap((m) => Object.values(m.files).map((f) => f.sha256)));
    expect(new Set(fs.readdirSync(path.join(store, 'objects')))).toEqual(used);
  });

  it.each(FIXTURE.ids as { id: string; valid: boolean }[])('id %j', ({ id, valid }) => {
    expect(isCheckpointId(id)).toBe(valid);
  });

  it('a new id is one', () => {
    expect(isCheckpointId(newCheckpointId())).toBe(true);
  });

  it.each(FIXTURE.labels as { message: string; label: string }[])('label for %j', ({ message, label }) => {
    expect(turnLabel(message)).toBe(label);
  });

  it.each(FIXTURE.plan_lines as { plan: never; lines: string[] }[])('plan lines', ({ plan, lines }) => {
    expect(planLines(plan)).toEqual(lines);
  });

  it.each(FIXTURE.restored_sentences as { written: number; removed: number; sentence: string }[])(
    'restored sentence %j',
    ({ written, removed, sentence }) => {
      expect(restoredSentence(written, removed)).toBe(sentence);
    },
  );

  it.each(FIXTURE.off as { case: string; folder: string; off: boolean }[])('undo is off: $case', ({ folder, off }) => {
    const base = path.dirname(root);
    fs.mkdirSync(path.join(base, 'home', 'project'), { recursive: true });
    fs.mkdirSync(path.join(base, 'elsewhere'), { recursive: true });
    const reason = snapshotsOffReason(path.join(base, folder), path.join(base, 'home'));
    expect(reason).toBe(off ? UNDO_OFF_HERE : null);
  });

  it('the root is off', () => {
    expect(snapshotsOffReason('/')).toBe(UNDO_OFF_HERE);
  });

  it('the words', () => {
    expect({
      nothing_to_undo: UNDO_WORDS.nothingToUndo,
      nothing_changed: UNDO_WORDS.nothingChanged,
      undo_header: UNDO_WORDS.undoHeader,
      confirm: UNDO_WORDS.confirm,
      left_as_is: UNDO_WORDS.leftAsIs,
      no_snapshots: UNDO_WORDS.noSnapshots,
      rewind_title: UNDO_WORDS.rewindTitle,
      rewind_hint: UNDO_WORDS.rewindHint,
      rewind_same: UNDO_WORDS.rewindSame,
      rewind_missing: UNDO_WORDS.rewindMissing('7'),
      rewind_missing_hint: UNDO_WORDS.rewindMissingHint,
      failed: UNDO_WORDS.failed('a.txt', 'Permission denied'),
      snapshot_failed: UNDO_WORDS.snapshotFailed('Permission denied'),
      off_here: UNDO_OFF_HERE,
      rewind_header: rewindHeader('2 min ago', 'before "plan the launch"'),
    }).toEqual(FIXTURE.words);
  });
});

describe('what keeps a restore safe', () => {
  let home: string | undefined;
  let profile: string | undefined;

  beforeEach(() => {
    home = process.env.HOME;
    profile = process.env.WEBAGENTS_PROFILE;
  });

  afterEach(() => {
    if (home === undefined) delete process.env.HOME;
    else process.env.HOME = home;
    if (profile !== undefined) process.env.WEBAGENTS_PROFILE = profile;
  });

  it('a link made since is removed and the folder put back, without touching its target', () => {
    const base = tempDir('wa-cp-link-');
    const agent = path.join(base, 'agent');
    const outside = path.join(base, 'outside');
    const own = path.join(base, 'store');
    fs.mkdirSync(path.join(agent, 'sub'), { recursive: true });
    fs.writeFileSync(path.join(agent, 'sub', 'file.txt'), 'mine\n');
    fs.mkdirSync(outside);
    const m = takeSnapshot(agent, 'before', { store: own });
    fs.rmSync(path.join(agent, 'sub'), { recursive: true });
    fs.symlinkSync(outside, path.join(agent, 'sub'));

    const result = restoreSnapshot(agent, m.id, 'before /undo', { store: own });

    expect(fs.readdirSync(outside)).toEqual([]);
    expect(result.removed).toContain('sub');
    expect(fs.lstatSync(path.join(agent, 'sub')).isSymbolicLink()).toBe(false);
    expect(fs.readFileSync(path.join(agent, 'sub', 'file.txt'), 'utf8')).toBe('mine\n');
  });

  it('nothing is written through a symlinked folder', () => {
    const base = tempDir('wa-cp-through-');
    const agent = path.join(base, 'agent');
    const outside = path.join(base, 'outside');
    const own = path.join(base, 'store');
    fs.mkdirSync(path.join(agent, 'sub'), { recursive: true });
    fs.writeFileSync(path.join(agent, 'sub', 'file.txt'), 'mine\n');
    fs.mkdirSync(outside);
    const m = takeSnapshot(agent, 'before', { store: own });
    fs.rmSync(path.join(agent, 'sub'), { recursive: true });
    fs.symlinkSync(outside, path.join(agent, 'sub'));
    fs.writeFileSync(path.join(own, `${m.id}.json`), JSON.stringify({ ...m, links: { sub: outside } }));

    const result = restoreSnapshot(agent, m.id, 'before /undo', { store: own });

    expect(fs.readdirSync(outside)).toEqual([]);
    expect(result.failed.map((f) => f.path)).toEqual(['sub/file.txt']);
  });

  it('the store is owner-only and outside the folder', () => {
    process.env.HOME = tempDir('wa-cp-home-');
    delete process.env.WEBAGENTS_PROFILE;
    const m = takeSnapshot(root, 'before');
    const own = checkpointsDir(root);
    expect(own.startsWith(path.join(process.env.HOME, '.webagents', 'checkpoints'))).toBe(true);
    expect(fs.existsSync(path.join(root, '.webagents'))).toBe(false);
    expect(fs.statSync(path.join(own, `${m.id}.json`)).mode & 0o777).toBe(0o600);
    for (const name of fs.readdirSync(path.join(own, 'objects'))) {
      expect(fs.statSync(path.join(own, 'objects', name)).mode & 0o777).toBe(0o600);
    }
  });
});
