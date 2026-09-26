/**
 * Snapshots of an agent's folder, for `/undo` and `/rewind` (2026-09-25).
 *
 * WHY. An agent with `filesystem` or `shell` edits the folder it works in, and
 * a person had no way to take an edit back. Both SDKs shipped a `checkpoint`
 * skill; neither worked: the Python one could not be reached from the chat,
 * and its restore brought deleted files back and wrote through symlinks; the
 * TypeScript one could not be named in an agent file, and handed every caller
 * of a served agent unscoped tools that restored any readable folder over the
 * working one (S-249). Both are gone. The chat takes the snapshot itself,
 * before each message to an agent that can change files, and only the person
 * at the terminal restores one.
 *
 * WHERE. Under the profile, never in the folder, so file tools cannot reach
 * them and a repository cannot commit them:
 * `~/.webagents[-profile]/checkpoints/<folder>/` (the folder named as
 * `sessions.ts` names it), holding `objects/<sha256>` (each content once) and
 * one `<id>.json` manifest per snapshot. The last `KEEP` are kept.
 *
 * WHAT. Every regular file under the folder, by path, less `.git`, `.webagents`,
 * `node_modules`, `.venv`, `venv` and `__pycache__` at any depth. A symlink is
 * recorded as a link and never followed. A file over `MAX_FILE_BYTES`, or past
 * `MAX_FILES` / `MAX_TOTAL_BYTES`, is listed as skipped, and a restore leaves
 * it alone. Only files whose size or modification time changed since the last
 * snapshot are read again, except one changed within `RACY_MS` of the scan:
 * that one is kept without its time, so the next scan reads it again (two
 * writes of the same size inside one tick of the file system's clock look the
 * same, and HFS+ keeps whole seconds). A snapshot identical to the last one is
 * not kept twice.
 *
 * RESTORE puts every file back as the snapshot has it, removes files the
 * snapshot does not have (made after it), and first takes a snapshot of how
 * things are, so a restore can itself be undone. A file is written to a
 * temporary name and renamed into place; nothing is written through a
 * symlink, or outside the folder.
 *
 * ONE FORMAT IN BOTH SDKS: `python/webagents/cli/checkpoints.py` reads and
 * writes the same files, and `python/tests/fixtures/checkpoints/checkpoints.json`
 * holds the cases both run.
 */

import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { createHash, randomBytes } from 'node:crypto';

import { globalDir, profileName } from './config-store';
import { slugFor } from './sessions';

export const EXCLUDED_DIRS: readonly string[] = ['.git', '.webagents', 'node_modules', '.venv', 'venv', '__pycache__'];
export const MAX_FILE_BYTES = 10 * 1024 * 1024;
export const MAX_FILES = 20_000;
export const MAX_TOTAL_BYTES = 200 * 1024 * 1024;
/** Snapshots kept per folder. */
export const KEEP = 50;
/** A file changed this close to a scan is read again by the next one. */
export const RACY_MS = 2000;

export const SKIPPED_LARGE = 'larger than 10 MB';
export const SKIPPED_FULL = 'past what a snapshot keeps (20000 files, 200 MB)';

const ID_RE = /^cp_\d{8}T\d{6}Z_[0-9a-f]{8}$/;

export interface FileEntry {
  sha256: string;
  size: number;
  mode: number;
  mtime_ms: number;
}

export interface Manifest {
  version: 1;
  id: string;
  created_at: string;
  label: string;
  files: Record<string, FileEntry>;
  links: Record<string, string>;
  skipped: Record<string, string>;
}

/** Where `folder`'s snapshots are kept (file comment). */
export function checkpointsDir(folder: string, profile?: string): string {
  let real: string;
  try {
    real = fs.realpathSync(folder);
  } catch {
    real = path.resolve(folder);
  }
  return path.join(globalDir(profileName(profile)), 'checkpoints', slugFor(real));
}

/** Why snapshots are off for `folder`, or null when they are on: never for the home folder or one above it. */
export const UNDO_OFF_HERE = '/undo is off in your home folder and above: start the chat in a project folder to use it.';

export function snapshotsOffReason(folder: string, home: string = os.homedir()): string | null {
  const real = (p: string) => {
    try {
      return fs.realpathSync(p);
    } catch {
      return path.resolve(p);
    }
  };
  const here = real(folder);
  const mine = real(home);
  if (here === path.parse(here).root) return UNDO_OFF_HERE;
  if (mine === here || mine.startsWith(here.endsWith(path.sep) ? here : here + path.sep)) return UNDO_OFF_HERE;
  return null;
}

/** `cp_<UTC time>_<8 hex>`. */
export function newCheckpointId(now: Date = new Date()): string {
  const stamp = now.toISOString().replace(/[-:]/g, '').replace(/\.\d+Z$/, 'Z');
  return `cp_${stamp}_${randomBytes(4).toString('hex')}`;
}

export function isCheckpointId(id: string): boolean {
  return ID_RE.test(id);
}

function writePrivate(file: string, data: string | Buffer): void {
  const temp = `${file}.${process.pid}.${randomBytes(4).toString('hex')}.tmp`;
  fs.writeFileSync(temp, data, { mode: 0o600 });
  fs.renameSync(temp, file);
}

/** The folder's files, links and skipped paths (file comment, WHAT), re-reading only what changed since `previous`. */
export function scanFolder(
  folder: string,
  store: string,
  previous?: Manifest,
): Pick<Manifest, 'files' | 'links' | 'skipped'> {
  const files: Record<string, FileEntry> = {};
  const links: Record<string, string> = {};
  const skipped: Record<string, string> = {};
  let count = 0;
  let total = 0;
  const objects = path.join(store, 'objects');
  const started = Date.now();

  const walk = (dir: string, prefix: string): void => {
    let names: string[];
    try {
      names = fs.readdirSync(dir).sort();
    } catch {
      return;
    }
    for (const name of names) {
      const full = path.join(dir, name);
      const rel = prefix ? `${prefix}/${name}` : name;
      let stat: fs.Stats;
      try {
        stat = fs.lstatSync(full);
      } catch {
        continue;
      }
      if (stat.isSymbolicLink()) {
        try {
          links[rel] = fs.readlinkSync(full);
        } catch {
          // An unreadable link is not recorded.
        }
      } else if (stat.isDirectory()) {
        if (!EXCLUDED_DIRS.includes(name)) walk(full, rel);
      } else if (stat.isFile()) {
        if (stat.size > MAX_FILE_BYTES) {
          skipped[rel] = SKIPPED_LARGE;
          continue;
        }
        if (count + 1 > MAX_FILES || total + stat.size > MAX_TOTAL_BYTES) {
          skipped[rel] = SKIPPED_FULL;
          continue;
        }
        const mtime = Math.floor(stat.mtimeMs);
        const before = previous?.files[rel];
        let sha256: string;
        if (before && before.mtime_ms && before.size === stat.size && before.mtime_ms === mtime && fs.existsSync(path.join(objects, before.sha256))) {
          sha256 = before.sha256;
        } else {
          let data: Buffer;
          try {
            data = fs.readFileSync(full);
          } catch {
            continue;
          }
          sha256 = createHash('sha256').update(data).digest('hex');
          const object = path.join(objects, sha256);
          if (!fs.existsSync(object)) {
            fs.mkdirSync(objects, { recursive: true, mode: 0o700 });
            writePrivate(object, data);
          }
        }
        files[rel] = { sha256, size: stat.size, mode: stat.mode & 0o777, mtime_ms: mtime < started - RACY_MS ? mtime : 0 };
        count += 1;
        total += stat.size;
      }
    }
  };
  walk(folder, '');
  return { files, links, skipped };
}

function sameState(a: Pick<Manifest, 'files' | 'links' | 'skipped'>, b: Pick<Manifest, 'files' | 'links' | 'skipped'>): boolean {
  const content = (m: Pick<Manifest, 'files' | 'links' | 'skipped'>) =>
    JSON.stringify({
      files: Object.fromEntries(Object.entries(m.files).sort().map(([k, v]) => [k, [v.sha256, v.mode]])),
      links: Object.fromEntries(Object.entries(m.links).sort()),
      skipped: Object.fromEntries(Object.entries(m.skipped).sort()),
    });
  return content(a) === content(b);
}

/** Every snapshot of the folder, newest first. */
export function listCheckpoints(store: string): Manifest[] {
  let names: string[];
  try {
    names = fs.readdirSync(store).filter((n) => n.endsWith('.json') && isCheckpointId(n.slice(0, -'.json'.length)));
  } catch {
    return [];
  }
  const out: Manifest[] = [];
  for (const name of names) {
    try {
      const data = JSON.parse(fs.readFileSync(path.join(store, name), 'utf-8')) as Manifest;
      if (data && data.version === 1 && isCheckpointId(data.id)) out.push(data);
    } catch {
      // A manifest that cannot be read is not a snapshot to offer.
    }
  }
  return out.sort((a, b) => (a.created_at < b.created_at ? 1 : a.created_at > b.created_at ? -1 : b.id.localeCompare(a.id)));
}

/** Keep the newest `KEEP` manifests, and the objects they use. */
function prune(store: string): void {
  const all = listCheckpoints(store);
  for (const old of all.slice(KEEP)) {
    try {
      fs.unlinkSync(path.join(store, `${old.id}.json`));
    } catch {
      // Already gone.
    }
  }
  const used = new Set(all.slice(0, KEEP).flatMap((m) => Object.values(m.files).map((f) => f.sha256)));
  const objects = path.join(store, 'objects');
  let names: string[] = [];
  try {
    names = fs.readdirSync(objects);
  } catch {
    return;
  }
  for (const name of names) {
    if (!used.has(name)) {
      try {
        fs.unlinkSync(path.join(objects, name));
      } catch {
        // Already gone.
      }
    }
  }
}

/**
 * Take a snapshot of `folder`, labelled `label`. Answers the snapshot, or the
 * newest one when nothing changed since it (file comment).
 */
export function takeSnapshot(folder: string, label: string, options: { store?: string; now?: Date } = {}): Manifest {
  const store = options.store ?? checkpointsDir(folder);
  fs.mkdirSync(store, { recursive: true, mode: 0o700 });
  const latest = listCheckpoints(store)[0];
  const state = scanFolder(folder, store, latest);
  if (latest && sameState(latest, state)) return latest;
  const now = options.now ?? new Date();
  const manifest: Manifest = { version: 1, id: newCheckpointId(now), created_at: now.toISOString(), label, ...state };
  writePrivate(path.join(store, `${manifest.id}.json`), `${JSON.stringify(manifest, null, 2)}\n`);
  prune(store);
  return manifest;
}

/** What a restore would do: files it writes back, files it removes, files it leaves alone. */
export interface RestorePlan {
  write: string[];
  remove: string[];
  link: string[];
  keep: string[];
}

/** The difference between how the folder is (`now`) and a snapshot (`target`), as a restore would settle it. */
export function planRestore(target: Manifest, now: Pick<Manifest, 'files' | 'links' | 'skipped'>): RestorePlan {
  const write = Object.keys(target.files)
    .filter((rel) => now.files[rel]?.sha256 !== target.files[rel].sha256 || now.files[rel]?.mode !== target.files[rel].mode || rel in now.links)
    .sort();
  const link = Object.keys(target.links)
    .filter((rel) => now.links[rel] !== target.links[rel])
    .sort();
  const remove = [...Object.keys(now.files), ...Object.keys(now.links)]
    .filter((rel) => !(rel in target.files) && !(rel in target.links) && !(rel in target.skipped))
    .sort();
  const keep = Object.keys(target.skipped).sort();
  return { write, remove, link, keep };
}

/** Whether `rel` is a plain relative path inside the folder. */
function safeRelative(rel: string): boolean {
  if (!rel || rel.startsWith('/') || rel.includes('\\')) return false;
  return rel.split('/').every((part) => part !== '' && part !== '.' && part !== '..');
}

/** Whether writing at `rel` stays inside `folder`: no part of the way there is a symlink. */
function insideFolder(folder: string, rel: string): boolean {
  const parts = rel.split('/');
  let at = folder;
  for (const part of parts.slice(0, -1)) {
    at = path.join(at, part);
    try {
      const stat = fs.lstatSync(at);
      if (stat.isSymbolicLink() || !stat.isDirectory()) return false;
    } catch {
      return true; // Not there yet: it will be made as a directory.
    }
  }
  return true;
}

export interface RestoreResult {
  written: string[];
  removed: string[];
  kept: string[];
  /** Paths it could not put back, and why. */
  failed: { path: string; reason: string }[];
  /** The snapshot of how things were, taken first. */
  before: Manifest;
}

/**
 * Put `folder` back as snapshot `id` has it (file comment, RESTORE), after
 * taking a snapshot labelled `beforeLabel`.
 */
export function restoreSnapshot(folder: string, id: string, beforeLabel: string, options: { store?: string } = {}): RestoreResult {
  if (!isCheckpointId(id)) throw new Error(`There is no snapshot ${id}.`);
  const store = options.store ?? checkpointsDir(folder);
  const target = JSON.parse(fs.readFileSync(path.join(store, `${id}.json`), 'utf-8')) as Manifest;
  const before = takeSnapshot(folder, beforeLabel, { store });
  const plan = planRestore(target, before);
  const failed: { path: string; reason: string }[] = [];
  const written: string[] = [];
  const removed: string[] = [];

  for (const rel of plan.remove) {
    if (!safeRelative(rel) || !insideFolder(folder, rel)) continue;
    try {
      fs.unlinkSync(path.join(folder, ...rel.split('/')));
      removed.push(rel);
    } catch (error) {
      failed.push({ path: rel, reason: (error as Error).message });
    }
  }
  for (const rel of plan.write) {
    if (!safeRelative(rel) || !insideFolder(folder, rel)) {
      failed.push({ path: rel, reason: 'the way to it is not a folder of its own' });
      continue;
    }
    const file = path.join(folder, ...rel.split('/'));
    const entry = target.files[rel];
    try {
      const data = fs.readFileSync(path.join(store, 'objects', entry.sha256));
      fs.mkdirSync(path.dirname(file), { recursive: true });
      const temp = `${file}.${process.pid}.${randomBytes(4).toString('hex')}.tmp`;
      fs.writeFileSync(temp, data, { mode: entry.mode });
      fs.chmodSync(temp, entry.mode);
      // A link where the file belongs is removed, never written through.
      try {
        if (fs.lstatSync(file).isSymbolicLink()) fs.unlinkSync(file);
      } catch {
        // Nothing there.
      }
      fs.renameSync(temp, file);
      written.push(rel);
    } catch (error) {
      failed.push({ path: rel, reason: (error as Error).message });
    }
  }
  for (const rel of plan.link) {
    if (!safeRelative(rel) || !insideFolder(folder, rel)) {
      failed.push({ path: rel, reason: 'the way to it is not a folder of its own' });
      continue;
    }
    const file = path.join(folder, ...rel.split('/'));
    try {
      fs.mkdirSync(path.dirname(file), { recursive: true });
      try {
        fs.unlinkSync(file);
      } catch {
        // Nothing there.
      }
      fs.symlinkSync(target.links[rel], file);
      written.push(rel);
    } catch (error) {
      failed.push({ path: rel, reason: (error as Error).message });
    }
  }
  return { written: written.sort(), removed, kept: plan.keep, failed, before };
}

// ============================================================================
// The words (`python/tests/fixtures/checkpoints/checkpoints.json`)
// ============================================================================

export const UNDO_WORDS = {
  nothingToUndo: 'Nothing to undo in this conversation.',
  nothingChanged: 'Nothing to undo: the folder is as it was before your last message.',
  undoHeader: "Undo your last message's changes to this folder:",
  confirm: 'Put these back? [y/N] ',
  leftAsIs: 'Left as it is.',
  noSnapshots: 'No snapshots of this folder yet.',
  rewindTitle: 'Snapshots of this folder',
  rewindHint: 'Put the folder back with /rewind <number>.',
  rewindSame: 'Nothing to put back: the folder is as that snapshot has it.',
  rewindMissing: (pick: string) => `There is no snapshot ${pick}.`,
  rewindMissingHint: 'Type /rewind to see the list.',
  failed: (rel: string, reason: string) => `Could not put back ${rel}: ${reason}.`,
  snapshotFailed: (reason: string) => `Snapshots are off for this conversation: ${reason}.`,
} as const;

/** What a snapshot taken before a message is called: `before "<the message, cut to 50>"`. */
export function turnLabel(message: string): string {
  const line = message.replace(/\s+/g, ' ').trim();
  return `before "${line.length > 50 ? `${line.slice(0, 49)}…` : line}"`;
}

/** The header of a `/rewind <n>` confirmation. */
export function rewindHeader(when: string, label: string): string {
  return `Put the folder back as it was ${when} (${label}):`;
}

/** The lines a confirmation lists: what goes back, what goes, at most `max`. */
export function planLines(plan: RestorePlan, max = 12): string[] {
  const rows = [
    ...[...plan.write, ...plan.link].sort().map((rel) => `  restore  ${rel}`),
    ...plan.remove.map((rel) => `  remove   ${rel}`),
  ];
  return rows.length > max ? [...rows.slice(0, max), `  and ${rows.length - max} more`] : rows;
}

/** `Put back 2 files and removed 1 file made since.` */
export function restoredSentence(written: number, removed: number): string {
  const files = (n: number) => `${n} ${n === 1 ? 'file' : 'files'}`;
  const parts = [
    ...(written ? [`put back ${files(written)}`] : []),
    ...(removed ? [`removed ${files(removed)} made since`] : []),
  ];
  const text = parts.join(' and ');
  return `${text.charAt(0).toUpperCase()}${text.slice(1)}.`;
}

/** Whether a plan changes anything. */
export function planChangesAnything(plan: RestorePlan): boolean {
  return plan.write.length + plan.remove.length + plan.link.length > 0;
}
