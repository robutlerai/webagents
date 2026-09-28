/**
 * The two sets of names the file tools guard (2026-09-27, the agent-secrets
 * lane of the gap-closure build). The Python twin is
 * `agents/skills/local/filesystem/agent_secrets_guard.py`; the shared fixture
 * `python/tests/fixtures/agent_secrets/file_tools.json` pins the names, the
 * sentences and the cases, and both suites read it.
 *
 * WHY. The file tools allowed everything under the agent's folder, and the
 * folder is where the secrets are: the CLI loads provider keys from its
 * `.env` (`cli/config-store.ts`), and `.webagents/` holds the schedule state
 * and, for a day, a served agent's private signing key. The default agent
 * pairs these tools with `rest_request`, which posts to any public URL, so a
 * prompt injection in the owner's own session could read a key and send it
 * out (S-312). The same tools could also rewrite the agent's control files:
 * the real-model pass wrote `AGENT.md` to `preset: unrestricted`, planted an
 * `AGENT-evil.md`, `WEBAGENTS.md`, `mcp.json`, another skill's `SKILL.md`,
 * `.env` and a `.git/hooks` file through `write_file`, each a persistent
 * change the daemon reloads on the write itself (S-314). S-283 had
 * write-denied exactly that set for confined shell commands; the file tools
 * were a second path to the same files, with nothing in the way but the
 * model's own caution.
 *
 * TWO SETS, TWO ANSWERS.
 *
 *   - SECRETS (`.env`, `.env.*`, anything under `.webagents/`): never read,
 *     written, searched or moved by a file tool, for any caller, with one
 *     sentence that says why. A listing may still show the names.
 *   - CONTROL FILES: the sandbox's `ESCALATION_DENY` and `AGENT_FILE_PATTERNS`
 *     (imported from `sandbox/policy`, never copied, so the two lists cannot
 *     drift) plus every `.env*` name and the running agent's own file. A
 *     write to one of these happens only with the owner's yes in the
 *     interactive chat, which shows the diff; `serve()`, the daemon, `-p` and
 *     any caller who is not the owner get the refusal sentence. Reads are
 *     fine: the built-in agent's job of writing agent files keeps working,
 *     behind the prompt.
 *
 * HOW A NAME MATCHES. On any component of the path, not only at the folder's
 * root: the daemon serves `sub/AGENT-helper.md` as readily as `AGENT.md`, a
 * nested repository's `.git/hooks` runs on the owner's next git command all
 * the same, and a whitelisted `~` reaches `~/.bashrc`. Symbolic links are
 * resolved first (on the deepest ancestor that exists, so a file about to be
 * created is judged by where it would land), and both the path as written
 * and the real path are checked, so a link named `config.txt` that points at
 * `.env` is refused and a link named `.env` that points elsewhere still asks.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';

import { AGENT_FILE_PATTERNS, ESCALATION_DENY, matchesAgentFilePattern } from '../../sandbox/policy';

/** The secrets set (the fixture's `secrets`): exact names, name prefixes and folder names. */
export const SECRET_NAMES: readonly string[] = ['.env'];
export const SECRET_PREFIXES: readonly string[] = ['.env.'];
export const SECRET_FOLDERS: readonly string[] = ['.webagents'];

/** Every `.env*` name is a control file too (the fixture's `control.prefixes`). */
export const CONTROL_PREFIXES: readonly string[] = ['.env'];

const SECRET_REFUSAL =
  'Refused: {path} is where the agent\'s secrets live (.env files and the .webagents folder), and the file tools never read, write, search or move them.';
const CONTROL_REFUSAL =
  'Refused: {path} is one of the agent\'s control files (its agent files, WEBAGENTS.md, mcp.json, its skills, git hooks and .env), which the file tools change only when the owner says yes in the interactive chat.';
/** What the chat prints above the diff, and asks below it (the fixture's `control.header` and `control.question`). */
export const CONTROL_HEADER = 'The agent wants to change {path}, one of its control files:';
export const CONTROL_QUESTION = 'Make this change? [y/N] ';
const CONTROL_DECLINED = 'The owner declined the change to {path}; nothing was written.';

/** The chat's yes/no for a control-file write: the file as the model named it, and the unified diff to show. */
export type ConfirmControlWrite = (file: string, diff: string) => Promise<boolean>;

export function secretRefusal(shown: string): string {
  return SECRET_REFUSAL.replace('{path}', shown);
}

export function controlRefusal(shown: string): string {
  return CONTROL_REFUSAL.replace('{path}', shown);
}

export function controlDeclined(shown: string): string {
  return CONTROL_DECLINED.replace('{path}', shown);
}

/** The control entries, as the sandbox denies them (`ESCALATION_DENY`), for the fixture to pin. */
export function controlEntries(): string[] {
  return [...ESCALATION_DENY];
}

/** The agent-file name patterns (`AGENT_FILE_PATTERNS`), for the fixture to pin. */
export function controlPatterns(): string[] {
  return [...AGENT_FILE_PATTERNS];
}

/**
 * The real path of `target`: symbolic links resolved on the deepest ancestor
 * that exists, the rest appended as written. Never throws.
 */
export function realPathOf(target: string): string {
  const absolute = path.resolve(target);
  try {
    return fs.realpathSync.native(absolute);
  } catch {
    // Absent, or unreadable: judge it by where it would land.
  }
  const parent = path.dirname(absolute);
  if (parent === absolute) return absolute;
  return path.join(realPathOf(parent), path.basename(absolute));
}

function segmentsOf(target: string): string[] {
  return path.resolve(target).split(/[\\/]+/).filter((segment) => segment.length > 0);
}

/** Whether `entry` (one or more segments, `/`-joined) appears as consecutive segments anywhere in `segments`. */
function containsEntry(segments: readonly string[], entry: string): boolean {
  const wanted = entry.split('/').filter((segment) => segment.length > 0);
  if (wanted.length === 0) return false;
  for (let start = 0; start + wanted.length <= segments.length; start++) {
    let matched = true;
    for (let offset = 0; offset < wanted.length; offset++) {
      if (segments[start + offset] !== wanted[offset]) {
        matched = false;
        break;
      }
    }
    if (matched) return true;
  }
  return false;
}

function segmentsAreSecret(segments: readonly string[]): boolean {
  return segments.some(
    (segment) =>
      SECRET_NAMES.includes(segment) ||
      SECRET_FOLDERS.includes(segment) ||
      SECRET_PREFIXES.some((prefix) => segment.startsWith(prefix)),
  );
}

function segmentsAreControl(segments: readonly string[]): boolean {
  if (ESCALATION_DENY.some((entry) => containsEntry(segments, entry))) return true;
  return segments.some(
    (segment) => matchesAgentFilePattern(segment) || CONTROL_PREFIXES.some((prefix) => segment.startsWith(prefix)),
  );
}

/** Whether `target`, as written or where it really points, is in the secrets set (S-312). */
export function isSecretPath(target: string): boolean {
  return segmentsAreSecret(segmentsOf(target)) || segmentsAreSecret(segmentsOf(realPathOf(target)));
}

/**
 * Whether `target`, as written or where it really points, is a control file
 * (S-314): a sandbox deny entry, an agent-file name, a `.env*` name, or the
 * running agent's own file when the loader named it.
 */
export function isControlPath(target: string, agentFile?: string): boolean {
  const real = realPathOf(target);
  if (segmentsAreControl(segmentsOf(target)) || segmentsAreControl(segmentsOf(real))) return true;
  if (agentFile) {
    const own = realPathOf(agentFile);
    if (real === own || path.resolve(target) === path.resolve(agentFile)) return true;
  }
  return false;
}

/** More lines than this on a side, and the diff is a count rather than a table the chat could not show anyway. */
const DIFF_LINE_LIMIT = 4000;

/**
 * A unified diff of `before` to `after` for the chat to show, three lines of
 * context, headers naming `shown`. A new file is all additions. Very large
 * files get a summary instead of a diff.
 */
export function unifiedDiff(before: string, after: string, shown: string): string {
  const a = before === '' ? [] : before.split('\n');
  const b = after === '' ? [] : after.split('\n');
  const header = [`--- ${shown}`, `+++ ${shown}`];
  if (a.length > DIFF_LINE_LIMIT || b.length > DIFF_LINE_LIMIT) {
    return [...header, `@@ ${a.length} lines replaced by ${b.length} lines (too large to show) @@`].join('\n');
  }
  // Longest common subsequence by dynamic programming: the files a control
  // file can be are small, and this needs no dependency.
  const n = a.length;
  const m = b.length;
  const width = m + 1;
  const table = new Uint32Array((n + 1) * width);
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      table[i * width + j] =
        a[i] === b[j] ? table[(i + 1) * width + j + 1] + 1 : Math.max(table[(i + 1) * width + j], table[i * width + j + 1]);
    }
  }
  type Op = { kind: ' ' | '-' | '+'; text: string; ai: number; bi: number };
  const ops: Op[] = [];
  let i = 0;
  let j = 0;
  // A removal goes before the addition that replaces it (2026-09-28, the e2e
  // pass): ties went to `+` first, so a changed line read `+new` then `-old`,
  // the reverse of every diff a person has seen, the Python chat's included.
  while (i < n || j < m) {
    if (i < n && j < m && a[i] === b[j]) {
      ops.push({ kind: ' ', text: a[i], ai: i, bi: j });
      i++;
      j++;
    } else if (i < n && (j >= m || table[(i + 1) * width + j] >= table[i * width + j + 1])) {
      ops.push({ kind: '-', text: a[i], ai: i, bi: j });
      i++;
    } else {
      ops.push({ kind: '+', text: b[j], ai: i, bi: j });
      j++;
    }
  }
  const context = 3;
  const lines = [...header];
  let index = 0;
  while (index < ops.length) {
    if (ops[index].kind === ' ') {
      index++;
      continue;
    }
    const start = Math.max(0, index - context);
    let end = index;
    let quiet = 0;
    while (end < ops.length && quiet < context * 2) {
      if (ops[end].kind === ' ') quiet++;
      else quiet = 0;
      end++;
    }
    // Trim trailing context beyond `context` lines.
    while (end > index && ops[end - 1].kind === ' ' && countTrailingContext(ops, index, end) > context) end--;
    const hunk = ops.slice(start, end);
    const removed = hunk.filter((op) => op.kind !== '+').length;
    const added = hunk.filter((op) => op.kind !== '-').length;
    const aStart = hunk[0].ai + (removed === 0 ? 0 : 1);
    const bStart = hunk[0].bi + (added === 0 ? 0 : 1);
    lines.push(`@@ -${aStart},${removed} +${bStart},${added} @@`);
    for (const op of hunk) lines.push(`${op.kind}${op.text}`);
    index = end;
  }
  return lines.join('\n');
}

function countTrailingContext(ops: readonly { kind: string }[], from: number, to: number): number {
  let count = 0;
  for (let k = to - 1; k >= from && ops[k].kind === ' '; k--) count++;
  return count;
}
