/**
 * Post-build script: copies runtime assets from src/ into dist/.
 *
 * `tsc` emits .js and .d.ts and nothing else, so any file the SDK READS at
 * runtime rather than imports has to be copied here or it simply is not in the
 * published package.
 *
 * WHY THIS EXISTS (2026-09-23). `src/agents/ROBUTLER.md` is the embedded
 * default agent. `getRobutlerContent()` (`src/agents/index.ts:24`) reads it
 * with a module-relative path, so from `dist/agents/index.js` it looks for
 * `dist/agents/ROBUTLER.md`. Nothing ever put it there. In the published
 * package the read threw ENOENT, `src/cli/app.ts:215-218` caught it and printed
 * a warning, `instructions` stayed undefined, and every `webagents chat` and
 * every `robutler` session ran with NO SYSTEM PROMPT. The failure was silent in
 * the sense that mattered: the REPL started and answered, just as a bare model.
 *
 * DELIBERATELY AN ALLOWLIST, not a glob over `src/**\/*.md`. There are a dozen
 * README.md files under `src/skills/messaging/` that are documentation for
 * humans reading the repo, not runtime assets, and shipping them would grow the
 * tarball for nothing. If you add a file the SDK reads at runtime, add it here
 * too; the check below fails the build if a listed file is missing from src,
 * so a rename cannot quietly reintroduce the original bug.
 */
import { copyFile, mkdir, stat } from 'fs/promises';
import { dirname, join, resolve } from 'path';

const root = resolve(new URL('..', import.meta.url).pathname);

/** Paths relative to src/, copied to the same path under dist/. */
const RUNTIME_ASSETS = [
  'agents/ROBUTLER.md',
];

async function exists(p) {
  try { await stat(p); return true; } catch { return false; }
}

let copied = 0;
const missing = [];

for (const rel of RUNTIME_ASSETS) {
  const from = join(root, 'src', rel);
  const to = join(root, 'dist', rel);
  if (!(await exists(from))) {
    missing.push(rel);
    continue;
  }
  await mkdir(dirname(to), { recursive: true });
  await copyFile(from, to);
  copied++;
}

if (missing.length) {
  console.error(
    `copy-assets: ${missing.length} listed asset(s) missing from src/: ${missing.join(', ')}\n` +
    'Either the file moved (update RUNTIME_ASSETS) or it was deleted (remove it here).',
  );
  process.exit(1);
}

console.log(`copy-assets: copied ${copied} runtime asset(s) into dist/`);
