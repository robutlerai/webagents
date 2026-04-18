/**
 * Dev watcher: `tsc --watch` + post-build `fix-extensions.mjs` on every
 * successful incremental compile.
 *
 * Why this exists: tsc has no post-emit hook, and `fix-extensions.mjs`
 * (which adds `.js` to relative imports for Node ESM) only runs on the
 * one-shot `build` script. Without re-running it, every incremental
 * dist update produced by `tsc --watch` is missing extensions and crashes
 * on `import`. This wrapper spawns tsc, scans its stdout for the
 * "Found N errors. Watching for file changes." completion line, and runs
 * fix-extensions after each one.
 *
 * Used by:
 *   - `pnpm dev` in this package (local dev)
 *   - `infrastructure/applications/portal/local-dev/kustomization.yaml`
 *     (portal `--dev` pod startup)
 *
 * Note: also needed by the portal's tsx watch (which now watches
 * `dist/**\/*.js`) — restarts only fire after fix-extensions has finished
 * patching the new files, eliminating the stale-dist race.
 */
import { spawn } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, resolve } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const fixExtensionsPath = resolve(__dirname, 'fix-extensions.mjs');

// tsc prints this line after every successful incremental compile (with
// or without errors). We trigger fix-extensions on either — even with type
// errors, tsc still emits .js, so the dist needs patching.
const COMPLETE_RE = /Found \d+ error(?:s)?\. Watching for file changes\./;

let fixInFlight = false;
let fixQueued = false;

async function runFixExtensions() {
  if (fixInFlight) {
    fixQueued = true;
    return;
  }
  fixInFlight = true;
  try {
    const fixUrl = pathToFileURL(fixExtensionsPath).href;
    // Bust the import cache by appending a query so each run re-evaluates.
    await import(`${fixUrl}?t=${Date.now()}`);
  } catch (err) {
    process.stderr.write(`[dev-watch] fix-extensions failed: ${err?.message ?? err}\n`);
  } finally {
    fixInFlight = false;
    if (fixQueued) {
      fixQueued = false;
      runFixExtensions();
    }
  }
}

const tsc = spawn(
  process.execPath,
  [resolve(__dirname, '..', 'node_modules', 'typescript', 'bin', 'tsc'), '--watch', '--preserveWatchOutput'],
  { stdio: ['inherit', 'pipe', 'inherit'] },
);

let buffer = '';
tsc.stdout.on('data', (chunk) => {
  process.stdout.write(chunk);
  buffer += chunk.toString();
  // Keep the tail bounded; we only care about recent lines.
  if (buffer.length > 4096) buffer = buffer.slice(-2048);
  if (COMPLETE_RE.test(buffer)) {
    buffer = '';
    runFixExtensions();
  }
});

tsc.on('exit', (code) => {
  process.exit(code ?? 0);
});

for (const sig of ['SIGINT', 'SIGTERM']) {
  process.on(sig, () => {
    tsc.kill(sig);
  });
}
