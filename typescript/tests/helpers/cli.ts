/**
 * Two helpers for tests that touch the disk or start the CLI (2026-09-25):
 * temporary folders that are removed again, and the CLI started with this
 * repo's own tsx.
 *
 * WHY. Six CLI tests spawned `npx tsx src/cli/index.ts` with HOME pointed at a
 * new temporary folder, and nineteen test files never removed the folders they
 * made. With a fresh HOME npx has no cache, so every spawn downloaded tsx and
 * esbuild again, about 86 MB into that HOME, and left it there. By 2026-09-25
 * the system temp folder held about 15 GB of them, and the disk filled up in
 * the middle of a portal test run (every failure was ENOSPC, vitest's own
 * cache). Two rules now:
 *
 *  - `tempDirs()` gives a test file a maker of folders that are all removed
 *    after that file's tests, however they ended;
 *  - the CLI runs as `node <tsx's cli> src/cli/index.ts ...` (`TSX_CLI`): a tsx
 *    that is already installed, so it never depends on the child's HOME, the
 *    network or npx.
 *
 * WHICH TSX. The first one, looking from the working directory and then from
 * here, whose esbuild has a binary for this machine. In the portal checkout
 * `webagents/typescript/node_modules` held an esbuild without the macOS binary
 * (2026-09-25), so its tsx could not start; `npx` hid that by downloading a
 * fresh tsx every time. The portal's own tsx, found from the working directory
 * when the portal runs these tests, is the one that works there.
 */

import { afterAll } from 'vitest';
import * as fs from 'node:fs';
import { createRequire } from 'node:module';
import * as os from 'node:os';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

function usableTsx(): string {
  const tried: string[] = [];
  for (const from of [path.join(process.cwd(), 'package.json'), fileURLToPath(import.meta.url)]) {
    let cli: string;
    try {
      cli = createRequire(from).resolve('tsx/cli');
    } catch {
      continue;
    }
    tried.push(cli);
    try {
      const esbuild = createRequire(cli).resolve('esbuild');
      createRequire(esbuild).resolve(`@esbuild/${process.platform}-${process.arch}/package.json`);
      return cli;
    } catch {
      // This install's esbuild has no binary for this machine; try the next.
    }
  }
  throw new Error(
    `No installed tsx whose esbuild runs on ${process.platform}-${process.arch} (tried: ${tried.join(', ') || 'none found'}).`,
  );
}

/** tsx's command-line entry: an installed one that runs on this machine (file comment). */
export const TSX_CLI = usableTsx();

/** The CLI's source, which the tests run through tsx. */
export const CLI_SOURCE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../src/cli/index.ts');

/**
 * A maker of temporary folders (`tempDir('wa-project-')`), all removed after
 * the calling file's tests. Call it once at the top of the test file, or
 * inside a `describe` to scope the removal to that block.
 */
export function tempDirs(): (prefix: string) => string {
  const made: string[] = [];
  afterAll(() => {
    for (const dir of made.splice(0)) fs.rmSync(dir, { recursive: true, force: true });
  });
  return (prefix: string) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
    made.push(dir);
    return dir;
  };
}
