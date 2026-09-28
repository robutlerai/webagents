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

/** Why no tsx can run the CLI here, or null when one can (`TSX_CLI`). */
export const TSX_PROBLEM: string | null = (() => {
  try {
    usableTsx();
    return null;
  } catch (error) {
    return (error as Error).message;
  }
})();

/** tsx's command-line entry: an installed one that runs on this machine (file comment); '' when none can, see `TSX_PROBLEM`. */
export const TSX_CLI = TSX_PROBLEM === null ? usableTsx() : '';

/** The CLI's source, which the tests run through tsx. */
export const CLI_SOURCE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../src/cli/index.ts');

/**
 * THE SDK'S OWN tsconfig, PASSED TO tsx (2026-09-26). tsx reads the tsconfig
 * nearest the WORKING DIRECTORY, not the file it runs. From the portal root
 * (whose vitest collects these tests too) that was the portal's tsconfig,
 * under which the SDK's decorators compiled to nothing: the spawned CLI
 * built an agent with no tools and no model ("No LLM skill available"), and
 * eight tests failed there for that reason alone. `CLI_ARGS` carries the
 * flag, so the CLI compiles the same wherever the tests run.
 */
export const SDK_TSCONFIG = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../tsconfig.json');

/** `process.execPath`'s arguments to run the CLI: tsx, the SDK's tsconfig, the CLI source. */
export const CLI_ARGS: readonly string[] = [TSX_CLI, '--tsconfig', SDK_TSCONFIG, CLI_SOURCE];

/**
 * Why a test that spawns the CLI and talks MCP to it cannot run here, or
 * null when it can (2026-09-26): a tsx that works on this machine, and the
 * MCP client modules the test drives it with, resolved from the SDK's own
 * dependencies. A missing prerequisite is a stated skip, never a quiet pass
 * and never a failure that says nothing about its cause.
 */
export function cliPrerequisite(needsMcpClient = true): string | null {
  if (TSX_PROBLEM) return `no usable tsx: ${TSX_PROBLEM}`;
  if (!needsMcpClient) return null;
  const require = createRequire(import.meta.url);
  for (const entry of ['client/index.js', 'client/stdio.js', 'client/streamableHttp.js', 'types.js']) {
    try {
      require.resolve(`@modelcontextprotocol/sdk/${entry}`);
    } catch (error) {
      return `@modelcontextprotocol/sdk/${entry} cannot be resolved from the SDK's node_modules: ${(error as Error).message}`;
    }
  }
  return null;
}

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
