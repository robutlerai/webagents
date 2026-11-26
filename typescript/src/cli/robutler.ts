#!/usr/bin/env node
/**
 * Robutler CLI
 *
 * `robutler` is `webagents -a robutler`: the chat with the built-in
 * assistant, or one prompt with `-p`. `robutler app ...` is its own
 * (`app-command.ts`); `login`, `logout`, `whoami` and `cas` are the main CLI's. Since 2026-09-25 it runs through the
 * main CLI (`robutler-args.ts` says why), as the Python package's does.
 *
 * IT PARSES ITS ARGUMENTS NOW (2026-09-23). This used to be six lines that
 * constructed a REPL and ran it, with no argument handling of any kind, so
 * `robutler -p "hello"`, `robutler --model ...` and `robutler --help` were all
 * silently ignored and dropped the user into an interactive session instead.
 * A bin that accepts anything and honours nothing is worse than one that
 * refuses: the user cannot tell their flag did nothing.
 *
 * The model default was also the literal string `'default'`, which is truthy
 * and was passed through as the API `model` field. Omitted now, so the agent's
 * own model applies.
 */

import { USAGE, robutlerCommand } from './robutler-args.js';

/**
 * Parsed here, run by the main CLI (2026-09-25, `robutler-args.ts`): the
 * arguments become `webagents -a robutler ...` and `./index.js` runs them, as
 * it parses `process.argv` when it loads.
 */
/**
 * `robutler app ...`, with what it needs from this machine: the portal the
 * CLI is pointed at, the stored sign-in (or `WEBAGENTS_TOKEN`), the browser,
 * standard input for a secret's value, and the app's draft as a project.
 */
async function runAppCommand(argv: string[]): Promise<number> {
  const [{ runApp }, { resolvePlatformUrl }, { getToken }, { openInBrowser }, { promptSecret }] = await Promise.all([
    import('./app-command.js'),
    import('./config-store.js'),
    import('./credentials.js'),
    import('./browser-login.js'),
    import('./prompt.js'),
  ]);
  const [baseUrl] = resolvePlatformUrl();
  const token = await getToken().catch(() => null);
  return runApp(argv, {
    cwd: process.cwd(),
    env: process.env,
    baseUrl,
    token,
    out: (line) => process.stdout.write(`${line}\n`),
    err: (line) => process.stderr.write(`${line}\n`),
    openBrowser: openInBrowser,
    readSecret: async (prompt) => {
      // Typed with echo off at a terminal; otherwise all of standard input, with nothing printed.
      if (process.stdin.isTTY) return (await promptSecret(prompt, { output: process.stderr })) ?? '';
      const chunks: Buffer[] = [];
      for await (const chunk of process.stdin) chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk)));
      return Buffer.concat(chunks).toString('utf8');
    },
    openDraft: async (appId) => {
      const [{ CasFs }, { DirChunkCache }] = await Promise.all([import('../casfs/index.js'), import('../casfs/local.js')]);
      return CasFs.platform({ baseUrl, token, cache: new DirChunkCache() }).open(appId);
    },
  });
}

async function main(): Promise<void> {
  const command = robutlerCommand(process.argv.slice(2));
  if (command.kind === 'help') {
    console.log(USAGE);
    process.exit(0);
  }
  if (command.kind === 'error') {
    console.error(`${command.message}\n`);
    console.log(USAGE);
    process.exit(2);
  }
  if (command.kind === 'app') {
    process.exit(await runAppCommand(command.argv));
  }
  process.argv = [process.argv[0], process.argv[1], ...command.argv];
  await import('./index.js');
}

main().catch((error) => {
  console.error('Error:', (error as Error).message);
  process.exit(1);
});
