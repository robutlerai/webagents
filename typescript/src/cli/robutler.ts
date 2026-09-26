#!/usr/bin/env node
/**
 * Robutler CLI
 *
 * `robutler` is `webagents -a robutler`: the chat with the built-in
 * assistant, or one prompt with `-p`. Since 2026-09-25 it runs through the
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
  process.argv = [process.argv[0], process.argv[1], ...command.argv];
  await import('./index.js');
}

main().catch((error) => {
  console.error('Error:', (error as Error).message);
  process.exit(1);
});
