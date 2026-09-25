#!/usr/bin/env node
/**
 * Robutler CLI
 *
 * Alias for `webagents connect` - starts an interactive session with the
 * default robutler agent.
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

import { InteractiveREPL } from './app';
import { setRequestErrorDetail } from '../skills/llm/request';

const USAGE = `robutler - interactive session with the default robutler agent

Usage:
  robutler [options]

Options:
  -p, --prompt <text>   Send one prompt, print the reply, exit
  -m, --model <model>   Model to use, as provider/model
  -a, --agent <name>    Agent name to load (AGENT-<name>.md)
      --json            Print the reply as JSON (with -p)
  -h, --help            Show this help

For the full command surface, use \`webagents\`.
`;

interface Args {
  prompt?: string;
  model?: string;
  agent?: string;
  json: boolean;
}

type ParseResult =
  | { ok: true; args: Args }
  /** `help` exits 0; a bad option exits 2. Two different things. */
  | { ok: false; reason: 'help' | 'bad-option' };

/** Minimal parser: this bin has one job and does not need commander. */
function parseArgs(argv: string[]): ParseResult {
  const args: Args = { json: false };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => argv[++i];
    switch (arg) {
      case '-h':
      case '--help':
        return { ok: false, reason: 'help' };
      case '-p':
      case '--prompt':
        args.prompt = next();
        break;
      case '-m':
      case '--model':
        args.model = next();
        break;
      case '-a':
      case '--agent':
        args.agent = next();
        break;
      case '--json':
        args.json = true;
        break;
      default:
        // Refuse rather than ignore.
        console.error(`Unknown option: ${arg}\n`);
        return { ok: false, reason: 'bad-option' };
    }
  }
  return { ok: true, args };
}

async function main() {
  const parsed = parseArgs(process.argv.slice(2));
  if (!parsed.ok) {
    console.log(USAGE);
    process.exit(parsed.reason === 'help' ? 0 : 2);
  }
  const args = parsed.args;

  const config: { model?: string; agentName: string } = { agentName: args.agent ?? 'robutler' };
  if (args.model) config.model = args.model;

  // A chat at a terminal: a failed model request names the server and the
  // reason (see request.ts; a server never does, S-228).
  setRequestErrorDetail(true);
  const repl = new InteractiveREPL(config);

  if (args.prompt) {
    await repl.initialize();
    const response = await repl.sendMessage(args.prompt);
    console.log(args.json ? JSON.stringify(response, null, 2) : response.content);
    return;
  }

  await repl.run();
}

main().catch((error) => {
  console.error('Error:', error.message);
  process.exit(1);
});
