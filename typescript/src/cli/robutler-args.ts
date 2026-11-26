/**
 * The `robutler` command's options, and the `webagents` command line they
 * become (2026-09-25).
 *
 * `robutler` is `webagents -a robutler`: the chat with the built-in assistant.
 * It has its own short help and options (below), and it runs through the main
 * CLI, so `robutler -p` answers exactly as `webagents -a robutler -p` does,
 * failure hints included. It used to run its own REPL, with its own `-p` path
 * that printed a bare `Error:` line and a `--json` shape nothing else used.
 *
 * The Python package's `robutler` (`python/webagents/robutler_entry.py`) is
 * the same, parse for parse: both are held to
 * `python/tests/fixtures/cli/robutler.json`.
 */

export const USAGE = `robutler - interactive session with the default robutler agent

Usage:
  robutler [options]

Options:
  -p, --prompt <text>   Send one prompt, print the reply, exit
  -m, --model <model>   Model to use, as provider/model
  -a, --agent <name>    Agent name to load (AGENT-<name>.md)
      --json            Print the reply as JSON (with -p)
  -V, --version         Show the version
  -h, --help            Show this help

  robutler login | logout | whoami   Your sign-in on Robutler
  robutler app <command> ...         Your apps: create, link, push, status, release, hooks, secrets, logs, token
  robutler cas <command> ...         Project files: ls, cat, mounts, mount, unmount, update

For the full command surface, use \`webagents\`.
`;

export type RobutlerCommand =
  /** Print USAGE, exit 0. */
  | { kind: 'help' }
  /** Print the message and USAGE, exit 2: an option this command does not take. */
  | { kind: 'error'; message: string }
  /** Run `webagents` with these arguments. */
  | { kind: 'run'; argv: string[] }
  /** `robutler app ...`: these are the arguments after `app`, run by `app-command.ts`. */
  | { kind: 'app'; argv: string[] };

/** The main CLI's commands `robutler` offers under their own names. */
const PASSED_THROUGH: ReadonlySet<string> = new Set(['cas', 'login', 'logout', 'whoami']);

/** What `robutler <argv>` means. Refuses what it does not know, rather than ignoring it. */
export function robutlerCommand(argv: string[]): RobutlerCommand {
  // `robutler app ...` is this command's own: the apps of the signed-in person.
  if (argv[0] === 'app') return { kind: 'app', argv: argv.slice(1) };
  // `robutler cas ...` and the sign-in commands are the main CLI's, as they are.
  if (argv[0] !== undefined && PASSED_THROUGH.has(argv[0])) return { kind: 'run', argv };
  let agent = 'robutler';
  let model: string | undefined;
  let prompt: string | undefined;
  let json = false;
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const value = (): string | undefined => (i + 1 < argv.length ? argv[++i] : undefined);
    switch (arg) {
      case '-h':
      case '--help':
        return { kind: 'help' };
      case '-V':
      case '--version':
        return { kind: 'run', argv: ['--version'] };
      case '-p':
      case '--prompt':
      case '-m':
      case '--model':
      case '-a':
      case '--agent': {
        const given = value();
        if (given === undefined) return { kind: 'error', message: `Missing value for ${arg}` };
        if (arg === '-p' || arg === '--prompt') prompt = given;
        else if (arg === '-m' || arg === '--model') model = given;
        else agent = given;
        break;
      }
      case '--json':
        json = true;
        break;
      default:
        return { kind: 'error', message: `Unknown option: ${arg}` };
    }
  }
  const run = ['-a', agent];
  if (model !== undefined) run.push('-m', model);
  if (prompt !== undefined) {
    run.push('-p', prompt);
    // `--json` shapes a reply; without `-p` there is none to shape.
    if (json) run.push('--output-format', 'json');
  }
  return { kind: 'run', argv: run };
}
