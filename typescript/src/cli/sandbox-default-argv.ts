/**
 * Global options typed after the subcommand (the sandbox-default lane,
 * 2026-09-27). `webagents login --profile local` answered "unknown option
 * '--profile'": commander's positional options bind a program option to the
 * place before the command name, and the Python CLI (click) does the same.
 * Both CLIs now lift the two global options a person types anywhere,
 * `--profile <name>` (also `--profile=<name>`) and `--no-sandbox`, to the
 * front of the arguments before parsing, wherever they appear, so
 * `webagents login --profile local` is `webagents --profile local login`.
 * Nothing past `--` is touched, and nothing else moves. The Python twin is
 * `cli/sandbox_default_argv.py`; the fixture
 * `python/tests/fixtures/cli/sandbox_default_global_options.json` pins the
 * cases for both.
 */

/**
 * The global options lifted to the front: the ones a person reasonably types
 * last. `--json` since 2026-09-28 (B10): `doctor --json` and `sandbox setup
 * --json` answered "unknown option '--json'".
 */
export const HOISTED_OPTIONS = ['--profile', '--no-sandbox', '--json', '--max-tool-rounds'] as const;

/** `argv` without the program and script names, with the hoisted options moved to the front, in the order found. */
export function hoistGlobalOptions(argv: readonly string[]): string[] {
  const front: string[] = [];
  const rest: string[] = [];
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === '--') {
      rest.push(...argv.slice(index));
      break;
    }
    if (arg === '--no-sandbox' || arg === '--json') {
      front.push(arg);
      continue;
    }
    if (arg === '--profile' || arg === '--max-tool-rounds') {
      // A value must follow; without one the parser says so, as before.
      if (index + 1 < argv.length) {
        front.push(arg, argv[index + 1]);
        index += 1;
      } else {
        rest.push(arg);
      }
      continue;
    }
    if (arg.startsWith('--profile=') || arg.startsWith('--max-tool-rounds=')) {
      front.push(arg);
      continue;
    }
    rest.push(arg);
  }
  return [...front, ...rest];
}
