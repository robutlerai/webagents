"""
Global options typed after the subcommand (the sandbox-default lane,
2026-09-27). `webagents login --profile local` answered "No such option:
--profile": click binds a group option to the place before the command name,
and the TypeScript CLI (commander, positional options) does the same. Both
CLIs now lift the two global options a person types anywhere, `--profile
<name>` (also `--profile=<name>`) and `--no-sandbox`, to the front of the
arguments before parsing, wherever they appear, so `webagents login --profile
local` is `webagents --profile local login`. Nothing past `--` is touched,
and nothing else moves. The TypeScript twin is `cli/sandbox-default-argv.ts`;
the fixture `tests/fixtures/cli/sandbox_default_global_options.json` pins the
cases for both.
"""

from __future__ import annotations

from typing import List, Sequence

#: The global options lifted to the front: the ones a person reasonably types
#: last. `--json` since 2026-09-28 (B10): `doctor --json` and `sandbox setup
#: --json` answered "unknown option '--json'".
HOISTED_OPTIONS = ("--profile", "--no-sandbox", "--json", "--max-tool-rounds")


def hoist_global_options(argv: Sequence[str]) -> List[str]:
    """`argv` without the program name, with the hoisted options moved to the front, in the order found."""
    front: List[str] = []
    rest: List[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--":
            rest.extend(argv[index:])
            break
        if arg in ("--no-sandbox", "--json"):
            front.append(arg)
        elif arg in ("--profile", "--max-tool-rounds"):
            # A value must follow; without one the parser says so, as before.
            if index + 1 < len(argv):
                front.extend([arg, argv[index + 1]])
                index += 1
            else:
                rest.append(arg)
        elif arg.startswith("--profile=") or arg.startswith("--max-tool-rounds="):
            front.append(arg)
        else:
            rest.append(arg)
        index += 1
    return [*front, *rest]
