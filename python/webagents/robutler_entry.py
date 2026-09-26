"""
Robutler CLI Entry Point

`robutler` is `webagents -a robutler`: the chat with the built-in assistant,
or one prompt with `-p`, and the TypeScript package's `robutler` command
(`typescript/src/cli/robutler-args.ts`), parse for parse. Both are held to
`tests/fixtures/cli/robutler.json`.

IT READS ITS ARGUMENTS (2026-09-25). It started the chat whatever it was given,
so `robutler --help` opened a chat and `robutler -p "..."` ignored the prompt:
the bug the TypeScript command lost on 2026-09-23. Now the options are parsed
here, anything else is refused with the help, and the main CLI runs the
result, so `robutler -p` answers exactly as `webagents -a robutler -p` does.
"""

import sys
from typing import List, Optional, Tuple, Union

USAGE = """robutler - interactive session with the default robutler agent

Usage:
  robutler [options]

Options:
  -p, --prompt <text>   Send one prompt, print the reply, exit
  -m, --model <model>   Model to use, as provider/model
  -a, --agent <name>    Agent name to load (AGENT-<name>.md)
      --json            Print the reply as JSON (with -p)
  -h, --help            Show this help

For the full command surface, use `webagents`.
"""

#: ("help",), ("error", message) or ("run", webagents arguments).
RobutlerCommand = Union[Tuple[str], Tuple[str, str], Tuple[str, List[str]]]

_VALUED = {"-p": "prompt", "--prompt": "prompt", "-m": "model", "--model": "model", "-a": "agent", "--agent": "agent"}


def robutler_command(argv: List[str]) -> RobutlerCommand:
    """What `robutler <argv>` means. Refuses what it does not know, rather than ignoring it."""
    values = {"agent": "robutler", "model": None, "prompt": None}
    json_reply = False
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-h", "--help"):
            return ("help",)
        if arg in _VALUED:
            if i + 1 >= len(argv):
                return ("error", f"Missing value for {arg}")
            values[_VALUED[arg]] = argv[i + 1]
            i += 2
            continue
        if arg == "--json":
            json_reply = True
            i += 1
            continue
        return ("error", f"Unknown option: {arg}")
    run = ["-a", values["agent"]]
    model: Optional[str] = values["model"]
    prompt: Optional[str] = values["prompt"]
    if model is not None:
        run += ["-m", model]
    if prompt is not None:
        run += ["-p", prompt]
        # `--json` shapes a reply; without `-p` there is none to shape.
        if json_reply:
            run += ["--output-format", "json"]
    return ("run", run)


def main() -> None:
    """Entry point for the robutler command."""
    command = robutler_command(sys.argv[1:])
    if command[0] == "help":
        print(USAGE)
        raise SystemExit(0)
    if command[0] == "error":
        print(f"{command[1]}\n", file=sys.stderr)
        print(USAGE)
        raise SystemExit(2)
    from webagents.cli.main import cli

    sys.argv = ["webagents", *command[1]]
    cli()


if __name__ == "__main__":
    main()
