"""
The `webagents` command groups that have modules of their own: `config`,
`daemon` and `secrets`. Every other command is defined in `cli/main.py`.

The command surface is the TypeScript CLI's (`typescript/src/cli/index.ts`),
and `tests/cli/test_cli_parity.py` holds the two together. On 2026-09-24 the
Python-only groups went (`agent`, `auth`, `checkpoint`, `session`, `skill`,
`template`, `ui`, `register`, `deploy`, and `doctor` as a group): the chat
builds its agent in its own process and has `/resume` and `/new`, `publish`
replaced `deploy`, and `webagents -p` replaced `run`.
"""

from . import config, daemon, secrets

__all__ = ["config", "daemon", "secrets"]
