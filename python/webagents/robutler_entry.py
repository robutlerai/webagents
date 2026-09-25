"""
Robutler CLI Entry Point

`robutler`: the chat with the built-in assistant, wherever you are; the same
as `webagents -a robutler`, and the TypeScript package's `robutler` command.
"""


def main() -> None:
    """Entry point for the robutler command."""
    from .cli.repl.session import start_repl

    start_repl(agent_path=None, chosen=True)


if __name__ == "__main__":
    main()
