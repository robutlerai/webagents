"""
Robutler CLI Entry Point

Alias for `webagents connect` - starts interactive session with the default agent.
"""


def main():
    """Entry point for the robutler command."""
    from .cli.commands.agent import connect_command
    connect_command(agent=None, use_tui=True)


if __name__ == "__main__":
    main()
