"""
Builtin embedded agents.

These agents are bundled with the webagents package and serve as defaults
when no local agent files are present.
"""

from pathlib import Path

BUILTIN_AGENTS_DIR = Path(__file__).parent


def get_robutler_path() -> Path:
    """Get the path to the embedded ROBUTLER.md agent."""
    return BUILTIN_AGENTS_DIR / "ROBUTLER.md"
