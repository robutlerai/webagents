"""
Importing the SDK does not print its internals into the caller's stdout (2026-09-24).

Import-time logging was INFO on stdout, so the README's first example printed
system-prompt sizes and skill initialisation ahead of its answer. A library is
quiet by default now; a server's log is its output, so `create_server` raises
it to INFO unless the application configured logging itself. Checked in fresh
interpreters, because logging configuration is process-global.
"""

import subprocess
import sys

LIBRARY_USE = r"""
import logging
from webagents import BaseAgent
agent = BaseAgent(name="quiet", instructions="x", model="openai/gpt-4o-mini")
logging.getLogger("webagents.base_agent").info("internal detail")
print("ANSWER")
"""

SERVER_USE = r"""
import logging
from webagents import BaseAgent, create_server
agent = BaseAgent(name="loud", instructions="x", model="openai/gpt-4o-mini")
create_server(agents=[agent], heartbeat=False)
print("INFO enabled:", logging.getLogger("webagents").isEnabledFor(logging.INFO))
"""

APP_CHOSE = r"""
import logging
from webagents.utils.logging import setup_logging
setup_logging(level="ERROR")
from webagents import BaseAgent, create_server
create_server(agents=[BaseAgent(name="x", instructions="x", model="openai/gpt-4o-mini")], heartbeat=False)
print("INFO enabled:", logging.getLogger("webagents").isEnabledFor(logging.INFO))
"""


def _run(code, **env):
    import os

    environment = {k: v for k, v in os.environ.items() if k != "WEBAGENTS_LOG_LEVEL"}
    environment.update(env)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=environment)


def test_library_use_keeps_stdout_for_the_caller():
    out = _run(LIBRARY_USE, OPENAI_API_KEY="dummy")
    # THE BUG: "BaseAgent created ..." and friends came first, on stdout.
    assert out.stdout == "ANSWER\n"


def test_a_server_still_logs_at_info():
    out = _run(SERVER_USE, OPENAI_API_KEY="dummy")
    assert "INFO enabled: True" in out.stdout


def test_a_server_does_not_override_what_the_application_chose():
    out = _run(APP_CHOSE, OPENAI_API_KEY="dummy")
    assert "INFO enabled: False" in out.stdout
