"""
`-p` and `doctor` keep their log in the profile's folder, as the chat does
(the ptypass-fixes lane, 2026-09-27, brief item 13). Both wrote
`~/.webagents/logs/repl.log` whatever `--profile` said, beside another
profile's history. The TypeScript CLI keeps no such log.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

AGENT = "---\nname: logged\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n---\nHi.\n"


@pytest.mark.parametrize("argv", [["doctor"], ["-p", "hi"]], ids=["doctor", "-p"])
def test_the_log_follows_the_profile(tmp_path, argv):
    project = tmp_path / "project"
    project.mkdir()
    (project / "AGENT.md").write_text(AGENT)
    home = tmp_path / "home"
    home.mkdir()
    env = {key: value for key, value in os.environ.items() if not key.endswith("_API_KEY") and key not in ("WEBAGENTS_TOKEN", "WEBAGENTS_PROFILE", "WEBAGENTS_DEBUG")}
    env.update(HOME=str(home), WEBAGENTS_SECRETS_BACKEND="file", ROBUTLER_API_URL="http://127.0.0.1:9")
    subprocess.run([sys.executable, "-m", "webagents", "--profile", "ptftest", *argv], cwd=project, env=env, capture_output=True, text=True, timeout=180)
    assert (home / ".webagents-ptftest" / "logs" / "repl.log").exists()
    assert not (home / ".webagents" / "logs" / "repl.log").exists()
