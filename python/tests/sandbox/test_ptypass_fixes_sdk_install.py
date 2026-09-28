"""
S-316, pinned and proved (the ptypass-fixes lane, 2026-09-27).

Under `development` the agent's folder is a write root. When the SDK was
installed there (a project-local `.venv`, or `node_modules` for the
TypeScript CLI), a confined command could rewrite the SDK's own code, the
interpreter's start-up files and srt itself, which runs OUTSIDE the sandbox
for every command: the next command or the next `webagents` start ran what
was planted, unconfined. The install is now write-denied whenever it lies
inside a write root, and nothing else is: a command may still write a
different venv the project keeps.

What these tests hold, from `tests/fixtures/sandbox/srt.json`
(`sdk_install_deny`):

  * the rule itself, case by case (`install_write_denies`);
  * where this process's install is (`sdk_install_paths`) and that the
    settings deny it when the folder holds it (`build_settings`);
  * the `install` line `doctor` and `webagents sandbox setup` print;
  * WITH REAL srt: an agent folder holding its own `.venv`, with a copy of
    webagents (and so its bundled engine) inside it, run from that venv's
    interpreter. A confined command cannot plant a `.pth` file, append to the
    SDK's code or to srt's `cli.js`, and can still write another venv and a
    plain file. A control run with the deny switched off writes all three,
    so the test would catch the hole coming back.

The TypeScript twin is
`typescript/tests/unit/sandbox/ptypass-fixes-sdk-install.test.ts`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from types import SimpleNamespace

import pytest

from webagents.sandbox import backend_status, default_policy, install_write_denies, sandbox_available
from webagents.sandbox import srt

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "sandbox" / "srt.json").read_text())["sdk_install_deny"]
requires_backend = pytest.mark.skipif(not sandbox_available(), reason=f"no srt here: {backend_status()['reason']}")

PACKAGE = Path(srt.__file__).resolve().parents[1]  # the webagents package folder
SDK_ROOT = PACKAGE.parent  # python/, which holds .venv and webagents/


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=[c["name"] for c in FIXTURE["cases"]])
def test_the_rule_case_by_case(case):
    assert install_write_denies(case["write_roots"], case["installs"]) == case["denied"]


def test_the_install_of_this_process_is_found():
    installs = srt.sdk_install_paths()
    assert os.path.realpath(sys.prefix) in installs or any(os.path.realpath(sys.prefix).startswith(p + "/") for p in installs)
    assert any(str(PACKAGE) == p or str(PACKAGE).startswith(p + "/") for p in installs)
    status = backend_status()
    if status.get("node"):
        node_dir = os.path.dirname(os.path.realpath(status["node"]))
        assert any(node_dir == p or node_dir.startswith(p + "/") for p in installs)


def test_the_settings_deny_it_only_when_the_folder_holds_it(tmp_path):
    held = srt.build_settings(default_policy(cwd=str(SDK_ROOT)))["filesystem"]["denyWrite"]
    assert str(PACKAGE) in held
    if os.path.realpath(sys.prefix).startswith(str(SDK_ROOT) + "/"):
        assert os.path.realpath(sys.prefix) in held
    elsewhere = srt.build_settings(default_policy(cwd=str(tmp_path)))["filesystem"]["denyWrite"]
    assert not any(entry == str(PACKAGE) or entry.startswith(str(PACKAGE) + "/") for entry in elsewhere)


def test_the_install_line_says_it_in_the_fixtures_words(tmp_path):
    report = FIXTURE["report"]
    found = srt.install_inside_check([str(SDK_ROOT)])
    assert found is not None
    assert found["name"] == report["name"] and found["status"] == report["status"] and found["fix"] == report["fix"]
    head, tail = report["detail"].split("{where}")
    assert found["detail"].startswith(head) and found["detail"].endswith(tail)
    assert "webagents" in found["detail"][len(head) : len(found["detail"]) - len(tail)].split(", ")
    assert srt.install_inside_check([str(tmp_path)]) is None
    itself = srt.install_inside_check([str(PACKAGE)])
    assert itself is not None and f"({report['is_the_folder']})" in itself["detail"]


def test_setup_says_it_before_the_confined_check(monkeypatch, tmp_path):
    monkeypatch.chdir(SDK_ROOT)
    names = [c["name"] for c in srt.setup_checks()]
    assert "install" in names and names.index("install") < names.index("confined")
    monkeypatch.chdir(tmp_path)
    assert "install" not in [c["name"] for c in srt.setup_checks()]


def test_doctor_says_it_for_a_confined_policy_only(tmp_path):
    from webagents.cli.doctor import _install_check

    def built(policy):
        return SimpleNamespace(agent=SimpleNamespace(skills={"shell": SimpleNamespace(policy=policy)}))

    line = _install_check(built(default_policy(cwd=str(SDK_ROOT))))
    assert line is not None and line.name == "install" and line.status == "warn"
    assert _install_check(built(default_policy(cwd=str(tmp_path)))) is None
    from webagents.sandbox import policy_from_metadata

    assert _install_check(built(policy_from_metadata("off", cwd=str(SDK_ROOT)))) is None


PROBE = r"""
import json, os, sys
import webagents
import webagents.sandbox.srt as engine
from webagents.sandbox import default_policy, run_sandboxed

if sys.argv[1] == "control":
    # The hole as it was: no install deny.
    engine.sdk_install_paths = lambda *args, **kwargs: []
agent = os.path.realpath(os.getcwd())
site = next(p for p in sys.path if p.endswith("site-packages") and os.path.realpath(p).startswith(agent + "/"))
status = engine.backend_status()
policy = default_policy(cwd=agent)

def attempt(command):
    result = run_sandboxed(command, policy, timeout=60)
    return result.stdout + result.stderr

print(json.dumps({
    "webagents": os.path.realpath(webagents.__file__),
    "prefix": os.path.realpath(sys.prefix),
    "engine": status["path"],
    "available": status["available"],
    "pth": attempt(f"echo 'import os' > {site}/ptypass_fixes_planted.pth && echo WROTE"),
    "sdk": attempt(f"echo '# planted' >> {site}/webagents/__init__.py && echo WROTE"),
    "engine_write": attempt(f"echo '// planted' >> {status['path']} && echo WROTE"),
    "other_venv": attempt("mkdir -p other-venv/lib && echo x > other-venv/lib/ok.py && echo WROTE"),
    "plain": attempt("echo x > plain.txt && echo WROTE"),
}))
"""


def _agent_with_its_own_venv(tmp_path: Path) -> Path:
    """An agent folder holding `.venv`, a real venv of this interpreter with a
    copy of webagents in its site-packages (the bundled engine with it) and a
    `.pth` line reaching this venv's site-packages for the dependencies."""
    agent = tmp_path / "agent"
    agent.mkdir()
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(agent / ".venv")], check=True, timeout=120)
    python = agent / ".venv" / "bin" / "python"
    site = Path(subprocess.run([str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], check=True, capture_output=True, text=True).stdout.strip())
    shutil.copytree(PACKAGE, site / "webagents", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (site / "ptypass_fixes_deps.pth").write_text(sysconfig.get_paths()["purelib"] + "\n")
    return agent


def _probe(agent: Path, mode: str) -> dict:
    done = subprocess.run([str(agent / ".venv" / "bin" / "python"), "-c", PROBE, mode], cwd=str(agent), capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@requires_backend
def test_real_srt_a_project_venv_inside_the_agent_folder(tmp_path):
    agent = _agent_with_its_own_venv(tmp_path)
    venv = str(agent.resolve() / ".venv")
    site = next((agent / ".venv" / "lib").glob("python*/site-packages")).resolve()
    init_before = (site / "webagents" / "__init__.py").read_text()

    out = _probe(agent, "enforced")
    # The copy inside the agent folder is what ran, engine included.
    assert out["available"] is True
    assert out["webagents"].startswith(venv + "/") and out["prefix"] == venv and out["engine"].startswith(venv + "/")
    engine_before = Path(out["engine"]).read_text()
    for key in ("pth", "sdk", "engine_write"):
        assert "WROTE" not in out[key], (key, out[key])
    assert not (site / "ptypass_fixes_planted.pth").exists()
    assert (site / "webagents" / "__init__.py").read_text() == init_before
    assert Path(out["engine"]).read_text() == engine_before
    # A different venv the project keeps, and a plain file, are still writable.
    assert "WROTE" in out["other_venv"] and (agent / "other-venv" / "lib" / "ok.py").exists()
    assert "WROTE" in out["plain"]

    # The control: without the install deny the same commands write all three.
    control = _probe(agent, "control")
    for key in ("pth", "sdk", "engine_write"):
        assert "WROTE" in control[key], (key, control[key])
    assert (site / "ptypass_fixes_planted.pth").exists()
