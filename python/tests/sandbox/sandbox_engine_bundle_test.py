"""
The sandbox engine that ships inside this package (the sandbox-engine lane,
2026-09-27), against `tests/fixtures/sandbox/sandbox_engine.json`.

WHY. The sandbox is on by default and fails closed, and a pip package cannot
declare an npm dependency, so a Python user without Node and a global srt had
every shell command refused. srt 0.0.77 and its dependencies are vendored
under `webagents/sandbox/sandbox_engine/node_modules/` by
`scripts/sandbox_engine_vendor.py`, and node comes from PATH (20.11 or later)
or the `nodejs-wheel-binaries` dependency. This file holds that together:

  * the vendored tree is byte for byte its provenance record, and the record
    is the TypeScript lockfile's srt closure (a lockfile bump without a
    re-vendor fails here);
  * `locate_srt` prefers the bundled copy, `WEBAGENTS_SRT_CLI` still wins,
    and a bundled copy of another version is refused;
  * node is chosen PATH first, then the wheel's, never downloaded;
  * the bundled engine confines for real, and under `--debug` prints the
    `[SandboxDebug] Connection blocked to host:port` lines the chat's
    ask-on-first-use reads;
  * pyproject declares the node wheel with its markers and ships the tree.
"""

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import pytest

from webagents.sandbox import backend_status, default_policy, run_sandboxed, sandbox_available
from webagents.sandbox import srt as engine

PY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = json.loads((PY_ROOT / "tests" / "fixtures" / "sandbox" / "sandbox_engine.json").read_text())

requires_backend = pytest.mark.skipif(not sandbox_available(), reason=f"no srt here: {backend_status()['reason']}")


def _vendor_script():
    spec = importlib.util.spec_from_file_location("sandbox_engine_vendor", PY_ROOT / "scripts" / "sandbox_engine_vendor.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def clean_status(monkeypatch):
    monkeypatch.delenv(engine.ENV_CLI, raising=False)
    monkeypatch.delenv(engine.ENV_NODE, raising=False)
    engine.reset_backend_status()
    yield
    engine.reset_backend_status()


def _fake_node(folder: Path, version: str) -> Path:
    """An executable that answers `--version` as a node of `version` would."""
    folder.mkdir(parents=True, exist_ok=True)
    node = folder / "node"
    node.write_text(f"#!/bin/sh\necho {version}\n")
    node.chmod(0o755)
    return node


class TestTheVendoredTree:
    def test_it_matches_its_record_and_the_lockfile(self):
        assert _vendor_script().check() == []

    def test_the_bundled_cli_is_the_pinned_version(self):
        package = Path(engine.BUNDLED_CLI).parents[1]
        assert json.loads((package / "package.json").read_text())["version"] == engine.SRT_VERSION
        assert Path(engine.BUNDLED_CLI).is_file()
        assert Path(engine.BUNDLED_CLI).relative_to(PY_ROOT).as_posix() == FIXTURE["engine"]["python"]["bundled"]

    def test_the_licenses_and_the_notice_ship(self):
        record = json.loads((Path(engine.ENGINE_DIR) / "sandbox_engine_provenance.json").read_text())
        for package in record["packages"]:
            assert "LICENSE" in package["files"], package["name"]
            assert package["integrity"].startswith("sha512-")
            assert package["tarball"].startswith("https://registry.npmjs.org/")
        srt = next(p for p in record["packages"] if p["name"] == engine.SRT_PACKAGE)
        assert srt["license"] == "Apache-2.0"
        notice = (Path(engine.ENGINE_DIR) / "NOTICE").read_text()
        assert f"{engine.SRT_PACKAGE} {engine.SRT_VERSION}" in notice and "Apache-2.0" in notice

    def test_macos_and_linux_helpers_are_kept_and_windows_is_not(self):
        record = json.loads((Path(engine.ENGINE_DIR) / "sandbox_engine_provenance.json").read_text())
        srt = next(p for p in record["packages"] if p["name"] == engine.SRT_PACKAGE)
        for kept in ("vendor/seccomp/x64/apply-seccomp", "vendor/seccomp/arm64/apply-seccomp", "vendor/java-proxy-agent/srt-proxy-agent.jar", "dist/cli.js"):
            assert kept in srt["files"], kept
        assert not any(name.startswith("vendor/srt-win/") for name in srt["files"])
        helper = Path(engine.ENGINE_DIR) / "node_modules" / engine.SRT_PACKAGE / "vendor" / "seccomp" / "x64" / "apply-seccomp"
        assert os.access(helper, os.X_OK)


class TestLocating:
    def test_the_bundled_copy_is_used_when_nothing_names_another(self, clean_status):
        cli, cli_from, reason, fix = engine.locate_cli({})
        assert (reason, fix) == ("", "")
        assert cli == os.path.realpath(engine.BUNDLED_CLI)
        assert cli_from == FIXTURE["engine"]["python"]["cli_from"]["bundled"]

    def test_webagents_srt_cli_still_wins(self, tmp_path, clean_status):
        package = tmp_path / "node_modules" / "@anthropic-ai" / "sandbox-runtime"
        (package / "dist").mkdir(parents=True)
        (package / "dist" / "cli.js").write_text("process.exit(0)\n")
        (package / "package.json").write_text(json.dumps({"name": engine.SRT_PACKAGE, "version": engine.SRT_VERSION}))
        cli, cli_from, _reason, _fix = engine.locate_cli({engine.ENV_CLI: str(package / "dist" / "cli.js")})
        assert cli == os.path.realpath(package / "dist" / "cli.js")
        assert cli_from == FIXTURE["engine"]["python"]["cli_from"]["env"]

    def test_a_named_install_that_is_not_there_says_so_with_its_fix(self, tmp_path, clean_status):
        value = str(tmp_path / "nowhere" / "cli.js")
        cli, _from, reason, fix = engine.locate_cli({engine.ENV_CLI: value})
        assert cli is None
        assert reason == FIXTURE["reasons"]["engine_env_missing"].format(package=engine.SRT_PACKAGE, version=engine.SRT_VERSION, value=value)
        assert fix == FIXTURE["fixes"]["engine_env"].format(version=engine.SRT_VERSION)

    def test_a_bundled_copy_of_another_version_is_refused(self, tmp_path, monkeypatch, clean_status):
        package = tmp_path / "node_modules" / "@anthropic-ai" / "sandbox-runtime"
        (package / "dist").mkdir(parents=True)
        (package / "dist" / "cli.js").write_text("process.exit(0)\n")
        (package / "package.json").write_text(json.dumps({"name": engine.SRT_PACKAGE, "version": "0.0.1"}))
        monkeypatch.setattr(engine, "BUNDLED_CLI", str(package / "dist" / "cli.js"))
        cli, _from, reason, fix = engine.locate_cli({})
        assert cli is None
        assert reason == FIXTURE["reasons"]["wrong_version"].format(cli=os.path.realpath(package / "dist" / "cli.js"), package=engine.SRT_PACKAGE, found="0.0.1", version=engine.SRT_VERSION)
        assert fix == FIXTURE["fixes"]["engine_bundled_python"]

    def test_a_missing_bundled_copy_names_the_reinstall(self, tmp_path, monkeypatch, clean_status):
        monkeypatch.setattr(engine, "BUNDLED_CLI", str(tmp_path / "gone" / "cli.js"))
        cli, _from, reason, fix = engine.locate_cli({})
        assert cli is None
        assert reason == FIXTURE["reasons"]["engine_bundled_missing_python"].format(package=engine.SRT_PACKAGE, version=engine.SRT_VERSION, path=str(tmp_path / "gone" / "cli.js"))
        assert fix == FIXTURE["fixes"]["engine_bundled_python"]


class TestChoosingNode:
    @pytest.mark.parametrize("case", FIXTURE["node"]["version_cases"])
    def test_versions_are_read_as_the_fixture_says(self, case):
        version = engine.parse_node_version(case["output"])
        assert (list(version) if version else None) == case["version"]
        assert (version is not None and version >= engine.NODE_MINIMUM) is case["enough"]

    def test_the_minimum_is_srts_own(self):
        package = Path(engine.BUNDLED_CLI).parents[1]
        engines = json.loads((package / "package.json").read_text())["engines"]["node"]
        assert engines == ">=" + FIXTURE["node"]["minimum"]
        assert ".".join(map(str, engine.NODE_MINIMUM)) == FIXTURE["node"]["minimum"]

    def test_node_from_path_when_it_is_recent_enough(self, tmp_path, monkeypatch):
        path_node = _fake_node(tmp_path / "bin", "v24.7.0")
        monkeypatch.setattr(engine, "wheel_node", lambda: None)
        node, node_from, reason, _fix = engine.choose_node({"PATH": str(path_node.parent)})
        assert (node, node_from, reason) == (os.path.realpath(path_node), FIXTURE["node"]["python_from"]["path"], "")

    def test_the_wheel_node_when_path_has_an_old_one(self, tmp_path, monkeypatch):
        old = _fake_node(tmp_path / "bin", "v18.19.1")
        wheel = _fake_node(tmp_path / "site" / "nodejs_wheel" / "bin", "v24.19.0")
        monkeypatch.setattr(engine, "wheel_node", lambda: str(wheel))
        node, node_from, _reason, _fix = engine.choose_node({"PATH": str(old.parent)})
        assert (node, node_from) == (os.path.realpath(wheel), FIXTURE["node"]["python_from"]["wheel"])

    def test_the_wheel_node_is_found_in_site_packages_without_importing_it(self, tmp_path, monkeypatch):
        package = tmp_path / "site" / "nodejs_wheel"
        wheel = _fake_node(package / "bin", "v24.19.0")
        # An __init__ that would fail loudly if anything imported it.
        (package / "__init__.py").write_text("raise RuntimeError('imported')\n")
        monkeypatch.setattr(engine, "_site_roots", lambda: [str(tmp_path / "site")])
        monkeypatch.delitem(sys.modules, "nodejs_wheel", raising=False)
        assert engine.wheel_node() == str(wheel)
        assert "nodejs_wheel" not in sys.modules

    def test_a_node_planted_on_sys_path_or_the_current_folder_is_not_the_wheels(self, tmp_path, monkeypatch):
        # `python -m webagents` puts the current folder on sys.path, and a
        # confined command can write the agent's folder.
        planted = _fake_node(tmp_path / "agent" / "nodejs_wheel" / "bin", "v24.19.0")
        monkeypatch.syspath_prepend(str(tmp_path / "agent"))
        monkeypatch.chdir(tmp_path / "agent")
        assert str(planted) != engine.wheel_node()
        assert all(not root.startswith(str(tmp_path.resolve())) for root in engine._site_roots())

    def test_relative_path_entries_are_not_searched(self, tmp_path, monkeypatch):
        planted = _fake_node(tmp_path / "agent" / "bin", "v24.7.0")
        monkeypatch.chdir(tmp_path / "agent")
        monkeypatch.setattr(engine, "wheel_node", lambda: None)
        node, _from, reason, _fix = engine.choose_node({"PATH": f"bin{os.pathsep}.{os.pathsep}"})
        assert node is None and str(planted) not in reason
        assert FIXTURE["reasons"]["node_looked_path_none"] in reason

    def test_no_node_anywhere_says_what_was_looked_at_and_the_fix(self, tmp_path, monkeypatch):
        monkeypatch.setattr(engine, "wheel_node", lambda: None)
        node, _from, reason, fix = engine.choose_node({"PATH": str(tmp_path)})
        minimum = FIXTURE["node"]["minimum"]
        looked = f"{FIXTURE['reasons']['node_looked_path_none']}; {FIXTURE['reasons']['node_looked_wheel_none']}"
        assert node is None
        assert reason == FIXTURE["reasons"]["node_missing"].format(minimum=minimum, looked=looked)
        assert fix == FIXTURE["fixes"]["node_missing"].format(minimum=minimum)

    def test_an_old_path_node_and_no_wheel(self, tmp_path, monkeypatch):
        old = _fake_node(tmp_path / "bin", "v18.19.1")
        monkeypatch.setattr(engine, "wheel_node", lambda: None)
        _node, _from, reason, _fix = engine.choose_node({"PATH": str(old.parent)})
        looked = f"{FIXTURE['reasons']['node_looked_path_old'].format(found='v18.19.1')}; {FIXTURE['reasons']['node_looked_wheel_none']}"
        assert reason == FIXTURE["reasons"]["node_missing"].format(minimum=FIXTURE["node"]["minimum"], looked=looked)

    def test_an_explicit_node_too_old_is_refused_not_passed_over(self, tmp_path, monkeypatch):
        old = _fake_node(tmp_path / "bin", "v18.19.1")
        monkeypatch.setattr(engine, "wheel_node", lambda: str(_fake_node(tmp_path / "wheel", "v24.19.0")))
        node, _from, reason, fix = engine.choose_node({engine.ENV_NODE: str(old), "PATH": ""})
        minimum = FIXTURE["node"]["minimum"]
        detail = FIXTURE["reasons"]["node_too_old"].format(path=str(old), found="v18.19.1", minimum=minimum)
        assert node is None
        assert reason == FIXTURE["reasons"]["node_env"].format(detail=detail)
        assert fix == FIXTURE["fixes"]["node_env"].format(minimum=minimum)

    def test_node_version_runs_with_an_empty_environment(self, tmp_path, monkeypatch):
        # NODE_OPTIONS in the environment would run code inside that node.
        probe = tmp_path / "bin" / "node"
        probe.parent.mkdir(parents=True)
        probe.write_text('#!/bin/sh\nif [ -n "$NODE_OPTIONS$HOME" ]; then echo v1.0.0; else echo v22.12.0; fi\n')
        probe.chmod(0o755)
        monkeypatch.setenv("NODE_OPTIONS", "--require /tmp/evil.js")
        assert engine._node_version(str(probe)) == (22, 12, 0)


class TestPackaging:
    # Read as text: `tomllib` is 3.11+, and this package supports 3.10.
    PYPROJECT = (PY_ROOT / "pyproject.toml").read_text()

    def test_pyproject_declares_the_node_wheel_for_macos_and_linux(self):
        node = re.findall(r'^\s*"(nodejs-wheel-binaries[^"]*)",', self.PYPROJECT, re.M)
        assert len(node) == 1
        requirement, _, marker = node[0].partition(";")
        assert requirement.strip() == "nodejs-wheel-binaries>=22.12,<25"
        for fragment in ("sys_platform == 'darwin'", "sys_platform == 'linux'", "platform_machine == 'arm64'", "platform_machine == 'x86_64'", "platform_machine == 'aarch64'"):
            assert fragment in marker
        assert "win32" not in marker

    def test_the_tree_ships_despite_the_gitignore(self):
        section = self.PYPROJECT.split("[tool.hatch.build]", 1)[1].split("\n[", 1)[0]
        assert 'artifacts = ["webagents/sandbox/sandbox_engine/**"]' in section
        ignore = (Path(engine.ENGINE_DIR) / ".gitignore").read_text().split()
        assert "!dist/" in ignore and "!lib/" in ignore


@requires_backend
class TestTheBundledEngineConfinesForReal:
    def test_it_is_the_engine_in_use(self):
        if os.environ.get(engine.ENV_CLI):
            pytest.skip("WEBAGENTS_SRT_CLI names another install")
        status = backend_status()
        assert status["path"] == os.path.realpath(engine.BUNDLED_CLI)
        assert "cli.js from the engine that ships with webagents" in status["found"]

    def test_a_confined_command_runs_and_the_network_is_refused(self, tmp_path):
        result = run_sandboxed("echo confined-ok; curl -sS -m 10 https://example.com/ 2>&1 | head -1", default_policy(cwd=str(tmp_path)), timeout=60)
        assert "confined-ok" in result.stdout
        assert "403" in result.stdout

    def test_debug_prints_the_blocked_line_the_chat_reads(self, tmp_path):
        result = run_sandboxed("curl -sS -m 10 https://example.com/ >/dev/null 2>&1; true", default_policy(cwd=str(tmp_path)), timeout=60, capture_refusals=True)
        assert result.refused_hosts == ["example.com"]
        assert result.stderr == ""
