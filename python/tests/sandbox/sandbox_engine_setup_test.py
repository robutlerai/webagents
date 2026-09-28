"""
`webagents sandbox setup`, the refusal sentence, and doctor's `sandbox` fix
(the sandbox-engine lane, 2026-09-27), against
`tests/fixtures/sandbox/sandbox_engine.json`, which the TypeScript suite
(`tests/unit/sandbox/sandbox-engine-setup.test.ts`) reads too.

The engine ships inside both packages, so what can still be missing is the
machine's: the Linux programs, named with this distribution's install line;
what a container must allow; WSL 2 on native Windows; a macOS TMPDIR too long
for srt's socket. `setup_checks` says which, and runs a real confined `true`.
The refusal a shell command gets carries the same reason and fix, then the
opt-out (`UNAVAILABLE_TAIL`). The platform cases are simulated here; the
confined `true` runs for real where srt can.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from webagents.sandbox import UNAVAILABLE_TAIL, backend_status, sandbox_available, unavailable_fix, unavailable_message
from webagents.sandbox import srt as engine

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FIXTURE = json.loads((FIXTURES / "sandbox" / "sandbox_engine.json").read_text())
SRT_FIXTURE = json.loads((FIXTURES / "sandbox" / "srt.json").read_text())
CHAT_FIXTURE = json.loads((FIXTURES / "cli" / "chat_edits.json").read_text())

requires_backend = pytest.mark.skipif(not sandbox_available(), reason=f"no srt here: {backend_status()['reason']}")


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.delenv(engine.ENV_CLI, raising=False)
    monkeypatch.delenv(engine.ENV_NODE, raising=False)
    engine.reset_backend_status()
    yield monkeypatch
    engine.reset_backend_status()


class TestTheWords:
    def test_the_tail_names_the_check_and_the_opt_out(self):
        assert UNAVAILABLE_TAIL == SRT_FIXTURE["refusals"]["unavailable_tail"]
        assert "webagents sandbox setup" in UNAVAILABLE_TAIL
        assert "--no-sandbox" in UNAVAILABLE_TAIL and "`sandbox: off`" in UNAVAILABLE_TAIL
        assert "npm install" not in UNAVAILABLE_TAIL
        assert SRT_FIXTURE["refusals"]["unavailable"] == FIXTURE["unavailable"]["template"].replace("{tail}", UNAVAILABLE_TAIL)

    def test_the_refusal_is_reason_fix_then_tail(self):
        example = FIXTURE["unavailable"]["example"]
        assert unavailable_message(example) == FIXTURE["unavailable"]["template"].format(tail=UNAVAILABLE_TAIL, **example)
        assert unavailable_message({"reason": example["reason"], "fix": ""}) == FIXTURE["unavailable"]["template_no_fix"].format(reason=example["reason"], tail=UNAVAILABLE_TAIL)

    def test_doctor_and_the_chat_say_the_same_fix(self):
        assert unavailable_fix({"fix": "`sudo apt-get install socat`"}) == FIXTURE["doctor"]["fix"].format(fix="`sudo apt-get install socat`")
        assert unavailable_fix({}) == FIXTURE["doctor"]["fix_no_fix"]
        assert CHAT_FIXTURE["words"]["sandboxFix"] == FIXTURE["doctor"]["fix"]
        from webagents.cli.repl.chat_words import CHAT_WORDS

        assert CHAT_WORDS["sandboxFix"] == FIXTURE["doctor"]["fix"]


class TestTheMachine:
    @pytest.mark.parametrize("case", FIXTURE["install"]["cases"])
    def test_the_install_line_is_this_distributions(self, case):
        assert engine.linux_install_fix(case["os_release"], case["missing"]) == case["fix"]

    def test_the_packages_are_the_fixtures(self):
        assert engine.LINUX_PACKAGES == FIXTURE["install"]["packages"]

    def test_a_container_is_recognised(self, tmp_path):
        assert engine.in_container(str(tmp_path), environ={}) is False
        (tmp_path / ".dockerenv").write_text("")
        assert engine.in_container(str(tmp_path), environ={}) is True
        other = tmp_path / "k8s"
        (other / "proc" / "1").mkdir(parents=True)
        (other / "proc" / "1" / "cgroup").write_text("0::/kubepods/besteffort/pod1234\n")
        assert engine.in_container(str(other), environ={}) is True
        assert engine.in_container(str(tmp_path / "none"), environ={"container": "podman"}) is True

    @pytest.mark.parametrize("check", FIXTURE["namespaces"]["checks"])
    def test_a_namespace_restriction_names_its_fix(self, tmp_path, check):
        target = tmp_path / "proc" / "sys" / check["file"]
        target.parent.mkdir(parents=True)
        target.write_text(check["restricts_when"] + "\n")
        where, value, fix = engine.namespace_restriction(str(tmp_path))
        assert (where, value, fix) == (f"/proc/sys/{check['file']}", check["restricts_when"], FIXTURE["fixes"][check["fix"]])

    def test_no_restriction_says_nothing(self, tmp_path):
        target = tmp_path / "proc" / "sys" / "kernel" / "apparmor_restrict_unprivileged_userns"
        target.parent.mkdir(parents=True)
        target.write_text("0\n")
        assert engine.namespace_restriction(str(tmp_path)) == ("", "", "")

    def test_a_long_tmpdir_is_caught(self, tmp_path):
        assert engine.tmpdir_too_long("/tmp/wa") is False
        deep = tmp_path / ("x" * 60)
        deep.mkdir()
        assert engine.tmpdir_too_long(str(deep)) is True
        # The path srt builds, measured (the ptypass-fixes lane, 2026-09-27;
        # fixture `tmpdir`): the preflight folder as spelled, 5-digit pid.
        assert engine.srt_socket_probe("/tmp/wa") == "/tmp/wa/webagents-srt-preflight-XXXXXXXX/" + FIXTURE["tmpdir"]["socket"]
        for case in FIXTURE["tmpdir"]["cases"]:
            assert engine.tmpdir_too_long(case["tmpdir"]) is case["too_long"], case["name"]


class TestTheRefusalOnEachMachine:
    def test_native_windows_points_at_wsl(self, fresh):
        fresh.setattr(engine.platform, "system", lambda: "Windows")
        status = backend_status()
        assert (status["available"], status["reason"], status["fix"]) == (False, FIXTURE["reasons"]["windows"], FIXTURE["fixes"]["windows"])
        assert unavailable_message(status).startswith(f"{FIXTURE['reasons']['windows']}: {FIXTURE['fixes']['windows']}. ")
        checks = engine.setup_checks()
        assert [c["name"] for c in checks] == ["platform", "confined"]
        assert checks[0] == {"name": "platform", "status": "fail", "detail": FIXTURE["reasons"]["windows"], "fix": FIXTURE["fixes"]["windows"]}
        assert checks[1]["detail"] == FIXTURE["setup"]["not_run"]

    def test_another_platform_says_which(self, fresh):
        fresh.setattr(engine.platform, "system", lambda: "FreeBSD")
        status = backend_status()
        assert status["reason"] == FIXTURE["reasons"]["platform"].format(platform="FreeBSD")
        assert status["fix"] == FIXTURE["fixes"]["platform"]

    def test_linux_without_the_programs_names_them_and_the_install_line(self, fresh):
        fresh.setattr(engine.platform, "system", lambda: "Linux")
        fresh.setattr(engine, "_linux_programs", lambda: ({"rg": "/usr/bin/rg"}, ["bwrap", "socat"]))
        fresh.setattr(engine, "_os_release", lambda root="/": "ID=ubuntu\nID_LIKE=debian\nPRETTY_NAME=\"Ubuntu 24.04 LTS\"\n")
        status = backend_status()
        assert status["reason"] == FIXTURE["reasons"]["programs"].format(programs="bwrap, socat")
        assert status["fix"] == "`sudo apt-get install bubblewrap socat`"
        message = unavailable_message(status)
        assert message == FIXTURE["unavailable"]["template"].format(reason=status["reason"], fix=status["fix"], tail=UNAVAILABLE_TAIL)
        checks = {c["name"]: c for c in engine.setup_checks()}
        assert checks["platform"]["detail"] == "Linux (Ubuntu 24.04 LTS)"
        assert checks["programs"] == {"name": "programs", "status": "fail", "detail": status["reason"], "fix": status["fix"]}
        assert checks["confined"] == {"name": "confined", "status": "fail", "detail": FIXTURE["setup"]["not_run"]}

    def test_a_failed_confined_true_in_a_container_says_what_it_must_allow(self, fresh):
        fresh.setattr(engine.platform, "system", lambda: "Linux")
        fresh.setattr(engine, "_linux_programs", lambda: ({"bwrap": "/usr/bin/bwrap", "socat": "/usr/bin/socat", "rg": "/usr/bin/rg"}, []))
        fresh.setattr(engine, "_os_release", lambda root="/": "ID=debian\n")
        fresh.setattr(engine, "_preflight", lambda location, deps: "srt cannot start a sandbox here: bwrap: No permissions to create a new namespace")
        fresh.setattr(engine, "in_container", lambda root="/", environ=None: True)
        fresh.setattr(engine, "namespace_restriction", lambda root="/": ("", "", ""))
        status = backend_status()
        assert status["fix"] == FIXTURE["fixes"]["container"]
        checks = engine.setup_checks()
        names = [c["name"] for c in checks]
        assert names == [n for n in FIXTURE["setup"]["checks"] if n in names] and names[-1] == "confined"
        by_name = {c["name"]: c for c in checks}
        assert by_name["container"] == {"name": "container", "status": "fail", "detail": FIXTURE["setup"]["fail"]["container"], "fix": FIXTURE["fixes"]["container"]}
        assert by_name["confined"]["detail"] == FIXTURE["setup"]["fail"]["confined"].format(reason=status["reason"])

    def test_a_namespace_restriction_is_the_fix_outside_a_container(self, fresh):
        fresh.setattr(engine.platform, "system", lambda: "Linux")
        fresh.setattr(engine, "_linux_programs", lambda: ({"bwrap": "/usr/bin/bwrap", "socat": "/usr/bin/socat", "rg": "/usr/bin/rg"}, []))
        fresh.setattr(engine, "_preflight", lambda location, deps: "srt cannot start a sandbox here: bwrap: setting up uid map: Permission denied")
        fresh.setattr(engine, "in_container", lambda root="/", environ=None: False)
        fresh.setattr(engine, "namespace_restriction", lambda root="/": ("/proc/sys/kernel/apparmor_restrict_unprivileged_userns", "1", FIXTURE["fixes"]["userns_apparmor"]))
        assert backend_status()["fix"] == FIXTURE["fixes"]["userns_apparmor"]
        by_name = {c["name"]: c for c in engine.setup_checks()}
        assert by_name["namespaces"]["detail"] == FIXTURE["setup"]["fail"]["namespaces"].format(file="/proc/sys/kernel/apparmor_restrict_unprivileged_userns", value="1")

    def test_a_long_macos_tmpdir_is_the_fix(self, fresh):
        fresh.setattr(engine.platform, "system", lambda: "Darwin")
        fresh.setattr(engine, "_preflight", lambda location, deps: "srt cannot start a sandbox here: listen EINVAL")
        fresh.setattr(engine, "tmpdir_too_long", lambda tmpdir=None: True)
        assert backend_status()["fix"] == FIXTURE["fixes"]["tmpdir"]
        by_name = {c["name"]: c for c in engine.setup_checks()}
        assert by_name["tmpdir"]["status"] == "fail" and by_name["tmpdir"]["fix"] == FIXTURE["fixes"]["tmpdir"]

    def test_a_shell_command_gets_the_machine_specific_refusal(self, fresh, tmp_path):
        import asyncio

        from webagents.agents.skills.local.shell.skill import ShellSkill

        fresh.setattr(engine.platform, "system", lambda: "Linux")
        fresh.setattr(engine, "_linux_programs", lambda: ({}, ["bwrap", "socat", "rg"]))
        fresh.setattr(engine, "_os_release", lambda root="/": "ID=fedora\n")
        out = asyncio.run(ShellSkill({"base_dir": str(tmp_path), "env": {}}).run_command("echo hi"))
        reason = FIXTURE["reasons"]["programs"].format(programs="bwrap, socat, rg")
        assert out == "Access denied: " + FIXTURE["unavailable"]["template"].format(reason=reason, fix="`sudo dnf install bubblewrap socat ripgrep`", tail=UNAVAILABLE_TAIL)


class TestTheCommand:
    def test_it_is_a_group_with_setup(self):
        from webagents.cli.main import COMMAND_ORDER, app

        assert "sandbox" in COMMAND_ORDER
        result = CliRunner().invoke(app, ["sandbox", "-h"])
        assert result.exit_code == 0
        assert FIXTURE["setup"]["description"] in result.output

    def test_the_checks_come_in_the_fixtures_order(self, fresh):
        names = [c["name"] for c in engine.setup_checks()]
        assert names == [n for n in FIXTURE["setup"]["checks"] if n in names]
        assert names[0] == "platform" and names[-1] == "confined"

    def test_a_named_install_that_is_missing_fails_the_command(self, fresh, tmp_path):
        from webagents.cli.main import app

        fresh.setenv(engine.ENV_CLI, str(tmp_path / "nowhere.js"))
        result = CliRunner().invoke(app, ["sandbox", "setup"])
        assert result.exit_code == 1
        assert FIXTURE["fixes"]["engine_env"].format(version=engine.SRT_VERSION) in result.output
        assert FIXTURE["setup"]["not_run"] in result.output

    def test_json_is_doctors_document(self, fresh, tmp_path):
        from webagents.cli.main import app

        fresh.setenv(engine.ENV_CLI, str(tmp_path / "nowhere.js"))
        result = CliRunner().invoke(app, ["--json", "sandbox", "setup"])
        document = json.loads(result.output)
        checks = document["data"]["checks"]
        assert [c["name"] for c in checks][-1] == "confined"
        assert all(set(c) <= {"name", "status", "detail", "fix"} for c in checks)

    @requires_backend
    def test_here_it_runs_a_confined_true(self, fresh, tmp_path):
        # From a folder that holds no part of the install: run from this
        # checkout, whose `.venv` and `webagents/` are the SDK's own install,
        # `install` warns (S-316, the ptypass-fixes lane, 2026-09-27).
        fresh.chdir(tmp_path)
        checks = engine.setup_checks()
        assert checks[-1] == {"name": "confined", "status": "ok", "detail": FIXTURE["setup"]["ok"]["confined"]}
        engine_check = next(c for c in checks if c["name"] == "engine")
        assert engine_check["detail"] == FIXTURE["setup"]["ok"]["engine"].format(version=engine.SRT_VERSION, **{"from": FIXTURE["engine"]["python"]["cli_from"]["bundled"]})
        assert all(c["status"] == "ok" for c in checks)
