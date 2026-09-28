"""
S-217: `sandbox:` is enforced by the kernel, not by reading the command.

The defect these pin: `sandbox:` was parsed, validated, merged through the
context hierarchy and applied to NOTHING, while `ShellSkill`'s own allow-list
looked like it was doing the work. An allow-list cannot do that work, and the
first two tests here demonstrate why before the rest assert the replacement.

Every enforcement test is skipped where no backend exists, because a test that
silently passes on a machine that cannot enforce anything would be the same
class of lie as the bug.

THE ENGINE IS srt SINCE 2026-09-26 (`webagents/sandbox/srt.py`; found through
`WEBAGENTS_SRT_CLI`, see `conftest.py`). The enforcement cases below are the
same, because what a sandbox must stop did not change; the engine's own
contract (settings file, environment, timeouts, locating, versions, the
`network:` list) is in `test_srt_engine_w1_sandbox.py`, against the shared
fixture `tests/fixtures/sandbox/srt.json` that the TypeScript suite reads too.
"""

import asyncio
import base64
import os
import platform
import subprocess

import pytest

from webagents.sandbox import (
    SandboxPolicy,
    SandboxUnavailable,
    backend_status,
    policy_from_metadata,
    run_sandboxed,
    sandbox_available,
)

requires_backend = pytest.mark.skipif(
    not sandbox_available(),
    reason=f"no OS sandbox backend here: {backend_status()['reason']}",
)


@pytest.fixture
def tree(tmp_path):
    """A working directory, and a secret OUTSIDE it."""
    work = tmp_path / "work"
    secrets = tmp_path / "secrets"
    work.mkdir()
    secrets.mkdir()
    (secrets / "token").write_text("TOP-SECRET\n")
    (work / ".git" / "hooks").mkdir(parents=True)
    return work, secrets


def strict(work):
    return policy_from_metadata(
        {"preset": "strict", "allowed_folders": [str(work)]}, cwd=str(work)
    )


class TestWhyAnAllowListIsNotABoundary:
    """Not a test of our code. A test of the premise our code rests on."""

    def test_shlex_lets_a_substitution_through(self):
        import shlex

        # `shlex` treats `$(...)` as a literal token, so an argv[0] allow-list
        # sees `echo` and approves. `shell=True` expands it afterwards.
        parts = shlex.split("echo $(id -un)")
        assert parts[0] == "echo"

        ran = subprocess.run(
            "echo $(id -un)", shell=True, capture_output=True, text=True
        ).stdout.strip()
        assert ran and ran != "$(id -un)", "the shell expanded what the guard did not see"

    def test_the_guard_and_the_kernel_disagree(self):
        import shlex

        for command in ("echo `id -un`", "echo x && echo CHAINED"):
            assert shlex.split(command)[0] == "echo"  # approved
            out = subprocess.run(
                command, shell=True, capture_output=True, text=True
            ).stdout
            assert "echo" not in out or "CHAINED" in out  # something else ran


@requires_backend
class TestTheKernelStopsWhatTheAllowListLetThrough:
    def test_a_legitimate_write_inside_the_root_succeeds(self, tree):
        work, _ = tree
        result = run_sandboxed(f"echo ok > {work}/f && cat {work}/f", strict(work), timeout=20)
        assert result.stdout.strip() == "ok"

    def test_a_write_outside_the_root_is_refused(self, tree):
        work, secrets = tree
        result = run_sandboxed(
            f"echo pwned > {secrets}/escaped && echo WROTE", strict(work), timeout=20
        )
        assert "WROTE" not in result.stdout
        assert not (secrets / "escaped").exists()

    def test_a_read_through_command_substitution_is_refused(self, tree):
        work, secrets = tree
        result = run_sandboxed(f"cat $(echo {secrets}/token)", strict(work), timeout=20)
        assert "TOP-SECRET" not in result.stdout

    def test_a_base64_pipeline_is_refused(self, tree):
        work, secrets = tree
        payload = base64.b64encode(f"cat {secrets}/token".encode()).decode()
        result = run_sandboxed(
            f"echo {payload} | base64 -d | sh", strict(work), timeout=20
        )
        assert "TOP-SECRET" not in result.stdout

    def test_a_python_subprocess_is_refused(self, tree):
        # The restriction is INHERITED, which is what makes it hold for
        # `python -c` and for anything else the command spawns.
        work, secrets = tree
        result = run_sandboxed(
            f"python3 -c \"print(open('{secrets}/token').read())\"", strict(work), timeout=30
        )
        assert "TOP-SECRET" not in result.stdout

    def test_the_escalation_set_stays_unwritable_inside_the_root(self, tree):
        # A command that can write `.git/hooks` grants itself the NEXT command.
        work, _ = tree
        result = run_sandboxed(
            f"echo evil > {work}/.git/hooks/pre-commit && echo WROTE", strict(work), timeout=20
        )
        assert "WROTE" not in result.stdout
        assert not (work / ".git" / "hooks" / "pre-commit").exists()

    def test_network_is_refused_at_the_socket_layer(self, tree):
        # By IP, not by hostname: this must not be passing because DNS failed.
        # `-f`: srt's proxy answers a 403 for a host outside the allowlist,
        # which is a successful HTTP exchange to curl without it.
        work, _ = tree
        result = run_sandboxed("curl -sf -m 4 http://1.1.1.1 && echo NET", strict(work), timeout=20)
        assert "NET" not in result.stdout

    @pytest.mark.skipif(platform.system() != "Darwin", reason="Seatbelt-specific")
    def test_the_sandbox_cannot_widen_itself(self, tree):
        work, secrets = tree
        result = run_sandboxed(
            f"/usr/bin/sandbox-exec -p '(version 1)(allow default)' -- cat {secrets}/token",
            strict(work),
            timeout=20,
        )
        assert "TOP-SECRET" not in result.stdout


@requires_backend
class TestPresets:
    def test_development_confines_writes_but_not_reads(self, tree):
        # The documented asymmetry: scoping reads is expensive and opt-in.
        work, secrets = tree
        policy = policy_from_metadata(
            {"preset": "development", "allowed_folders": [str(work)]}, cwd=str(work)
        )
        assert not policy.scoped_reads

        denied = run_sandboxed(f"echo x > {secrets}/nope && echo WROTE", policy, timeout=20)
        assert "WROTE" not in denied.stdout

    def test_unrestricted_allows_network(self, tree):
        # And confines nothing: srt has no allow-all network entry, so the
        # preset is an explicit opt-out (fixture `unrestricted.decision`).
        work, secrets = tree
        policy = policy_from_metadata(
            {"preset": "unrestricted", "allowed_folders": [str(work)]}, cwd=str(work)
        )
        assert policy.network is True
        assert policy.confined is False
        assert run_sandboxed(f"cat {secrets}/token", policy, timeout=20).stdout.strip() == "TOP-SECRET"


class TestThePolicyItself:
    def test_no_declaration_is_not_an_empty_policy(self):
        # "nothing was asked for" and "deny everything" are different answers.
        assert policy_from_metadata(None) is None

    def test_an_unknown_preset_is_refused(self):
        with pytest.raises(ValueError, match="unknown sandbox preset"):
            policy_from_metadata({"preset": "paranoid"})

    def test_paths_are_realpathed(self, tmp_path):
        # `/tmp`, `/etc` and `/var` are symlinks into `/private` on macOS, so a
        # rule written against the symlink matches nothing.
        policy = policy_from_metadata({"allowed_folders": ["/tmp"]}, cwd=str(tmp_path))
        assert os.path.realpath("/tmp") in policy.write_roots

    def test_the_scratch_directory_is_private_not_the_whole_tmpdir(self, tmp_path):
        # An earlier version allowed `$TMPDIR` ITSELF as a write root, so an
        # agent working anywhere under the temp area had its siblings made
        # writable and readable. Caught by the first run of these tests.
        #
        # The working directory is deliberately OUTSIDE the temp base here:
        # under `development` the cwd is legitimately writable, and mixing the
        # two is what made the first version of this test wrong.
        temp_base = tmp_path / "tmpbase"
        work = tmp_path / "elsewhere"
        temp_base.mkdir()
        work.mkdir()
        sibling = temp_base / "someone-elses-tempdir"
        sibling.mkdir()

        policy = policy_from_metadata(
            {"allowed_folders": []}, cwd=str(work), tmpdir=str(temp_base)
        )

        assert str(temp_base.resolve()) not in policy.write_roots
        assert str(sibling.resolve()) not in policy.write_roots
        # Only OUR subdirectory of it.
        assert policy.scratch == str((temp_base / "webagents-sandbox").resolve())
        assert policy.scratch in policy.write_roots

    def test_unenforceable_subkeys_are_reported(self):
        policy = policy_from_metadata({"allowed_imports": ["os"]})
        assert "allowed_imports" in policy.unenforceable

    def test_the_escalation_set_is_derived_per_root(self):
        policy = SandboxPolicy(write_roots=["/a", "/b"])
        denied = policy.deny_writes
        assert "/a/.git/hooks" in denied and "/b/.git/hooks" in denied


class TestItFailsClosed:
    def test_an_unavailable_backend_refuses_rather_than_degrades(self, monkeypatch, tmp_path):
        # Claude Code's default is to warn and run unsandboxed. For a DECLARED
        # restriction that is the defect, not the fallback.
        import webagents.sandbox.runner as runner

        monkeypatch.setattr(
            runner, "backend_status",
            lambda: {"available": False, "reason": "no backend here", "backend": None, "path": None},
        )
        with pytest.raises(SandboxUnavailable, match="was not run"):
            runner.run_sandboxed("echo hi", policy_from_metadata({}, cwd=str(tmp_path)))

    def test_the_backend_binary_is_pinned_by_absolute_path(self):
        # The process being confined is the one we distrust; resolving through
        # PATH would let it pick its own sandbox binary. srt itself runs with
        # a root-owned PATH, and node and cli.js are absolute.
        from webagents.sandbox.srt import ROOT_OWNED_BIN_DIRS, SRT_PATH

        assert all(entry.startswith("/") for entry in SRT_PATH.split(":"))
        assert all(entry.startswith("/") for entry in ROOT_OWNED_BIN_DIRS)
        status = backend_status()
        if status["available"]:
            assert os.path.isabs(str(status["node"])) and os.path.isabs(str(status["path"]))


class TestTheSkillUsesIt:
    def test_no_declaration_means_the_defaults(self, tmp_path):
        # On by default (2026-09-27): the defaults, from the agent's folder, no network.
        from webagents.agents.skills.local.shell.skill import ShellSkill

        skill = ShellSkill({"base_dir": str(tmp_path), "env": {}})
        assert skill.policy is not None and skill.policy.confined and skill.policy.preset == "development"
        assert skill.policy.network_domains == [] and skill.sandbox_origin == "default"
        assert skill.sandbox_state_line() == "development (default)"

    def test_a_declaration_produces_a_policy(self, tmp_path):
        from webagents.agents.skills.local.shell.skill import ShellSkill

        skill = ShellSkill(
            {"base_dir": str(tmp_path), "sandbox": {"preset": "strict", "allowed_folders": [str(tmp_path)]}}
        )
        assert skill.policy is not None and skill.policy.scoped_reads

    def test_a_malformed_declaration_refuses_instead_of_running_free(self, tmp_path):
        from webagents.agents.skills.local.shell.skill import ShellSkill

        skill = ShellSkill({"base_dir": str(tmp_path), "sandbox": {"preset": "nope"}})
        out = asyncio.run(skill.run_command("echo hi"))
        assert "Access denied" in out

    @requires_backend
    def test_the_skill_blocks_an_exfiltration(self, tree):
        from webagents.agents.skills.local.shell.skill import ShellSkill

        work, secrets = tree
        skill = ShellSkill(
            {"base_dir": str(work), "sandbox": {"preset": "strict", "allowed_folders": [str(work)]}}
        )
        out = asyncio.run(skill.run_command(f"cat $(echo {secrets}/token)", timeout=20))
        assert "TOP-SECRET" not in out

    def test_the_declaration_reaches_the_skill_from_the_agent_file(self, tmp_path):
        # The wiring is the whole fix: without it the key is parsed and
        # applied to nothing, which is S-217 exactly. One loader builds the
        # skills for the daemon and the chat (`cli/agent_builder.py`), so this
        # is checked by what the loader builds, not by reading its source.
        from webagents.cli.agent_builder import load_skills
        from webagents.cli.loader.schema import SandboxConfig

        skills = load_skills(
            ["shell"], agent_name="helper", agent_path=tmp_path / "AGENT.md", sandbox=SandboxConfig(preset="strict")
        )
        assert skills["shell"].policy is not None and skills["shell"].sandbox_origin == "agent file"
        # Without a declaration the loader hands nothing, and the shell takes the defaults.
        assert load_skills(["shell"], agent_name="helper", agent_path=tmp_path / "AGENT.md")["shell"].sandbox_origin == "default"

        from webagents.server.extensions import local_file_source

        # The daemon uses the same loader.
        daemon_skills = local_file_source.LocalFileSource._load_skills(
            None, ["shell"], "helper", tmp_path / "AGENT.md", SandboxConfig(preset="strict")
        )
        assert daemon_skills["shell"].policy is not None


class TestSandboxIsNoLongerReportedAsInert:
    def test_sandbox_left_the_inert_set(self):
        from webagents.cli.loader.schema import INERT_FIELDS

        assert "sandbox" not in INERT_FIELDS

    def test_but_allowed_imports_is_still_reported(self):
        from webagents.cli.loader.schema import INERT_SANDBOX_FIELDS

        assert "allowed_imports" in INERT_SANDBOX_FIELDS


class TestDoctorTellsTheTruthAboutIt:
    """`doctor`'s sandbox check, for an agent whose file declares one."""

    def _sandbox_check(self, tmp_path, monkeypatch):
        (tmp_path / "AGENT.md").write_text(
            "---\nname: a\nskills:\n  - shell\nsandbox:\n  preset: strict\n---\n\nBody.\n"
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
        from webagents.cli.doctor import run_checks

        return next(c for c in run_checks(tmp_path) if c.name == "sandbox")

    def test_the_backend_is_reported(self, tmp_path, monkeypatch):
        check = self._sandbox_check(tmp_path, monkeypatch)
        if sandbox_available():
            assert check.status == "ok"
            assert backend_status()["backend"] in check.detail
            assert "strict" in check.detail
        else:
            # A machine with no backend REFUSES sandboxed commands, so knowing
            # before you write the declaration is the point.
            assert check.fix

    def test_an_unavailable_backend_fails_with_the_consequence(self, tmp_path, monkeypatch):
        import webagents.sandbox as sandbox_pkg

        monkeypatch.setattr(
            sandbox_pkg, "backend_status",
            lambda: {"available": False, "reason": "nothing here", "backend": None, "path": None},
        )
        check = self._sandbox_check(tmp_path, monkeypatch)
        assert check.status == "fail"
        assert "shell commands are refused" in check.detail
        assert check.fix


class TestSecretsAreWithheld:
    """S-220. The first version confined files and network and then ran the
    command with the agent's ENTIRE environment, so under the strictest preset
    `echo $OPENAI_API_KEY` printed the key into the tool result."""

    def test_secret_looking_names_are_withheld(self):
        from webagents.sandbox.runner import scrub_environment

        kept, withheld = scrub_environment(
            {
                "OPENAI_API_KEY": "sk-x",
                "GH_TOKEN": "ghp-x",
                "DB_PASSWORD": "p",
                "AWS_SECRET_ACCESS_KEY": "s",
                "PATH": "/usr/bin",
                "HOME": "/home/x",
            }
        )
        assert set(kept) == {"PATH", "HOME"}
        assert set(withheld) == {"OPENAI_API_KEY", "GH_TOKEN", "DB_PASSWORD", "AWS_SECRET_ACCESS_KEY"}

    def test_an_explicit_passthrough_lets_exactly_that_one_through(self):
        from webagents.sandbox.runner import scrub_environment

        kept, _ = scrub_environment(
            {"GH_TOKEN": "ghp-x", "OPENAI_API_KEY": "sk-x"}, passthrough=["GH_TOKEN"]
        )
        assert kept == {"GH_TOKEN": "ghp-x"}

    def test_the_passthrough_list_reaches_the_policy(self):
        policy = policy_from_metadata({"env_passthrough": ["GH_TOKEN"]})
        assert policy.env_passthrough == ["GH_TOKEN"]

    def test_the_schema_accepts_it(self):
        # The loader rejects unknown keys, so a field the policy reads but the
        # schema refuses would be unreachable from an agent file.
        from webagents.cli.loader.schema import AgentMetadata

        meta = AgentMetadata(name="a", sandbox={"env_passthrough": ["GH_TOKEN"]})
        assert meta.sandbox.env_passthrough == ["GH_TOKEN"]

    @requires_backend
    def test_a_sandboxed_command_cannot_read_a_key(self, tree, monkeypatch):
        work, _ = tree
        monkeypatch.setenv("OPENAI_API_KEY", "sk-FAKE-canary-not-real")
        result = run_sandboxed('echo "k=[$OPENAI_API_KEY]"', strict(work), timeout=20)
        assert "sk-FAKE-canary" not in result.stdout
        assert "k=[]" in result.stdout
