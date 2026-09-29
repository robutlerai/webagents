"""
The granular `sandbox:` shape, its defaults and aliases, the host groups, the
built-in denies as settings, the state strings, the refusal hints and srt's
refused-host log line (the sandbox-default lane, 2026-09-27), against the
shared fixture `tests/fixtures/sandbox/srt.json` that
`typescript/tests/unit/sandbox/sandbox-default-shape.test.ts` reads too.
Nothing here runs srt; the real-srt proofs are in
`test_sandbox_default_denies.py`.
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import pytest

from webagents.agents.skills.local.shell.skill import NO_SANDBOX_WARNING, UNRESTRICTED_WARNING, ShellSkill
from webagents.cli.loader.schema import AgentFormatError, AgentMetadata, SandboxConfig
from webagents.sandbox import (
    CREDENTIAL_DIRS,
    HOME_ENV_DENY,
    HOST_GROUPS,
    PROFILE_DIR_PATTERN,
    REFUSAL_HINTS,
    ROOT_READ_DENY,
    ROOT_READ_DENY_PATTERNS,
    SANDBOX_ACCEPTED_KEYS,
    SANDBOX_FILES_KEYS,
    SANDBOX_KEYS,
    SANDBOX_NETWORK_KEYS,
    UNAVAILABLE_TAIL,
    SandboxDeclarationError,
    env_from_dotenv,
    expand_hosts,
    home_env_denies,
    is_sandbox_off,
    no_sandbox_requested,
    normalize_declaration,
    policy_from_metadata,
    profile_dir_denies,
    refusal_kind,
    refused_hosts_from_srt_log,
    root_read_denies,
    sandbox_state,
    walk_home_env_files,
)
from webagents.sandbox import srt as engine

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "sandbox" / "srt.json").read_text())


class TestTheShape:
    def test_the_keys_aliases_and_nested_keys(self):
        assert list(SANDBOX_KEYS) == FIXTURE["schema"]["keys"]
        assert sorted(SandboxConfig.model_fields) == FIXTURE["schema"]["keys"]
        assert list(SANDBOX_ACCEPTED_KEYS) == FIXTURE["schema"]["accepted_keys"]
        assert list(SANDBOX_FILES_KEYS) == FIXTURE["schema"]["files_keys"]
        assert list(SANDBOX_NETWORK_KEYS) == FIXTURE["schema"]["network_keys"]
        assert {k: list(v) for k, v in HOST_GROUPS.items()} == {k: v for k, v in FIXTURE["host_groups"].items() if k != "about"}

    def test_the_defaults_aliases_and_opt_out_normalise_as_the_fixture_pins_them(self, tmp_path):
        assert normalize_declaration({}) == FIXTURE["schema"]["defaults"]["declaration"]
        assert normalize_declaration({"preset": "strict"})["files"]["read"] == FIXTURE["schema"]["defaults"]["strict_read"]
        for case in FIXTURE["schema"]["alias_cases"]:
            assert normalize_declaration(case["declared"]) == case["normalised"], case
            # The loader's model normalises the same way, and dumps the same shape.
            assert AgentMetadata(name="a", sandbox=case["declared"]).sandbox.model_dump() == case["normalised"], case
        assert is_sandbox_off(False) and is_sandbox_off("off") and is_sandbox_off("OFF")
        assert not (is_sandbox_off({}) or is_sandbox_off("on") or is_sandbox_off(True))
        assert normalize_declaration(True) == FIXTURE["schema"]["defaults"]["declaration"]
        assert normalize_declaration("on") == FIXTURE["schema"]["defaults"]["declaration"]
        assert policy_from_metadata("off", cwd=str(tmp_path)).confined is False
        # PyYAML reads `sandbox: off` as False; the loader takes it as the opt-out.
        assert AgentMetadata(name="a", sandbox=False).sandbox.preset == "unrestricted"

    def test_an_unknown_key_at_any_level_is_refused_with_the_fixture_sentence(self):
        for case in FIXTURE["unknown_key"]["cases"]:
            with pytest.raises(SandboxDeclarationError) as caught:
                normalize_declaration(case["declared"])
            assert str(caught.value) == case["message"]
            with pytest.raises(AgentFormatError) as loaded:
                AgentMetadata(name="a", sandbox=case["declared"])
            assert str(loaded.value) == case["message"]
        with pytest.raises(SandboxDeclarationError, match="old spelling of files.write"):
            normalize_declaration({"allowed_folders": ["."], "files": {"write": ["."]}})
        with pytest.raises(SandboxDeclarationError, match="old spelling of env"):
            normalize_declaration({"env_passthrough": ["A"], "env": ["B"]})
        with pytest.raises(SandboxDeclarationError, match="network.local must be true or false"):
            normalize_declaration({"network": {"local": "yes"}})
        with pytest.raises(SandboxDeclarationError):
            normalize_declaration([])

    def test_host_groups_expand_to_exactly_the_fixture_hosts(self, tmp_path):
        for group, hosts in HOST_GROUPS.items():
            assert expand_hosts([group]) == list(hosts)
        assert expand_hosts(["npm", "example.com", "registry.npmjs.org"]) == ["registry.npmjs.org", "example.com"]
        with pytest.raises(ValueError, match="network entry"):
            expand_hosts(["*"])
        policy = policy_from_metadata({"network": {"hosts": ["github", "pypi"]}}, cwd=str(tmp_path))
        assert policy.network_domains == [*HOST_GROUPS["github"], *HOST_GROUPS["pypi"]]


class TestTheBuiltInDenies:
    def test_the_lists_and_the_enumeration_on_linux(self, tmp_path):
        assert list(CREDENTIAL_DIRS) == FIXTURE["credential_dirs_unreadable_under_development"]
        assert PROFILE_DIR_PATTERN == FIXTURE["builtin_denies"]["profile_dirs"]["pattern"]
        assert list(ROOT_READ_DENY) == FIXTURE["builtin_denies"]["root_read_deny"]["literal"]
        assert list(ROOT_READ_DENY_PATTERNS) == FIXTURE["builtin_denies"]["root_read_deny"]["patterns"]
        home = tmp_path / "home"
        (home / ".webagents-local").mkdir(parents=True)
        (home / ".webagents-team").mkdir()
        (home / ".webagents-note").write_text("a file, not a profile")
        assert profile_dir_denies(str(home), "Darwin") == [str(home / ".webagents-*")]
        assert profile_dir_denies(str(home), "Linux") == [str(home / ".webagents-local"), str(home / ".webagents-team")]
        root = tmp_path / "root"
        root.mkdir()
        (root / ".env.local").write_text("A=1")
        (root / ".envelope").write_text("not env")
        assert root_read_denies(str(root), "Darwin") == [str(root / ".env"), str(root / ".webagents"), str(root / ".env.*")]
        assert root_read_denies(str(root), "Linux") == [str(root / ".env.local")]
        (root / ".env").write_text("B=2")
        assert root_read_denies(str(root), "Linux") == [str(root / ".env"), str(root / ".env.local")]

    def test_every_env_under_home_is_denied_globs_on_macos_a_bounded_cached_walk_on_linux(self, tmp_path):
        """S-343 (2026-09-29). The TypeScript twin is the same test in
        `sandbox-default-shape.test.ts`."""
        from webagents.sandbox.policy import (
            HOME_ENV_WALK_BUDGET,
            HOME_ENV_WALK_CACHE_SECONDS,
            HOME_ENV_WALK_DEPTH,
            HOME_ENV_WALK_SKIP,
        )

        pinned = FIXTURE["builtin_denies"]["home_env_deny"]
        assert list(HOME_ENV_DENY) == pinned["globs"]
        assert HOME_ENV_WALK_DEPTH == pinned["linux"]["depth"]
        assert HOME_ENV_WALK_BUDGET == pinned["linux"]["budget"]
        assert HOME_ENV_WALK_CACHE_SECONDS == pinned["linux"]["cache_seconds"]
        assert list(HOME_ENV_WALK_SKIP) == pinned["linux"]["skip"]

        home = Path(os.path.realpath(tmp_path)) / "home"

        def file(relative):
            path = home / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("SECRET=1")
            return str(path)

        found = [
            file(".env"),
            file("dev/project/.env"),
            file("dev/project/.env.local"),
            file("a/b/c/d/.env"),  # four levels down: the deepest the walk lists
        ]
        file("dev/project/.envelope")  # not a .env file
        file("a/b/c/d/e/.env")  # five levels down: past the walk
        file("dev/project/node_modules/pkg/.env")  # inside a skipped folder
        file("Library/Application Support/x/.env")
        (home / "link-to-dev").symlink_to(home / "dev")  # links to folders are not followed

        assert home_env_denies(str(home), "Darwin") == [str(home / "**/.env"), str(home / "**/.env.*")]
        assert walk_home_env_files(str(home)) == sorted(found)
        # Reused for a minute, then walked again.
        now = 1_000_000.0
        assert home_env_denies(str(home), "Linux", now) == sorted(found)
        later = file("dev/second/.env")
        assert later not in home_env_denies(str(home), "Linux", now + 1)
        assert later in home_env_denies(str(home), "Linux", now + HOME_ENV_WALK_CACHE_SECONDS + 1)

    def test_a_development_policy_gets_the_wider_list_and_the_home_env_denies_and_strict_does_not(self, tmp_path, monkeypatch):
        home = Path(os.path.realpath(tmp_path)) / "home"
        work = home / "work" / "agent"
        work.mkdir(parents=True)
        (home / "work" / "other").mkdir(parents=True)
        (home / "work" / "other" / ".env").write_text("K=1")
        monkeypatch.setenv("HOME", str(home))
        development = policy_from_metadata({}, cwd=str(work))
        for system in ("Darwin", "Linux"):
            monkeypatch.setattr(platform, "system", lambda system=system: system)
            denied = development.deny_reads
            for relative in (".config/gh", ".git-credentials", ".zsh_history", ".codex/auth.json", ".claude/projects"):
                assert str(home / relative) in denied, (system, relative)
            # Another program's folder stays readable where it holds more than secrets.
            assert str(home / ".claude") not in denied and str(home / ".cargo") not in denied
            if system == "Darwin":
                assert str(home / "**/.env") in denied
            else:
                assert str(home / "work" / "other" / ".env") in denied
        strict = policy_from_metadata({"preset": "strict"}, cwd=str(work))
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        assert strict.deny_reads[0] == "/" and str(home / "**/.env") not in strict.deny_reads

    def test_the_granular_keys_reach_the_settings(self, tmp_path, monkeypatch):
        home, work, cwd = tmp_path / "home", tmp_path / "work", tmp_path / "cwd"
        for folder in (home, work, cwd, home / "notes"):
            folder.mkdir()
        monkeypatch.setenv("HOME", str(home))
        granular = FIXTURE["settings_cases_granular"]

        def sub(value):
            return json.loads(
                json.dumps(value)
                .replace("{work}", os.path.realpath(work))
                .replace("{home}", os.path.realpath(home))
                .replace("{cwd}", os.path.realpath(cwd))
            )

        policy = policy_from_metadata(sub(granular["declared"]), cwd=str(cwd), tmpdir=str(tmp_path / "tmp"))
        settings = engine.build_settings(policy, deps={})

        def fill(value):
            return json.loads(json.dumps(sub(value)).replace("{scratch}", policy.scratch))

        def check(expectations):
            for key, expected in expectations.items():
                section, _, field = key.partition(".")
                includes = field.endswith("_includes")
                actual = settings[section][field[: -len("_includes")] if includes else field]
                if includes:
                    assert all(item in actual for item in fill(expected)), (key, actual)
                else:
                    assert actual == fill(expected), (key, actual)

        check(granular["expect"])
        if platform.system() == "Darwin":
            check(granular["darwin"])
            default = engine.build_settings(policy_from_metadata({}, cwd=str(cwd), tmpdir=str(tmp_path / "tmp")), deps={})
            for entry in fill(granular["default_denies_darwin"]):
                assert entry in default["filesystem"]["denyRead"], entry
        else:
            assert "network.sockets" in policy.unenforceable
            assert "allowUnixSockets" not in settings["network"]
        for key in FIXTURE["never_set"]:
            assert key not in settings["network"] and key not in settings["filesystem"]


class TestTheStateTheOptOutsAndTheFailClosedSentence:
    def test_the_state_is_spelled_from_the_origin(self, tmp_path, capsys):
        status = FIXTURE["status"]
        by_default = ShellSkill({"base_dir": str(tmp_path), "env": {}})
        assert by_default.sandbox_state_line() == status["default"]
        assert sandbox_state(by_default.policy, "default") == status["default"]
        declared = ShellSkill({"base_dir": str(tmp_path), "sandbox": {"preset": "strict"}, "env": {}})
        assert declared.sandbox_state_line() == status["agent_file"].replace("{preset}", "strict")
        off = ShellSkill({"base_dir": str(tmp_path), "sandbox": False, "env": {}})
        assert off.sandbox_state_line() == status["off_agent_file"]
        flag = ShellSkill({"base_dir": str(tmp_path), "sandbox": {"preset": "strict"}, "env": {status["env_flag"]: "1"}})
        assert flag.sandbox_state_line() == status["off_flag"] and flag.sandbox_origin == "--no-sandbox"
        assert UNRESTRICTED_WARNING == status["warnings"]["off_agent_file"] == FIXTURE["unrestricted"]["warning"]
        assert NO_SANDBOX_WARNING == status["warnings"]["off_flag"]
        err = capsys.readouterr().err
        assert UNRESTRICTED_WARNING in err and NO_SANDBOX_WARNING in err
        assert no_sandbox_requested({"WEBAGENTS_NO_SANDBOX": "yes"}) and no_sandbox_requested({"WEBAGENTS_NO_SANDBOX": "1"})
        assert not (no_sandbox_requested({"WEBAGENTS_NO_SANDBOX": "0"}) or no_sandbox_requested({}))

    def test_the_fail_closed_sentence_names_the_opt_out(self):
        assert UNAVAILABLE_TAIL == FIXTURE["refusals"]["unavailable_tail"]
        # The sandbox-engine lane (2026-09-27): the reason, this machine's fix, then the tail.
        assert FIXTURE["refusals"]["unavailable"] == f"{{reason}}: {{fix}}. {UNAVAILABLE_TAIL}"


class TestTheHintsAndTheRefusedHostLog:
    def test_the_sentences_and_the_fixture_outputs(self):
        assert REFUSAL_HINTS == FIXTURE["hints"]["sentences"]
        for case in FIXTURE["hints"]["cases"]:
            assert refusal_kind(case["command"], case["output"]) == case["kind"], case

    def test_refused_hosts_come_from_srts_own_lines_only(self):
        for case in FIXTURE["refused_hosts"]["cases"]:
            assert refused_hosts_from_srt_log(case["stderr"]) == case["hosts"]

    def test_the_wrapper_has_the_node_switch_and_merges_stderr_when_capturing(self):
        wrapped = FIXTURE["no_proxy"]["wrapped"]
        assert engine.wrapped_command(wrapped["command"], wrapped["path"], wrapped["network"]) == wrapped["text"]
        assert engine.wrapped_command(wrapped["command"], wrapped["path"], wrapped["network"], True) == wrapped["merged_text"]

    def test_a_listed_env_name_comes_from_the_dotenv_files_and_nothing_else(self, tmp_path, monkeypatch):
        cwd = tmp_path / "cwd"
        home = tmp_path / "home"
        (home / ".webagents-local").mkdir(parents=True)
        cwd.mkdir()
        (cwd / ".env").write_text('# keys\nPROBE_API_KEY="from-project"\nPLAIN=plain\nOTHER_TOKEN=other\n')
        (home / ".webagents-local" / ".env").write_text("PROFILE_SECRET=from-profile\nPROBE_API_KEY=shadowed\n")
        monkeypatch.setenv("HOME", str(home))
        env = {"WEBAGENTS_PROFILE": "local", "OTHER_TOKEN": "from-process"}
        assert env_from_dotenv(["PROBE_API_KEY", "PROFILE_SECRET", "OTHER_TOKEN", "MISSING"], str(cwd), env) == {
            "PROBE_API_KEY": "from-project",
            "PROFILE_SECRET": "from-profile",
        }
        assert env_from_dotenv([], str(cwd), env) == {}
        assert env_from_dotenv(["probe_api_key"], str(cwd), env) == {"probe_api_key": "from-project"}
