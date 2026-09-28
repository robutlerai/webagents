"""
`webagents skills add <source>`, `skills remove` of an installed SKILL.md
skill, `skills list` and `doctor` for SKILL.md skills (gap-closure plan item
1.4, 2026-09-26), against `tests/fixtures/skillmd/skillmd.json` (`install`,
`doctor`), which the TypeScript suite reads too
(`tests/unit/cli/skillmd-install.test.ts`).

The source is a git repository the test makes from `fixtures/skillmd/repo`
and reaches over a `file://` URL: nothing here touches the internet. What is
proved: the sources a name can be, the sample skill installed from the
repository with its file list shown and the lock recording the commit and
the digest, `--skill`, `--yes` required without a terminal, the skipped
skill reported and not installed, the limits, a skill the lock does not
know left alone, removal, and the same skill then loading into an agent
built from the folder with the catalog and activation the fixture pins.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from webagents.agents.skills.local.skillmd import SKILL_KEY
from webagents.agents.skills.local.skillmd.skillmd_install import (
    BINARY_EXTENSIONS,
    DOWNLOAD_LIMIT,
    EXTRACTED_LIMIT,
    FILE_LIMIT,
    LOCK_ENTRY_KEYS,
    LOCK_FILE,
    LOCK_VERSION,
    PLUGIN_MANIFESTS,
    SCRIPT_DIRS,
    SCRIPT_EXTENSIONS,
    SEARCH_DEPTH,
    SEARCH_ROOTS,
    install_from_source,
    list_files,
    locate_skills,
    parse_source,
    read_lock,
    tree_digest,
)
from webagents.cli.skills_edit import skills_command

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "skillmd"
FIXTURE = json.loads((FIXTURES / "skillmd.json").read_text())
INSTALL = FIXTURE["install"]
MESSAGES = INSTALL["messages"]
REPO = FIXTURES / FIXTURE["sample"]["repo"]

GIT_ENV = dict(
    GIT_AUTHOR_NAME="fixture", GIT_AUTHOR_EMAIL="fixture@example.invalid",
    GIT_COMMITTER_NAME="fixture", GIT_COMMITTER_EMAIL="fixture@example.invalid",
    GIT_AUTHOR_DATE="2026-09-26T00:00:00Z", GIT_COMMITTER_DATE="2026-09-26T00:00:00Z",
)


@pytest.fixture(autouse=True)
def _no_profile(monkeypatch):
    monkeypatch.delenv("WEBAGENTS_PROFILE", raising=False)


@pytest.fixture
def repo(tmp_path):
    """The fixture repository as a git repository, and its file:// URL."""
    source = tmp_path / "src"
    shutil.copytree(REPO, source)
    env = {**os.environ, **GIT_ENV}
    for args in (["init", "-q", "-b", "main"], ["add", "."], ["commit", "-q", "-m", "fixture skills"]):
        subprocess.run(["git", *args], cwd=source, env=env, check=True, capture_output=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, capture_output=True, text=True, check=True).stdout.strip()
    return {"path": source, "url": f"file://{source}", "sha": sha}


@pytest.fixture
def folder(tmp_path):
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "AGENT.md").write_text("---\nname: bot\nskills:\n  - openai\nsandbox:\n  preset: development\n---\n\nBody.\n")
    return agent


def _run(action, names, folder, **kwargs):
    out, err = [], []
    code = skills_command(action, names, folder=folder, out=out.append, err=err.append, **kwargs)
    return code, out, err


class TestTheFixtureIsTheContract:
    def test_limits_search_rules_and_flags(self):
        limits = INSTALL["limits"]
        assert (DOWNLOAD_LIMIT, EXTRACTED_LIMIT, FILE_LIMIT) == (limits["download_bytes"], limits["extracted_bytes"], limits["files"])
        assert list(SEARCH_ROOTS) == INSTALL["search"]["roots"]
        assert SEARCH_DEPTH == INSTALL["search"]["depth"]
        assert list(PLUGIN_MANIFESTS) == INSTALL["search"]["plugin_manifests"]
        assert list(SCRIPT_DIRS) == INSTALL["flags"]["script_dirs"]
        assert list(SCRIPT_EXTENSIONS) == INSTALL["flags"]["script_extensions"]
        assert list(BINARY_EXTENSIONS) == INSTALL["flags"]["binary_extensions"]
        assert LOCK_FILE == INSTALL_LOCK_FILE()
        assert LOCK_VERSION == INSTALL["lock"]["version"]
        assert list(LOCK_ENTRY_KEYS) == INSTALL["lock"]["entry_keys"]

    @pytest.mark.parametrize("case", INSTALL["sources"], ids=[c["text"] for c in INSTALL["sources"]])
    def test_the_sources(self, case, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)  # no `owner/repo` folder exists here
        source = parse_source(case["text"])
        assert source.kind == case["kind"]
        if case["kind"] == "git":
            assert (source.url, source.ref, source.subpath) == (case["url"], case["ref"], case["subpath"])
        elif case["kind"] == "local":
            assert source.path == case["path"]

    def test_a_folder_that_exists_is_a_local_source_and_a_bare_word_is_a_name(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "skills" / "pdf").mkdir(parents=True)
        assert parse_source("skills/pdf").kind == "local"
        assert parse_source("pdf").kind == "name"
        (tmp_path / "shell").mkdir()
        assert parse_source("shell").kind == "name"

    def test_the_digest_vector(self, tmp_path):
        vector = INSTALL["lock"]["digest_vector"]
        for relative, text in vector["files"].items():
            target = tmp_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(text.encode())
        assert tree_digest(str(tmp_path)) == vector["tree"]
        assert tree_digest(str(tmp_path), list(vector["files"])) == vector["tree"]
        # The definition, spelled out: sha256 over "path\n<sha256 of content>\n" per sorted file.
        digest = hashlib.sha256()
        for relative in sorted(vector["files"]):
            digest.update(f"{relative}\n{hashlib.sha256(vector['files'][relative].encode()).hexdigest()}\n".encode())
        assert vector["tree"] == "sha256:" + digest.hexdigest()


def INSTALL_LOCK_FILE() -> str:
    return FIXTURE["layout"]["lock_file"].replace("/", os.sep)


class TestLocating:
    def test_skills_sh_rules_find_the_sample_repository(self, repo):
        located, skipped = locate_skills(str(repo["path"]), None, "src")
        assert [name for name, _ in located] == ["pdf", "xlsx"]
        assert [(s.name, s.problem) for s in skipped] == [("broken", "no_description")]

    def test_a_subpath_narrows_to_one_skill(self, repo):
        located, _ = locate_skills(str(repo["path"]), "skills/pdf", "src")
        assert [name for name, _ in located] == ["pdf"]
        located, _ = locate_skills(str(repo["path"]), "skills", "src")
        assert [name for name, _ in located] == ["pdf", "xlsx"]
        assert locate_skills(str(repo["path"]), "../outside", "src") == ([], [])

    def test_a_root_skill_and_nested_and_dot_agent_folders(self, tmp_path):
        root = tmp_path / "one-skill"
        root.mkdir()
        (root / "SKILL.md").write_text("---\nname: one-skill\ndescription: d\n---\n")
        (root / ".cursor" / "skills" / "deep").mkdir(parents=True)
        (root / ".cursor" / "skills" / "deep" / "SKILL.md").write_text("---\ndescription: d\n---\n")
        (root / "skills" / "cat" / "sub" / "nested").mkdir(parents=True)
        (root / "skills" / "cat" / "sub" / "nested" / "SKILL.md").write_text("---\ndescription: d\n---\n")
        (root / "skills" / "a" / "b" / "c" / "toodeep").mkdir(parents=True)
        (root / "skills" / "a" / "b" / "c" / "toodeep" / "SKILL.md").write_text("---\ndescription: d\n---\n")
        located, _ = locate_skills(str(root), None, "one-skill")
        assert [name for name, _ in located] == ["one-skill", "nested", "deep"]

    def test_files_are_flagged(self, repo):
        files, links = list_files(str(repo["path"] / "skills" / "pdf"))
        assert links == []
        by_path = {f.path: f for f in files}
        assert sorted(by_path) == sorted(FIXTURE["sample"]["pdf"]["files"] + ["SKILL.md"])
        assert by_path["scripts/fill_form.py"].script and not by_path["scripts/fill_form.py"].binary
        assert not by_path["forms.md"].script and not by_path["forms.md"].binary
        (repo["path"] / "skills" / "pdf" / "tool.bin").write_bytes(b"\x00\x01")
        (repo["path"] / "skills" / "pdf" / "run.sh").write_text("echo\n")
        files, _ = list_files(str(repo["path"] / "skills" / "pdf"))
        by_path = {f.path: f for f in files}
        assert by_path["tool.bin"].binary and not by_path["tool.bin"].script
        assert by_path["run.sh"].script


class TestInstalling:
    def test_installs_the_sample_skill_and_records_the_lock(self, repo, folder):
        code, out, err = _run("add", [repo["url"]], folder, yes=True, skill="pdf")
        assert code == 0, (out, err)
        assert err == [MESSAGES["skipped"].format(name="broken", reason=FIXTURE["sample"]["skipped"][0]["reason"])]
        assert out[0] == MESSAGES["found"].format(count=1, source=repo["url"], names="pdf")
        assert out[1].startswith("pdf: Work with PDF files")
        files = FIXTURE["sample"]["pdf"]["files"] + ["SKILL.md"]
        listed = out[2:2 + len(files)]
        assert [line.split(" (")[0].strip() for line in listed] == sorted(files)
        for line in listed:
            path = line.split(" (")[0].strip()
            size = os.path.getsize(REPO / "skills" / "pdf" / path)
            flag = MESSAGES["script_flag"] if path.startswith("scripts/") else ""
            assert line == MESSAGES["file_line"].format(path=path, size=size) + flag
        assert out[-1] == MESSAGES["installed_git"].format(name="pdf", files=len(files), sha7=repo["sha"][:7])
        installed = folder / ".agents" / "skills" / "pdf"
        assert (installed / "SKILL.md").read_bytes() == (REPO / "skills" / "pdf" / "SKILL.md").read_bytes()
        assert (installed / "scripts" / "probe.py").is_file()
        assert not (folder / ".agents" / "skills" / "xlsx").exists()
        lock = json.loads((folder / LOCK_FILE).read_text())
        assert lock["version"] == LOCK_VERSION
        entry = lock["skills"]["pdf"]
        assert list(entry) == sorted(LOCK_ENTRY_KEYS)
        assert entry["source"] == repo["url"] and entry["url"] == repo["url"] and entry["ref"] is None
        assert entry["subpath"] == "skills/pdf"
        assert entry["commit"] == repo["sha"]
        assert entry["files"] == sorted(files)
        assert entry["tree"] == tree_digest(str(installed)) == tree_digest(str(REPO / "skills" / "pdf"))
        assert entry["installed_at"].endswith("Z")
        # The agent file was not edited: every agent in the folder finds .agents/skills on its own.
        assert (folder / "AGENT.md").read_text().count("pdf") == 0

    def test_installs_every_skill_without_skill(self, repo, folder):
        code, out, _ = _run("add", [repo["url"]], folder, yes=True)
        assert code == 0
        assert out[0] == MESSAGES["found"].format(count=2, source=repo["url"], names="pdf, xlsx")
        assert sorted(read_lock(str(folder))["skills"]) == ["pdf", "xlsx"]

    def test_a_tree_url_needs_a_forge_so_a_local_repo_uses_skill(self, repo, folder):
        code, out, err = _run("add", [repo["url"]], folder, yes=True, skill="nope")
        assert code == 1
        assert err[-1] == MESSAGES["no_such_skill"].format(name="nope", source=repo["url"], names="pdf, xlsx")
        assert not (folder / ".agents").exists()

    def test_yes_is_required_without_a_terminal(self, repo, folder):
        code, out, err = _run("add", [repo["url"]], folder, yes=False, tty=False)
        assert code == 1
        assert err[-1] == MESSAGES["no_tty"]
        assert out[0].startswith("Found ")
        assert not (folder / ".agents").exists()

    def test_at_a_terminal_the_person_is_asked(self, repo, folder):
        asked = []

        def decline(question):
            asked.append(question)
            return False

        code, out, _ = _run("add", [repo["url"]], folder, yes=False, tty=True, confirm=decline, skill="pdf")
        assert code == 0
        assert asked == [MESSAGES["confirm"].format(names="pdf")]
        assert out[-1] == MESSAGES["not_installed"]
        assert not (folder / ".agents").exists()
        code, out, _ = _run("add", [repo["url"]], folder, yes=False, tty=True, confirm=lambda q: True, skill="pdf")
        assert code == 0 and out[-1].startswith("Installed pdf")

    def test_a_local_folder_source(self, repo, folder):
        code, out, _ = _run("add", [str(repo["path"] / "skills" / "pdf")], folder, yes=True)
        assert code == 0
        assert out[-1] == MESSAGES["installed_local"].format(name="pdf", files=6)
        entry = read_lock(str(folder))["skills"]["pdf"]
        assert entry["commit"] is None and entry["url"] is None

    def test_a_skill_the_lock_does_not_know_is_left_alone(self, repo, folder):
        mine = folder / ".agents" / "skills" / "pdf"
        mine.mkdir(parents=True)
        (mine / "SKILL.md").write_text("---\ndescription: mine\n---\nMine.\n")
        code, _, err = _run("add", [repo["url"]], folder, yes=True, skill="pdf")
        assert code == 1
        assert err[-1] == MESSAGES["exists"].format(name="pdf")
        assert (mine / "SKILL.md").read_text().endswith("Mine.\n")

    def test_a_reinstall_replaces_what_the_lock_knows(self, repo, folder):
        assert _run("add", [repo["url"]], folder, yes=True, skill="pdf")[0] == 0
        (folder / ".agents" / "skills" / "pdf" / "extra.md").write_text("stale")
        assert _run("add", [repo["url"]], folder, yes=True, skill="pdf")[0] == 0
        assert not (folder / ".agents" / "skills" / "pdf" / "extra.md").exists()

    def test_a_symbolic_link_refuses_the_skill(self, repo, folder):
        os.symlink("/etc/hosts", repo["path"] / "skills" / "xlsx" / "hosts")
        env = {**os.environ, **GIT_ENV}
        subprocess.run(["git", "add", "."], cwd=repo["path"], env=env, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-q", "-m", "link"], cwd=repo["path"], env=env, check=True, capture_output=True)
        code, _, err = _run("add", [repo["url"]], folder, yes=True, skill="xlsx")
        assert code == 1
        assert err[-1] == MESSAGES["symlink"].format(name="xlsx", path="hosts")

    def test_the_file_count_limit(self, tmp_path, folder):
        source = tmp_path / "many"
        (source / "assets").mkdir(parents=True)
        (source / "SKILL.md").write_text("---\nname: many\ndescription: d\n---\n")
        for i in range(FILE_LIMIT):
            (source / "assets" / f"f{i}.txt").write_text("x")
        code, _, err = _run("add", [str(source)], folder, yes=True)
        assert code == 1
        assert err[-1] == MESSAGES["too_many_files"].format(names="many", count=FILE_LIMIT + 1)
        assert not (folder / ".agents").exists()

    def test_the_extracted_size_limit(self, tmp_path, folder, monkeypatch):
        source = tmp_path / "big"
        source.mkdir()
        (source / "SKILL.md").write_text("---\nname: big\ndescription: d\n---\n")
        (source / "blob.txt").write_bytes(b"x" * 2048)
        monkeypatch.setattr("webagents.agents.skills.local.skillmd.skillmd_install.EXTRACTED_LIMIT", 1024)
        code, _, err = _run("add", [str(source)], folder, yes=True)
        assert code == 1
        assert err[-1].startswith("Refused: big would install ") and err[-1].endswith(", more than the 25 MiB limit.")

    def test_the_download_limit(self, repo, folder, monkeypatch):
        monkeypatch.setattr("webagents.agents.skills.local.skillmd.skillmd_install.DOWNLOAD_LIMIT", 10)
        code, _, err = _run("add", [repo["url"]], folder, yes=True)
        assert code == 1
        assert err[-1].startswith(f"Refused: fetching {repo['url']} took ") and err[-1].endswith(", more than the 10 MiB limit.")
        assert not (folder / ".agents").exists()

    def test_a_source_that_cannot_be_fetched(self, folder, tmp_path):
        url = f"file://{tmp_path / 'nowhere'}"
        code, _, err = _run("add", [url], folder, yes=True)
        assert code == 1
        assert err[-1].startswith(MESSAGES["fetch_failed"].split("{detail}")[0].format(source=url))
        code, _, err = _run("add", [str(tmp_path / "nofolder")], folder, yes=True)
        assert code == 1
        assert err[-1] == MESSAGES["not_a_folder"].format(source=str(tmp_path / "nofolder"))

    def test_a_repository_without_skills(self, tmp_path, folder):
        source = tmp_path / "empty"
        source.mkdir()
        (source / "README.md").write_text("nothing")
        code, _, err = _run("add", [str(source)], folder, yes=True)
        assert code == 1
        assert err[-1] == MESSAGES["none_found"].format(source=str(source))

    def test_names_and_sources_in_one_command(self, repo, folder):
        from webagents.cli.skills_edit import SkillsFacts

        facts = SkillsFacts(has_key=lambda v: True, signed_in=True)
        code, out, _ = _run("add", ["shell", repo["url"]], folder, yes=True, skill="xlsx", facts=lambda: facts)
        assert code == 0
        assert out[0] == "Added shell to AGENT.md."
        assert out[1] == "Skills: openai, shell"
        assert out[2] == MESSAGES["found"].format(count=1, source=repo["url"], names="xlsx")
        assert (folder / "AGENT.md").read_text().count("- shell") == 1


class TestRemoving:
    def test_removes_an_installed_skill_and_its_lock_entry(self, repo, folder):
        assert _run("add", [repo["url"]], folder, yes=True)[0] == 0
        code, out, err = _run("remove", ["pdf"], folder)
        assert code == 0 and err == []
        assert out == [MESSAGES["removed"].format(name="pdf")]
        assert not (folder / ".agents" / "skills" / "pdf").exists()
        assert sorted(read_lock(str(folder))["skills"]) == ["xlsx"]

    def test_a_coded_name_still_goes_to_the_agent_file(self, repo, folder):
        assert _run("add", [repo["url"]], folder, yes=True)[0] == 0
        code, out, _ = _run("remove", ["openai"], folder)
        assert code == 0
        assert out == ["Removed openai from AGENT.md.", "Skills: none"]
        assert (folder / ".agents" / "skills" / "pdf").exists()

    def test_a_source_is_not_a_name_to_remove(self, repo, folder):
        code, _, err = _run("remove", [repo["url"]], folder)
        assert code == 1
        assert err == [f"{repo['url']} is not a skill name; remove takes the names `skills list` shows."]


class TestTheInstalledSkillRunsInAnAgent:
    """The same public-style skill, installed by the CLI, loads into the
    agent built from the folder with the catalog and activation the fixture
    pins: the acceptance test of the lane, run identically in TypeScript."""

    def test_the_agent_built_from_the_folder_has_the_skill(self, repo, folder):
        from webagents.access.caller import LOCAL_OWNER
        from webagents.cli.agent_builder import build_agent
        from webagents.server.context.context_vars import CONTEXT, create_context, set_context

        assert _run("add", [repo["url"]], folder, yes=True, skill="pdf")[0] == 0
        built = asyncio.run(build_agent(folder / "AGENT.md", working_dir=folder, initialize=False))
        skill = built.agent.skills[SKILL_KEY]
        assert list(skill.skills) == ["pdf"]
        location = os.path.realpath(folder / ".agents" / "skills" / "pdf" / "SKILL.md")
        directory = os.path.dirname(location)
        token = CONTEXT.set(None)
        try:
            context = create_context(messages=[], agent=built.agent)
            context.auth = LOCAL_OWNER
            set_context(context)
            catalog = skill.skills_catalog()
            assert catalog.startswith(FIXTURE["catalog"]["preamble"])
            assert f"<name>pdf</name>" in catalog and f"<location>{location}</location>" in catalog
            assert asyncio.run(skill.activate_skill(name="pdf")) == FIXTURE["sample"]["activation"].replace("{pdf_dir}", directory)
        finally:
            CONTEXT.reset(token)
        names = {t["name"] for t in built.agent.get_tools_for_scope("owner")}
        assert set(FIXTURE["tools"]["names"]) <= names
        assert not (set(FIXTURE["tools"]["names"]) & {t["name"] for t in built.agent.get_tools_for_scope("user")})


class TestListingAndDoctor:
    def test_skills_list_shows_the_two_kinds_apart(self, repo, folder, monkeypatch):
        from typer.testing import CliRunner

        from webagents.cli.main import app

        monkeypatch.chdir(folder)
        before = CliRunner().invoke(app, ["skills", "list"]).output
        assert "  shell\n" in before
        assert MESSAGES["list_header"] in before
        assert MESSAGES["list_none"] in before
        assert MESSAGES["list_hint"].format(command="webagents skills add <owner/repo | git URL | folder>") in before
        assert _run("add", [repo["url"]], folder, yes=True)[0] == 0
        broken = folder / ".agents" / "skills" / "broken"
        broken.mkdir()
        (broken / "SKILL.md").write_text("---\nname: broken\n---\n")
        after = CliRunner().invoke(app, ["skills", "list"]).output
        location = os.path.realpath(folder / ".agents" / "skills" / "pdf" / "SKILL.md")
        assert MESSAGES["list_line"].format(name="pdf", location=location) in after
        assert MESSAGES["list_skipped"].format(name="broken", reason=FIXTURE["sample"]["skipped"][0]["reason"]) in after
        assert MESSAGES["list_none"] not in after

    def test_doctor_reports_the_skills_and_the_skipped_ones(self, repo, folder, monkeypatch):
        from webagents.cli.doctor import run_checks

        doctor = FIXTURE["doctor"]
        monkeypatch.chdir(folder)
        checks = {c.name: c for c in run_checks(folder)}
        assert checks[doctor["check"]].status == "ok"
        assert checks[doctor["check"]].detail == doctor["none"]
        assert _run("add", [repo["url"]], folder, yes=True)[0] == 0
        checks = {c.name: c for c in run_checks(folder)}
        assert checks[doctor["check"]].status == "ok"
        assert checks[doctor["check"]].detail == doctor["some"].format(count=2, s="s", names="pdf, xlsx")
        broken = folder / ".agents" / "skills" / "broken"
        broken.mkdir()
        (broken / "SKILL.md").write_text("---\nname: broken\n---\n")
        checks = {c.name: c for c in run_checks(folder)}
        assert checks[doctor["check"]].status == "warn"
        assert checks[doctor["check"]].detail == doctor["some"].format(count=2, s="s", names="pdf, xlsx") + doctor["skipped"].format(name="broken", reason=FIXTURE["sample"]["skipped"][0]["reason"])
        # The fix names the folders (2026-09-26).
        assert checks[doctor["check"]].fix == doctor["fix"].format(names="broken")

    def test_agent_skills_that_is_not_a_list_is_refused_with_the_fixture_sentence(self, tmp_path):
        from webagents.cli.loader.agent_md import AgentFile
        from webagents.cli.loader.schema import AgentFormatError

        file = tmp_path / "AGENT.md"
        file.write_text("---\nname: a\nagent_skills: ./skills\n---\n")
        with pytest.raises(AgentFormatError) as caught:
            AgentFile(file)
        assert str(caught.value) == f"{file}: " + FIXTURE["load_messages"]["agent_skills_not_a_list"]
