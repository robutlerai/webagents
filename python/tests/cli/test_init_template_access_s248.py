"""
`webagents init --template tool-agent` ships an `access:` block that keeps
shell and filesystem owner-only (S-248, 2026-09-26), byte for byte what the
TypeScript CLI writes (`tests/fixtures/cli/init_templates.json`), and the file
it writes loads: the block parses, and it scopes both skills' tools to the
`trusted` group, which nobody but the owner is in until the owner names
someone.
"""

import asyncio
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from webagents.agents.core.scopes import scope_allows
from webagents.cli.main import app

runner = CliRunner()
FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "init_templates.json").read_text())
HOST_TOOLS = ["run_command", "list_directory", "read_file", "write_file", "glob", "search_file_content", "replace"]


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("ROBUTLER_API_URL", "http://127.0.0.1:9")
    from webagents.agents.skills.core.llm.providers import LLM_PROVIDERS

    # No provider key at all (B3: `init` names a keyed provider's model), and
    # setenv THEN delenv so a key another test loaded into os.environ is
    # restored rather than leaked.
    keys = [v for p in LLM_PROVIDERS for v in p.env_vars] + ["GOOGLE_API_KEY"]
    for name in ("WEBAGENTS_PROFILE", "WEBAGENTS_TOKEN", *keys):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path / "work")


def expected_agent_md(name: str, template: str) -> str:
    t = FIXTURE["templates"][template]
    # The file's description is the template's, and the tool-agent template
    # ships a sandbox block before its access block (2026-09-26). With no
    # provider key here it names no model and no provider skill, and says
    # what runs (2026-09-28, `no_model`).
    lines = ["---", f"name: {name}", f"description: {t['description']}", *FIXTURE["no_model"]["comment"]]
    if t["skills"]:
        lines += ["skills:", *[f"  - {skill}" for skill in t["skills"]]]
    lines += t["sandbox"]
    lines += t["access"]
    lines += ["---", "", f"# {name}", "", "You are a helpful assistant.", ""]
    text = "\n".join(lines)
    assert text == FIXTURE["agent_md"][template].replace("{name}", name)
    return text


def test_agent_markdown_renders_each_template_byte_for_byte():
    """The renderer both `init` and the chat's /agent new use (spec 3.3)."""
    from webagents.cli.init_templates import agent_markdown

    for template, text in FIXTURE["agent_md"].items():
        if template == "about":
            continue
        assert agent_markdown("demo", template) == text.replace("{name}", "demo")


def test_agent_markdown_names_the_provider_the_chat_runs_on():
    """With a model, the chat writes that provider's skill in place of openai."""
    from webagents.cli.init_templates import agent_markdown

    wm = FIXTURE["with_model"]
    assert agent_markdown(wm["name"], wm["template"], wm["model"]) == wm["agent_md"]


def test_init_with_a_key_names_that_providers_model(monkeypatch):
    """B3 (2026-09-28): a key here, that provider's model; none, Robutler's choice."""
    from webagents.cli import model_access
    from webagents.cli.init_templates import init_model

    monkeypatch.setattr(model_access, "_client_installed", lambda provider: True)
    wk = FIXTURE["with_key"]
    for name, value in wk["env"].items():
        monkeypatch.setenv(name, value)
    assert init_model() == wk["model"]
    result = runner.invoke(app, ["init", wk["name"], "-t", wk["template"]])
    assert result.exit_code == 0, result.output
    assert Path(f"{wk['name']}/AGENT.md").read_text() == wk["agent_md"]
    for name in wk["env"]:
        monkeypatch.delenv(name)
    assert init_model() is None


@pytest.mark.parametrize("case", FIXTURE["init_line"]["cases"], ids=lambda c: c["line"][:40])
def test_the_last_line_names_the_way_in_only_when_one_is_needed(case):
    from webagents.cli.init_templates import init_line

    assert init_line(case["model"], case["keyed"], case["signed_in"]) == case["line"]


def test_a_file_with_no_model_runs_as_the_chat_does(monkeypatch):
    """Robutler's choice while no key is set, the key's model once one is (B3):
    the quickstart's order, `init` then `secrets set`, keeps working."""
    from webagents.cli import model_access
    from webagents.cli.agent_builder import build_agent

    monkeypatch.setattr(model_access, "is_signed_in", lambda: True)
    monkeypatch.setattr(model_access, "_client_installed", lambda provider: True)
    monkeypatch.setattr("webagents.cli.credentials.get_token", lambda *a, **k: "jwt.login")
    monkeypatch.setattr("webagents.cli.commands.secrets.load_into_environment", lambda: 0)
    assert runner.invoke(app, ["init", "tools", "-t", "tool-agent"]).exit_code == 0
    path = Path("tools/AGENT.md").resolve()
    built = asyncio.run(build_agent(path, working_dir=path.parent, initialize=False))
    assert built.model_problem is None
    assert built.model_label == f"{FIXTURE['no_model']['runs_on_without_a_key']} via Robutler"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    keyed = asyncio.run(build_agent(path, working_dir=path.parent, initialize=False))
    assert keyed.model_label == FIXTURE["with_key"]["model"]


def test_writes_the_file_the_shared_fixture_describes():
    result = runner.invoke(app, ["init", "tools", "-t", "tool-agent"])
    assert result.exit_code == 0, result.output
    assert Path("tools/AGENT.md").read_text() == expected_agent_md("tools", "tool-agent")


def test_the_chatbot_template_carries_no_access_block():
    assert runner.invoke(app, ["init", "chat", "-t", "chatbot"]).exit_code == 0
    assert Path("chat/AGENT.md").read_text() == expected_agent_md("chat", "chatbot")
    assert FIXTURE["templates"]["chatbot"]["access"] == []


def test_the_written_block_loads_and_keeps_shell_and_filesystem_owner_only(tmp_path):
    from webagents.cli.agent_builder import build_agent

    assert runner.invoke(app, ["init", "tools", "-t", "tool-agent"]).exit_code == 0
    path = Path("tools/AGENT.md").resolve()
    built = asyncio.run(build_agent(path, working_dir=path.parent, initialize=False))
    assert "access" in built.agent.skills
    scopes = {t["name"]: t.get("scope") for t in built.agent.get_all_tools()}
    for name in HOST_TOOLS:
        assert scopes[name] == ["group:trusted"], name
        assert scope_allows(scopes[name], {"owner"})
        assert not scope_allows(scopes[name], {"user"})
        assert not scope_allows(scopes[name], set())
        # The group the owner would name callers in: a member gets the tools.
        assert scope_allows(scopes[name], {"group:trusted"})


def test_the_block_that_restores_the_old_open_tools_grants_them_to_the_default_group(tmp_path):
    from webagents.cli.agent_builder import build_agent

    folder = tmp_path / "open"
    folder.mkdir()
    lines = ["---", "name: open", "skills:", "  - filesystem", "  - shell", *FIXTURE["restores_open_tools"]["access"], "---", "Open.", ""]
    (folder / "AGENT.md").write_text("\n".join(lines))
    built = asyncio.run(build_agent(folder / "AGENT.md", working_dir=folder, initialize=False))
    scopes = {t["name"]: t.get("scope") for t in built.agent.get_all_tools()}
    assert scopes["run_command"] == ["group:everyone"]
    # The access skill places every caller the block names no other group for in `everyone`.
    assert scope_allows(scopes["run_command"], {"group:everyone"})
