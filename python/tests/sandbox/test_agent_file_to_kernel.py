"""
The `sandbox:` declaration, EXECUTED from an agent file to the kernel
(2026-09-24).

S-217 was never that the kernel could not confine a command. It was that the
declaration in an agent file reached nothing: `sandbox:` was parsed and applied
to no skill. The fix lives in one line of
`LocalFileSource._load_skills`, and until this file the only test of that line
was `test_the_declaration_reaches_the_skill_from_the_agent_file`, which READS
THE SOURCE for the string `config["sandbox"] = sandbox`. It would pass against
code that assigns the value and then overwrites it, or never calls the
function. (The loader's own suite, `tests/server/plugins/`, imports a module
that does not exist and skips on every run, so nothing executed this path.)

These go the way the daemon goes: an `AGENT-*.md` in a watched directory,
`LocalFileSource.get_agent`, the agent's own `shell` skill, a real command,
and the real OS sandbox. The control case, the same file with no `sandbox:`,
is what shows the assertions can tell enforcement from its absence.
"""

import pytest

from webagents.sandbox import backend_status, sandbox_available
from webagents.server.extensions.local_file_source import LocalFileSource
from webagents.server.storage.json_store import JSONMetadataStore

requires_backend = pytest.mark.skipif(
    not sandbox_available(),
    reason=f"no OS sandbox backend here: {backend_status()['reason']}",
)

DUMMY_SECRET = "DUMMY-SECRET-NOT-A-KEY"


@pytest.fixture
def layout(tmp_path):
    """A watched project directory, and a sibling directory outside it."""
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    (outside / "token").write_text(DUMMY_SECRET)
    return tmp_path, project.resolve(), outside.resolve()


@pytest.fixture(autouse=True)
def model_credentials(monkeypatch):
    # Loading constructs the declared model's client. No call is made; a dummy
    # key is enough, and it must not be a real one.
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-not-a-key")


def _write_agent(project, name, sandbox_block):
    # `model:` as `webagents init` writes it. A file with NO model defaults to
    # the Google skill, which cannot even be constructed without the optional
    # google-genai package; that is a separate defect, reported, not this test.
    (project / f"AGENT-{name}.md").write_text(
        f"---\nname: {name}\nmodel: openai/gpt-4o-mini\nskills:\n  - shell\n"
        f"{sandbox_block}---\nA test agent.\n"
    )


async def _shell_of(tmp_path, project, name):
    source = LocalFileSource(
        watch_dirs=[project], metadata_store=JSONMetadataStore(data_dir=tmp_path / "data")
    )
    agent = await source.get_agent(name)
    assert agent is not None, f"the daemon's loader did not produce {name}"
    shell = agent.skills.get("shell")
    assert shell is not None, f"{name} has no shell skill: {sorted(agent.skills)}"
    return shell


@requires_backend
@pytest.mark.asyncio
async def test_a_declared_sandbox_confines_the_agents_shell_tool(layout):
    tmp_path, project, outside = layout
    _write_agent(
        project,
        "boxed",
        f"sandbox:\n  preset: strict\n  allowed_folders:\n    - {project}\n",
    )
    shell = await _shell_of(tmp_path, project, "boxed")

    # The declaration arrived, as a policy, on the skill the agent will call.
    assert shell.policy is not None
    assert shell.policy.scoped_reads

    # Legitimate work inside the declared folder still happens.
    await shell.run_command(f"echo ok > {project}/inside.txt", timeout=20)
    assert (project / "inside.txt").read_text().strip() == "ok"

    # A write outside it does not, whatever the command said.
    await shell.run_command(f"echo leaked > {outside}/leak.txt", timeout=20)
    assert not (outside / "leak.txt").exists()

    # Nor does a read outside it, under strict. `$(echo ...)` hides the path
    # from the skill's own in-process check, which refuses an EXISTING
    # absolute path outside its folder before anything runs. Without it this
    # assertion passes on that heuristic and says nothing about the kernel (it
    # did, in the first version of this test).
    out = await shell.run_command(f"cat $(echo {outside})/token", timeout=20)
    assert DUMMY_SECRET not in out


@requires_backend
@pytest.mark.asyncio
async def test_without_a_declaration_the_same_command_is_not_confined(layout):
    """The control. Without it, a loader that confined EVERYTHING, or a shell
    skill that refused every write, would pass the test above for the wrong
    reason."""
    tmp_path, project, outside = layout
    _write_agent(project, "open", "")
    shell = await _shell_of(tmp_path, project, "open")

    assert shell.policy is None
    await shell.run_command(f"echo written > {outside}/leak.txt", timeout=20)
    assert (outside / "leak.txt").read_text().strip() == "written"

    # The same read that the sandboxed agent is refused.
    out = await shell.run_command(f"cat $(echo {outside})/token", timeout=20)
    assert DUMMY_SECRET in out


@pytest.mark.asyncio
async def test_a_malformed_declaration_refuses_rather_than_running_free(layout):
    """Fail closed from the file too: a typo in `preset` must not mean 'none'."""
    tmp_path, project, outside = layout
    _write_agent(project, "typo", "sandbox:\n  preset: stirct\n")
    shell = await _shell_of(tmp_path, project, "typo")

    out = await shell.run_command(f"echo leaked > {outside}/leak.txt", timeout=20)
    assert "Access denied" in out
    assert not (outside / "leak.txt").exists()
