"""
`webagents doctor` (2026-09-24): what stands between this folder and a
running agent, and the one command that fixes each thing.

The same checks, names and words as the TypeScript CLI's doctor
(`typescript/src/cli/doctor.ts`): runtime, agent, model, sign-in, keys,
keychain, sandbox, skills, mcp, config. The model check is the chat's own decision
(`agent_builder.build_agent`), so doctor and the chat never disagree about
whether the agent can run. Where the two SDKs differ in fact, the check says
the fact: the runtime is Python here, and this SDK has a sandbox.

It replaced a longer doctor of fifteen checks (daemon reachability, docker,
dotenv files, optional extras and more) whose report did not match the
TypeScript one in a single line.
"""

from __future__ import annotations

import asyncio
import platform
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config_store import cli_command

# `keychain` (keychain-ux, 2026-09-27): where the sign-in and keys live, which
# program last used them and whether the next use makes macOS ask. Its words
# are the store's (`keychain_ux.DOCTOR_WORDS`, fixture `keychain_ux`).

OK, WARN, FAIL = "ok", "warn", "fail"
MARKS = {OK: "✓", WARN: "▲", FAIL: "✗"}

#: The mcp check's words (S-292), the same in the TypeScript doctor and pinned
#: by `tests/fixtures/cli/secrets.json` (`doctor`).
MCP_CHECK_WORDS = {
    "notUsed": "not used: the agent names no MCP servers",
    "connected": "{count} connected: {names}",
    "fixEntry": "Fix the server's entry in the agent file",
    "fixCommand": "{command}: {hint}",
    "fixLiteral": "${secret:{name}} in the agent file, then `{hint}`",
    # A `${env:NAME}` that is not set names the variable (2026-09-26): the fix
    # line said only "Fix the server's entry in the agent file".
    "fixEnv": "Set {name} in the environment, or store it with `{hint}` and write ${secret:{name}} in the agent file",
    # A server that answered 401 or 403 wants a bearer token (2026-09-29,
    # `mcp/connect_errors.py`): the recipe, never "fix the server's entry".
    "fixCredential": "Authorization: Bearer ${secret:{name}} in {server}'s headers, then `{hint}`",
}


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        if out["fix"] is None:
            del out["fix"]
        return out


def _model_words(built: Any) -> str:
    """How the agent reaches its model, as `doctor` and `/status` say it."""
    access = built.access
    if built.model_problem or (access is not None and access.kind == "none"):
        kind = getattr(access, "kind", None)
        reason = "not signed in" if kind == "proxy" else (getattr(access, "reason", "") or "no model")
        return f"none ({reason})"
    if access is not None and access.kind == "proxy":
        return f"{built.model_label}, paid from your Robutler credits"
    local = access.local_route() if access is not None and hasattr(access, "local_route") else None
    if local:
        return local
    from .agent_builder import provider_env_var

    env_var = provider_env_var(built)
    return f"{built.model_label}, with your {env_var}" if env_var else built.model_label


def _local_model_check(built: Any, model_check: Check) -> Check:
    """The model check for an `ollama/<model>` agent (plan item 2.8): whether
    Ollama answers at its address and serves the model, in the fixture's
    words (`ollama_model_check`); any other route keeps `model_check`."""
    access = getattr(built, "access", None)
    if built is None or built.model_problem or access is None or access.kind != "direct":
        return model_check
    provider = getattr(access, "provider", None)
    if provider is None or getattr(provider, "credential", None) != "none" or not access.model:
        return model_check
    from webagents.agents.skills.core.llm.ollama.probe import ollama_model_check, probe_ollama
    from webagents.agents.skills.core.llm.providers import provider_base_url

    base = provider_base_url(provider) or ""
    words = ollama_model_check(access.model, base, probe_ollama(base))
    return Check("model", str(words["status"]), str(words["detail"]), words.get("fix"))


def _sandbox_check(built: Any) -> Check:
    """The sandbox check, the TypeScript doctor's words (`cli/doctor.ts`,
    `sandboxCheck`): the state (preset and origin) from the shell's own
    policy, the engine's status, and SKILL.md scripts counted as confined
    commands (the sandbox-default lane, 2026-09-27)."""
    from webagents.sandbox import backend_status, sandbox_state, unavailable_fix

    skills = built.agent.skills if built is not None else {}
    shell = skills.get("shell")
    if shell is None:
        # SKILL.md scripts are confined commands too: an agent that runs them
        # is not one that "cannot run commands".
        scripts = skills.get("agent_skills")
        script_state = scripts.script_state() if scripts is not None and hasattr(scripts, "script_state") else None
        if not script_state:
            return Check("sandbox", OK, "not needed: the agent cannot run commands")
        status = backend_status()
        if not status.get("available"):
            # What THIS machine lacks and its fix, as `webagents sandbox setup`
            # says them (the sandbox-engine lane, 2026-09-27).
            return Check("sandbox", FAIL, f"{script_state} for SKILL.md scripts, but {status.get('reason') or 'no sandbox backend here'}: scripts are refused", unavailable_fix(status))
        found = f" ({status['found']})" if status.get("found") else ""
        return Check("sandbox", OK, f"{script_state} for SKILL.md scripts, enforced by {status.get('backend')} {status.get('version')}{found}")
    sandbox_error = getattr(shell, "sandbox_error", None)
    if sandbox_error:
        return Check("sandbox", FAIL, f"invalid: {sandbox_error}: shell commands are refused", "Fix the `sandbox:` section of the agent file")
    policy = getattr(shell, "policy", None)
    origin = getattr(shell, "sandbox_origin", None) or "default"
    state = sandbox_state(policy, origin)
    if policy is None or not getattr(policy, "confined", True):
        # An opt-out is reported as what it is, not as a sandbox.
        if origin == "--no-sandbox":
            return Check("sandbox", WARN, f"{state}: not confined; shell commands run with your permissions for this run", "Run without --no-sandbox to confine them")
        return Check("sandbox", WARN, f"{state}: not confined; shell commands run with your permissions", "Remove `sandbox: off` (or `preset: unrestricted`) from the agent file to confine them")
    status = backend_status()
    if not status.get("available"):
        return Check("sandbox", FAIL, f"{state}, but {status.get('reason') or 'no sandbox backend here'}: shell commands are refused", unavailable_fix(status))
    found = f" ({status['found']})" if status.get("found") else ""
    return Check("sandbox", OK, f"{state}, enforced by {status.get('backend')} {status.get('version')}{found}")


def _install_check(built: Any) -> Optional[Check]:
    """The `install` line (S-316, the ptypass-fixes lane, 2026-09-27): when
    webagents, or the node srt runs on, is installed inside a folder the
    agent's confined commands may write, the sandbox write-denies it, and
    this says so, with why a confined `pip install` into that environment
    fails. The words are `sandbox.srt.install_inside_check`'s, the TypeScript
    doctor's too (fixture `sandbox/srt.json` `sdk_install_deny.report`)."""
    from webagents.sandbox.srt import install_inside_check

    shell = (built.agent.skills if built is not None else {}).get("shell")
    policy = getattr(shell, "policy", None)
    if policy is None or not getattr(policy, "confined", False):
        return None
    roots = [root for root in [policy.cwd, *policy.write_roots] if root and root != policy.scratch]
    found = install_inside_check(roots)
    return Check(found["name"], found["status"], found["detail"], found.get("fix")) if found else None


def _mcp_check(built: Any) -> Check:
    """The MCP servers check (S-292, 2026-09-26), the TypeScript doctor's
    words (`cli/doctor.ts`, `mcpCheck`), from the skill's own report: a server
    whose `${secret:NAME}` is not stored, or whose entry was refused, fails
    the check with the sentence and the `webagents secrets set` command; a
    literal that looks like a key is a warning with the reference to use.
    Never a value: the report masks them."""
    skill = built.agent.skills.get("mcp") if built is not None else None
    return mcp_check(skill.server_report() if skill is not None and hasattr(skill, "server_report") else [])


def mcp_check(report: List[Dict[str, Any]]) -> Check:
    """The `mcp` check from the skill's `server_report()` rows (`_mcp_check`)."""
    if not report:
        return Check("mcp", OK, MCP_CHECK_WORDS["notUsed"])
    broken = [row for row in report if row.get("rejected") or row.get("error")]
    if broken:
        from webagents.agents.skills.local.mcp.connect_errors import command_hint, credential_secret_name

        missing = list(dict.fromkeys(name for row in broken for name in row.get("missing_secrets", [])))
        unset = list(dict.fromkeys(name for row in broken for name in row.get("missing_env") or []))
        wants_token = [row["name"] for row in broken if row.get("needs_credential")]
        fixes = [f"`{cli_command(f'secrets set {name}')}`" for name in missing] + [
            MCP_CHECK_WORDS["fixEnv"].replace("{name}", name).replace("{hint}", cli_command(f"secrets set {name}")) for name in unset
        ] + [
            MCP_CHECK_WORDS["fixCredential"]
            .replace("{name}", credential_secret_name(server))
            .replace("{server}", server)
            .replace("{hint}", cli_command(f"secrets set {credential_secret_name(server)}"))
            for server in wants_token
        ] + [
            # A command that is not there: what to install (2026-09-29).
            MCP_CHECK_WORDS["fixCommand"].replace("{command}", command).replace("{hint}", command_hint(command))
            for command in dict.fromkeys(row["needs_command"] for row in broken if row.get("needs_command"))
        ]
        return Check(
            "mcp",
            FAIL,
            "; ".join(f"{row['name']}: {row.get('rejected') or row.get('error')}" for row in broken),
            ", ".join(fixes) if fixes else MCP_CHECK_WORDS["fixEntry"],
        )
    warned = [row for row in report if row.get("warnings")]
    if warned:
        suggested = list(dict.fromkeys(
            name for row in warned for warning in row["warnings"] for name in re.findall(r"\$\{secret:([A-Za-z0-9_]+)\}", warning)
        ))
        return Check(
            "mcp",
            WARN,
            "; ".join(f"{row['name']} {warning}" for row in warned for warning in row["warnings"]),
            "; ".join(
                MCP_CHECK_WORDS["fixLiteral"].replace("{name}", name).replace("{hint}", cli_command(f"secrets set {name}"))
                for name in suggested
            ),
        )
    names = [row["name"] for row in report]
    count = f"{len(names)} server{'' if len(names) == 1 else 's'}"
    return Check("mcp", OK, MCP_CHECK_WORDS["connected"].replace("{count}", count).replace("{names}", ", ".join(names)))


def _skillmd_check(folder: Path, agent_file: Optional[Path] = None) -> Check:
    """The SKILL.md skills check (plan item 1.4, 2026-09-26), the TypeScript
    doctor's words (`skillmd_loader.doctor_report`): what the folder's agent
    would load from `.agents/skills` and its `agent_skills:`, and every folder
    that was skipped or warned about, since a skipped skill is otherwise only
    a line at start-up. `agent_file` is the file `-a` chose."""
    from webagents.agents.skills.local.skillmd.skillmd_loader import discover_skills, doctor_report

    from .agent_files import default_agent_file

    explicit: List[str] = []
    file = agent_file if agent_file is not None else default_agent_file(folder)
    if file is not None:
        try:
            from .loader.hierarchy import load_agent

            explicit = list(load_agent(file).metadata.agent_skills or [])
        except Exception:  # noqa: BLE001 - the agent check above already reports a broken file
            explicit = []
    report = doctor_report(discover_skills(str(folder), explicit))
    return Check("skills", str(report["status"]), str(report["detail"]), report["fix"])


async def _agent_checks(folder: Path, agent_file: Optional[Path]) -> tuple[List[Check], Any, Check]:
    """The agent and model checks, the built agent, and the mcp check, for
    `agent_file` (the file `-a` chose, else the folder's default; None means
    the built-in assistant).

    THE MCP SERVERS ARE CLOSED HERE, in the task that opened them, before the
    loop ends (2026-09-26, the e2e run): the agent was built for its facts and
    never cleaned up, so a stdio server's client generator was collected when
    the loop closed and asyncio printed "an error occurred during closing of
    asynchronous generator" above the report. The mcp check is read first,
    since the report says which servers ARE connected.
    """
    from .agent_builder import build_agent

    try:
        # The opt-out is said by the sandbox line below, not above the
        # report as well (the ptypass-fixes lane, 2026-09-27).
        from webagents.agents.skills.local.shell.skill import opt_out_announcer

        with opt_out_announcer(lambda _message, _origin: None):
            built = await build_agent(agent_file, working_dir=folder)
    except Exception as error:  # noqa: BLE001 - a broken agent file is what doctor is for
        return [Check("agent", FAIL, str(error), "Fix the agent file, or start over with `webagents init`.")], None, _mcp_check(None)
    checks = [
        Check("agent", OK, f"{built.name} ({built.file.name})" if built.file else "none here; the built-in assistant runs"),
    ]
    model_ok = not built.model_problem and not (built.access is not None and built.access.kind == "none")
    checks.append(
        _local_model_check(
            built,
            Check("model", OK if model_ok else FAIL, _model_words(built), None if model_ok else f"`{cli_command('login')}`, or `{cli_command('secrets set <NAME>')}`"),
        )
    )
    mcp = _mcp_check(built)
    skill = built.agent.skills.get("mcp") if getattr(built.agent, "skills", None) else None
    if skill is not None and hasattr(skill, "cleanup"):
        try:
            await skill.cleanup()
        except Exception:  # noqa: BLE001 - a close that fails must not fail the report
            pass
    return checks, built, mcp


def run_checks(folder: Optional[Path] = None, agent: Optional[str] = None) -> List[Check]:
    """The checks for `folder`. `agent` is `-a <name>` (2026-09-26): the
    folder's agent to check, as the chat's `-a` picks it; an unknown name
    raises `AgentNotFound` with its sentence, before anything is built."""
    from .account import who_am_i
    from .agent_files import agent_file_for, default_agent_file

    folder = folder or Path.cwd()
    agent_file = agent_file_for(folder, agent) if agent else default_agent_file(folder)
    version = platform.python_version()
    recent = sys.version_info >= (3, 10)
    checks = [Check("runtime", OK if recent else WARN, f"Python {version}", None if recent else "Install Python 3.10 or later.")]

    # Before anything reads the keychain (the agent's keys, the sign-in):
    # whether the NEXT use will ask is a fact about now, and the reads below
    # may answer it (keychain-ux, 2026-09-27).
    keychain_line = keychain_check()

    agent_checks, built, mcp = asyncio.run(_agent_checks(folder, agent_file))
    checks.extend(agent_checks)

    who = who_am_i()
    if who.ok:
        checks.append(Check("sign-in", OK, f"@{who.username} on {re.sub(r'^https?://', '', who.platform)}"))
    else:
        fix = re.sub(r"\.$", "", re.sub(r"^Run ", "", who.fix)) if who.fix else None
        checks.append(Check("sign-in", WARN, who.message, fix))

    try:
        from .commands.secrets import _store

        keychain = bool(_store(quiet=True).keystore)
        checks.append(
            Check("keys", OK, "stored in your keychain")
            if keychain
            else Check("keys", WARN, "stored in an owner-only file (no keychain on this machine)")
        )
    except Exception as error:  # noqa: BLE001 - reported, not raised
        checks.append(Check("keys", WARN, f"the key store cannot be opened: {error}"))
    checks.append(keychain_line)

    checks.append(_sandbox_check(built))
    install = _install_check(built)
    if install is not None:
        checks.append(install)
    checks.append(_skillmd_check(folder, agent_file))
    # Read inside the loop, before the servers were closed (`_agent_checks`).
    checks.append(mcp)

    from .config_store import ConfigStore

    problems = ConfigStore().validate()
    checks.append(
        Check("config", FAIL, f"{len(problems)} problem{'' if len(problems) == 1 else 's'}", "`webagents config validate`")
        if problems
        else Check("config", OK, "valid")
    )
    return checks


def keychain_facts() -> Dict[str, Any]:
    """What the `keychain` line and the chat's /status `Keychain` row say,
    found without reading a value: where the sign-in and keys live, which
    program last used this CLI's items, whether the next use will ask, an
    earlier version's items, and what a run with nobody to answer could not
    read (keychain-ux, 2026-09-27; the TypeScript `keychainFacts`)."""
    import os

    from webagents.agents.skills.local.secrets.keychain_ux import KeychainRecord, doctor_facts, legacy_service_name, record_path

    from .commands.secrets import NAMESPACE as KEYS_NAMESPACE
    from .commands.secrets import _store as keys_store
    from .config_store import global_dir, profile_name, scoped_namespace
    from .credentials import CLI_NAMESPACE, TOKEN_ENV_VAR, TOKEN_KEY
    from .credentials import _store as token_store

    profile = profile_name()
    token_namespace = scoped_namespace(CLI_NAMESPACE, profile)
    stores = [token_store(profile), keys_store(quiet=True)]
    legacy_keys = stores[1].legacy_index_names()
    facts = doctor_facts(
        stores,
        KeychainRecord(record_path(global_dir(profile) / "secrets")),
        {
            legacy_service_name(token_namespace): [TOKEN_KEY],
            legacy_service_name(scoped_namespace(KEYS_NAMESPACE, profile)): legacy_keys,
        },
        env_token=bool(os.environ.get(TOKEN_ENV_VAR)),
    )
    facts["command"] = cli_command("whoami")
    return facts


def keychain_check() -> Check:
    """The `keychain` line (`keychain_facts`), in the fixture's words."""
    from webagents.agents.skills.local.secrets.keychain_ux import doctor_line

    try:
        line = doctor_line(keychain_facts())
    except Exception as error:  # noqa: BLE001 - reported, not raised
        return Check("keychain", WARN, f"cannot be checked: {error}")
    return Check(line["name"], line["status"], line["detail"], line.get("fix"))


def report_lines(checks: List[Check]) -> List[str]:
    """The report, as the TypeScript doctor prints it."""
    width = max(len(c.name) for c in checks) + 3
    lines = ["Checks"]
    for c in checks:
        lines.append(f"  {MARKS[c.status]} {c.name.ljust(width)}{c.detail}")
    to_fix = [c for c in checks if c.status != OK and c.fix]
    if to_fix:
        lines += ["", "To fix"]
        for c in to_fix:
            lines.append(f"  {c.name.ljust(width + 2)}{c.fix}")
    return lines
