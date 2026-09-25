"""
`webagents doctor` (2026-09-24): what stands between this folder and a
running agent, and the one command that fixes each thing.

The same checks, names and words as the TypeScript CLI's doctor
(`typescript/src/cli/doctor.ts`): runtime, agent, model, sign-in, keys,
sandbox, config. The model check is the chat's own decision
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

OK, WARN, FAIL = "ok", "warn", "fail"
MARKS = {OK: "✓", WARN: "▲", FAIL: "✗"}


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
    from .agent_builder import provider_env_var

    env_var = provider_env_var(built)
    return f"{built.model_label}, with your {env_var}" if env_var else built.model_label


def _sandbox_check(built: Any) -> Check:
    shell = built.agent.skills.get("shell") if built is not None else None
    if shell is None:
        return Check("sandbox", OK, "not needed: the agent cannot run commands")
    if getattr(shell, "policy", None) is None:
        return Check(
            "sandbox",
            WARN,
            "off: shell commands run with your permissions",
            "Add a `sandbox:` section to the agent file",
        )
    from webagents.sandbox import backend_status

    status = backend_status()
    preset = getattr(built.sandbox, "preset", None) or "custom"
    if not status.get("available"):
        return Check(
            "sandbox",
            FAIL,
            f"{preset}, but {status.get('reason') or 'no sandbox backend here'}: shell commands are refused",
            "Install bubblewrap (Linux), or remove `sandbox:` to run commands unconfined",
        )
    return Check("sandbox", OK, f"{preset}, enforced by {status.get('backend')}")


async def _agent_checks(folder: Path) -> tuple[List[Check], Any]:
    from .agent_builder import build_agent
    from .agent_files import default_agent_file

    try:
        built = await build_agent(default_agent_file(folder), working_dir=folder)
    except Exception as error:  # noqa: BLE001 - a broken agent file is what doctor is for
        return [Check("agent", FAIL, str(error), "Fix the agent file, or start over with `webagents init`.")], None
    checks = [
        Check("agent", OK, f"{built.name} ({built.file.name})" if built.file else "none here; the built-in assistant runs"),
    ]
    model_ok = not built.model_problem and not (built.access is not None and built.access.kind == "none")
    checks.append(
        Check("model", OK if model_ok else FAIL, _model_words(built), None if model_ok else f"`{cli_command('login')}`, or `{cli_command('secrets set <NAME>')}`")
    )
    return checks, built


def run_checks(folder: Optional[Path] = None) -> List[Check]:
    from .account import who_am_i

    folder = folder or Path.cwd()
    version = platform.python_version()
    recent = sys.version_info >= (3, 10)
    checks = [Check("runtime", OK if recent else WARN, f"Python {version}", None if recent else "Install Python 3.10 or later.")]

    agent_checks, built = asyncio.run(_agent_checks(folder))
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

    checks.append(_sandbox_check(built))

    from .config_store import ConfigStore

    problems = ConfigStore().validate()
    checks.append(
        Check("config", FAIL, f"{len(problems)} problem{'' if len(problems) == 1 else 's'}", "`webagents config validate`")
        if problems
        else Check("config", OK, "valid")
    )
    return checks


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
