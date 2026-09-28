"""
The skill an agent carries for its SKILL.md skills (gap-closure plan item
1.4, 2026-09-26). The TypeScript twin is
`typescript/src/skills/skillmd/skillmd-skill.ts`; both run
`tests/fixtures/skillmd/skillmd.json`.

PROGRESSIVE DISCLOSURE, three tiers, as agentskills.io describes it:

  1. The catalog. A system-prompt section listing each skill's name,
     description and location in the `<available_skills>` format, about a
     hundred tokens per skill, omitted when there are none. It is shown to a
     caller only when that caller may call `activate_skill`, so a stranger
     an `access:` block keeps away from the tools does not read a menu it
     cannot order from.
  2. Activation. `activate_skill(name)`, `name` an enum of the loaded skills,
     returns the body of SKILL.md wrapped in `<skill_content name="...">`,
     then the skill's folder and the files it bundles, LISTED and not read.
     It never injects twice: an earlier tool result in the conversation that
     opens the same tag is what "already active" means, so the check needs
     no state and survives a restart of the daemon.
  3. Resources. `read_skill_file` reads one bundled text file, confined to
     the skill's folder by real path; `run_skill_script` runs one bundled
     script through the kernel sandbox (`webagents.sandbox`, srt), with the
     skill's folder readable and write-denied, the agent's own `sandbox:`
     folders and `network:` list, and nothing else. An agent that declares no
     sandbox gets a synthesised `strict` policy with no writable folder but
     the private scratch; one that declares `unrestricted` is refused,
     because a skill fetched from a git repository is third-party code and
     runs confined or not at all.

OWNER-ONLY BY DEFAULT (S-248, ADR-0045). All three tools are `scope="owner"`
until the agent file hands `agent_skills` (the whole skill) or a tool's name
to a group with `access: tools:`. The tools are built per instance rather
than with `@tool`, because the `name` enum is this agent's list of skills;
they are set as attributes in `__init__` so the agent's decorator scan finds
them at construction and a strict `access.tools` pass can name them.

`!`cmd`` substitutions and `$ARGUMENTS` in a body are text: nothing here
runs or expands them.
"""

from __future__ import annotations

import dataclasses
import os
import shlex
import subprocess
from typing import Any, Callable, Dict, List, Optional, Sequence

from webagents.agents.tools.decorators import prompt

from ...base import Skill
from .skillmd_loader import (
    SkillMd,
    SkippedSkill,
    activation_marker,
    activation_text,
    catalog_text,
)

#: The name the agent's skills dict (and `access: tools:`) knows this skill by.
SKILL_KEY = "agent_skills"

TOOL_NAMES = ("activate_skill", "read_skill_file", "run_skill_script")

#: How a bundled script is run, by its extension; an executable file with
#: another extension runs on its own.
INTERPRETERS = {".py": "python3", ".sh": "bash", ".js": "node", ".mjs": "node", ".cjs": "node"}

DEFAULT_TIMEOUT = 60
MAX_TIMEOUT = 300

#: The largest bundled file `read_skill_file` returns.
READ_MAX_BYTES = 256 * 1024

ACTIVATE_DESCRIPTION = (
    "Load the full instructions of one available skill (see <available_skills>) into this conversation. "
    "Call it once per skill, before doing what the skill covers; the result also lists the files the skill bundles."
)
READ_DESCRIPTION = (
    "Read a text file bundled with an activated skill, by its path inside the skill's folder "
    "(for example references/api.md). Only files inside that folder can be read."
)
RUN_DESCRIPTION = (
    "Run a script bundled with an activated skill inside the sandbox and return its output. "
    "script is the path inside the skill's folder (for example scripts/fill_form.py); "
    "args are its command-line arguments; a relative path in args resolves against the agent's folder."
)

UNRESTRICTED_REFUSAL = (
    "Access denied: skill scripts run only in a sandbox, and this agent's sandbox is unrestricted, which is no sandbox"
)


def _tool_function(fn: Callable[..., Any], name: str, description: str, parameters: Dict[str, Any]) -> Callable[..., Any]:
    """A plain function over the bound method `fn`, carrying what
    `BaseAgent.register_tool` reads, as the MCP skill builds its dynamic
    tools: the OpenAI-style definition, the name, the description and the
    owner scope. A function rather than the method itself, because a bound
    method takes no attributes."""

    async def tool_function(**kwargs: Any) -> Any:
        return await fn(**kwargs)

    tool_function.__name__ = name
    tool_function.__qualname__ = name
    tool_function.__doc__ = description
    tool_function._webagents_is_tool = True  # type: ignore[attr-defined]
    tool_function._webagents_tool_definition = {  # type: ignore[attr-defined]
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }
    tool_function._tool_name = name  # type: ignore[attr-defined]
    tool_function._tool_description = description  # type: ignore[attr-defined]
    tool_function._tool_scope = "owner"  # type: ignore[attr-defined]
    tool_function._tool_scope_was_set = True  # type: ignore[attr-defined]
    tool_function._tool_provides = None  # type: ignore[attr-defined]
    return tool_function


def tool_parameters(names: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """The three tools' JSON schemas for a list of skill names (pinned by the fixture)."""
    enum = sorted(names)
    return {
        "activate_skill": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": enum, "description": "The skill's name, as listed in <available_skills>."},
            },
            "required": ["name"],
        },
        "read_skill_file": {
            "type": "object",
            "properties": {
                "skill": {"type": "string", "enum": enum, "description": "The skill's name."},
                "path": {"type": "string", "description": "The file's path inside the skill's folder."},
            },
            "required": ["skill", "path"],
        },
        "run_skill_script": {
            "type": "object",
            "properties": {
                "skill": {"type": "string", "enum": enum, "description": "The skill's name."},
                "script": {"type": "string", "description": "The script's path inside the skill's folder."},
                "args": {"type": "array", "items": {"type": "string"}, "description": "Command-line arguments for the script."},
                "timeout": {"type": "number", "description": "Seconds before the script is stopped (default 60, at most 300)."},
            },
            "required": ["skill", "script"],
        },
    }


def _inside(target: str, root: str) -> bool:
    root = root.rstrip(os.sep)
    return target == root or target.startswith(root + os.sep)


def _looks_binary(head: bytes) -> bool:
    return b"\x00" in head


class SkillMdSkill(Skill):
    """SKILL.md skills: a catalog in the prompt, activation on demand, and the
    skill's files and scripts behind two confined tools (module docstring)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {}, scope="owner")
        config = config or {}
        loaded: List[SkillMd] = list(config.get("skills") or [])
        self.skills: Dict[str, SkillMd] = {skill.name: skill for skill in sorted(loaded, key=lambda s: s.name)}
        self.skipped: List[SkippedSkill] = list(config.get("skipped") or [])
        self.warnings: List[str] = list(config.get("warnings") or [])
        self.agent_dir: str = os.path.realpath(str(config.get("agent_dir") or config.get("agent_path") or os.getcwd()))
        #: The agent file's `sandbox:` declaration (a `SandboxConfig` or a mapping), or None.
        self.sandbox_declaration = config.get("sandbox")
        self.timeout_limit = int(config.get("max_timeout") or MAX_TIMEOUT)
        parameters = tool_parameters(list(self.skills))
        self.activate_skill = _tool_function(self._activate, "activate_skill", ACTIVATE_DESCRIPTION, parameters["activate_skill"])
        self.read_skill_file = _tool_function(self._read, "read_skill_file", READ_DESCRIPTION, parameters["read_skill_file"])
        self.run_skill_script = _tool_function(self._run, "run_skill_script", RUN_DESCRIPTION, parameters["run_skill_script"])

    # ------------------------------------------------------------------
    # Tier 1: the catalog
    # ------------------------------------------------------------------

    def _activate_scope(self, context: Any) -> Any:
        """The scope `activate_skill` is registered with on the agent, which
        `access: tools:` may have rewritten; `owner` before the agent exists.
        The run's `context.agent` is the agent before `initialize` has set
        `self.agent` (skills start on the first turn)."""
        agent = self.agent if self.agent is not None else getattr(context, "agent", None)
        if agent is not None and hasattr(agent, "get_all_tools"):
            for config in agent.get_all_tools():
                if config.get("name") == "activate_skill" and config.get("source") in (self.skill_name, SKILL_KEY):
                    return config.get("scope") or "owner"
        return "owner"

    def _caller_may_activate(self, context: Any) -> bool:
        """Outside any run there is no caller: that is the process itself."""
        if context is None:
            return True
        from webagents.agents.core.scopes import scope_allows

        return scope_allows(self._activate_scope(context), context.auth_scopes)

    @prompt(priority=60, scope="all")
    def skills_catalog(self, context: Any = None) -> str:
        """The `<available_skills>` section, for a caller who may activate one."""
        if not self.skills or not self._caller_may_activate(context):
            return ""
        return catalog_text(list(self.skills.values()))

    # ------------------------------------------------------------------
    # Tier 2: activation
    # ------------------------------------------------------------------

    def _unknown(self, name: Any) -> str:
        available = ", ".join(self.skills) if self.skills else "none"
        return f'No skill called "{name}". Available: {available}'

    def _already_active(self, name: str) -> bool:
        """Whether an earlier tool result in this conversation opens the
        skill's tag: the conversation is the state."""
        context = self.get_context()
        messages = getattr(context, "messages", None) if context is not None else None
        marker = activation_marker(name)
        for message in messages or []:
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            content = message.get("content")
            if isinstance(content, str) and marker in content:
                return True
            if isinstance(content, list):
                for part in content:
                    text = part.get("text") if isinstance(part, dict) else None
                    if isinstance(text, str) and marker in text:
                        return True
        return False

    async def _activate(self, name: str = "", **_ignored: Any) -> str:
        skill = self.skills.get(str(name))
        if skill is None:
            return self._unknown(name)
        if self._already_active(skill.name):
            return f'Skill "{skill.name}" is already active in this conversation; its instructions are above.'
        return activation_text(skill)

    # ------------------------------------------------------------------
    # Tier 3: resources
    # ------------------------------------------------------------------

    def _resolve_inside(self, skill: SkillMd, relative: str) -> Optional[str]:
        """The real path of `relative` inside the skill's folder, or None when
        it is not a regular file there (a symbolic link out of the folder
        resolves outside it and is refused the same way)."""
        candidate = os.path.realpath(os.path.join(skill.directory, str(relative)))
        if not _inside(candidate, skill.directory) or not os.path.isfile(candidate):
            return None
        return candidate

    async def _read(self, skill: str = "", path: str = "", **_ignored: Any) -> str:
        loaded = self.skills.get(str(skill))
        if loaded is None:
            return self._unknown(skill)
        target = self._resolve_inside(loaded, path)
        if target is None:
            return f"Access denied: {path} is not a file inside the skill folder"
        try:
            size = os.path.getsize(target)
            if size > READ_MAX_BYTES:
                return f"Access denied: {path} is larger than 256 KiB"
            with open(target, "rb") as handle:
                data = handle.read()
        except OSError as error:
            return f"Error reading {path}: {error.strerror or error}"
        if _looks_binary(data[:8192]):
            return f"Access denied: {path} is not a text file"
        return data.decode("utf-8", errors="replace")

    def script_policy(self, skill: SkillMd):
        """The policy a script of `skill` runs under, or the refusal text.

        The agent's own `sandbox:` when it declares one; a synthesised
        `strict` policy with no writable folder (only the private scratch)
        when it declares none; a refusal for `unrestricted`. In every case
        the running skill's folder is readable, and EVERY skill's folder is
        write-denied (S-283, 2026-09-26): only the running skill's folder
        was, and the review's `pdf` script wrote `.agents/skills/other/
        SKILL.md`, instructions the next activation of `other` would have
        trusted. The agent's own files (`AGENT.md`, `WEBAGENTS.md`,
        `mcp.json`, `.agents/skills`) are denied by the policy's escalation
        set on top.
        """
        from webagents.sandbox import policy_from_metadata

        declared = self.sandbox_declaration
        try:
            if declared is None:
                base = policy_from_metadata({"preset": "strict", "allowed_folders": []}, cwd=self.agent_dir)
            else:
                base = policy_from_metadata(declared, cwd=self.agent_dir)
        except ValueError as error:
            return f"Access denied: invalid sandbox declaration: {error}"
        if base is None or not base.confined:
            return UNRESTRICTED_REFUSAL
        read_roots = list(base.read_roots)
        if base.scoped_reads and skill.directory not in read_roots:
            read_roots.append(skill.directory)
        read_only = list(base.read_only)
        for each in [skill, *self.skills.values()]:
            if each.directory not in read_only:
                read_only.append(each.directory)
        return dataclasses.replace(base, read_roots=read_roots, read_only=read_only)

    def script_state(self) -> str:
        """The state scripts run under, as the status row prints it: the agent
        file's preset (or `off`, which refuses them) with `(agent file)`, or
        `strict (default)`. For `doctor` and `/sandbox`: an agent that runs
        SKILL.md scripts is not one that cannot run commands. `--no-sandbox`
        does not apply to scripts."""
        if not self.skills:
            return ""
        from webagents.sandbox import policy_from_metadata, sandbox_state

        declared = self.sandbox_declaration is not None
        try:
            policy = policy_from_metadata(self.sandbox_declaration if declared else {"preset": "strict", "allowed_folders": []}, cwd=self.agent_dir)
        except ValueError:
            return "invalid (agent file)"
        return sandbox_state(policy, "agent file" if declared else "default")

    def script_command(self, target: str, args: Sequence[str]) -> Optional[str]:
        """The shell line for a bundled script, or None when it is not runnable."""
        extension = os.path.splitext(target)[1].lower()
        interpreter = INTERPRETERS.get(extension)
        if interpreter is None:
            if not os.access(target, os.X_OK):
                return None
            argv = [target, *args]
        else:
            argv = [interpreter, target, *args]
        return " ".join(shlex.quote(str(part)) for part in argv)

    async def _run(self, skill: str = "", script: str = "", args: Optional[Sequence[str]] = None, timeout: Any = None, **_ignored: Any) -> str:
        loaded = self.skills.get(str(skill))
        if loaded is None:
            return self._unknown(skill)
        target = self._resolve_inside(loaded, script)
        if target is None:
            return f"Access denied: {script} is not a file inside the skill folder"
        command = self.script_command(target, [str(a) for a in (args or [])])
        if command is None:
            return f"Access denied: {script} is not a script this agent can run (.py, .sh, .js, .mjs, .cjs, or an executable file)"
        policy = self.script_policy(loaded)
        if isinstance(policy, str):
            return policy
        try:
            seconds = int(float(timeout)) if timeout is not None else DEFAULT_TIMEOUT
        except (TypeError, ValueError):
            seconds = DEFAULT_TIMEOUT
        seconds = max(1, min(seconds, self.timeout_limit))

        from webagents.sandbox import INTERRUPTED_RESULT, CommandInterrupted, SandboxUnavailable, run_interruptibly, run_sandboxed

        try:
            # In a worker thread that the turn's Esc or Ctrl+C stops, process
            # group and all (the ptypass-fixes lane, 2026-09-27). This called
            # `run_sandboxed` on the event loop itself, so the chat could not
            # even see the keypress until the script ended or timed out.
            result = await run_interruptibly(run_sandboxed, command, policy, timeout=seconds)
        except SandboxUnavailable as error:
            return f"Access denied: {error}"
        except subprocess.TimeoutExpired:
            return f"Script timed out after {seconds}s"
        except CommandInterrupted:
            return INTERRUPTED_RESULT
        except Exception as error:  # noqa: BLE001 - reported, not raised
            return f"Error running {script}: {error}"
        output = result.stdout or ""
        if result.stderr:
            output += f"\nStderr: {result.stderr}"
        if result.returncode != 0:
            output += f"\nExit code: {result.returncode}"
        # A script is a confined command too: the same one sentence names the switch (fixture `hints`).
        from webagents.sandbox import refusal_hint

        hint = refusal_hint(command, f"{result.stdout or ''}\n{result.stderr or ''}")
        if hint:
            output += f"\n{hint}"
        return output if output else "(No output)"

    # ------------------------------------------------------------------
    # For doctor and `skills list`
    # ------------------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        return {
            "skills": [skill.name for skill in self.skills.values()],
            "skipped": [{"name": s.name, "location": s.location, "problem": s.problem, "reason": s.reason} for s in self.skipped],
            "warnings": list(self.warnings) + [f"{skill.name}: {w}" for skill in self.skills.values() for w in skill.warnings],
        }
