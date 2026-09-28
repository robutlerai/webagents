"""
The agent files `webagents init` and the chat's `/agent new` write (2026-09-26,
interactive-mode spec 3.3): one table of templates and one function that
renders AGENT.md, so the two commands cannot drift.

Byte for byte what the TypeScript CLI writes (`cli/init-templates.ts`), pinned
by `python/tests/fixtures/cli/init_templates.json` (`agent_md`, `with_model`,
`with_key`). The chat passes the provider it runs on when this machine holds
that provider's key, so the new agent runs at once; `init` passes the model
`init_model` finds the same way.

NO KEY, THE CHAT'S OWN RULE (B3, 2026-09-28). With no model the file named
`openai/gpt-4o-mini` and listed `openai`, whatever the machine had. A new
developer with no OpenAI key then ran OpenAI's model through Robutler, and when
that route failed, `/model` refused every other provider because the file
named `openai`. With no provider key the file now names no model and no
provider skill, so it runs as the zero-config chat does: Robutler's choice,
`auto/balanced`, until a provider key is set, then that key's default model
(the quickstart's order, `init` then `secrets set`, keeps working). Two
comment lines say so, and how to pin a model.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Tuple

#: The `access:` block the tool-agent template ships with (S-248): shell and
#: filesystem stay owner-only until the owner names callers in `trusted`.
TOOL_AGENT_ACCESS: List[str] = [
    "access:",
    "  # shell and filesystem run as you, so only you (and admins) can use them.",
    "  # To let other callers use them, name the callers in the group:",
    "  # user:@handle, agent:https://host/**, key:<thumbprint>, domain:host.",
    "  groups:",
    "    trusted: []",
    "  tools:",
    "    trusted: [filesystem, shell]",
]

#: The `sandbox:` block the tool-agent template ships with (2026-09-26, the
#: e2e run): a template that runs shell commands declared no sandbox, so its
#: own `doctor` warned "off: shell commands run with your permissions" on the
#: first run. `development` confines writes to the folder and allows no
#: network until hosts are listed.
TOOL_AGENT_SANDBOX: List[str] = [
    "sandbox:",
    "  # Shell commands run confined: writes stay in this folder, no network.",
    "  # List hosts under network: to let commands reach them.",
    "  preset: development",
]

#: What `init` can make: description, skills, the `sandbox:` lines and the
#: `access:` lines. The descriptions are the files' `description:` too
#: (2026-09-26): the file said "A tool-agent agent", which described nothing.
#: No colon in them: a plain YAML scalar cannot hold `: `.
INIT_TEMPLATES: Dict[str, Tuple[str, List[str], List[str], List[str]]] = {
    "chatbot": ("A chat agent with one model and no tools", [], [], []),
    "tool-agent": (
        "Reads and writes files and runs shell commands",
        ["filesystem", "shell"],
        TOOL_AGENT_SANDBOX,
        TOOL_AGENT_ACCESS,
    ),
}

#: What a new agent with no provider key runs on while none is set: the
#: zero-config chat's choice (`model_access.PROXY_DEFAULT_MODEL`).
ROBUTLER_CHOICE_MODEL = "auto/balanced"

#: The lines a file with no `model:` carries in its place, so it says what runs.
NO_MODEL_COMMENT: List[str] = [
    "# No model named: a provider key's default model when one is set, else",
    "# Robutler's choice (auto/balanced). Add a model: line to pin one.",
]

#: The names, in the order `templates list` shows them.
TEMPLATE_NAMES: List[str] = list(INIT_TEMPLATES)

#: What `/agent new` accepts as a name (fixture `name_grammar`).
AGENT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")


def init_model(has_key: Optional[Callable[[str], bool]] = None) -> Optional[str]:
    """The model `init` writes: the default model of the first provider whose
    key this machine holds (the shell, or `secrets set`) and whose client is
    installed, as the zero-config chat would run it; None, Robutler's choice,
    with no key. The TypeScript `initModel` answers the same."""
    from webagents.agents.skills.core.llm.providers import LLM_PROVIDERS

    from .model_access import usable_directly

    if has_key is None:
        from .skills_edit import machine_facts

        has_key = machine_facts().has_key
    for provider in LLM_PROVIDERS:
        if provider.credential != "api_key" or not provider.default_model:
            continue
        if any(has_key(variable) for variable in provider.env_vars) and usable_directly(provider, {v: "set" for v in provider.env_vars}):
            return f"{provider.id}/{provider.default_model}"
    return None


def init_line(model: str, keyed: bool, signed_in: bool) -> str:
    """The last line `init` prints: what the new agent runs on and, only when
    it is needed, the way in (fixture `init_line`; the TypeScript `initLine`).
    It said "add your key ..., or sign in ..." to everyone, the signed in
    included (2026-09-28)."""
    from webagents.agents.skills.core.llm.providers import find_provider

    from .config_store import cli_command

    if keyed:
        provider = find_provider(model.split("/")[0])
        key = provider.env_vars[0] if provider is not None and provider.env_vars else "its key"
        return f"It runs on {model} with your {key}."
    if signed_in:
        return f"It runs on {model} through Robutler, paid from your Robutler credits, until a provider key is set."
    return (
        f"It runs on {model} through Robutler once you sign in with `{cli_command('login')}`, "
        f"or on your own provider key: `{cli_command('secrets set OPENAI_API_KEY')}`."
    )


def agent_markdown(name: str, template: str, model: Optional[str] = None) -> str:
    """AGENT.md for `name` from `template` (module docstring). With `model`
    (`provider/model`), that provider's skill leads the list and the model line
    names it; a model only Robutler serves (`auto/...`) is named with no
    provider skill; without one, the file names no model and says what runs."""
    from webagents.agents.skills.core.llm.providers import find_provider

    if template not in INIT_TEMPLATES:
        raise ValueError(f"Unknown template '{template}'.")
    description, skills, sandbox, access = INIT_TEMPLATES[template]
    skills = list(skills)
    prefix = model.split("/")[0] if model else ""
    lines = ["---", f"name: {name}", f"description: {description}"]
    if not model:
        lines += NO_MODEL_COMMENT
    elif prefix in ("auto", "proxy", "robutler"):
        lines.append(f"model: {model}")
    else:
        provider = find_provider(prefix)
        skills = [provider.id if provider is not None else "openai"] + skills
        lines.append(f"model: {model}")
    if skills:
        lines.append("skills:")
        lines += [f"  - {skill}" for skill in skills]
    lines += sandbox
    lines += access
    lines += ["---", "", f"# {name}", "", "You are a helpful assistant.", ""]
    return "\n".join(lines)
