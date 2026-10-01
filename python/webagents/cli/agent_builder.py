"""
Building a CLI agent from its file (2026-09-24): one place for the daemon and
the chat.

The daemon (`server/extensions/local_file_source.py`) built agents with its
own skill table, and the chat went through the daemon to reach them. The chat
now builds the agent itself, in its own process, as the TypeScript chat does,
so the two must build the SAME agent: the skills the file lists (none when
it lists none), the file's `sandbox:` handed to the shell, and the model decided
by `model_access.choose_model_access`.

What the chat gets that the daemon does not need: an agent is built even when
no model can run (no key, not signed in). The chat then says so and offers the
ways out, rather than having nothing to show `/tools` or `/status` with.
"""

from __future__ import annotations

import importlib
import os
import sys
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Collection, Dict, List, Optional, Sequence, Union

logger = logging.getLogger("webagents.cli.agent_builder")

#: Every skill an agent file can name, by the name it uses.
SKILL_CLASSES: Dict[str, str] = {
    "filesystem": "webagents.agents.skills.local.filesystem.skill.FilesystemSkill",
    "shell": "webagents.agents.skills.local.shell.skill.ShellSkill",
    "rag": "webagents.agents.skills.local.rag.skill.LocalRagSkill",
    "session": "webagents.agents.skills.local.session.skill.SessionSkill",
    # Memory scoped by verified caller (plan item 2.1, 2026-09-26): `- memory`
    # or `- memory: {local, portal, ...}`, the TypeScript `memory` loader's
    # twin, pinned by `tests/fixtures/memory_tool/definition.json`.
    "memory": "webagents.agents.skills.local.memory.caller_scoped.MemorySkill",
    # LLM skills. A provider's other names (`llm`, `gemini`, `claude`, `grok`)
    # resolve to these through the provider registry, as in TypeScript.
    "google": "webagents.agents.skills.core.llm.google.skill.GoogleAISkill",
    # Robutler's models, with the sign-in or `ROBUTLER_LLM_PROXY_URL`
    # (`load_skills` gives it the socket and the sign-in, as TypeScript does).
    "proxy": "webagents.agents.skills.core.llm.proxy.skill.LLMProxySkill",
    "openai": "webagents.agents.skills.core.llm.openai.skill.OpenAISkill",
    # Local models through Ollama (plan item 2.8, 2026-09-26).
    "ollama": "webagents.agents.skills.core.llm.ollama.skill.OllamaSkill",
    "anthropic": "webagents.agents.skills.core.llm.anthropic.skill.AnthropicSkill",
    "xai": "webagents.agents.skills.core.llm.xai.skill.XAISkill",
    "fireworks": "webagents.agents.skills.core.llm.fireworks.skill.FireworksAISkill",
    # Local skills
    "web": "webagents.agents.skills.local.web.skill.WebSkill",
    "rest": "webagents.agents.skills.local.rest.skill.RestSkill",
    "todo": "webagents.agents.skills.local.todo.skill.TodoSkill",
    # The platform search, the TypeScript `search` tool (2026-09-25).
    "discovery": "webagents.agents.skills.robutler.discovery.skill.DiscoverySkill",
    # TrustFlow (plan items 2.5 and 2.7, 2026-09-26): the `trust` tool and the
    # agent's own signed record, the TypeScript `trust` loader's twin, pinned
    # by `tests/fixtures/trust/trust_tool_definition.json`.
    "trust": "webagents.agents.skills.robutler.trust.trust_skill.TrustSkill",
    "mcp": "webagents.agents.skills.local.mcp.skill.LocalMcpSkill",
    "sandbox": "webagents.agents.skills.local.sandbox.skill.SandboxSkill",
    # Transport skills - always available
    "completions": "webagents.agents.skills.core.transport.completions.skill.CompletionsTransportSkill",
    "a2a": "webagents.agents.skills.core.transport.a2a.skill.A2ATransportSkill",
    "realtime": "webagents.agents.skills.core.transport.realtime.skill.RealtimeTransportSkill",
    "acp": "webagents.agents.skills.core.transport.acp.skill.ACPTransportSkill",
}


def skill_name_of(item: Union[str, Dict[str, Any]]) -> Optional[str]:
    """The name a `skills:` entry uses: `shell`, or the key of `{shell: {...}}`."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict) and len(item) == 1:
        return next(iter(item))
    return None


def load_skills(
    skills_config: List[Union[str, Dict[str, Any]]],
    agent_name: str,
    agent_path: Optional[Path] = None,
    sandbox: Any = None,
    model: Optional[str] = None,
    report: Optional[Dict[str, List[Any]]] = None,
    person_token: Optional[Any] = None,
    agent_skills: Optional[Sequence[str]] = None,
    confirm_control_write: Optional[Any] = None,
    for_callers: bool = False,
) -> Dict[str, Any]:
    """Instantiate the skills a file lists. Unknown names and failing skills are skipped,
    and recorded in `report` (`unknown`: names; `failed`: `(name, reason)`) for the
    caller to say, as the TypeScript `resolveSkillsByName` returns them.

    SKILL.md skills come too (plan item 1.4, 2026-09-26), when `agent_path`
    names where the agent lives: `.agents/skills/*` beside it and the folders
    `agent_skills` (the file's `agent_skills:`) names, as ONE skill under
    `agent_skills` (`SkillMdSkill`), owner-only until `access: tools:` opens
    it. A skill that cannot load is reported in `report["skillmd_skipped"]`
    and `report["skillmd_warnings"]`, never fatal, as `resolveSkillsByName`
    reports it in `skillmd`.

    A model provider's skill (`openai`, `anthropic`, ...) runs `model` when it
    is that provider's, else the provider's default model, unless its own
    config names one. THE FILE'S `model:` NEVER REACHED IT (2026-09-24): the
    skill was built with no config and ran its hard-coded default, so an agent
    declaring `openai/gpt-4o-mini` with `skills: [openai]` was billed for
    `gpt-4o`, and a second LLM skill built from `model:` sat beside it unused.
    The TypeScript SDK passes the model the same way (`resolveSkillsByName`).

    `for_callers` (`serve`, `mcp serve`, the daemon): a listed `proxy` skill is
    built with NO sign-in (S-327), as `model_access.proxy_skill_for` builds
    one for other callers' turns.
    """
    from webagents.agents.skills.core.llm.providers import find_provider

    from .model_access import model_for_named_provider

    loaded_skills: Dict[str, Any] = {}

    for item in skills_config:
        skill_name = None
        config: Dict[str, Any] = {}

        if isinstance(item, str):
            skill_name = item
        elif isinstance(item, dict):
            # item is like {"filesystem": {"whitelist": [...]}}
            # or {"mcp": {"sqlite": {...}}}
            if len(item) == 1:
                skill_name = list(item.keys())[0]
                if skill_name == "mcp":
                    # The entry, in either shape, under `mcp` for the skill's
                    # normalizer (`skills/local/mcp/config.py`, pinned by the
                    # shared fixture). Passing the `mcpServers` shape UNWRAPPED
                    # here made the skill scan its keys for servers one level
                    # too high, so that shape connected to nothing (2026-09-26).
                    config = {"mcp": item[skill_name] or {}}
                else:
                    config = item[skill_name] or {}

        if skill_name == "mcp" and "references" not in config:
            # An agent FILE, run by its local owner, is the one place
            # `${env:NAME}` and `${secret:NAME}` resolve (S-295): the sources
            # are the owner's own environment and keystore. FOR BOTH SHAPES
            # (2026-09-27, the final e2e re-run): a bare `- mcp`, which reads
            # the folder's `mcp.json`, got no sources, so a reference written
            # there was used as its literal bytes, where TypeScript
            # (`skills/resolve.ts`) resolved it. Pinned by the shared fixture
            # `mcp_tool/config_shapes.json`, `secret_refs.folder_mcp_json`.
            from webagents.agents.skills.local.mcp.skill import owner_reference_sources

            config["references"] = owner_reference_sources()

        provider = find_provider(skill_name) if skill_name else None
        if provider is not None and provider.id in SKILL_CLASSES:
            # A provider's other name (`llm`, `gemini`, `claude`) is the provider.
            skill_name = provider.id
        if not skill_name or skill_name not in SKILL_CLASSES:
            if report is not None and skill_name:
                report.setdefault("unknown", []).append(skill_name)
            continue

        if skill_name == "proxy":
            # The TypeScript loader's rule: the model as written, less a
            # `proxy/` prefix, which names the route rather than the model.
            from .model_access import platform_llm_url

            if not config.get("model") and model:
                config["model"] = model[len("proxy/"):] if model.startswith("proxy/") else model
            config.setdefault("proxy_url", platform_llm_url())
            if for_callers:
                # Other callers' turns never run on the sign-in (S-327): each
                # is paid by the caller's payment token (`callers_pay`), and a
                # `platform_token` the file wrote itself is not used either.
                config.pop("platform_token", None)
                config["callers_pay"] = True
            else:
                from .credentials import get_token

                config.setdefault("platform_token", get_token)
        elif provider is not None and not config.get("model"):
            chosen = model_for_named_provider(provider, model)
            if chosen:
                config["model"] = chosen.split("/", 1)[1] if chosen.startswith(f"{provider.id}/") else chosen

        # `discovery` searches as the person at this terminal when the agent
        # has no platform credential of its own. ONLY the chat and `-p` pass
        # `person_token` (`build_agent`); `serve` and the daemon must not, or
        # every caller would search as the owner (the skill's `person_token`).
        if skill_name == "discovery" and person_token is not None:
            config.setdefault("person_token", person_token)

        # Inject agent name into config if needed (e.g. for session skill)
        config["agent_name"] = agent_name
        # Pass agent DIRECTORY, not the file path. Not to `discovery`, whose
        # `agent_path` is the server's URL prefix (the path its signatures
        # name); the folder there would sign for a URL nothing serves.
        if skill_name != "discovery":
            config["agent_path"] = str(agent_path.parent) if agent_path else None

        # Inject agent directory into filesystem/shell config
        if agent_path:
            agent_dir = str(agent_path.parent.resolve())

            if skill_name == "filesystem":
                whitelist = config.get("whitelist", [])
                if agent_dir not in whitelist:
                    whitelist.append(agent_dir)
                config["whitelist"] = whitelist
                config["base_dir"] = agent_dir
                # The agent's own file, so a write to it asks under any name
                # (S-314), and the chat's yes/no for such a write: ONLY the
                # interactive chat passes it; `-p`, `serve` and the daemon
                # leave it unset, and the skill refuses the write instead.
                config["agent_file"] = str(agent_path.resolve())
                if confirm_control_write is not None and "confirm_control_write" not in config:
                    config["confirm_control_write"] = confirm_control_write
            elif skill_name == "shell":
                config["base_dir"] = agent_dir
                # The agent file's own `sandbox:`, so `ShellSkill` can
                # enforce it at the OS level rather than with a string
                # check. A skill-level `sandbox` key still wins, because a
                # more specific declaration should.
                if sandbox is not None and "sandbox" not in config:
                    config["sandbox"] = sandbox

        try:
            module_path, class_name = SKILL_CLASSES[skill_name].rsplit(".", 1)
            module = importlib.import_module(module_path)
            skill_class = getattr(module, class_name)
            loaded_skills[skill_name] = skill_class(config)
        except Exception as exc:  # noqa: BLE001 - a skill that cannot load is skipped, and said
            logger.warning(f"{agent_name}: skill {skill_name} did not load: {exc}")
            if report is not None:
                report.setdefault("failed", []).append((skill_name, str(exc)))

    if agent_path is not None:
        from webagents.agents.skills.local.skillmd import SKILL_KEY, SkillMdSkill, discover_skills

        agent_dir = str(agent_path.parent.resolve())
        found = discover_skills(agent_dir, list(agent_skills or []))
        if found.skills:
            loaded_skills[SKILL_KEY] = SkillMdSkill({
                "skills": found.skills,
                "skipped": found.skipped,
                "warnings": found.warnings,
                "agent_dir": agent_dir,
                "sandbox": sandbox,
                "agent_name": agent_name,
            })
        if report is not None:
            if found.skipped:
                report.setdefault("skillmd_skipped", []).extend(found.skipped)
            if found.warnings:
                report.setdefault("skillmd_warnings", []).extend(found.warnings)

    return loaded_skills


def say_skillmd_report(report: Dict[str, List[Any]], say: Any = None) -> None:
    """What did not load among the SKILL.md skills, one line each, in the
    words the TypeScript loaders print (pinned by
    `tests/fixtures/skillmd/skillmd.json`, `messages.load_skipped` and
    `messages.load_warning`)."""
    emit = say or (lambda line: print(line, file=sys.stderr))
    for skipped in report.get("skillmd_skipped", []):
        emit(f"SKILL.md skill {skipped.name} at {skipped.location} skipped: {skipped.reason}")
    for warning in report.get("skillmd_warnings", []):
        emit(f"SKILL.md skills: {warning}")


@dataclass
class BuiltAgent:
    """An agent the chat built, and what the chat says about it."""

    agent: Any
    name: str
    description: str = ""
    #: The agent file; None for the built-in agent.
    file: Optional[Path] = None
    #: How the model is reached (`model_access.ModelAccess`); None when the file names its LLM skill.
    access: Any = None
    #: The model as the card and the footer show it.
    model_label: str = ""
    #: Why there is no model to run on, naming the ways out; None when there is one.
    model_problem: Optional[str] = None
    #: The file's `sandbox:` declaration, or None.
    sandbox: Any = None
    skills: List[str] = field(default_factory=list)
    #: The skill names the agent file declares (the built-in one's too), for `/skills` (2026-09-27).
    declared_skills: List[str] = field(default_factory=list)
    #: Where the chat keeps conversations: `local`, or on Robutler too (`session: {backend: robutler}`).
    session_backend: str = "local"
    #: The file's `access:` block, parsed (`webagents.access.policy.AccessPolicy`),
    #: for the chat's `/access` (interactive-mode spec 3.7); None without one.
    access_policy: Any = None


def _named_skill_model(skills: Dict[str, Any]) -> Optional[str]:
    """`provider/model` of the model provider skill the file names, if any."""
    from webagents.agents.skills.core.llm.providers import find_provider

    for name, skill in skills.items():
        provider = find_provider(name)
        own = getattr(skill, "model", None)
        if provider is not None and own:
            return own if "/" in own else f"{provider.id}/{own}"
    return None


def llm_skill_for_model(model: str, *, for_callers: bool = False) -> Any:
    """The provider skill for `provider/model` (a fallback, or a direct
    primary): the construction `BaseAgent._process_model_parameter` does from
    a model string, plus Robutler's models through the proxy (with no sign-in
    for other callers' turns, S-327). Raises with the reason when this SDK has
    no client for the provider."""
    prefix, _, bare = model.partition("/")
    if prefix in ("auto", "proxy", "robutler"):
        from .model_access import PROXY_DEFAULT_MODEL, proxy_skill_for

        return proxy_skill_for(model if prefix == "auto" else (bare or PROXY_DEFAULT_MODEL), for_callers=for_callers)
    from webagents.agents.skills.core.llm.providers import find_provider

    provider = find_provider(prefix)
    if provider is None or provider.id not in SKILL_CLASSES:
        raise ValueError(f"this SDK has no client for {prefix or model}")
    module_path, class_name = SKILL_CLASSES[provider.id].rsplit(".", 1)
    skill_class = getattr(importlib.import_module(module_path), class_name)
    return skill_class({"model": bare})


def apply_fallback_models(
    skills: Dict[str, Any],
    fallback_models: Sequence[str],
    primary_model: Optional[str],
    agent_name: str,
    *,
    for_callers: bool = False,
) -> "tuple[Optional[str], List[tuple[str, str]]]":
    """The agent's LLM skill wrapped with its `fallback_models:` (plan item
    2.8, 2026-09-26; `llm/failover.py`), the TypeScript `withFallbackModels`.

    `primary_model` is the `provider/model` `BaseAgent` would build the
    primary from (a direct model); None when a skill in `skills` carries it
    (the file's own LLM skill, or Robutler's proxy). The primary and every
    fallback that can be built here form the chain, which takes the
    primary's place in `skills`. Returns the model to build the agent with
    (None once the chain carries the primary) and the fallbacks that could
    not be built, `(name, reason)`, said by the caller and never fatal.

    `for_callers` (`serve`, the daemon): a fallback to Robutler's models needs
    the agent's own platform credential, never the sign-in (S-327).
    """
    failed: List[tuple[str, str]] = []
    if not fallback_models:
        return primary_model, failed
    from webagents.agents.skills.core.llm.failover import FailoverLLMSkill
    from webagents.agents.skills.core.llm.providers import find_provider

    from .model_access import (
        LLM_SKILL_NAMES,
        _why_not_direct,
        agent_has_own_platform_credential,
        is_signed_in,
        usable_directly,
    )

    key = "llm"
    if primary_model:
        try:
            primary_skill = llm_skill_for_model(primary_model, for_callers=for_callers)
        except Exception as exc:  # noqa: BLE001 - said, and the agent runs as before
            failed.append((f"fallback for {primary_model}", str(exc)))
            return primary_model, failed
        label = primary_model
    else:
        key = next((name for name in skills if name in LLM_SKILL_NAMES), None)
        if key is None:
            return primary_model, failed
        primary_skill = skills[key]
        own = getattr(primary_skill, "model", None) or "unknown"
        if type(primary_skill).__name__ == "LLMProxySkill" or "/" in own:
            label = own
        else:
            provider = find_provider(key)
            label = f"{provider.id}/{own}" if provider is not None else own
    chain = [(label, primary_skill)]
    for fallback in fallback_models:
        prefix = fallback.partition("/")[0]
        if prefix in ("auto", "proxy", "robutler"):
            if for_callers and not agent_has_own_platform_credential(agent_name):
                failed.append((f"fallback {fallback}", "Robutler's models need the agent's own platform credential here"))
                continue
            if not for_callers and not is_signed_in():
                failed.append((f"fallback {fallback}", "Robutler's models need a sign-in"))
                continue
        else:
            provider = find_provider(prefix)
            if provider is None or provider.id not in SKILL_CLASSES:
                failed.append((f"fallback {fallback}", f"this SDK has no client for {prefix or fallback}"))
                continue
            if not usable_directly(provider):
                failed.append((f"fallback {fallback}", _why_not_direct(provider, None)))
                continue
        try:
            chain.append((fallback, llm_skill_for_model(fallback, for_callers=for_callers)))
        except Exception as exc:  # noqa: BLE001 - said, and the chain goes on without it
            failed.append((f"fallback {fallback}", str(exc)))
    if len(chain) == 1:
        return primary_model, failed
    skills[key] = FailoverLLMSkill(chain)
    logger.info(f"{agent_name}: model failover over {', '.join(model for model, _ in chain)}")
    return None, failed


def provider_env_var(built: BuiltAgent) -> Optional[str]:
    """The key the agent's model runs on (`OPENAI_API_KEY`), for "with your ..."
    in `/status` and `doctor`: the provider the model decision chose, or the
    LLM skill the file names when its own choice stands. The first of the
    provider's variables that is set, else its first."""
    import os

    from webagents.agents.skills.core.llm.providers import find_provider

    provider = getattr(built.access, "provider", None) if built.access is not None else None
    if provider is None and built.access is None:
        provider = next((p for p in (find_provider(name) for name in built.skills) if p is not None), None)
    env_vars = getattr(provider, "env_vars", None) or ()
    if not env_vars or getattr(provider, "credential", "api_key") != "api_key":
        return None
    return next((name for name in env_vars if os.environ.get(name)), env_vars[0])


async def build_agent(
    agent_file: Optional[Path],
    *,
    working_dir: Path,
    model: Optional[str] = None,
    exclude: Collection[str] = ("session",),
    bare: bool = False,
    initialize: bool = True,
    strict_skills: bool = False,
    person_token: Optional[Any] = None,
    confirm_control_write: Optional[Any] = None,
    for_callers: bool = False,
) -> BuiltAgent:
    """Build the agent in `agent_file`, or the built-in one, the way the daemon does.

    `model` is a model the person chose (`/model`), ahead of the file's.
    `exclude` drops skills by name: the chat keeps its own conversations, so
    the session skill (which saves them from its hooks) would be a second
    writer. `bare`, with no file, builds an agent called `agent` with no
    instructions and no tools (`serve` with nothing to serve, as the
    TypeScript `serve` does) rather than the built-in assistant.
    `initialize=False` leaves the skills to start on first use, in the event
    loop that serves them (`serve`). `person_token` answers the signed-in
    person's platform token, for `discovery` to search with when the agent
    has none of its own: the chat and `-p` pass it, `serve` never does.
    `confirm_control_write` is the chat's yes/no (shown a diff) for a file-tool
    write to one of the agent's control files (S-314): ONLY the interactive
    chat passes it; everywhere else the file tools refuse such a write.
    `for_callers` is `serve` and `mcp serve` (S-327): the turns are other
    callers', so the model never runs on the sign-in
    (`model_access.choose_model_access`).
    """
    from webagents.agents.core.base_agent import BaseAgent

    from .loader import AgentFormatError
    from .preamble import with_cli_preamble
    from .loader.hierarchy import load_agent
    from .model_access import ModelUnavailable, choose_model_access

    if agent_file is not None:
        merged = load_agent(agent_file)
        name = merged.metadata.name or merged.name
        skill_anchor = agent_file
    elif bare:
        return await _bare_agent(working_dir, model, initialize, for_callers=for_callers)
    else:
        from webagents.agents.builtin import get_robutler_path

        merged = load_agent(get_robutler_path())
        name = "robutler"
        # The built-in agent works in the folder the chat started in.
        skill_anchor = working_dir / "AGENT.md"

    # The skills the file names, and none when it names none, as in the
    # TypeScript SDK (see `server/extensions/local_file_source.py`).
    skills_list = [s for s in (merged.metadata.skills or []) if skill_name_of(s) not in set(exclude)]
    # Where the chat keeps conversations: the `session` entry says, and the
    # chat leaves the skill itself out (`exclude`), since it keeps them itself.
    session_entry = next(
        (s for s in (merged.metadata.skills or []) if (skill_name_of(s) or "").lower() == "session"), None
    )
    session_config = session_entry.get(skill_name_of(session_entry)) if isinstance(session_entry, dict) else None
    session_backend = (
        "robutler" if isinstance(session_config, dict) and session_config.get("backend") == "robutler" else "local"
    )
    report: Dict[str, List[Any]] = {}
    skills = load_skills(
        skills_list,
        agent_name=name,
        agent_path=skill_anchor,
        sandbox=merged.metadata.sandbox,
        model=model or merged.metadata.model,
        report=report,
        person_token=person_token,
        agent_skills=merged.metadata.agent_skills,
        confirm_control_write=confirm_control_write,
        for_callers=for_callers,
    )
    say_skillmd_report(report)
    # What did not load, said as the TypeScript CLI says it (`cli/app.ts`,
    # `serve-action.ts`): `serve` refuses a file naming a skill that does not
    # exist, and the chat and `-p` skip it with a line. The embedded
    # assistant's own skills are ours to fix, so they are said only when
    # debugging.
    unknown = report.get("unknown", [])
    if unknown and strict_skills:
        raise AgentFormatError(
            f"Unknown skill(s) in agent config: {', '.join(unknown)}. "
            "Run `webagents skills list` for the available names."
        )
    if agent_file is not None or os.environ.get("WEBAGENTS_DEBUG"):
        where = str(agent_file) if agent_file is not None else "the embedded agent"
        for unknown_name in unknown:
            print(f'Unknown skill "{unknown_name}" in {where}; skipping.', file=sys.stderr)
    for failed_name, reason in report.get("failed", []):
        print(f'Skill "{failed_name}" failed to load: {reason}', file=sys.stderr)
    # Who may call it, and what each group gets (ADR-0045).
    from webagents.access.install import add_access, finish_access
    from webagents.access.policy import AccessConfigError

    try:
        access_policy = add_access(skills, merged.metadata.access, agent_file)
    except AccessConfigError as e:
        raise AgentFormatError(f"{agent_file}: {e}") from None

    problem: Optional[str] = None
    access = None
    try:
        access = choose_model_access(model or merged.metadata.model, skills, name, for_callers=for_callers)
    except ModelUnavailable as unavailable:
        access = unavailable.access
        problem = str(unavailable)

    # With the file's own LLM skill standing (`access is None`), that skill
    # carries the model (`load_skills`); a `model=` here would build a second one.
    agent_model = access.model if access is not None and access.kind == "direct" else None

    # `fallback_models:` (plan item 2.8): the model's skill becomes a chain
    # that moves to the next model on a provider error, with a note in the
    # transcript. A fallback that cannot be built here is said and left out.
    if problem is None and merged.metadata.fallback_models:
        agent_model, fallback_failed = apply_fallback_models(
            skills, merged.metadata.fallback_models, agent_model, name, for_callers=for_callers
        )
        for failed_name, reason in fallback_failed:
            print(f'Skill "{failed_name}" failed to load: {reason}', file=sys.stderr)

    agent = BaseAgent(
        name=name,
        # Where the instructions come from, the folder and the small-talk rule,
        # for an agent that can explore its folder (`preamble.py`).
        instructions=with_cli_preamble(merged.instructions, agent_file, merged.metadata.skills),
        skills=skills,
        scopes=merged.metadata.scopes or ["all"],
        model=agent_model,
    )
    # The file's `description:` on the agent (B7, 2026-09-28): the A2A card,
    # the listing and `/info` read `agent.description`, which `BaseAgent`
    # never set, so a Python agent's card said `"description": ""` where the
    # TypeScript one carried the file's sentence.
    agent.description = merged.metadata.description or ""
    # `observability: {otel: true}` in the file records the run as
    # OpenTelemetry spans (plan item 2.4, `webagents/observability/otel.py`).
    agent.observability = merged.metadata.observability if isinstance(merged.metadata.observability, dict) else None
    # The tool rounds a turn may run (2026-09-28, `agents/core/tool_budget.py`):
    # `--max-tool-rounds`, then the file's `max_tool_rounds`, then 50.
    from webagents.agents.core.tool_budget import effective_max_tool_rounds

    try:
        agent.max_tool_iterations, agent.max_tool_rounds_source = effective_max_tool_rounds(merged.metadata.max_tool_rounds)
    except ValueError as e:
        raise AgentFormatError(str(e)) from None
    # Compaction (2026-09-29, `agents/core/context_compaction.py`): the file's
    # `compaction:` block; without one, the `memory` skill's old
    # `compaction.threshold` (tokens) is its `at`, so a file that set it keeps it.
    from webagents.agents.core.context_compaction import CompactionPolicy, parse_policy

    policy = parse_policy(merged.metadata.compaction)
    if merged.metadata.compaction is None:
        memory = next((s for s in (skills or {}).values() if getattr(s, "compaction", None) and hasattr(s, "on_compaction")), None)
        threshold = (getattr(memory, "compaction", None) or {}).get("threshold") if memory is not None else None
        if threshold:
            policy = CompactionPolicy(at=float(threshold), hard=float(max(int(threshold) + 1, int(threshold * 1.2))))
    agent.compaction_policy = policy
    try:
        finish_access(agent, access_policy, skills)
    except AccessConfigError as e:
        raise AgentFormatError(f"{agent_file}: {e}") from None
    if initialize:
        await agent._ensure_skills_initialized()

    if problem is not None:
        label = model or merged.metadata.model or ""
    elif access is None:
        label = _named_skill_model(skills) or model or merged.metadata.model or ""
    else:
        label = access.describe()

    return BuiltAgent(
        agent=agent,
        name=name,
        description=merged.metadata.description or "",
        file=agent_file,
        access=access,
        model_label=label,
        model_problem=problem,
        sandbox=merged.metadata.sandbox,
        skills=list(skills),
        declared_skills=[name for name in (skill_name_of(s) for s in skills_list) if name],
        session_backend=session_backend,
        access_policy=access_policy,
    )


async def _bare_agent(working_dir: Path, model: Optional[str], initialize: bool, *, for_callers: bool = False) -> BuiltAgent:
    """An agent called `agent`: no instructions, no tools, and the usual model decision."""
    from webagents.agents.core.base_agent import BaseAgent

    from .model_access import ModelUnavailable, choose_model_access

    skills: Dict[str, Any] = {}
    problem: Optional[str] = None
    try:
        access = choose_model_access(model, skills, "agent", for_callers=for_callers)
    except ModelUnavailable as unavailable:
        access, problem = unavailable.access, str(unavailable)
    agent = BaseAgent(
        name="agent",
        instructions="",
        skills=skills,
        scopes=["all"],
        model=access.model if access is not None and access.kind == "direct" else None,
    )
    if initialize:
        await agent._ensure_skills_initialized()
    label = (model or "") if problem is not None else (access.describe() if access is not None else "")
    return BuiltAgent(agent=agent, name="agent", access=access, model_label=label, model_problem=problem, skills=list(skills))
