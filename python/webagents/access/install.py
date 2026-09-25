"""
Putting an agent file's `access:` block into effect (ADR-0045).

The loader calls `access_skill_for` with the block and the agent file, adds the
skill it returns under the name `access`, and after the agent is built calls
`apply_access_tools`, which gives every tool `access.tools` names the scopes of
the groups that name it. The access skill calls it again once skills have
started, for a tool a skill registers only then. The TypeScript twin is
`typescript/src/access/install.ts`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Collection, Dict, List, Optional

from .policy import AccessConfigError, AccessPolicy, parse_access


def instruction_texts(policy: AccessPolicy, agent_dir: Path) -> Dict[str, str]:
    """Each group's instructions file, read now, relative to the agent file."""
    texts: Dict[str, str] = {}
    for group, relative in policy.instructions.items():
        path = (agent_dir / relative).resolve()
        if not path.is_file():
            raise AccessConfigError(f"access.instructions.{group}: {relative} was not found next to the agent file.")
        texts[group] = path.read_text(encoding="utf-8").strip()
    return texts


def access_skill_for(raw: Any, agent_file: Optional[Path]):
    """`(AccessSkill, AccessPolicy)` for the block, or an `AccessConfigError`."""
    from webagents.agents.skills.local.access.skill import AccessSkill

    policy = parse_access(raw)
    agent_dir = agent_file.parent if agent_file is not None else Path.cwd()
    skill = AccessSkill({"policy": policy, "instruction_texts": instruction_texts(policy, agent_dir)})
    return skill, policy


def granted_scopes(policy: AccessPolicy) -> Dict[str, List[str]]:
    """Skill or tool name to the `group:<name>` scopes of the groups that name it."""
    out: Dict[str, List[str]] = {}
    for group, names in policy.tools.items():
        for name in names:
            scopes = out.setdefault(name, [])
            if f"group:{group}" not in scopes:
                scopes.append(f"group:{group}")
    return out


def apply_access_tools(agent: Any, policy: AccessPolicy, skill_names: Collection[str], *, strict: bool = True) -> None:
    """Give every tool the block names (by its skill's `skills:` name, or its own
    name) exactly the scopes of the groups naming it. `strict` refuses a name
    that is neither; the second, post-start pass is not strict."""
    grants = granted_scopes(policy)
    if not grants:
        return
    tools = agent.get_all_tools() if hasattr(agent, "get_all_tools") else []
    tool_names = {t.get("name") for t in tools}
    if strict:
        for group, names in policy.tools.items():
            for name in names:
                if name not in skill_names and name not in tool_names:
                    raise AccessConfigError(
                        f'access.tools.{group}: "{name}" is not a skill in this agent file or one of its tools.'
                    )
    with agent._registration_lock:
        for config in agent._registered_tools:
            scopes = grants.get(config.get("name")) or grants.get(config.get("source"))
            if scopes:
                config["scope"] = list(scopes)


def add_access(skills: Dict[str, Any], raw: Any, agent_file: Optional[Path]) -> Optional[AccessPolicy]:
    """Add the `access` skill to `skills` when the file has a block, and return the
    policy for `finish_access`; None, and `skills` untouched, when it has none.

    Every skill is renamed to its `skills:` name first, so a tool a skill
    registers only once started carries that name too, and `access.tools`
    can find it."""
    if raw is None:
        return None
    skill, policy = access_skill_for(raw, agent_file)
    for key, loaded in skills.items():
        if hasattr(loaded, "skill_name"):
            loaded.skill_name = key
    skill.skill_names = list(skills)
    skills["access"] = skill
    return policy


def finish_access(agent: Any, policy: Optional[AccessPolicy], skills: Dict[str, Any]) -> None:
    """After the agent is built: the scopes `access.tools` names."""
    if policy is not None:
        apply_access_tools(agent, policy, [k for k in skills if k != "access"], strict=True)
