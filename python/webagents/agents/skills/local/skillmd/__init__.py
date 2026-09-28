"""
SKILL.md skills (the Agent Skills format), gap-closure plan item 1.4,
2026-09-26. Three modules, each with a TypeScript twin under
`typescript/src/skills/skillmd/`:

- `skillmd_loader`: finding and parsing skills (`discover_skills`), and the
  words the model sees (the catalog and the activation text);
- `skillmd_skill`: the `SkillMdSkill` an agent carries, with the catalog in
  its system prompt and the `activate_skill`, `read_skill_file` and
  `run_skill_script` tools, owner-only unless `access: tools:` opens them;
- `skillmd_install`: `webagents skills add <git URL | owner/repo | path>`,
  which fetches at a commit, shows the files, installs into
  `.agents/skills/<name>` and records `.webagents/skills.lock`.

Every word and rule is pinned by `tests/fixtures/skillmd/skillmd.json`,
which both suites read.
"""

from .skillmd_loader import (
    AUTO_SKILLS_DIR,
    EXPLICIT_KEY,
    Discovery,
    SkillMd,
    SkippedSkill,
    activation_text,
    bundled_files,
    catalog_text,
    discover_skills,
    load_skill_dir,
    parse_skill_md,
)
from .skillmd_skill import SKILL_KEY, SkillMdSkill

__all__ = [
    "AUTO_SKILLS_DIR",
    "EXPLICIT_KEY",
    "SKILL_KEY",
    "Discovery",
    "SkillMd",
    "SkillMdSkill",
    "SkippedSkill",
    "activation_text",
    "bundled_files",
    "catalog_text",
    "discover_skills",
    "load_skill_dir",
    "parse_skill_md",
]
