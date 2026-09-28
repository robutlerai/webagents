---
title: SKILL.md Skills
description: Skills in the Agent Skills format (instructions, scripts and reference files in a folder), loaded when the model asks, with scripts run in the sandbox, the same in both SDKs, and installed from git at a pinned commit.
---

# SKILL.md Skills

An agent has two kinds of skill:

- **Coded skills**, named in `skills:`: typed tools, hooks, HTTP endpoints,
  pricing and scopes, written in the SDK's language. A Python skill runs in the
  Python SDK and a TypeScript one in the TypeScript SDK.
- **SKILL.md skills**, in the Agent Skills format: a folder holding a
  `SKILL.md` (a name, a description and instructions) and, optionally, scripts
  and reference files. They load and run the same way in both SDKs, and the
  same folder works in other agents that read the format.

`webagents skills list` shows both kinds, under separate headings.

## Where They Come From

- **`.agents/skills/<name>/SKILL.md`** beside the agent file is found without
  being named.
- **`agent_skills:`** in the agent file names other folders, relative to the
  file. Each is a skill folder, or a folder of skill folders:

```yaml
---
name: analyst
skills:
  - openai
  - shell
agent_skills:
  - ./team-skills
  - ../shared/pdf
---
```

`skills:` keeps meaning coded skills, so neither list can be mistaken for the
other.

## Writing One

```markdown
---
name: release-notes
description: Drafts release notes from the git history. Use when asked for release notes or a changelog.
---

# Release notes

1. Run `scripts/changes.sh <since>` to list the merged changes.
2. Group them under Added, Changed and Fixed, one line each.
3. Read `references/style.md` for the house style before writing.
```

The front matter takes `name` (at most 64 characters), `description` (at most
1,024), `license`, `compatibility`, `metadata` (strings) and `allowed-tools`.
The description is what the model reads to decide when to use the skill, so
say what it does and when. A skill is skipped only when its description is
missing or its front matter cannot be parsed; skips are reported when the
agent starts, in `webagents skills list` and in the `skills` check of
`webagents doctor`.

`allowed-tools` is a hint, never a grant: what the agent may run is decided by
its own skills, `access:` and `sandbox:`. Shell-substitution lines (`` !`cmd` ``)
and `$ARGUMENTS` in a SKILL.md are never executed.

## How the Model Uses Them

Skills load by progressive disclosure, so a large library costs little until
it is used:

1. The prompt lists each skill's name and description.
2. `activate_skill(name)` loads a skill's full instructions into the
   conversation, once, with the list of files it bundles.
3. `read_skill_file(skill, path)` reads a text file inside the skill's folder,
   up to 256 KiB.
4. `run_skill_script(skill, script, args, timeout)` runs a script from the
   folder (`.py`, `.sh`, `.js`, `.mjs`, `.cjs`, or an executable file) and
   returns its output. The timeout is 60 seconds unless the call asks for more,
   at most 300.

The three tools belong to one skill, `agent_skills`, and are the owner's. To
let other callers use SKILL.md skills, name it in the `access:` block:

```yaml
access:
  tools:
    partners: [agent_skills]
```

A caller who may not activate a skill does not see the list either.

## Scripts Always Run Confined

A skill's script runs through the sandbox (srt), under the agent's own
`sandbox:` and `network:` settings, or `strict` with no network when the agent
declares none. `unrestricted` is refused for scripts, and every skill folder
is read-only to them, their own included. A script that calls a service on
this machine goes through the sandbox's proxy (`HTTP_PROXY`), so the host must
be in `network:`. See [Sandbox](../cli/sandbox.md).

## Installing from Git

```bash
webagents skills add anthropics/skills --skill pdf
webagents skills add https://github.com/anthropics/skills/tree/main/skills/xlsx
webagents skills add git@github.com:your-org/skills.git
webagents skills add ./local-skills
```

A source is `owner/repo` on GitHub, a git URL, a GitHub `tree` or `blob` URL,
a `file://` URL or a folder. `--skill <name>` picks one skill from a source
that holds several.

- The repository is cloned shallow, at one commit, and searched where skill
  collections keep their skills (the root, `skills/`, `.agents/skills/`,
  `.claude/skills/` and the like, a few levels deep).
- It is limited to 10 MiB fetched, 25 MiB installed and 1,000 files. Symbolic
  links are refused.
- Every file is listed before anything is installed, with scripts and binaries
  flagged, and the command asks. Where nothing can answer, it needs `--yes`.
- The skill goes into `.agents/skills/<name>/`, named by its own `name:`, and
  `.webagents/skills.lock` records the source, the commit and a SHA-256 digest
  of the files. A folder the lock does not know is never replaced or removed.

```bash
webagents skills remove pdf     # removes only what the lock recorded
```

Installing a skill from someone else's repository is installing their code:
read what the listing shows, and remember that its scripts run with what your
`sandbox:` allows.

## In the Chat

`/skills` lists the agent's skills of both kinds, with each installed skill's
source and commit. `/skills add <source>` installs the same way, after the same
listing and question; the chat never accepts `--yes`. `/skills remove <name>`
removes an installed skill after asking. See [Chat](../cli/repl.md).
