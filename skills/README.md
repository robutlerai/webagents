# Skills

Ready-made skills for agents. Each folder is one skill: a `SKILL.md` with
instructions, sometimes with reference files beside it. They follow the open
Agent Skills format, so they work in webagents and in any other agent that
reads `SKILL.md`.

## Using one

Install a skill into an agent's folder:

```bash
webagents skills add robutlerai/webagents --skill security-review
```

The installer fetches this repository at one commit, lists every file the
skill brings, and installs it into `.agents/skills/<name>` only after you
confirm. The agent then finds it by itself; `webagents skills list` shows it.

## The skills

| Skill | What it does | From |
|---|---|---|
| `accessibility-audit` | A WCAG 2.2 audit by principle, with a prioritised fix table and advice for manual testing | [ClawHub](https://clawhub.ai/mohitagw15856/skills/accessibility-audit), by mohitagw15856 |
| `api-contract-review` | A breaking-change review of an existing API, before against after, citing the codebase's own conventions | [ClawHub](https://clawhub.ai/dennisrongo/skills/api-contract-review), by dennisrongo |
| `api-design-review` | A review of a proposed HTTP API on nine points, from resources and errors to pagination, versioning and rate limits, ending in a verdict | [ClawHub](https://clawhub.ai/archlab-space/skills/api-design-review), by archlab-space |
| `architecture-decision-record` | Architecture Decision Records: the context, the options with honest trade-offs, the consequences and what would make you revisit the decision | [ClawHub](https://clawhub.ai/mohitagw15856/skills/architecture-decision-record), by mohitagw15856 |
| `bug-report` | Bug reports someone else can reproduce, with observed facts kept apart from guesses | [ClawHub](https://clawhub.ai/mohitagw15856/skills/bug-report), by mohitagw15856 |
| `changelog-generator` | Changelog entries from a git log or release notes, written for the reader in the Keep a Changelog format | [ClawHub](https://clawhub.ai/mohitagw15856/skills/changelog-generator), by mohitagw15856 |
| `code-review-checklist` | A review checklist scaled to the language, the kind of change and its risk, ending in a clear decision | [ClawHub](https://clawhub.ai/mohitagw15856/skills/code-review-checklist), by mohitagw15856 |
| `coe-root-cause` | A correction-of-error analysis: evidence, why-questions down to a mechanism, and fixes that each come with a check | [ClawHub](https://clawhub.ai/ghitafilali/skills/coe-root-cause), by ghitafilali |
| `commit-messages` | Conventional Commits messages from the staged changes, in plain words | [ClawHub](https://clawhub.ai/athola/skills/nm-sanctum-commit-messages), by athola |
| `data-analysis` | Analysis that starts from the decision: metric definitions, method choice, charts and decision briefs | [ClawHub](https://clawhub.ai/ivangdavila/skills/data-analysis), by ivangdavila |
| `database-schema-design` | Schema design documents: entities, DDL with constraints, indexes tied to the queries, normalisation and migration notes | [ClawHub](https://clawhub.ai/mohitagw15856/skills/database-schema-design), by mohitagw15856 |
| `drawio` | draw.io diagrams (flowcharts, architecture, UML, ER, mind maps, networks) written as XML, with layout rules and a self-check | [ClawHub](https://clawhub.ai/bruc3van/skills/bruce-drawio), by bruc3van |
| `editor-in-chief` | Copy and line editing that keeps the author's voice, as a review or applied | [ClawHub](https://clawhub.ai/tangentus/skills/editor-in-chief), by tangentus |
| `experiment-designer` | A/B test design (hypothesis, one main metric, guardrails, sample size, success criteria fixed in advance) and reading the result | [ClawHub](https://clawhub.ai/mohitagw15856/skills/experiment-designer), by mohitagw15856 |
| `frontend-design` | Distinctive, intentional visual design for new or reworked interfaces: direction, type, layout, restraint | [anthropics/skills](https://github.com/anthropics/skills/tree/8a1541c4a3ffa5a20a5a91de0dcf3f0bab1d1ef4/skills/frontend-design), by Anthropic (Apache-2.0) |
| `gantt-roadmap` | A plan turned into a dated Mermaid Gantt chart with dependencies, milestones, the critical path and risks | [ClawHub](https://clawhub.ai/mohitagw15856/skills/gantt-roadmap), by mohitagw15856 |
| `git-workflows` | Advanced git with a recovery point first: interactive rebase, reflog, conflicts, bisect, worktrees, subtrees and submodules | [ClawHub](https://clawhub.ai/darinrowe/skills/git-workflows-pro), by darinrowe |
| `kubernetes-triage` | Kubernetes fault triage from the evidence you give it: the fault class, ranked hypotheses, the next checks | [ClawHub](https://clawhub.ai/ghostwritten/skills/kubernetes-triage-expert), by ghostwritten |
| `market-research` | Market sizing, competitor mapping and demand validation, with the evidence graded for confidence | [ClawHub](https://clawhub.ai/ivangdavila/skills/market-research), by ivangdavila |
| `marp` | Slide decks in Markdown with Marp: directives, themes, images, CSS, export to HTML, PDF or PPTX | [ClawHub](https://clawhub.ai/patoo0x/skills/marp), by patoo0x |
| `mcp-builder` | Building MCP servers in Python or TypeScript: tool design, the SDKs, testing and evaluations | [anthropics/skills](https://github.com/anthropics/skills/tree/8a1541c4a3ffa5a20a5a91de0dcf3f0bab1d1ef4/skills/mcp-builder), by Anthropic (Apache-2.0) |
| `observability` | Metrics, logs and traces: cardinality budgets, percentile traps, sampling, SLOs and burn-rate alerts | [ClawHub](https://clawhub.ai/ivangdavila/skills/observability), by ivangdavila |
| `pandoc` | Document conversion with pandoc (an open-source converter): formats, PDF engines, styling and fixes, with a helper script | [ClawHub](https://clawhub.ai/oliver-hrkltz/skills/pandoc), by oliver-hrkltz |
| `product-requirements-drafter` | Product requirements shaped by the kind of change, with testable requirements, metrics and flagged assumptions | [ClawHub](https://clawhub.ai/archlab-space/skills/product-requirements-drafter), by archlab-space |
| `python-testing` | pytest practice: test structure, fixtures, mocking, async tests, coverage and CI | [ClawHub](https://clawhub.ai/athola/skills/nm-parseltongue-python-testing), by athola |
| `research-claim-ledger` | A claim-by-claim evidence ledger for a draft: the source, where in it, a verdict and a fix | [ClawHub](https://clawhub.ai/zack-dev-cm/skills/research-claim-ledger), by zack-dev-cm |
| `security-review` | A defensive security review where every finding names an attack path, and the report says what it did not check | [ClawHub](https://clawhub.ai/dennisrongo/skills/security-review), by dennisrongo |
| `sql-review` | A review of SQL Server (T-SQL) changes against 17 patterns taken from real incidents; it reports first and fixes only when asked | [ClawHub](https://clawhub.ai/dennisrongo/skills/sql-review), by dennisrongo |
| `systematic-literature-review` | Systematic and scoping reviews to the PRISMA 2020 reporting standard, never inventing a citation | [ClawHub](https://clawhub.ai/archlab-space/skills/systematic-literature-review), by archlab-space |
| `upgrade-deps` | Dependency upgrades backed by evidence: each changelog read, each breaking API searched, one major version at a time with tests between | [ClawHub](https://clawhub.ai/dennisrongo/skills/upgrade-deps), by dennisrongo |
| `webapp-testing` | Testing local web apps with Playwright, with a helper that starts and stops the servers (needs Playwright, and `sandbox: network: local: true` to reach them) | [anthropics/skills](https://github.com/anthropics/skills/tree/8a1541c4a3ffa5a20a5a91de0dcf3f0bab1d1ef4/skills/webapp-testing), by Anthropic (Apache-2.0) |
| `word-docx` | How Word documents are built (runs, styles, numbering, sections, tracked changes, fields) and where they break | [ClawHub](https://clawhub.ai/ivangdavila/skills/word-docx), by ivangdavila |
| `writing-credibility-auditor` | An audit of text for unsupported claims, missing citations, fallacies, weasel words and misleading statistics, quoting each passage | [ClawHub](https://clawhub.ai/arbazex/skills/writing-credibility-auditor), by arbazex |

## Where they come from

Thirty were published by their authors on ClawHub under MIT-0; three come
from Anthropic's skills repository under the Apache License 2.0.
Each was read before it was taken, and the ClawHub ones were checked against
other sources for copied text.
[`PROVENANCE.json`](PROVENANCE.json) records, for each one, the ClawHub
listing, the author, the version, the checksum of every file as the author
published it and as it is here, and what was changed. The changes are small:
the front matter carries the name, the licence and the source; pointers to
tools and skills that are not here are taken out; punctuation follows this
repository's style. A file changed from an Apache-licensed original says so
in the file itself.

## Licence

MIT No Attribution (MIT-0), see [`LICENSE`](LICENSE): use, change and
republish those skills anywhere, with no notice required. The exceptions are
`frontend-design`, `mcp-builder` and `webapp-testing`, which keep Anthropic's
Apache License 2.0: keep their `LICENSE.txt` and the [`NOTICE`](NOTICE) when
you pass them on, and say what you changed. Each skill's front matter names
its licence and its author.

## Adding a skill

Open a pull request that adds `skills/<name>/SKILL.md`, and any reference
files beside it. The folder name and the `name:` in the front matter are the
same: lower-case letters, digits and single hyphens. By contributing you
license the skill under MIT-0 and confirm you have the right to.

A skill lands only if all of these hold:

- it does one useful thing an agent does worse without it;
- it downloads nothing and runs nothing from the network, and never tells
  the agent or the person to install something from a URL;
- any script is short, readable and does only what the skill says: no
  network calls to hosts the skill does not name, and no reading of
  credentials, keys, `.env` files, browser data or shell history;
- it holds no binaries, archives, minified or encoded code, and no hidden
  text;
- nothing in it was copied from a source whose licence does not allow MIT-0.

The SDK's tests load every skill here and check each file against its
checksum in `PROVENANCE.json`, so a change to a skill updates its record,
with a line saying why.
