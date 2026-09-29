---
name: research-claim-ledger
description: "Build source-backed research claim ledgers from drafts, literature matrices, notes, citation lists, PDFs, source packets, or reviewer comments. Use when the user needs to audit factual, numerical, causal, comparative, novelty, policy, or citation-dependent claims; find unsupported or overclaimed statements; prepare a supervisor/coauthor/shareable evidence receipt; repair a manuscript before submission; triage a literature review; or turn academic sources into a traceable claim-to-source table without running a full research-paper pipeline."
license: MIT-0
metadata:
  author: "zack-dev-cm"
  source: "https://clawhub.ai/zack-dev-cm/skills/research-claim-ledger"
  version: "1.0.0"
---

# Research Claim Ledger

Use this skill to produce a compact, shareable evidence ledger. The goal is not
to write the paper for the user. The goal is to make every important claim
auditable: what is being asserted, which source supports it, where the support
is located, how strong the support is, and what should be fixed next.

## Workflow

1. **Intake the materials.**
   Identify what the user supplied: draft text, claim list, literature matrix,
   bibliography, source PDFs, notes, reviewer comments, or only a topic. If
   only a topic is supplied, ask for either source material or permission to
   search current sources before building a ledger.

2. **Set the audit scope.**
   Choose one of these scopes unless the user specifies another:
   - `fast`: top 10 to 15 claims most likely to create citation risk.
   - `submission`: all high-risk claims in the abstract, introduction,
     findings/results, discussion, and conclusion.
   - `full`: every factual, numerical, causal, comparative, and novelty claim.
   - `repair`: only claims flagged by reviewers, supervisors, or prior audit.

3. **Extract claims before judging them.**
   List candidate claims first. Preserve the author's wording enough that the
   user can find the sentence again. Split compound sentences when one source
   supports part of the sentence but not the whole assertion.

4. **Map each claim to evidence.**
   For each claim, record cited source, uncited implied source, source type,
   DOI/URL when available, locator, and access status. A locator should be a
   page, table, figure, section, paragraph, quote fragment, timestamp, dataset
   row, or repository path. If there is no locator, mark `missing-locator`.

5. **Judge support conservatively.**
   Compare claim strength against the source. Watch for inflated causality,
   generalized populations, changed date ranges, cherry-picked numbers,
   secondary-source drift, stale fast-moving claims, and sources that merely
   mention a topic without supporting the sentence.

6. **Deliver the ledger and repair plan.**
   Always separate evidence, inference, and recommended edits. Give the user a
   short set of next actions rather than a long generic critique.

## Claim Types

Classify each claim with one or more types:

- `numeric`: percentages, counts, p-values, effect sizes, dates, rankings.
- `causal`: says X causes, drives, reduces, improves, predicts, or leads to Y.
- `comparative`: larger, smaller, better, worse, first, only, strongest, most.
- `trend`: increasing, declining, stable, accelerating, emerging, recent.
- `definition`: defines a term, framework, method, or construct.
- `novelty`: claims a gap, first contribution, unique dataset, or new method.
- `synthesis`: combines multiple sources into an interpretation.
- `policy`: makes a recommendation, implication, guideline, or warning.
- `uncited`: needs evidence but currently has no citation.

## Verdicts

Use these verdicts exactly:

- `supported`: source supports the claim at the stated strength.
- `weakly-supported`: source is relevant but too indirect, narrow, old, or weak.
- `overclaimed`: source supports a softer claim than the draft makes.
- `wrong-source`: cited source does not contain the needed support.
- `missing-locator`: source may support the claim, but no precise locator exists.
- `stale-source`: claim depends on current facts and the source may be outdated.
- `inaccessible`: source exists but cannot be checked from available materials.
- `unsupported`: no supplied or discovered source supports the claim.
- `needs-human-review`: judgment depends on field expertise, statistics, ethics,
  legal interpretation, medical judgment, or access to restricted materials.

## Source Strength

Use simple grades that work across fields:

- `A`: primary, recent, methodologically strong source for this exact claim.
- `B`: credible source that supports the claim with minor limits.
- `C`: useful context, but weak for direct support.
- `D`: poor fit, outdated, non-authoritative, or only usable as background.
- `NA`: no usable source yet.

Adjust by field. In medicine, law, finance, safety, and public policy, be stricter
about primary sources, dates, official guidance, and uncertainty.

## Output Template

Produce this structure unless the user asks for another format:

```markdown
## Claim Ledger Summary
- Scope:
- Materials checked:
- Claims checked:
- Supported:
- Repair before sharing/submission:
- Highest-risk pattern:

## Claim Ledger
| ID | Claim | Type | Source | Locator | Verdict | Strength | Fix |
|---|---|---|---|---|---|---|---|
| C1 | ... | numeric | Smith 2024 | p. 8, Table 2 | supported | A | Keep. |

## Unsupported Or Overclaimed Claims
| ID | Problem | Safer wording | Evidence needed |
|---|---|---|---|

## Source Gaps
| Gap | Why it matters | Best next source to find |
|---|---|---|

## Shareable Receipt
One paragraph the user can paste to a supervisor, coauthor, reviewer, or public
project note. Include what was checked, what changed, and what remains unverified.
```

Keep table cells concise. If the ledger is large, give the top-risk table first
and offer to continue in batches.

## Verification Rules

- Never fabricate references, DOIs, page numbers, quotations, data, policies, or
  access status.
- If current or external facts matter, verify with browsing and cite source
  links. Prefer primary papers, official datasets, publisher pages, DOI records,
  government or institution pages, standards bodies, and official docs.
- If the user supplied source files, use those first. Do not replace supplied
  sources with web snippets unless the user asks for source discovery.
- Quote only short fragments when needed to prove locator quality. Paraphrase
  the rest.
- Mark uncertainty explicitly. `inaccessible` and `needs-human-review` are valid
  outcomes, not failures.
- For medical, legal, financial, safety, or admissions decisions, do not give
  professional advice. Audit claim support and recommend qualified review.

## Repair Guidance

When a claim fails, choose one of four repairs:

- `soften`: reduce certainty or causal strength.
- `narrow`: match the population, method, date range, or context in the source.
- `replace-source`: find or request a better source.
- `remove`: delete the claim if it is nonessential and unsupported.

Prefer concrete rewrites:

```markdown
Original: X proves Y improves student outcomes.
Safer: In one quasi-experimental study of [population], X was associated with
[specific outcome], but causal interpretation is limited by [method limit].
```

## Good User Prompts

- "Use research-claim-ledger to audit this literature review before I send it
  to my supervisor."
- "Build a fast claim ledger from this draft and flag unsupported or overclaimed
  sentences."
- "Turn this Zotero/literature matrix into a source-to-claim map."
- "Check whether these reviewer objections are supported by the manuscript and
  cited sources."
- "Create a shareable evidence receipt for this research note."

## Boundaries

- Do not ghostwrite a full paper unless the user separately asks for writing.
- Do not promise publication, acceptance, grade improvement, legal compliance,
  diagnosis, financial outcome, or plagiarism clearance.
- Do not treat AI-generated text detectors as evidence of authorship.
- Do not hide AI use. If asked for disclosure, draft a transparent assistance
  note describing auditing, organization, or editing support.
