---
name: api-contract-review
description: "Review an API's contract as a promise to consumers: detects breaking changes by diffing the before/after surface (removed/renamed fields, type changes, tightened requiredness, status-code changes), and judges design by the repo's OWN precedent (error envelope, naming, pagination, auth placement) with every consistency finding citing the in-repo convention being violated. Covers versioning, idempotency on retryable writes, pagination on collections, and status-code semantics. Use this skill whenever the user says \"review this API\", \"review the endpoint\", \"is this a breaking change\", \"check backward compatibility\", \"review the contract\", or \"review this OpenAPI/swagger spec\", even if they don't name the skill. Distinct from a general code review (implementation quality); this reviews the SURFACE consumers depend on."
license: MIT-0
metadata:
  author: "dennisrongo"
  source: "https://clawhub.ai/dennisrongo/skills/api-contract-review"
  version: "1.0.0"
---

# API Contract Review

An API contract is a promise made to code you can't see and can't fix. This skill reviews the promise, not the implementation: what a consumer can observe: paths, methods, fields, types, requiredness, status codes, error shapes, headers, ordering and pagination semantics, and whether this change keeps, extends, or breaks it. Two evidence rules do the work: **breaking** requires a before/after diff of a consumer-visible element, and **inconsistent** requires a citation of the in-repo precedent being violated.

## When to use this skill

- The user says "review this API", "is this a breaking change", "check backward compat", "review the contract", "review this OpenAPI spec".
- A new endpoint, GraphQL type, gRPC service, webhook payload, or event schema is being added or changed.
- An endpoint is being designed and the contract deserves its own pass before implementation.

Do **not** auto-trigger for internal function signatures or module interfaces (that is general code review territory): this skill is for surfaces crossed by consumers who deploy independently: HTTP APIs, published events, webhooks, SDK-facing types.

## Workflow

1. **Establish the before and the after.** For a change: the old contract is `git show` of the previous handler/spec/DTO, the new one is the working tree: read both; a breaking-change verdict without the before-state in hand is a hypothesis. For a brand-new endpoint there is no "before", so the compat section reduces to forward-compat design (step 4). Identify the consumers if discoverable (other repos, mobile apps, webhook subscribers, "unknown external"): unknown consumers raise the cost of every breaking change and the report should say so.
2. **Diff the consumer-visible surface for breaking changes.** Breaking = an existing valid consumer interaction stops working or changes meaning. The checklist, each judged by before/after citation:
   - Removed or renamed: path, method, field, enum value, header.
   - Type changes (string→int, scalar→object, nullable→non-null in responses).
   - Requiredness tightened on **requests** (new required field/param, stricter validation rejecting previously-valid input).
   - Semantics changed under the same name: status code for the same condition, default value, sort order consumers observe, pagination behavior, error `code` values, ID format.
   - Response fields **removed or now-sometimes-absent** (additive response fields are non-breaking for tolerant readers, but check the repo's serializer isn't strict).
   - ❌ "Changing this field feels risky.": no before/after, not a verdict.
   - ✅ "Breaking: `status` response field was `\"active\"|\"disabled\"` (git show `UserDto.cs:14`), now adds `\"suspended\"`: consumers with exhaustive enum handling will throw. New enum values in responses are breaking unless the contract documents open enums; nothing in the spec says so."
3. **Judge design by local precedent, not by taste.** Before flagging anything as inconsistent, grep the sibling endpoints and read at least two. Then check the new surface against what THIS repo does: error envelope shape (find the canonical one; new endpoint must return it, not a fresh ad-hoc `{message}`: cite both), naming (`camelCase` vs `snake_case`, plural collections, ID field naming), auth placement (same middleware/guard pattern as siblings, compose with `security-review` if it's absent entirely), pagination style (cursor vs offset, envelope keys), timestamp format, route casing. A consistency finding without the cited precedent is an opinion; drop it or label it one.
4. **Check the design invariants that hurt later.** These apply even with zero consumers today, because they're near-impossible to retrofit:
   - **Collections paginate from day one**: an unpaginated list endpoint is a time bomb; adding pagination later breaks every consumer.
   - **Retryable writes are idempotent**: POSTs that create money-adjacent or non-deduplicable resources need an idempotency key or a natural dedup constraint; name which.
   - **Status codes carry their standard meaning**: 200-with-error-body, 404 for authz denials (unless deliberate resource-hiding, then say so), 500 for validation failures: each flagged with the observed code and the condition.
   - **Errors are machine-usable**: a stable `code` field, not prose-only messages consumers will regex.
   - **Nothing leaks that can't be unshipped**: internal IDs, stack traces, ORM entity fields serialized wholesale (grep: does the handler return the entity type or a DTO?). Every response field is a permanent promise; flag fields with no evident consumer need.
   - **Timestamps and money have explicit units/zones/currency**: `amount: 4200` with no currency or minor-unit convention is a production incident on layaway.
5. **Report in three ranked sections.** (a) **Breaking**: each with before/after citations and the migration path (version bump per the repo's existing strategy, additive alternative, deprecation window); (b) **Design**: invariant violations and precedent-cited inconsistencies; (c) **Questions**: semantics you couldn't determine from code (is this enum open? is this endpoint consumer-facing at all?). Zero findings is a valid outcome: an additive, precedent-matching endpoint should get a short pass, not invented nits. Never edit code or specs unprompted.

## Examples

### Example 1: "is this a breaking change?"

**User:** "I'm renaming `customer_id` to `customerId` in the orders response: breaking?"

**Agent:** Diffs before/after (breaking: field rename is remove+add for every existing reader), greps the repo: every other response uses `snake_case`, so the rename also breaks local convention rather than fixing it (cited: 9 sibling DTOs). Recommends: keep `customer_id`; if the target is a casing migration, that's an API-version-level change, not a field edit.

### Example 2: new endpoint pass

**User:** "Review the contract for the new `/api/reports` endpoints before I build them."

**Agent:** No before-state → forward-compat pass: flags the unpaginated `GET /reports` (day-one invariant), flags `POST /reports/generate` returning 200 with the result inline when generation takes minutes (should be 202 + status resource: cites the repo's existing export endpoint doing exactly that), confirms error envelope and naming match siblings, asks one question (open or closed enum for `format`?).

## Anti-patterns

- ❌ Declaring "breaking" or "safe" without reading the before-state: the diff of the surface is the entire evidence base.
- ❌ Imposing REST doctrine the repo doesn't follow ("should be HATEOAS", "must be plural nouns") when siblings consistently do otherwise: local consistency beats global convention; flag the doctrine mismatch once as a question, at most.
- ❌ Consistency findings with no cited precedent: grep the siblings first or don't flag it.
- ❌ Treating additive response fields as breaking, or new required request fields as safe: the asymmetry (requests: consumers write them; responses: consumers read them) is the whole compat model.
- ❌ Passing an unpaginated collection endpoint because "there won't be much data": volume assumptions don't survive; retrofitting pagination breaks every consumer.
- ❌ Reviewing the handler's implementation quality (naming, DRY, perf): that is a general code review's job; scope discipline keeps this report actionable.
- ✅ Before/after-cited breaking verdicts, precedent-cited consistency findings, day-one invariants checked, questions kept separate from findings.

## Notes

- Spec-first repos (OpenAPI/proto/GraphQL SDL): review the spec diff as the contract and verify the implementation actually matches it (spot-check one handler against its spec entry: drift between the two is itself a finding). Code-first repos: the serialized DTOs + routes are the contract.
- Deprecation over deletion: when a breaking change is genuinely wanted, the recommendation is the repo's existing versioning/deprecation mechanism if one exists (grep for it): inventing a versioning strategy is a design conversation, not a review finding.
- The risk lives in the unknown consumers, so compat verdicts get the re-derivation effort; "safe" claims are labeled by what was actually diffed; the report leads with the one change the user must not merge as-is.
