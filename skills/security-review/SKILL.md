---
name: security-review
description: "Attacker's-eye security review of a diff, branch, or module: walks a fixed vulnerability catalog (missing authz on new endpoints, injection, secrets in code/logs, trusting client-sent identity, SSRF, path traversal, insecure deserialization, mass assignment, broken crypto, unsafe redirects, dependency CVEs) where every finding must name a concrete attack path (attacker does X → gains Y) or be demoted to hardening advice. Never claims \"secure\", only \"nothing found in the classes checked\". Use this skill whenever the user says \"security review\", \"is this secure\", \"check for vulnerabilities\", \"audit the auth\", \"threat model this\", \"pentest mindset\", or \"check for injection\", even if they don't name the skill. Distinct from a general code review, where security is one lens; this is the dedicated deep pass."
license: MIT-0
metadata:
  author: "dennisrongo"
  source: "https://clawhub.ai/dennisrongo/skills/security-review"
  version: "1.0.0"
---

# Security Review

A defensive security audit of code you're shipping, run with an attacker's questions: what does this trust that it shouldn't, and what can I reach that I shouldn't. The output discipline is what makes it useful: every finding carries an attack path, every "pass" carries the evidence that was checked, and the report never claims more than it verified. "No vulnerabilities" is not a claim this skill can make; "none found in these classes, with these checks, and here's what wasn't checked" is.

## When to use this skill

- The user says "security review", "is this secure", "check for vulnerabilities", "audit the auth", "threat model this", "check for injection".
- New endpoints, auth changes, file handling, payment paths, or user-generated-content rendering are about to ship.
- A general code review surfaces a security concern that needs a dedicated pass.

Do **not** auto-trigger on every diff: a general code review carries a security lens for routine work; this skill is the deep pass when the change touches trust boundaries or the user asks. This skill reviews and reports; it never writes exploit tooling, and it never edits code unprompted.

## Workflow

1. **Fix the scope first.** A diff, a branch, a module, or an endpoint list: ask the user to name it if ambiguous (one question). Then map the trust boundaries inside that scope: every place data crosses from less-trusted to more-trusted (HTTP input, file upload, queue message, webhook, env/config, DB values rendered back out). Effort follows boundaries: a 500-line diff with one new endpoint gets most scrutiny on the endpoint.
2. **Walk the catalog against the scope.** For each class, the check is named: run it, don't vibe it:
   - **Missing authn/authz**: for every new/changed route or handler, locate the auth check (middleware registration, guard attribute, explicit session call) and cite `file:line`. Then check *object-level* authz: does the handler verify the caller may touch **this** record, or only that they're logged in? Absence of either is a finding. Compare against how sibling endpoints in the repo do it.
   - **Trusting client-sent identity**: grep handlers in scope for user/account/tenant IDs read from body, query, or headers and used in queries or writes. The ID must come from the session/token.
   - **Injection**: SQL/query built by string concatenation with external input (must be parameterized); shell commands from input; path traversal (user input joined into filesystem paths without normalization + prefix check); XSS (external input rendered without the framework's escaping: grep for `dangerouslySetInnerHTML`, `innerHTML`, `v-html`, `Html.Raw`, raw template filters).
   - **Secrets**: grep the scope for key/token/password-shaped literals and connection strings; check that secrets aren't logged (grep log calls near auth/config code) or committed in config. A finding is the **value** in the repo, not the reference: `process.env.STRIPE_KEY` / a secret-manager call is the correct pattern, never a finding; `sk_live_4eC39...` in code or a committed `.env` is. If a real secret is already committed, rotation is the fix: deleting the line doesn't un-leak it; git history keeps it.
   - **SSRF & redirects**: any outbound request whose URL contains external input; any redirect target taken from a parameter without an allowlist.
   - **Insecure deserialization / mass assignment**: deserializing external input into types with dangerous side effects; binding request bodies directly to DB entities so a caller can set `isAdmin`/`role`/`price` (check for an explicit DTO/allowlist between input and model).
   - **Crypto misuse**: hand-rolled hashing/encryption, fast hashes (MD5/SHA-x) for passwords instead of bcrypt/argon2/scrypt, `Math.random()`-class RNG for tokens, comparing secrets with `==` instead of constant-time compare.
   - **Dependency CVEs**: only via an actual tool run (`npm audit`, `dotnet list package --vulnerable`, `pip-audit`, `cargo audit`); quote the output. If no tool ran, the report says `dependencies: not checked`, never "dependencies look fine".
3. **Evidence-gate every finding.** A finding must state, in one sentence, the attack path: *who* (unauthenticated user / any logged-in user / tenant B / insider) does *what* → gains *what*. No constructible path → demote to **hardening** (still reported, clearly separated). Severity from the path itself: **critical** = unauthenticated or cross-tenant data access/mutation, RCE, secret exposure; **high** = authenticated privilege escalation or injection with real reachable input; **medium** = requires unusual preconditions; **hardening** = defense-in-depth with no current path.
   - ❌ "The `userId` parameter could be dangerous.": no actor, no gain, not a finding.
   - ✅ "`GET /api/invoices/{id}` checks login but not ownership (`InvoiceController.cs:41`: no tenant filter in the query): any logged-in user who increments `id` reads other customers' invoices. Critical (IDOR)."
4. **Verify before reporting.** For each finding, re-derive it: open the file, trace the input to the sink, confirm no sanitizer/guard sits between them (search the call chain, not just the hunk: the guard may live in middleware or a base class). A pattern-matched finding you didn't trace is labeled **unconfirmed** in the report, never stated in the same register as a traced one. Zero findings is a valid outcome and must be reported without padding.
5. **Report.** Findings ranked by severity, each with `file:line`, attack path, and a recommended fix direction (not applied). Then the coverage statement: which catalog classes were checked with what evidence, and which were **not** checked (e.g. "infra/deployment config out of scope, dependencies not audited: no lockfile in scope"). End by asking per-finding whether to draft the fix: never edit unprompted.

## Examples

### Example 1: new endpoint pass

**User:** "Security review the payments branch before I merge."

**Agent:** Scopes to the branch diff, maps boundaries (2 new endpoints, 1 webhook), finds the webhook skips signature verification (traced: raw body straight to the handler, `webhooks.ts:23`): critical, forged payment confirmations; finds sibling endpoints use `requireSession` but the new refund route registers outside the auth router: high; runs `npm audit` (quotes: 0 high). Coverage note: XSS class N/A (no rendering in scope), crypto N/A.

### Example 2: honest zero

**User:** "Check this date-formatting utility PR for vulnerabilities."

**Agent:** Walks the catalog; no external input reaches a sink (verified: all call sites pass server-generated dates). Report: zero findings, one hardening note (no input-length guard), explicit "not checked: dependencies (no manifest change)". No invented findings to look thorough.

## Anti-patterns

- ❌ Findings without an attack path: severity theater. If you can't say who gains what, it's hardening advice.
- ❌ Claiming "this code is secure": the only honest claim is "nothing found in the classes checked", plus the unchecked list.
- ❌ Reporting a pattern-match ("string concatenation near SQL") without tracing whether external input actually reaches it: half of these have a parameterizer or validator one frame up.
- ❌ Marking everything critical. The severity ladder is defined by the attack path; a finding needing admin access plus a misconfigured proxy is not critical.
- ❌ "Fixing" a committed secret by deleting the line: history retains it; the finding is "rotate this credential".
- ❌ Skipping object-level authz because authentication exists: IDOR is the most common real-world miss, and it lives exactly in that gap.
- ❌ Editing code or adding "quick fixes" unprompted: report, ask, then fix.
- ✅ Scope → boundaries → named checks with citations → attack-path-gated findings → coverage statement including what was NOT checked.

## Notes

- This skill is defensive: it reviews code the user owns or is authorized to audit. It doesn't produce working exploits: a one-sentence attack path is the proof standard, a PoC payload is not required and not offered beyond what's needed to demonstrate the flaw to the developer.
- Language/framework specifics come from the repo: find how THIS codebase does auth, validation, and escaping (grep for the middleware/guards), then hunt for the places in scope that deviate from it: deviation from the local safe pattern is the highest-yield query.
- Put the effort at the boundaries (step 1), re-derive every finding rather than recognising it and label what is unconfirmed out loud (step 4), and attack your own report: the finding you're most confident in is the one to re-trace.
