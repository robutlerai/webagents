---
name: sql-review
description: "Pre-commit SQL code review (SQL Server, T-SQL) for uncommitted database changes. Detects antipatterns that cause real production incidents: broken TRY/CATCH error handling, swallowed errors, retry patterns without loops, tables missing PK/indexes, type-mismatch truncation, dynamic-SQL injection, NOLOCK in write paths, UPDATE/DELETE without WHERE, hardcoded env values, cross-DB refs, and more (full catalog in the skill body). Reports BLOCKER/WARN/INFO findings with file:line citations and fixes. Never edits SQL. Use this skill whenever the user says \"review my SQL\", \"review the SQL diff\", \"lint the SQL\", \"check my SQL changes\", \"SQL pre-commit check\", \"audit my stored proc\", or asks about SQL antipatterns in a diff, even if they don't name the skill. Distinct from a general code review: this carries SQL-specific incident patterns."
license: MIT-0
metadata:
  author: "dennisrongo"
  source: "https://clawhub.ai/dennisrongo/skills/sql-review"
  version: "1.0.1"
---

# SQL Review

Review uncommitted SQL changes against a catalog of patterns that cause real production incidents: masked errors, unindexed new tables, silent truncation, SQL injection, deadlock-prone CATCH blocks, and surface findings as recommendations the user can act on, **without editing SQL unprompted**.

## When to use this skill

- "review my SQL" / "review the SQL diff"
- "lint the SQL" / "SQL pre-commit check" / "check my SQL changes"
- "audit my stored proc" / "any SQL antipatterns in this diff"
- Before committing a `.sql` change touching a stored procedure, function, or table

Do **not** auto-trigger when the user is asking for a *general* code review on a mixed diff; that is a general code review's job. This skill is specifically for SQL-heavy diffs where the antipattern catalog applies.

## Hard rule: surface, don't refactor

If you find issues, **do not start editing the SQL**. Produce the findings report first. After delivering it, ask the user per finding: *"Want me to fix #N?"*, and wait for an explicit yes. Pre-authorization ("fix them all") proceeds through the list; absent that, default to ask-per-fix.

SQL changes are particularly sensitive because they often run against production data in a single non-reversible deployment. A drive-by refactor mid-review is more dangerous here than in application code.

## Workflow

1. **Snapshot the SQL diff.** Run in parallel:
   - `git status --short`: see what's modified, staged, untracked.
   - `git diff -- '*.sql'`: unstaged SQL changes.
   - `git diff --staged -- '*.sql'`: staged SQL changes.
   - If user passed an explicit range (e.g. `main..HEAD`), use that instead: `git diff <range> -- '*.sql'`.
2. **Triage files into two buckets:**
   - **New files** (status `A` or `??`): review the entire file. The whole file is new code; pre-existing antipatterns elsewhere don't apply.
   - **Modified files** (status `M`): review only the changed hunks. The rest of the file may have legacy antipatterns the user isn't introducing (e.g. existing repos commonly have dozens of SPs with the `sp_send_dbmail`-in-CATCH pattern; flagging all of them on an unrelated edit creates noise).
3. **For each check in [What to check](#what-to-check), look for the pattern** in the relevant scope (whole file for new files, diff hunks for modified files). Use ripgrep: don't try to fully parse SQL.
4. **Categorize each finding** as `BLOCKER` / `WARN` / `INFO` (see [Categories](#categories)).
5. **Cite `file:line`** for every finding. Line numbers come from the diff or the current file content.
6. **Produce the report** in the [Output format](#output-format).
7. **Offer to fix** the suggested findings. Wait for the user's selection before editing.

## Categories

- **`BLOCKER`**: would cause a production incident: silent data loss, masked errors, SQL injection, unindexed table that will grow, broken transaction handling. Fix before commit.
- **`WARN`**: likely a bug or maintenance hazard: NOLOCK in write paths, hardcoded environment values, missing `SET NOCOUNT ON`, cross-DB references. Should fix but won't necessarily incident.
- **`INFO`**: style / convention / hygiene: vestigial comments, cursor defaults, missing `GRANT EXECUTE` where the project's pattern exists.

Lead each finding with the *why*. "This swallows the original error and surfaces a misleading permission denial" beats "add `;THROW;`".

## Pattern hit ≠ finding

The 17 checks are pattern matches. A ripgrep hit is a **candidate**: before reporting it, `Read` the whole procedure/batch it sits in; severity depends on context the pattern can't see. Concrete demotions:

- `WITH (NOLOCK)` hit, but the proc is a read-only reporting proc with no `INSERT`/`UPDATE`/`DELETE` → **WARN** at most, not BLOCKER (check #13 blocks only when the same proc writes the tables it dirty-reads).
- `DECLARE @retry` with no `WHILE @retry` in sight → check for a `GOTO`-label retry loop consuming it before flagging; only report when nothing reads the variable.
- New `CREATE TABLE` with no index, but it's a static lookup/config table seeded with a handful of rows in the same diff → **WARN** with a one-line reason, not BLOCKER (the deadlock incident needs growth under concurrent writes).

Every reported finding must:
1. **Quote the offending SQL verbatim**: the exact line(s) from the file, not a paraphrase. Severity turns on exact tokens.
2. For `BLOCKER` / `WARN`: state the concrete production incident in one sentence (*this state → this outcome*, e.g. "first deadlock under load → SP silently gives up, rows never written"). If you can't write that sentence after reading the full proc, demote one level.

Zero findings is a valid outcome: a clean diff gets a clean report. Never stretch INFO items into WARNs to fill sections; thoroughness is the 17 checks you ran, which the report shows.

## What to check

Use ripgrep for detection. The patterns below are starting points: adapt to the actual diff text.

### Error-handling antipatterns (BLOCKER class)

1. **`sp_send_dbmail` inside `CATCH`**: masks the real exception with a permission/configuration error when the executing login lacks `EXECUTE` on `msdb.dbo.sp_send_dbmail`. The real underlying error is silently lost.
   - Pattern: `rg -n -B 20 'sp_send_dbmail' <file>` → check whether the surrounding context is a `BEGIN CATCH`.
   - **Fix:** Replace with `;THROW;` to re-raise, or write the error info to a local log table. Never `sp_send_dbmail` from inside operational SPs.

2. **Broken retry pattern**: `DECLARE @retry INT = N;` exists, but no `WHILE @retry` loop wraps the `TRY`/`CATCH`. The retry variable is decremented in the CATCH but never read, so the SP gives up on the first deadlock instead of retrying.
   - Pattern: `rg -n '@retry' <file>` → confirm there's a matching `WHILE.*@retry` somewhere in the same SP body.
   - **Fix:** Add the missing `WHILE @retry > 0 BEGIN ... END` around the TRY/CATCH, or remove the vestigial variable entirely.

3. **Empty / swallowing `CATCH` blocks**: `BEGIN CATCH ... END CATCH` with no `THROW`, no `RAISERROR`, and no `INSERT INTO <error_log_table>`. Silent failure → silent data loss.
   - Pattern: `rg -n -A 30 'BEGIN CATCH' <file>` → check each match for at least one of `THROW`, `RAISERROR`, or `INSERT INTO`.
   - **Fix:** End the CATCH with `;THROW;` unless there's a deliberate reason to suppress (and document that reason).

4. **`BEGIN TRANSACTION` without `TRY`/`CATCH` + `XACT_STATE` handling**: an error mid-transaction leaves an open transaction on the connection.
   - Pattern: `rg -n 'BEGIN TRAN' <file>` → confirm the same SP has `BEGIN TRY` and `XACT_STATE()` checks in CATCH.
   - **Fix:** Wrap in `BEGIN TRY / BEGIN TRANSACTION ... COMMIT TRANSACTION / END TRY / BEGIN CATCH / IF XACT_STATE() = -1 ROLLBACK; IF XACT_STATE() = 1 COMMIT; ;THROW; / END CATCH`.

5. **Vestigial control-flow comments**: `-- end while loop` with no `WHILE` keyword, `-- retry on deadlock` with no retry loop, etc. Strong signal that a refactor left dead code behind.
   - Pattern: `rg -n -- '-- end (while|for|repeat)' <file>` → check for matching opening keyword.
   - **Fix:** Either restore the missing loop or delete the misleading comment.

### Schema / performance antipatterns (BLOCKER for new tables, WARN otherwise)

6. **New `CREATE TABLE` without a primary key, clustered index, or any index**: heap tables will deadlock-and-table-scan once they grow. This is the exact pattern that caused real production deadlocks.
   - Pattern: `rg -n -A 50 'CREATE TABLE' <file>` → confirm at least one of `PRIMARY KEY`, `CLUSTERED INDEX`, or `CREATE INDEX` exists on that table within the diff or in a sibling file.
   - **Fix:** Add `PRIMARY KEY CLUSTERED` on the natural ID column. If the table is append-mostly with no natural ID, add a clustered index on the column the populating SPs filter by in `WHERE`.

7. **Parameter-to-column type-width mismatches**: passing `@x VARCHAR(50)` into a column declared `VARCHAR(3)` causes silent truncation (or runtime error 8152 depending on `ANSI_WARNINGS`).
   - Pattern: cross-reference the SP's `@param VARCHAR(N)` declarations against the target table's column widths. Flag any narrowing.
   - **Fix:** Match widths between parameter and column, or add explicit `LEFT(@x, 3)` truncation at the boundary so it's intentional and visible.

8. **Implicit type conversions in `JOIN` / `WHERE`**: `WHERE intCol = @varcharParam` defeats indexes and produces table scans.
   - Pattern: `rg -n -i 'WHERE\s+\w+\s*=\s*@\w+' <file>` → cross-check parameter type vs. column type.
   - **Fix:** Match the parameter type to the column type. If they're authoritatively different, cast the *parameter*, not the column (casting the column kills the index).

### Security antipatterns (BLOCKER)

9. **Dynamic SQL via string concatenation passed to `EXEC()`** without `sp_executesql @stmt, @params` parameterization: SQL injection.
   - Pattern: `rg -n 'EXEC\s*\(' <file>` → confirm any string-concat dynamic SQL nearby uses `sp_executesql` with parameters, not `EXEC(@strSQL)` with values inlined.
   - **Fix:** Switch to `EXEC sp_executesql @stmt = @strSQL, @params = N'@p1 INT, @p2 VARCHAR(50)', @p1 = @value1, @p2 = @value2;`.

10. **Hardcoded environment-specific values**: email addresses, server names, file paths, IP addresses, connection strings.
    - Pattern: `rg -nE "'[\w.-]+@[\w.-]+\.\w+'|@server_name\s*=\s*'\w+'|\\\\\\\\[\\w.-]+\\\\" <file>` (emails, linked servers, UNC paths).
    - **Fix:** Move env-specific values into a configuration table or pass them as parameters from the application layer.

11. **Cross-database references**: `msdb.dbo.*`, `master.dbo.*`, linked server `[srv].db.dbo.*`. These require permissions that may not exist in all environments (the canonical example: a SP calling `msdb.dbo.sp_send_dbmail` from a login that lacks EXECUTE on it).
    - Pattern: `rg -n 'msdb\.|master\.dbo\.|\[\w+\]\.\w+\.\w+\.' <file>`.
    - **Fix:** Move the cross-DB call out of operational SPs into infrastructure-owned code. If it must stay, document the required permission in a comment so deployments to new environments include the GRANT.

### Correctness / data-safety antipatterns (BLOCKER)

12. **`UPDATE` or `DELETE` without a `WHERE` clause**: almost always a bug; will affect every row.
    - Pattern: `rg -n -B 0 -A 20 'UPDATE\s+\w+' <file>` → confirm a `WHERE` clause is present and not commented out.
    - Pattern: `rg -n -B 0 -A 10 'DELETE\s+FROM' <file>` → same.
    - **Fix:** Add the `WHERE` clause. If "every row" really is intended (TRUNCATE-style), use `TRUNCATE TABLE` and add a comment explaining why.

13. **`NOLOCK` / `READ UNCOMMITTED` inside transactional write paths**: `WITH (NOLOCK)` is fine for ad-hoc reporting reads, but in an `UPDATE`/`INSERT`/`DELETE` SP it can read uncommitted data and corrupt the write.
    - Pattern: `rg -n -i 'NOLOCK|READ UNCOMMITTED' <file>` → check whether the same SP has `INSERT`, `UPDATE`, or `DELETE` against the same tables.
    - **Fix:** Remove the hint on tables being modified. Consider `SNAPSHOT` isolation if the goal is to avoid blocking.

14. **`TRUNCATE TABLE` or `DROP` without `IF EXISTS`** in deploy scripts that are supposed to be idempotent (re-runnable). Many repos use flat idempotent `Deploy_*.sql` files where re-runs must not fail on missing objects.
    - Pattern: `rg -n -i 'DROP (TABLE|PROC|FUNCTION|VIEW)' <file>` → confirm `IF EXISTS` or `OBJECT_ID(...) IS NOT NULL` guard.
    - **Fix:** `IF OBJECT_ID('dbo.X', 'U') IS NOT NULL DROP TABLE dbo.X;` or `DROP TABLE IF EXISTS dbo.X;` (SQL 2016+).

### Style / convention (WARN/INFO)

15. **Missing `SET NOCOUNT ON`** at the top of a new `CREATE PROC`: causes extra round-trips to the client and breaks some ORM drivers that don't handle row-count messages.
    - Pattern: `rg -n -A 5 'CREATE PROC' <file>` → confirm `SET NOCOUNT ON` appears within the first ~10 lines of the procedure body.
    - **Fix:** Add `SET NOCOUNT ON;` as the first statement after `AS BEGIN`.

16. **Cursors without `READ_ONLY FORWARD_ONLY LOCAL`**: default cursors are slow and persist beyond the current batch.
    - Pattern: `rg -n 'DECLARE\s+\w+\s+CURSOR\b' <file>` → confirm `READ_ONLY FORWARD_ONLY LOCAL` (or `FAST_FORWARD`) is specified.
    - **Fix:** `DECLARE c CURSOR READ_ONLY FORWARD_ONLY LOCAL FOR SELECT ...`.

17. **Missing `GRANT EXECUTE` at end of `CREATE PROC`**: only flag when the rest of the repo has a consistent `GRANT EXECUTE ON ... TO <role>` pattern at the end of every SP file. Detect the pattern by sampling a few neighboring `.sql` files; if every existing SP grants to the same role and this new SP doesn't, flag it.
    - Pattern: `rg -n 'GRANT\s+EXECUTE' <file>` → confirm present if the repo convention requires it.
    - **Fix:** Append `GRANT EXECUTE ON [dbo].[<proc>] TO [<app_role>] GO` matching the role the rest of the repo grants to.

## Output format

```
# SQL review: uncommitted changes

**SQL files changed:** <N> (<S new>, <M modified>)

---

## Verdict
**Ship / Fix blockers first**

<one-paragraph summary of the change and overall state>

---

## Blockers (<N>)

### B1. <short title>: `path/to/file.sql:<line>`
**Evidence:** `<offending SQL fragment, quoted verbatim>`
**Why:** <root cause / production consequence: one line>
**Fix:** <concrete recommendation: one line>

### B2. ...

## Warnings (<N>)

### W1. <short title>: `path/to/file.sql:<line>`
**Evidence:** `<offending SQL fragment, quoted verbatim>`
**Why:** ...
**Fix:** ...

## Info (<N>)
- `path:line`: <one-liner>

---

## Summary
- BLOCKER: <count>
- WARN: <count>
- INFO: <count>

## Fixes I can apply

If you want, I can apply any of: B1, B2, W1. Which? (or "all", or "none")
```

End with the offer. Wait for the user's choice.

## Examples

### Example 1: New stored proc with mixed issues

**User:** "review my SQL"

**Agent:**
- Runs `git diff --staged -- '*.sql'` and `git diff -- '*.sql'` in parallel.
- Finds one new file `Stored Procedures/dbo.usp_NewSave.sql`.
- Spots:
  - `B1`: `BEGIN CATCH` without `;THROW;` (line 88), swallowing.
  - `B2`: calls `EXEC msdb.dbo.sp_send_dbmail` inside that CATCH (line 92), masking risk.
  - `W1`: missing `SET NOCOUNT ON` at line 6.
  - `W2`: `WITH (NOLOCK)` on the table being UPDATEd (line 45).
  - `I1`: missing `GRANT EXECUTE` at end of file (the repo's other SPs all end with it).
- Reports, then asks: "Want me to fix B1, B2, W1, W2, I1?"

### Example 2: Adding a new table: the canonical incident pattern

**User:** "review the SQL diff before I push"

**Agent:**
- Spots `Tables/dbo.NewAuditLog.sql` (new file): `CREATE TABLE` declared but **no PRIMARY KEY, no CLUSTERED INDEX, no CREATE INDEX**.
- Flags as `B1`: "Heap table will table-scan on every read/UPDATE. Once the table grows past a few hundred thousand rows under concurrent writes, SQL Server's optimizer picks parallel plans and the workers deadlock on heap pages: a recurring production incident pattern."
- Fix: "Add `PRIMARY KEY CLUSTERED (id)` or a clustered index on the lookup column."
- Also flags the SP that populates the new table (`B2`: implicit type conversion `WHERE recordID = @recordID` where the new table's `recordID` is `INT` but the param is declared `VARCHAR`).

### Example 3: Modified existing SP: flag only new badness, not legacy

**User:** "lint the SQL: just the staged hunks"

**Agent:**
- The file `usp_UpdateBigOldSP.sql` is modified but huge. The user's diff added a 20-line block that does `UPDATE dbo.Customers SET ...` without a `WHERE` clause.
- **Only flags the new block's `UPDATE` without `WHERE` as `B1`**: does NOT flag the legacy `sp_send_dbmail` call elsewhere in the file (which was already there before the user's edit).
- Notes in the report footer: "1 pre-existing antipattern observed in this file but outside the diff: flagging suppressed per scope. Full-file audit available on request."

## Anti-patterns

- ❌ Trying to parse SQL with a full grammar: use ripgrep on the diff text. The skill is heuristic by design.
- ❌ Flagging every existing antipattern in a modified file. Stay scoped to the diff for modified files; full-file scan only for new files.
- ❌ Editing SQL during the review pass. Report first, **always**.
- ❌ Suggesting autofixes for `BLOCKER` items without asking. SQL deploys to production; user confirmation gates the change.
- ❌ Listing a `nit` when there's a `BLOCKER` next to it. Lead with severity.
- ❌ Hand-wavy findings without `file:line`.
- ❌ Reporting a raw ripgrep hit without reading the surrounding procedure. A pattern match is a candidate; the proc's full context sets the severity (see [Pattern hit ≠ finding](#pattern-hit--finding)).
- ❌ Paraphrasing the offending SQL instead of quoting it verbatim. `WITH (NOLOCK)` vs `READPAST`, `@retry` vs `@retryCount`: the exact token is the finding.
- ❌ A `BLOCKER` with no one-sentence production-incident scenario. Can't name the incident? It's a `WARN`.
- ❌ Padding a clean diff with stretched findings. Zero findings is a valid report.
- ❌ Inventing antipatterns not in [What to check](#what-to-check). This is a fixed catalog: extending it is a change to this skill, not a per-run improvisation.
- ❌ Running on non-SQL files. The skill is `.sql`-scoped; mixed diffs get a general code review for the non-SQL parts.
- ✅ Diff-scoped scan for modified files, full scan for new files, findings categorized + cited + offered as fixes.

## Notes

- The check catalog is intentionally finite: 17 checks chosen because each one maps to a real production incident class. Don't try to be a full SQL linter (e.g. SQLFluff or sqlcheck): those exist and run from CI; this skill is the pre-commit human-pass complement focused on bug classes that linters miss.
- For repos that use flat idempotent `Deploy_*.sql` files (every deploy is a full re-run), check #14 (`DROP` / `TRUNCATE` without `IF EXISTS`) is critical: the deploy will fail on second run otherwise. For repos that use forward-only migrations, this check is lower priority.
- The "broken retry" pattern (check #2) is sneakier than it sounds: a `DECLARE @retry INT = 5;` followed by `SET @retry = @retry - 1;` in CATCH but no `WHILE` loop *looks* like deadlock handling. It isn't. Multiple SPs in real codebases have this bug from removed-loop refactors.
- If the user wants the broader cleanup (refactor every legacy SP in the repo with one of these issues), recommend opening a tracked task rather than doing it in the current review: these fixes ripple through deploy pipelines and need coordinated review.
- On a mixed diff (`.cs` + `.sql`), pair this review of the database changes with a general code review of the application code.
