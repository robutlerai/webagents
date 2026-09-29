---
name: git-workflows
description: "Handle advanced git workflows and recovery tasks. Use when the user needs help with interactive rebase, commit cleanup, conflict resolution, reflog recovery, cherry-pick, stash, worktree, bisect, submodule vs subtree decisions, sparse checkout, branch archaeology, or undoing dangerous history mistakes in real repositories."
license: MIT-0
metadata:
  author: "darinrowe"
  source: "https://clawhub.ai/darinrowe/skills/git-workflows-pro"
  version: "1.0.0"
---

# Git Workflows

Use this skill for non-trivial git work where safety, history clarity, or repository structure matters more than a single command.

Keep the main thread focused on the user’s goal. Prefer the smallest safe sequence of git operations.

## Core approach

Before suggesting commands, determine:

1. the user’s goal
2. whether history is already shared with others
3. whether the task changes commits, refs, working tree, or repository structure
4. whether recovery should be prepared first

If a step is destructive or hard to reverse, create a safety point first.

## Default safety rules

- Check `git status` before history edits.
- Check the current branch and upstream before rebases, resets, or force-pushes.
- Prefer non-destructive inspection first.
- Prefer `git switch` and `git restore` over older mixed forms when clarity matters.
- Create a recovery point before risky history surgery.
- Avoid rewriting shared history unless the user explicitly wants that tradeoff.
- When conflicts or recovery are involved, explain both the immediate fix and the rollback path.

## Recovery-first moves

Use these patterns early when risk is high:

```bash
git status
git branch backup/$(date +%Y%m%d-%H%M%S)-preop
```

For history recovery, inspect before changing anything:

```bash
git reflog --date=local --decorate -n 30
git log --oneline --graph --decorate -n 30
```

Read `references/recovery.md` for reflog-based recovery, reset recovery, branch recovery, and force-push mistakes.

## Common task families

### History cleanup

Use for:

- squashing fix commits
- rewording commit messages
- splitting a bad commit
- dropping accidental commits
- preparing a branch before merge

Prefer interactive rebase for local or not-yet-shared history.

### Bug origin hunting

Use `git bisect` when the user knows a good state and a bad state and needs to find the introducing commit.

### Parallel branch work

Use `git worktree` when the user needs two branches checked out at once, wants cleaner hotfix flow, or wants to avoid stashing.

### Conflict handling

Resolve carefully when rebasing, merging, cherry-picking, or applying stashes. Preserve user intent, not just file merge success.

### Repository archaeology

Use blame, pickaxe, grep, log graph, and path history when the user needs to answer:

- who changed this
- when did this break
- where did this line come from
- which commit removed this behavior

### Repository shape changes

Use extra care for:

- submodules
- subtrees
- sparse checkout
- worktree pruning
- branch renames
- default-branch migration

Read `references/advanced-patterns.md` when the task involves repository topology rather than simple day-to-day commits.

## Decision rules

### Rebase vs merge

- Prefer rebase for cleaning local feature-branch history.
- Prefer merge when preserving shared branch history matters.
- If the branch is already shared, explicitly call out the rewrite risk before rebasing.

### Subtree vs submodule

- Prefer subtree when the user wants simpler consumption and fewer contributor footguns.
- Prefer submodule when the user truly needs a pinned external repository boundary.

### Worktree vs stash

- Prefer worktree for medium or long parallel work.
- Prefer stash for short interruptions or tiny context switches.

### Reset vs restore vs revert

- Prefer `restore` for file-level undo.
- Prefer `reset` for local history and index surgery.
- Prefer `revert` for undoing commits that are already shared.

## Operating style

When guiding the user:

1. say what state to inspect first
2. state the safest command sequence
3. note whether history is being rewritten
4. give the rollback path when risk is non-trivial

Keep command sets short. Do not dump every git option unless needed.

## When to read references

- Read `references/recovery.md` for reflog recovery, accidental reset, branch resurrection, detached HEAD recovery, or force-push mistakes.
- Read `references/history-surgery.md` for interactive rebase, commit splitting, autosquash, cherry-pick cleanup, and safe force-push guidance.
- Read `references/advanced-patterns.md` for worktree, bisect, subtree, submodule, sparse checkout, and repository-structure decisions.

## Output template

When responding on a git workflow task, prefer this structure:

- **Goal**:
- **Risk level**:
- **Safe first check**:
- **Recommended commands**:
- **Rollback path**:
- **Shared-history warning**:
