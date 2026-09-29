# Advanced Patterns

Use this file for worktree, bisect, sparse checkout, subtree, submodule, and repository-shape decisions.

## Worktree

Use worktree when the user needs parallel checked-out branches.

```bash
git worktree add ../repo-hotfix hotfix/branch
git worktree list
git worktree remove ../repo-hotfix
```

Prefer worktree over stash when the parallel work will last more than a quick interruption.

## Bisect

Use bisect when the user can identify a known-good and known-bad revision.

```bash
git bisect start
git bisect bad
git bisect good <good-revision>
```

After each checkout, mark:

```bash
git bisect good
git bisect bad
```

Finish with:

```bash
git bisect reset
```

If a deterministic test command exists, consider automated bisect.

## Sparse checkout

Use sparse checkout when the repository is large and the user only needs a subset.

```bash
git sparse-checkout init --cone
git sparse-checkout set <dir1> <dir2>
```

Explain that this changes working tree visibility, not repository history.

## Subtree vs submodule

### Prefer subtree when

- contributors want simpler clone/update flows
- the dependency should feel integrated
- external pointer management would create friction

### Prefer submodule when

- the external repository must remain clearly separate
- pinning an exact external revision matters
- contributors can tolerate the extra workflow complexity

## Branch archaeology

Useful inspection commands:

```bash
git blame <file>
git log -- <path>
git log -S "needle" --oneline
git log -G "regex" --oneline
git show <commit> -- <path>
```

Use these when the user is asking when, where, why, or by whom something changed.
