# History Surgery

Use this file for commit cleanup, interactive rebase, autosquash, cherry-pick cleanup, and safe force-push guidance.

## Interactive rebase use cases

Use interactive rebase when the branch history is local or not yet shared and the user wants to:

- squash fix commits
- reorder commits
- reword messages
- split a commit
- drop accidental commits

Start with:

```bash
git status
git log --oneline --graph --decorate -n 20
git rebase -i <base>
```

Common bases:

```bash
git rebase -i HEAD~5
git rebase -i origin/main
git rebase -i main
```

## Rebase actions

- `pick`: keep commit as-is
- `reword`: edit message only
- `edit`: stop and amend or split
- `squash`: combine and keep message text
- `fixup`: combine and discard fix commit message
- `drop`: remove commit

## Split a commit

During `edit`:

```bash
git reset HEAD^
# stage subsets intentionally
git add -p
git commit -m "first part"
git add -A
git commit -m "second part"
git rebase --continue
```

## Autosquash flow

Create targeted fixup commits:

```bash
git commit --fixup <commit>
git rebase -i --autosquash <base>
```

Prefer this over manual squash ordering when the user is doing incremental cleanup.

## Cherry-pick guidance

Use cherry-pick for moving specific commits across branches without merging full branch history.

```bash
git cherry-pick <commit>
```

For ranges:

```bash
git cherry-pick <oldest>^..<newest>
```

If conflicts are likely, warn first and provide abort path:

```bash
git cherry-pick --abort
```

## Force-push guidance

If history was rewritten and remote update is intended, prefer:

```bash
git push --force-with-lease
```

Prefer `--force-with-lease` over `--force` unless the user explicitly understands the stronger overwrite behavior.

## Revert vs rewrite

- Use `revert` when the commit is already shared and should be undone safely.
- Use rebase/reset only when rewriting history is acceptable.
