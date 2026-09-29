# Recovery

Use this file when the user needs to recover from a git mistake or wants a rollback plan before risky history edits.

## Reflog first

Start with inspection:

```bash
git reflog --date=local --decorate -n 30
git log --oneline --graph --decorate -n 30
```

Reflog is often the fastest route after:

- `reset --hard`
- bad rebase
- deleted branch
- detached HEAD confusion
- accidental force-push

## Recover after `reset --hard`

```bash
git reflog --date=local --decorate -n 20
git reset --hard HEAD@{1}
```

Use the actual reflog entry that points at the lost state. Inspect first before resetting again.

## Recover deleted branch

Find the lost tip via reflog or `git fsck --lost-found` if needed, then recreate the branch:

```bash
git branch recovered/<name> <commit>
```

## Recover from bad rebase

If rebase is in progress and should be abandoned:

```bash
git rebase --abort
```

If the bad rebase already completed:

```bash
git reflog --date=local --decorate -n 30
git reset --hard <pre-rebase-commit>
```

If the branch was already force-pushed, warn clearly before rewriting again.

## Recover detached HEAD work

If commits were created in detached HEAD:

```bash
git log --oneline --decorate -n 10
git branch recovered/detached-work HEAD
```

Create a branch before switching away.

## Recover overwritten local changes

Check whether the changes are recoverable from:

- stash
- IDE local history
- reflog for committed states
- `git fsck --lost-found` for dangling objects

Do not promise recovery for uncommitted overwritten file edits unless there is evidence.

## Force-push mistakes

Inspect remote and local tips first:

```bash
git fetch --all --prune
git log --oneline --graph --decorate --all -n 40
```

Then recover by pushing the correct commit or restoring a saved branch ref. Prefer creating a recovery branch first.

## Safety pattern before risky operations

```bash
git status
git branch backup/$(date +%Y%m%d-%H%M%S)-preop
```

Use this before reset, rebase, branch surgery, or force-push.
