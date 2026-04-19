# Release Instructions

The `webagents` repo publishes two packages:

- **Python** — [`webagents`](https://pypi.org/project/webagents/) on PyPI, sources in [`python/`](python/)
- **TypeScript** — [`webagents`](https://www.npmjs.com/package/webagents) on npm, sources in [`typescript/`](typescript/)

Releases are driven by **git tags**:

| Tag pattern        | Workflow                                        | Publishes to |
| ------------------ | ----------------------------------------------- | ------------ |
| `python-v*`        | [`publish-python.yml`](.github/workflows/publish-python.yml)         | PyPI         |
| `typescript-v*`    | [`publish-typescript.yml`](.github/workflows/publish-typescript.yml) | npm          |

The two packages are versioned independently. They happen to currently share a version, but you can bump either on its own.

## Prerequisites

- Push access to `git@github.com:robutlerai/webagents.git`.
- Repository secrets configured:
  - `PYPI_API_TOKEN` — PyPI API token with upload permissions on the `webagents` project.
  - npm publishing uses **OIDC trusted publishing** (`npm publish --provenance`); the `webagents` package on npmjs.com must list this repo + workflow as a trusted publisher.
- Local tools (only needed if you don't use the script): `git`, `node` / `npm`, `python` + `build` + `twine`.

## Recommended: `scripts/release.sh`

The release script bumps versions, commits, tags, and pushes — the GitHub Actions workflows then build and publish.

```bash
# both packages, patch bump (default)
./scripts/release.sh

# python only, minor bump
./scripts/release.sh python minor

# typescript only, explicit version
./scripts/release.sh typescript 0.4.0

# both, explicit version
./scripts/release.sh both 0.4.0

# preview without writing/committing/pushing
./scripts/release.sh python 0.3.5 --dry-run
```

Positional args (both optional):

1. `target` — `python` | `typescript` | `both` (default `both`)
2. `version` — explicit `X.Y.Z`, or `patch` | `minor` | `major` (default `patch`)

Flags:

- `--dry-run` — print everything it would do, change nothing
- `--skip-checks` — skip clean-tree / branch / up-to-date checks
- `--remote <name>` — git remote to push to (default `origin`)
- `--branch <name>` — expected current branch (default `main`)

What the script does, in order:

1. Verifies clean working tree, current branch, and that `origin/<branch>` is in sync.
2. Computes the new version(s) and refuses if a matching tag already exists.
3. Updates `python/pyproject.toml` and/or runs `npm version --no-git-tag-version` in `typescript/`.
4. Creates a single commit (`Release: python X.Y.Z, typescript X.Y.Z`).
5. Creates annotated tags `python-v<ver>` / `typescript-v<ver>`.
6. Pushes the branch and then each tag, which triggers the publish workflows.

## Manual fallback

If you can't (or don't want to) use the script, do the equivalent by hand.

### Python

```bash
# 1. Bump version in python/pyproject.toml
$EDITOR python/pyproject.toml   # version = "X.Y.Z"

# 2. Commit, tag, push
git add python/pyproject.toml
git commit -m "Release: python X.Y.Z"
git tag -a python-vX.Y.Z -m python-vX.Y.Z
git push origin main
git push origin python-vX.Y.Z
```

### TypeScript

```bash
# 1. Bump version in typescript/package.json
( cd typescript && npm version X.Y.Z --no-git-tag-version )

# 2. Commit, tag, push
git add typescript/package.json
git commit -m "Release: typescript X.Y.Z"
git tag -a typescript-vX.Y.Z -m typescript-vX.Y.Z
git push origin main
git push origin typescript-vX.Y.Z
```

### Workflow dispatch (no tag)

You can also publish without tagging by running the workflow manually from the **Actions** tab:

- **Publish Python SDK to PyPI** → "Run workflow" → enter version (e.g. `0.3.5`)
- **Publish TypeScript SDK to npm** → "Run workflow" → enter version

The workflow rewrites the version on the fly but does **not** commit or tag — prefer the tagged path so the repo and the published artifact stay in sync.

### Fully local build (testing only)

```bash
# Python
cd python
pip install build twine
python -m build
twine check dist/*
twine upload --repository testpypi dist/*    # optional
twine upload dist/*

# TypeScript
cd typescript
pnpm install --frozen-lockfile
pnpm run build
npm publish --provenance --access public
```

## Version numbering

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR** — breaking changes
- **MINOR** — new features (backward compatible)
- **PATCH** — bug fixes (backward compatible)

## Pre-release checklist

- [ ] All tests pass (Python: `cd python && pytest`; TypeScript: `cd typescript && pnpm test:run`)
- [ ] Lint / format clean (Python: `black .` + `ruff check .`; TypeScript: `pnpm lint` + `pnpm typecheck`)
- [ ] Docs updated where relevant
- [ ] Working tree clean and `main` is up to date with `origin/main`

## Post-release

1. Watch the workflow:
   - https://github.com/robutlerai/webagents/actions/workflows/publish-python.yml
   - https://github.com/robutlerai/webagents/actions/workflows/publish-typescript.yml
2. Verify install:
   ```bash
   pip install --upgrade webagents==X.Y.Z
   npm view webagents@X.Y.Z
   ```
3. Announce on the relevant channels if it's a notable release.

## Troubleshooting

- **PyPI 403 / "Invalid or non-existent authentication"** — `PYPI_API_TOKEN` missing, expired, or scoped to the wrong project.
- **Version already exists** — both PyPI and npm forbid overwriting an existing version. Bump and re-tag.
- **npm `provenance` failure** — the `webagents` npm package isn't configured as a trusted publisher for this repo+workflow, or the workflow lacks `id-token: write` permission (it has it; check OIDC config on npmjs.com).
- **Tag already exists locally** — `git tag -d <tag>` to drop it, then re-run the script.
- **Workflow didn't trigger** — confirm the tag actually pushed (`git ls-remote --tags origin | grep <tag>`) and that the tag name matches the `python-v*` / `typescript-v*` patterns exactly.
