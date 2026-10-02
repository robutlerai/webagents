# Release Instructions

The `webagents` repo publishes two packages:

- **Python**: [`webagents`](https://pypi.org/project/webagents/) on PyPI, sources in [`python/`](python/)
- **TypeScript**: [`webagents`](https://www.npmjs.com/package/webagents) on npm, sources in [`typescript/`](typescript/)

Releases are driven by **git tags**:

| Tag pattern        | Workflow                                        | Publishes to |
| ------------------ | ----------------------------------------------- | ------------ |
| `python-v*`        | [`publish-python.yml`](.github/workflows/publish-python.yml)         | PyPI         |
| `typescript-v*`    | [`publish-typescript.yml`](.github/workflows/publish-typescript.yml) | npm          |

The two packages are versioned independently. They happen to currently share a version, but you can bump either on its own.

A pushed tag does not publish straight away. The publish workflow first checks that the tag names the version in the package file (`python/pyproject.toml` or `typescript/package.json`), and runs the whole CI workflow for that package (`ci-python.yml` or `ci-typescript.yml`) on the tagged commit. Nothing is published unless both pass. A manual dispatch runs the same CI before publishing.

## Prerequisites

- Push access to `git@github.com:robutlerai/webagents.git`.
- Repository secrets configured:
  - `PYPI_API_TOKEN`: PyPI API token with upload permissions on the `webagents` project.
  - npm publishing uses **OIDC trusted publishing** (`npm publish --provenance`); the `webagents` package on npmjs.com must list this repo + workflow as a trusted publisher.
- Local tools (only needed if you don't use the script): `git`, `node` / `npm`, `python` + `build` + `twine`.

## Recommended: `scripts/release.sh`

The release script bumps versions, commits, tags, and pushes; the GitHub Actions workflows then build and publish.

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

1. `target`: `python` | `typescript` | `both` (default `both`)
2. `version`: explicit `X.Y.Z`, or `patch` | `minor` | `major` (default `patch`)

Flags:

- `--dry-run`: print everything it would do, change nothing
- `--skip-checks`: skip clean-tree / branch / up-to-date checks
- `--remote <name>`: git remote to push to (default `origin`)
- `--branch <name>`: expected current branch (default `main`)

What the script does, in order:

1. Verifies clean working tree, current branch, and that `origin/<branch>` is in sync.
2. Computes the new version(s) and refuses if a matching tag already exists.
3. Renames the `## Unreleased` section of `CHANGELOG.md` to `## X.Y.Z (date)` (both versions when they differ), and stages it with the release commit.
4. Updates `python/pyproject.toml` and/or runs `npm version --no-git-tag-version` in `typescript/`.
5. Creates a single commit (`Release: python X.Y.Z, typescript X.Y.Z`).
6. Creates annotated tags `python-v<ver>` / `typescript-v<ver>`.
7. Pushes the branch and then each tag, which triggers the publish workflows.

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

The workflow runs CI on the dispatched commit first, then rewrites the version on the fly, but it does **not** commit or tag: prefer the tagged path so the repo and the published artifact stay in sync.

### The sandbox engine inside the Python package

The Python wheel carries srt (`@anthropic-ai/sandbox-runtime`) and its
dependencies under `python/webagents/sandbox/sandbox_engine/`, the files npm
publishes for the version the TypeScript lockfile pins. To move srt to a new
version: change `typescript/package.json` and its lockfile, `SRT_VERSION` in
both `srt` modules, then run `python scripts/sandbox_engine_vendor.py` from
`python/` (it checks each tarball against the lockfile's integrity and
rewrites the tree, `NOTICE` and `sandbox_engine_provenance.json`).
`tests/sandbox/sandbox_engine_bundle_test.py` fails until the tree matches the
lockfile. A built wheel should hold every file the record lists: the tree has
`dist/` and `lib/` folders, which `pyproject.toml` includes as hatch
`artifacts` despite `python/.gitignore`.

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

Both packages follow [Semantic Versioning](https://semver.org/). While a
package is below 1.0:

- **MINOR** (0.4.0 to 0.5.0) for any release that removes or renames a public
  API, changes a default, refuses input it used to accept, or needs a newer
  Robutler platform. The CHANGELOG lists each of these under **Breaking**.
- **PATCH** (0.4.0 to 0.4.1) for fixes and additions that change nothing an
  existing agent relies on.

From 1.0, a breaking change is MAJOR, an addition MINOR and a fix PATCH.

A version is never reused: npm and PyPI refuse it, and a yanked version stays
taken. The npm versions 1.0.0 and 1.0.1 are not part of the release line,
which continues from 0.x (see step 5).

## Release checklist

### 1. The platform goes first

The SDK calls platform routes, and reads fields the platform sends, that ship
with the platform. Before either package is published:

- [ ] Every platform change the release depends on is live on robutler.ai,
      with its database changes applied by the platform's own deployment. For
      the current Unreleased section that means: the caller assertion on
      Portal Connect turns, idempotent settles, caller-scoped memory, x402
      credit nonces, channel senders, TrustFlow lookups and records, and the
      budget tree.
- [ ] Check it on the live platform, not from a working tree. A release
      published against an older platform fails closed (see the top of the
      Unreleased section), which is safe but breaks paid and relayed calls
      until the platform catches up.
- [ ] `python/tests/test_endpoint_contract.py` passes: every platform path the
      SDK calls exists in the platform's route manifest
      (`python/tests/fixtures/portal_routes.json`, regenerated from the
      platform).

### 2. The CHANGELOG

- [ ] Rename `## Unreleased` to `## X.Y.Z (YYYY-MM-DD)` for the version being
      published, and start a new, empty `## Unreleased` above it. When the two
      packages ship different versions, say which entries belong to which.
- [ ] Sections in this order: **Breaking**, Added, Changed, Fixed, Security.
- [ ] Every entry is tagged **[py]**, **[ts]** or **[both]**, and every
      breaking entry says what changed and what to edit, with a snippet where
      one helps.
- [ ] Security fixes are described by what is now enforced; no exploit
      detail beyond what a user needs to judge the upgrade.

### 3. Versions, docs and tests

- [ ] `python/pyproject.toml` and `typescript/package.json` carry the new
      version (`scripts/release.sh` sets them).
- [ ] `integrations/acp-registry/webagents/agent.json` carries the TypeScript
      version, in `version` and in the `npx` package pin.
- [ ] The docs under `docs/` describe the release, and the generated snippets
      match the examples: `python3 scripts/sync_doc_examples.py --check`.
- [ ] CI is green on the release commit for both packages. It runs, for
      TypeScript, `pnpm run lint`, `pnpm run typecheck` and
      `pnpm run test:run`; for Python, `ruff check .`,
      `python scripts/lint_annotations.py webagents` and `pytest -q`. The
      publish workflow runs the same CI again on the tag.
- [ ] [SECURITY.md](SECURITY.md)'s supported-versions table still holds.

### 4. Publish

Tag and push with `scripts/release.sh` (above). Nothing is published unless
the tag names the version in the package file and CI passes on the tagged
commit.

### 5. Deprecate broken releases

A release with a serious bug or a security hole is deprecated once the release
that fixes it is out, so installers see the warning and the upgrade:

```bash
npm deprecate webagents@<version> "<what is wrong>; upgrade to X.Y.Z"
```

On PyPI, yank the release (the project's release page, **Options**, **Yank**)
with the same reason. A yanked release still installs when pinned exactly, and
never for a range.

To deprecate when the next release is out: npm `webagents` 0.3.6, 1.0.0 and
1.0.1, and PyPI `webagents` 0.3.6.

### 6. After publishing

1. Watch the workflow:
   - https://github.com/robutlerai/webagents/actions/workflows/publish-python.yml
   - https://github.com/robutlerai/webagents/actions/workflows/publish-typescript.yml
2. Verify install, and that a plain install resolves to the new version:
   ```bash
   pip install --upgrade webagents==X.Y.Z
   pip index versions webagents
   npm view webagents dist-tags        # latest must be X.Y.Z
   npm view webagents@X.Y.Z
   ```
3. Publish the GitHub release notes from the CHANGELOG section.
4. Update the ACP registry entry to the new version.
5. Announce on the relevant channels if it's a notable release.

## Troubleshooting

- **PyPI 403 / "Invalid or non-existent authentication"**: `PYPI_API_TOKEN` missing, expired, or scoped to the wrong project.
- **Version already exists**: both PyPI and npm forbid overwriting an existing version. Bump and re-tag.
- **npm `provenance` failure**: the `webagents` npm package isn't configured as a trusted publisher for this repo+workflow, or the workflow lacks `id-token: write` permission (it has it; check OIDC config on npmjs.com).
- **Tag already exists locally**: `git tag -d <tag>` to drop it, then re-run the script.
- **Workflow didn't trigger**: confirm the tag actually pushed (`git ls-remote --tags origin | grep <tag>`) and that the tag name matches the `python-v*` / `typescript-v*` patterns exactly.
