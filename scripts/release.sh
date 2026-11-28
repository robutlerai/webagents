#!/usr/bin/env bash
# WebAgents release script.
#
# Names the changelog's "Unreleased" section after the release, bumps the
# version in the selected package(s), commits, tags (`python-v<ver>` /
# `typescript-v<ver>` / `rust-v<ver>`) and pushes. The existing GitHub Actions
# workflows then publish to PyPI / npm / crates.io and create the GitHub
# release.
#
# Usage:
#   ./scripts/release.sh                        # both packages, patch bump
#   ./scripts/release.sh python                 # python only, patch bump
#   ./scripts/release.sh typescript minor       # typescript, minor bump
#   ./scripts/release.sh both 0.4.0             # python and typescript, explicit version
#   ./scripts/release.sh rust 0.2.0             # the Rust crates (webagents and robutler)
#   ./scripts/release.sh all minor              # python, typescript and rust
#   ./scripts/release.sh python 0.3.5 --dry-run # preview, no commit/push
#
# Flags:
#   --dry-run         Print actions without writing/committing/pushing
#   --skip-checks     Skip clean-tree / branch / up-to-date checks
#   --remote <name>   Git remote to push to (default: origin)
#   --branch <name>   Expected current branch (default: main)
#   -h | --help       Show help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TARGET="both"
VERSION_ARG="patch"
DRY_RUN=0
SKIP_CHECKS=0
REMOTE="origin"
BRANCH="main"

# ----------------------------- helpers -----------------------------

color() {
    # color <code> <text...>
    local code="$1"
    shift
    if [[ -t 1 ]]; then
        printf '\033[%sm%s\033[0m\n' "$code" "$*"
    else
        printf '%s\n' "$*"
    fi
}

info()  { color "0;36" "==> $*"; }
ok()    { color "0;32" "✓ $*"; }
warn()  { color "0;33" "! $*"; }
err()   { color "0;31" "✗ $*" 1>&2; }

die() {
    err "$*"
    exit 1
}

run() {
    # Execute a command (or print it in dry-run mode).
    if (( DRY_RUN )); then
        printf '[dry-run] %s\n' "$*"
    else
        eval "$@"
    fi
}

usage() {
    sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

validate_semver() {
    [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "Invalid version: '$1' (expected X.Y.Z)"
}

bump_version() {
    # bump_version <current> <patch|minor|major>
    local current="$1" kind="$2"
    validate_semver "$current"
    local major minor patch
    IFS='.' read -r major minor patch <<<"$current"
    case "$kind" in
        patch) patch=$((patch + 1));;
        minor) minor=$((minor + 1)); patch=0;;
        major) major=$((major + 1)); minor=0; patch=0;;
        *) die "Unknown bump kind: $kind";;
    esac
    printf '%s.%s.%s\n' "$major" "$minor" "$patch"
}

current_python_version() {
    grep -E '^version = ' "$REPO_ROOT/python/pyproject.toml" \
        | head -n1 \
        | sed -E 's/version = "(.*)"/\1/'
}

current_rust_version() {
    # The [workspace.package] version, which both crates inherit.
    sed -n '/^\[workspace.package\]/,/^\[/s/^version = "\(.*\)"$/\1/p' "$REPO_ROOT/rust/Cargo.toml" | head -n1
}

current_typescript_version() {
    if command -v node >/dev/null 2>&1; then
        node -p "require('$REPO_ROOT/typescript/package.json').version"
    else
        grep -E '"version"\s*:' "$REPO_ROOT/typescript/package.json" \
            | head -n1 \
            | sed -E 's/.*"version"\s*:\s*"([^"]+)".*/\1/'
    fi
}

resolve_new_version() {
    # resolve_new_version <current>
    # Uses global $VERSION_ARG.
    local current="$1"
    case "$VERSION_ARG" in
        patch|minor|major) bump_version "$current" "$VERSION_ARG";;
        *) validate_semver "$VERSION_ARG"; printf '%s\n' "$VERSION_ARG";;
    esac
}

tag_exists() {
    # tag_exists <tag>
    git -C "$REPO_ROOT" rev-parse --verify --quiet "refs/tags/$1" >/dev/null
}

update_python_version() {
    # update_python_version <new_version>
    local new="$1"
    if (( DRY_RUN )); then
        printf "[dry-run] sed -i.bak -E 's/^version = \".*\"/version = \"%s\"/' python/pyproject.toml\n" "$new"
        return
    fi
    local file="$REPO_ROOT/python/pyproject.toml"
    sed -i.bak -E "s/^version = \".*\"/version = \"$new\"/" "$file"
    rm -f "$file.bak"
    # The `robutler` package on PyPI follows: the same version, pinned to it.
    node "$REPO_ROOT/scripts/thin-packages.mjs" --only pypi --write >/dev/null
}

update_typescript_version() {
    # update_typescript_version <new_version>
    local new="$1"
    if (( DRY_RUN )); then
        printf '[dry-run] (cd typescript && npm version %s --no-git-tag-version)\n' "$new"
        return
    fi
    (cd "$REPO_ROOT/typescript" && npm version "$new" --no-git-tag-version >/dev/null)
    # The `robutler` package on npm follows: the same version, pinned to it.
    node "$REPO_ROOT/scripts/thin-packages.mjs" --only npm --write >/dev/null
}

update_rust_version() {
    # update_rust_version <new_version>
    # The workspace version, robutler's pin on webagents, and both crates'
    # entries in Cargo.lock, so the tagged commit builds with --locked.
    local new="$1"
    if (( DRY_RUN )); then
        printf '[dry-run] rust/Cargo.toml, rust/crates/robutler/Cargo.toml, rust/Cargo.lock: -> %s\n' "$new"
        return
    fi
    local root="$REPO_ROOT/rust" f
    sed -i.bak -E '/^\[workspace.package\]/,/^\[/s/^version = ".*"$/version = "'"$new"'"/' "$root/Cargo.toml"
    sed -i.bak -E 's/^(webagents = \{ version = )"[^"]*"/\1"'"$new"'"/' "$root/crates/robutler/Cargo.toml"
    awk -v v="$new" '
        /^name = "(webagents|robutler)"$/ { print; hit = 1; next }
        hit && /^version = / { print "version = \"" v "\""; hit = 0; next }
        { hit = 0; print }
    ' "$root/Cargo.lock" > "$root/Cargo.lock.new" && mv "$root/Cargo.lock.new" "$root/Cargo.lock"
    for f in "$root/Cargo.toml" "$root/crates/robutler/Cargo.toml"; do rm -f "$f.bak"; done
    [[ "$(current_rust_version)" == "$new" ]] || die "rust/Cargo.toml still says $(current_rust_version), expected $new"
}

changelog_heading() {
    # The heading the release gives CHANGELOG.md's "## Unreleased" section:
    # the version when one package is released or all released ones share it,
    # else each package with its own.
    local today; today="$(date +%Y-%m-%d)"
    local versions=() parts=() v
    [[ -n "$PY_NEW" ]] && { versions+=("$PY_NEW"); parts+=("python $PY_NEW"); }
    [[ -n "$TS_NEW" ]] && { versions+=("$TS_NEW"); parts+=("typescript $TS_NEW"); }
    [[ -n "$RS_NEW" ]] && { versions+=("$RS_NEW"); parts+=("rust $RS_NEW"); }
    for v in "${versions[@]}"; do
        if [[ "$v" != "${versions[0]}" ]]; then
            printf '## %s (%s)\n' "$(join_by ', ' "${parts[@]}")" "$today"
            return
        fi
    done
    printf '## %s (%s)\n' "${versions[0]}" "$today"
}

stamp_changelog() {
    # Rename "## Unreleased" in CHANGELOG.md to the release's heading. The
    # first step of a release, so the tagged commit's changelog names it; a
    # changelog with no such section is left alone, with a warning.
    local file="$REPO_ROOT/CHANGELOG.md" heading
    heading="$(changelog_heading)"
    if ! grep -qx '## Unreleased' "$file" 2>/dev/null; then
        warn "CHANGELOG.md has no '## Unreleased' section; left as it is"
        return 1
    fi
    if (( DRY_RUN )); then
        printf "[dry-run] CHANGELOG.md: '## Unreleased' -> '%s'\n" "$heading"
        return 0
    fi
    sed -i.bak -E "s/^## Unreleased\$/$heading/" "$file"
    rm -f "$file.bak"
}

join_by() {
    # join_by <sep> <parts...>
    local sep="$1"; shift
    local out="${1:-}"; shift || true
    while (( $# )); do out+="$sep$1"; shift; done
    printf '%s' "$out"
}

# ----------------------------- arg parsing -----------------------------

POSITIONAL=()
while (( $# )); do
    case "$1" in
        -h|--help) usage 0;;
        --dry-run) DRY_RUN=1; shift;;
        --skip-checks) SKIP_CHECKS=1; shift;;
        --remote) REMOTE="${2:?--remote needs a value}"; shift 2;;
        --branch) BRANCH="${2:?--branch needs a value}"; shift 2;;
        --) shift; while (( $# )); do POSITIONAL+=("$1"); shift; done;;
        -*) die "Unknown flag: $1 (use --help)";;
        *) POSITIONAL+=("$1"); shift;;
    esac
done

if (( ${#POSITIONAL[@]} >= 1 )); then
    TARGET="${POSITIONAL[0]}"
fi
if (( ${#POSITIONAL[@]} >= 2 )); then
    VERSION_ARG="${POSITIONAL[1]}"
fi
if (( ${#POSITIONAL[@]} > 2 )); then
    die "Too many positional args (got ${#POSITIONAL[@]}; expected at most 2)"
fi

case "$TARGET" in
    python|typescript|both|rust|all) ;;
    *) die "Invalid target '$TARGET' (expected python|typescript|both|rust|all)";;
esac

case "$VERSION_ARG" in
    patch|minor|major) ;;
    *) validate_semver "$VERSION_ARG";;
esac

# ----------------------------- pre-flight -----------------------------

cd "$REPO_ROOT"

if (( ! SKIP_CHECKS )); then
    info "Running pre-flight checks (use --skip-checks to bypass)"

    git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
        || die "Not inside a git repo: $REPO_ROOT"

    REMOTE_URL="$(git remote get-url "$REMOTE" 2>/dev/null || true)"
    [[ -n "$REMOTE_URL" ]] || die "Remote '$REMOTE' not configured"
    if ! [[ "$REMOTE_URL" == *robutlerai/webagents* ]]; then
        warn "Remote '$REMOTE' = $REMOTE_URL (expected robutlerai/webagents)"
    fi

    if [[ -n "$(git status --porcelain)" ]]; then
        die "Working tree is dirty. Commit or stash changes first."
    fi

    CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
    if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
        die "On branch '$CURRENT_BRANCH' but expected '$BRANCH' (use --branch to override)"
    fi

    info "Fetching $REMOTE"
    git fetch "$REMOTE" "$BRANCH" --tags
    LOCAL="$(git rev-parse "$BRANCH")"
    UPSTREAM="$(git rev-parse "$REMOTE/$BRANCH")"
    if [[ "$LOCAL" != "$UPSTREAM" ]]; then
        die "Local '$BRANCH' is not in sync with '$REMOTE/$BRANCH' (pull/rebase first)"
    fi

    ok "Pre-flight checks passed"
else
    warn "Skipping pre-flight checks"
fi

# ----------------------------- compute plan -----------------------------

PY_OLD=""; PY_NEW=""; PY_TAG=""
TS_OLD=""; TS_NEW=""; TS_TAG=""
RS_OLD=""; RS_NEW=""; RS_TAG=""

if [[ "$TARGET" == "python" || "$TARGET" == "both" || "$TARGET" == "all" ]]; then
    PY_OLD="$(current_python_version)"
    PY_NEW="$(resolve_new_version "$PY_OLD")"
    PY_TAG="python-v$PY_NEW"
    if tag_exists "$PY_TAG"; then
        die "Tag $PY_TAG already exists"
    fi
fi

if [[ "$TARGET" == "typescript" || "$TARGET" == "both" || "$TARGET" == "all" ]]; then
    TS_OLD="$(current_typescript_version)"
    TS_NEW="$(resolve_new_version "$TS_OLD")"
    TS_TAG="typescript-v$TS_NEW"
    if tag_exists "$TS_TAG"; then
        die "Tag $TS_TAG already exists"
    fi
fi

if [[ "$TARGET" == "rust" || "$TARGET" == "all" ]]; then
    RS_OLD="$(current_rust_version)"
    RS_NEW="$(resolve_new_version "$RS_OLD")"
    RS_TAG="rust-v$RS_NEW"
    if tag_exists "$RS_TAG"; then
        die "Tag $RS_TAG already exists"
    fi
fi

# `DRY_RUN` is 0 or 1, never empty: `${DRY_RUN:+...}` labelled every real run a dry run.
DRY_LABEL=""; (( DRY_RUN )) && DRY_LABEL=" (dry-run)"
info "Release plan$DRY_LABEL:"
[[ -n "$PY_NEW" ]] && printf '  python:     %s -> %s   (tag %s)\n' "$PY_OLD" "$PY_NEW" "$PY_TAG"
[[ -n "$TS_NEW" ]] && printf '  typescript: %s -> %s   (tag %s)\n' "$TS_OLD" "$TS_NEW" "$TS_TAG"
[[ -n "$RS_NEW" ]] && printf '  rust:       %s -> %s   (tag %s)\n' "$RS_OLD" "$RS_NEW" "$RS_TAG"
printf '  remote:     %s\n' "$REMOTE"
printf '  branch:     %s\n' "$BRANCH"

# ----------------------------- apply -----------------------------

FILES_TO_STAGE=()
COMMIT_PARTS=()
TAGS_TO_PUSH=()

info "Naming the changelog's Unreleased section: $(changelog_heading)"
if stamp_changelog; then
    FILES_TO_STAGE+=("CHANGELOG.md")
fi

if [[ -n "$PY_NEW" ]]; then
    info "Updating python version to $PY_NEW"
    update_python_version "$PY_NEW"
    FILES_TO_STAGE+=("python/pyproject.toml")
    FILES_TO_STAGE+=("packages/robutler-py/pyproject.toml")
    COMMIT_PARTS+=("python $PY_NEW")
    TAGS_TO_PUSH+=("$PY_TAG")
fi

if [[ -n "$TS_NEW" ]]; then
    info "Updating typescript version to $TS_NEW"
    update_typescript_version "$TS_NEW"
    FILES_TO_STAGE+=("typescript/package.json")
    FILES_TO_STAGE+=("packages/robutler-npm/package.json")
    COMMIT_PARTS+=("typescript $TS_NEW")
    TAGS_TO_PUSH+=("typescript-v$TS_NEW")
fi

if [[ -n "$RS_NEW" ]]; then
    info "Updating rust version to $RS_NEW"
    update_rust_version "$RS_NEW"
    FILES_TO_STAGE+=("rust/Cargo.toml" "rust/crates/robutler/Cargo.toml" "rust/Cargo.lock")
    COMMIT_PARTS+=("rust $RS_NEW")
    TAGS_TO_PUSH+=("$RS_TAG")
fi

COMMIT_MSG="Release: $(join_by ', ' "${COMMIT_PARTS[@]}")"

info "Committing: $COMMIT_MSG"
run "git add ${FILES_TO_STAGE[*]}"
run "git commit -m \"$COMMIT_MSG\""

for tag in "${TAGS_TO_PUSH[@]}"; do
    info "Creating tag $tag"
    run "git tag -a \"$tag\" -m \"$tag\""
done

info "Pushing branch $BRANCH to $REMOTE"
run "git push \"$REMOTE\" \"$BRANCH\""

for tag in "${TAGS_TO_PUSH[@]}"; do
    info "Pushing tag $tag"
    run "git push \"$REMOTE\" \"$tag\""
done

# ----------------------------- summary -----------------------------

ok "Done${DRY_LABEL:+ (dry-run, nothing was changed)}."
echo
echo "Watch the publish workflows here:"
[[ -n "$PY_NEW" ]] && echo "  https://github.com/robutlerai/webagents/actions/workflows/publish-python.yml"
[[ -n "$TS_NEW" ]] && echo "  https://github.com/robutlerai/webagents/actions/workflows/publish-typescript.yml"
[[ -n "$RS_NEW" ]] && echo "  https://github.com/robutlerai/webagents/actions/workflows/publish-rust.yml"
echo
# `if`, not `[[ ... ]] && echo`: as the script's last command, a false test
# made a python-only release exit 1 after everything had succeeded.
if [[ -n "$PY_NEW" ]]; then
    echo "After publish: pip install webagents==$PY_NEW"
fi
if [[ -n "$TS_NEW" ]]; then
    echo "After publish: npm install webagents@$TS_NEW"
    echo "After publish: brew upgrade webagents   (the tap follows npm in the same workflow run)"
fi
if [[ -n "$RS_NEW" ]]; then
    echo "After publish: cargo add webagents@$RS_NEW"
fi
