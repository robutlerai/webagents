#!/usr/bin/env bash
# The Homebrew tap, written from this repository.
#
# The tap (github.com/robutlerai/homebrew-tap, which is what
# `brew install robutlerai/tap/webagents` and `robutlerai/tap/robutler` read)
# holds nothing of its own: every file in it is a copy of
# integrations/homebrew/tap/, with each formula's `url` and `sha256` stamped
# for one published npm version. This script makes that copy into a checkout
# of the tap. It never commits and never pushes: the release workflow
# (publish-typescript.yml, job `homebrew`) does that after installing the
# result, and a person does it for the first push.
#
# The formulae in this repository carry @VERSION@ and @SHA256@, not a real
# release, on purpose. The sha256 is of the tarball npm serves, which exists
# only after the publish, so a number committed here would be one release
# stale by construction, and would look as if it mattered.
#
# TWO FORMULAE, ONE PACKAGE. `webagents` and `robutler` install the same npm
# package and differ in their name and description only. `robutler` cannot be
# a Homebrew alias of `webagents`: since Homebrew 6.0 a formula from a tap
# loads only once it is trusted, `brew install <owner>/<tap>/<name>` trusts
# the name it was given only when a formula FILE has that name, and an alias
# has none, so the install was refused. Nor can either depend on or declare a
# conflict with the other, since loading the other is refused the same way.
# So each stands alone, and this script refuses to write them if anything
# but the name and the description has drifted apart.
#
# Usage:
#   ./scripts/homebrew-tap.sh <tap-checkout>         # the version in typescript/package.json
#   ./scripts/homebrew-tap.sh <tap-checkout> 0.4.0   # an explicit published version
#
# Flags:
#   -h | --help       Show help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PACKAGE="webagents"
SOURCE="$REPO_ROOT/integrations/homebrew/tap"
FORMULAE=("Formula/webagents.rb" "Formula/robutler.rb")

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
err()   { color "0;31" "✗ $*" 1>&2; }

die() {
    err "$*"
    exit 1
}

usage() {
    sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

validate_semver() {
    [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "Invalid version: '$1' (expected X.Y.Z)"
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

formula_body() {
    # A formula without the two lines that may differ between them.
    sed -e '/^class [A-Z][A-Za-z]* < Formula$/d' -e '/^  desc "/d' "$1"
}

sha256_of() {
    # macOS has shasum; a slim Linux image may have only sha256sum.
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d' ' -f1
    else
        sha256sum "$1" | cut -d' ' -f1
    fi
}

# ----------------------------- arg parsing -----------------------------

POSITIONAL=()
while (( $# )); do
    case "$1" in
        -h|--help) usage 0;;
        -*) die "Unknown flag: $1 (use --help)";;
        *) POSITIONAL+=("$1"); shift;;
    esac
done

(( ${#POSITIONAL[@]} >= 1 )) || usage 1
(( ${#POSITIONAL[@]} <= 2 )) || die "Too many positional args (got ${#POSITIONAL[@]}; expected at most 2)"

TAP_DIR="${POSITIONAL[0]}"
VERSION="${POSITIONAL[1]:-$(current_typescript_version)}"
validate_semver "$VERSION"

[[ -d "$TAP_DIR" ]] || die "No such folder: $TAP_DIR (clone robutlerai/homebrew-tap there first)"
TAP_DIR="$(cd "$TAP_DIR" && pwd -P)"

# Stamping the source in place would turn the template into the stale number
# the header describes, and the next run would find nothing left to stamp.
[[ "$TAP_DIR" != "$(cd "$SOURCE" && pwd -P)" ]] \
    || die "$TAP_DIR is the tap's source in this repository; give a checkout of the tap itself"

for formula in "${FORMULAE[@]}"; do
    grep -q '@VERSION@' "$SOURCE/$formula" && grep -q '@SHA256@' "$SOURCE/$formula" \
        || die "$SOURCE/$formula has lost its @VERSION@ or @SHA256@ placeholder"
    if ! diff <(formula_body "$SOURCE/${FORMULAE[0]}") <(formula_body "$SOURCE/$formula") >/dev/null; then
        diff <(formula_body "$SOURCE/${FORMULAE[0]}") <(formula_body "$SOURCE/$formula") 1>&2 || true
        die "${FORMULAE[0]} and $formula differ in more than their name and description; make the same change in both"
    fi
done

# ----------------------------- the published tarball -----------------------------

URL="https://registry.npmjs.org/$PACKAGE/-/$PACKAGE-$VERSION.tgz"
TARBALL="$(mktemp)"
trap 'rm -f "$TARBALL"' EXIT

# The registry serves a new version's tarball within seconds of `npm publish`
# returning, not always at once, and this runs straight after it in the
# release workflow. Two minutes of patience before calling the version
# unpublished.
info "Fetching $URL"
fetched=0
for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
    if curl -fsSL -o "$TARBALL" "$URL"; then
        fetched=1
        break
    fi
    printf '    not there yet (attempt %s of 12)\n' "$attempt"
    sleep 10
done
(( fetched )) || die "$PACKAGE $VERSION is not on npm: $URL"

SHA256="$(sha256_of "$TARBALL")"
[[ "$SHA256" =~ ^[0-9a-f]{64}$ ]] || die "Could not compute the tarball's sha256"

# ----------------------------- write the tap -----------------------------

# Files the tap has and this folder does not (its .git above all) are left
# alone.
cp -RP "$SOURCE/." "$TAP_DIR/"

for formula in "${FORMULAE[@]}"; do
    sed -i.bak -e "s/@VERSION@/$VERSION/g" -e "s/@SHA256@/$SHA256/g" "$TAP_DIR/$formula"
    rm -f "$TAP_DIR/$formula.bak"
    if grep -qE '@VERSION@|@SHA256@' "$TAP_DIR/$formula"; then
        die "$TAP_DIR/$formula still has a placeholder after stamping"
    fi
done

ok "$PACKAGE $VERSION written to $TAP_DIR"
printf '  url:    %s\n' "$URL"
printf '  sha256: %s\n' "$SHA256"
