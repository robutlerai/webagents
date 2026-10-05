# Homebrew

```bash
brew install robutlerai/tap/webagents
```

installs the TypeScript package's two commands, `webagents` and `robutler`, on
Homebrew's own Node. `brew install robutlerai/tap/robutler` installs the same
package under the other name.

## The pieces

| Piece | Where | What it is |
| --- | --- | --- |
| The tap | [`robutlerai/homebrew-tap`](https://github.com/robutlerai/homebrew-tap) | The repository Homebrew reads. Written by the release, never edited by hand. |
| Its source | [`tap/`](tap/) | Every file of the tap: the two formulae, its README and its own test workflow. |
| The writer | [`scripts/homebrew-tap.sh`](../../scripts/homebrew-tap.sh) | Copies `tap/` into a checkout of the tap and stamps each formula's `url` and `sha256` for one published npm version. It does not commit or push. |
| The release job | `homebrew` in [`publish-typescript.yml`](../../.github/workflows/publish-typescript.yml) | After the npm publish: writes the tap, installs each formula from it on macOS, runs `brew test`, then pushes. |

## How the formulae are built

- **Two formulae, one package.** `webagents` and `robutler` install the same
  npm tarball with `npm install` into the formula's own folder and link every
  command the package declares in `bin`, so both commands arrive with either.
  The two files differ in their name and description and in nothing else, and
  the writer refuses to run when they do.
- **Why `robutler` is not an alias.** Homebrew loads a formula from a tap only
  once it is trusted. `brew install <owner>/<tap>/<name>` trusts the name it
  was given, and only when a formula file has that name, so an install through
  an alias is refused. A formula that depends on the other, or declares a
  conflict with it, is refused the same way, because Homebrew has to load the
  other one to check. So each formula stands alone.
- **One of the two at a time.** Both link the same two commands, so the second
  to be installed is not linked while the first is still installed.
- **Node comes with it.** The formulae depend on `node@24`, the Node line the
  package's tests run on. That formula is keg-only (not on `PATH`), so the
  install rewrites both commands' first line to point at it. A different `node`
  earlier on the user's `PATH` changes nothing.
- **Nothing is compiled,** so there are no bottles (Homebrew's prebuilt
  packages): an install is one `npm install`.
- **`@VERSION@` and `@SHA256@` in `tap/Formula/*.rb` are placeholders.** The
  sha256 is of the tarball npm serves, which exists only after the publish, so
  the writer fills both in. The files here are not installable as they stand.
- **On Linux the sandbox's programs are the system's.** The sandbox takes
  bubblewrap, socat and ripgrep from root-owned folders only (`/usr/bin` and
  the like), which Homebrew's prefix is not, so the formulae do not depend on
  Homebrew's copies of them. Their caveat points at `webagents sandbox setup`.

The tap is where the formulae live until the project meets homebrew-core's
notability bar. From then the `webagents` formula can be submitted there, with
`robutler` as an alias of it, and `brew install webagents` works without the
`robutlerai/tap/` prefix.

## Setting it up, once

1. Create the tap repository. The name must be `homebrew-tap` for
   `robutlerai/tap` to resolve to it, and it must be public.

   ```bash
   gh repo create robutlerai/homebrew-tap --public \
     --description "Homebrew formulae for WebAgents and Robutler"
   ```

2. Make the first push, for the version npm serves now.

   ```bash
   git clone git@github.com:robutlerai/homebrew-tap.git ../homebrew-tap
   ./scripts/homebrew-tap.sh ../homebrew-tap
   git -C ../homebrew-tap add -A
   git -C ../homebrew-tap commit -m "webagents $(node -p "require('./typescript/package.json').version")"
   git -C ../homebrew-tap push -u origin HEAD:main
   ```

3. Give the release workflow a key that can push to the tap and to nothing
   else: a deploy key on the tap, its private half a secret of this repository.

   ```bash
   ssh-keygen -t ed25519 -N "" -C "webagents release" -f ./homebrew-tap-key
   gh repo deploy-key add ./homebrew-tap-key.pub --repo robutlerai/homebrew-tap \
     --allow-write --title "webagents release workflow"
   gh secret set HOMEBREW_TAP_DEPLOY_KEY --repo robutlerai/webagents < ./homebrew-tap-key
   rm ./homebrew-tap-key ./homebrew-tap-key.pub
   ```

From then on every TypeScript release updates the tap.

## Changing a formula

Make the change in both `tap/Formula/webagents.rb` and
`tap/Formula/robutler.rb`. To try it without touching your own Homebrew, write
the tap into an empty folder and install from it inside Homebrew's Docker
image:

```bash
mkdir -p ../homebrew-tap-try
./scripts/homebrew-tap.sh ../homebrew-tap-try
docker run --rm -v "$(cd ../homebrew-tap-try && pwd):/tap:ro" ghcr.io/homebrew/brew:main bash -c '
  set -e
  taps="$(brew --repository)/Library/Taps/robutlerai"
  mkdir -p "$taps" && cp -RP /tap "$taps/homebrew-tap"
  brew style robutlerai/tap
  for formula in webagents robutler; do
    brew install "robutlerai/tap/$formula"
    brew test "robutlerai/tap/$formula"
    webagents --version
    brew uninstall "$formula"
  done
'
```

The change reaches the tap with the next TypeScript release. To send it
sooner, write the tap by hand for the version already published
([RELEASE.md](../../RELEASE.md#homebrew)).
