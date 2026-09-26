# Internal engineering notes

These are contributor notes, not product documentation. They live **here and
not under `docs/`** for one hard reason: everything under `docs/` is published.

The portal's fumadocs source (`source.config.ts`, `defineDocs({ dir:
'webagents/docs' })`) scans that tree recursively and serves every MDX file it
finds at `https://robutler.ai/develop/webagents/...`. There is no allowlist
step in that path. `docs/meta.json` controls only what appears in the sidebar;
a file left out of it is still a public 200.

That is how `docs/MANUAL_TESTING_GUIDE.md` (a QA checklist) and
`docs/internal/python-typescript-parity.md` (a matrix advertising unshipped SDK
gaps) ended up crawlable and, in the first case, submitted to the sitemap. The
portal patched over it with `Disallow` rules in `app/robots.ts` and left a note
saying the durable fix was to move the files out of the served tree. This
directory is that fix, applied 2026-09-22.

So: anything that should not be read by a stranger evaluating the SDK goes
here. Anything written for users goes in `docs/`, and gets a `meta.json` entry.

| File | What it is |
| --- | --- |
| `manual-testing-guide.md` | Step-by-step manual verification of skills and the daemon (`webagents daemon`) |
| `cli-test-coverage-assessment.md` | Where the Python and TypeScript CLI test suites are thin |
| `python-typescript-parity.md` | Source of truth for which features ship in which SDK; drives the "Coming soon" doc tabs |
