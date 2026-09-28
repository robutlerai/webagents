# Fixture skills repository

A git repository shaped like `anthropics/skills`: `skills/<name>/SKILL.md` with
bundled files, a script, and a `.claude-plugin/marketplace.json`. The tests of
both SDKs `git init` a copy of this folder and install from it over a `file://`
URL, never from the internet.

- `skills/pdf`: the sample skill, with a script (`scripts/fill_form.py`) and a
  sandbox probe (`scripts/probe.py`).
- `skills/xlsx`: a second skill, so `--skill` has something to choose.
- `skills/broken`: no description, so it is reported and never installed.
