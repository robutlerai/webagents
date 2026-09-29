"""
The repository's `skills/` folder (2026-09-29): SKILL.md skills under MIT-0,
so they can be used and republished anywhere, some written here and some
brought in from ClawHub after a review. `skills/README.md` says what a skill
must be to land here; this file holds every skill to it, so a skill that
stopped loading, or a file that changed after its review, fails here rather
than on someone's machine:

  * every skill loads with the SDK's own loader, under its folder's name, with
    no loader warnings;
  * `PROVENANCE.json` has a record for every skill and no other, and the
    sha256 of every file matches the record: a file edited, added or removed
    after the review is caught, and a change has to update the record (and say
    why) on purpose;
  * no symbolic links, no binaries, and each skill inside the installer's own
    limits (`skillmd_install.py`);
  * the folder's licence is MIT-0, and a skill that keeps another licence
    (Anthropic's Apache-2.0 ones) names it in its front matter, carries its
    LICENSE.txt, is listed in the folder's NOTICE, and every file changed from
    the original says so (Apache 2.0, section 4(b)).

The TypeScript twin is `typescript/tests/unit/skills/repo-skills-folder.test.ts`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pytest

from webagents.agents.skills.local.skillmd.skillmd_install import BINARY_EXTENSIONS, EXTRACTED_LIMIT, FILE_LIMIT
from webagents.agents.skills.local.skillmd.skillmd_loader import SkippedSkill, load_skill_dir

ROOT = Path(__file__).resolve().parents[4] / "skills"
PROVENANCE = json.loads((ROOT / "PROVENANCE.json").read_text())
SKILL_DIRS = sorted(entry.name for entry in ROOT.iterdir() if entry.is_dir())


def _files_in(directory: Path):
    """Every file under `directory`, relative with `/`, sorted; links reported, never followed."""
    files, links = [], []
    for current, folders, names in os.walk(directory, followlinks=False):
        for name in folders + names:
            full = Path(current) / name
            relative = full.relative_to(directory).as_posix()
            if full.is_symlink():
                links.append(relative)
            elif full.is_file():
                files.append(relative)
    return sorted(files), links


def test_the_folder_is_mit0_and_has_a_provenance_record_for_every_skill_and_no_other():
    assert (ROOT / "LICENSE").read_text().split("\n")[0] == "MIT No Attribution"
    assert PROVENANCE["license"] == "MIT-0"
    assert SKILL_DIRS
    assert sorted(PROVENANCE["skills"]) == SKILL_DIRS


@pytest.mark.parametrize("name", SKILL_DIRS)
def test_a_skill_loads_cleanly_matches_its_reviewed_files_and_holds_nothing_the_installer_refuses(name):
    directory = ROOT / name
    loaded = load_skill_dir(str(directory))
    assert not isinstance(loaded, SkippedSkill), loaded.reason if isinstance(loaded, SkippedSkill) else ""
    assert loaded.declared_name == name
    assert loaded.warnings == []

    files, links = _files_in(directory)
    assert links == []
    assert len(files) <= FILE_LIMIT
    record = PROVENANCE["skills"][name]
    assert sorted(record["files"]) == files
    total = 0
    for relative in files:
        data = (directory / relative).read_bytes()
        total += len(data)
        assert "sha256:" + hashlib.sha256(data).hexdigest() == record["files"][relative], relative
        assert os.path.splitext(relative)[1].lower() not in BINARY_EXTENSIONS, relative
    assert total <= EXTRACTED_LIMIT

    if record["license"].startswith("Apache-2.0"):
        assert loaded.license == "Apache-2.0"
        assert re.search(r"Apache License\s+Version 2\.0", (directory / "LICENSE.txt").read_text())
        assert name in (ROOT / "NOTICE").read_text()
        for relative in files:
            if relative.endswith(".md") and record["original_files"][relative] != record["files"][relative]:
                assert "Changed from Anthropic's original" in (directory / relative).read_text(), relative
    else:
        assert record["license"].startswith("MIT-0")
        assert loaded.license == "MIT-0"
