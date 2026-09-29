/**
 * The repository's `skills/` folder (2026-09-29): SKILL.md skills under
 * MIT-0, so they can be used and republished anywhere, some written here and
 * some brought in from ClawHub after a review. `skills/README.md` says what a
 * skill must be to land here; this file holds every skill to it, so a skill
 * that stopped loading, or a file that changed after its review, fails here
 * rather than on someone's machine:
 *
 *   - every skill loads with the SDK's own loader, under its folder's name,
 *     with no loader warnings;
 *   - `PROVENANCE.json` has a record for every skill and no other, and the
 *     sha256 of every file matches the record: a file edited, added or
 *     removed after the review is caught, and a change has to update the
 *     record (and say why) on purpose;
 *   - no symbolic links, no binaries, and each skill inside the installer's
 *     own limits (`skillmd-install.ts`);
 *   - the folder's licence is MIT-0, and a skill that keeps another licence
 *     (Anthropic's Apache-2.0 ones) names it in its front matter, carries its
 *     LICENSE.txt, is listed in the folder's NOTICE, and every file changed
 *     from the original says so (Apache 2.0, section 4(b)).
 *
 * The Python twin is `python/tests/skills/local/test_repo_skills_folder.py`.
 */

import { describe, expect, it } from 'vitest';
import { createHash } from 'node:crypto';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { BINARY_EXTENSIONS, EXTRACTED_LIMIT, FILE_LIMIT } from '../../../src/skills/skillmd/skillmd-install';
import { loadSkillDir } from '../../../src/skills/skillmd/skillmd-loader';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '../../../../skills');
const PROVENANCE = JSON.parse(fs.readFileSync(path.join(ROOT, 'PROVENANCE.json'), 'utf8')) as {
  license: string;
  skills: Record<string, { files: Record<string, string>; original_files: Record<string, string>; source: string; license: string }>;
};

/** Every file under `dir`, relative to it with `/`, sorted; links are reported, never followed. */
function filesIn(dir: string): { files: string[]; links: string[] } {
  const files: string[] = [];
  const links: string[] = [];
  const walk = (current: string) => {
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const full = path.join(current, entry.name);
      const relative = path.relative(dir, full).split(path.sep).join('/');
      if (entry.isSymbolicLink()) links.push(relative);
      else if (entry.isDirectory()) walk(full);
      else files.push(relative);
    }
  };
  walk(dir);
  return { files: files.sort(), links };
}

const skillDirs = fs
  .readdirSync(ROOT, { withFileTypes: true })
  .filter((entry) => entry.isDirectory())
  .map((entry) => entry.name)
  .sort();

describe("the repository's skills folder", () => {
  it('is MIT-0, and has a provenance record for every skill and no other', () => {
    expect(fs.readFileSync(path.join(ROOT, 'LICENSE'), 'utf8').split('\n')[0]).toBe('MIT No Attribution');
    expect(PROVENANCE.license).toBe('MIT-0');
    expect(skillDirs.length).toBeGreaterThan(0);
    expect(Object.keys(PROVENANCE.skills).sort()).toEqual(skillDirs);
  });

  for (const name of skillDirs) {
    it(`${name}: loads cleanly, matches its reviewed files, and holds nothing the installer would refuse`, () => {
      const dir = path.join(ROOT, name);
      const loaded = loadSkillDir(dir);
      expect('problem' in loaded ? loaded.reason : '').toBe('');
      if ('problem' in loaded) return;
      expect(loaded.declaredName).toBe(name);
      expect(loaded.warnings).toEqual([]);

      const { files, links } = filesIn(dir);
      expect(links).toEqual([]);
      expect(files.length).toBeLessThanOrEqual(FILE_LIMIT);
      const record = PROVENANCE.skills[name];
      expect(Object.keys(record.files).sort()).toEqual(files);
      let bytes = 0;
      for (const file of files) {
        const data = fs.readFileSync(path.join(dir, file));
        bytes += data.length;
        expect(`sha256:${createHash('sha256').update(data).digest('hex')}`, file).toBe(record.files[file]);
        expect((BINARY_EXTENSIONS as readonly string[]).includes(path.extname(file).toLowerCase()), file).toBe(false);
      }
      expect(bytes).toBeLessThanOrEqual(EXTRACTED_LIMIT);

      if (record.license.startsWith('Apache-2.0')) {
        expect(loaded.license).toBe('Apache-2.0');
        expect(fs.readFileSync(path.join(dir, 'LICENSE.txt'), 'utf8')).toMatch(/Apache License\s+Version 2\.0/);
        expect(fs.readFileSync(path.join(ROOT, 'NOTICE'), 'utf8')).toContain(name);
        for (const file of files) {
          if (!file.endsWith('.md') || record.original_files[file] === record.files[file]) continue;
          expect(fs.readFileSync(path.join(dir, file), 'utf8'), file).toContain("Changed from Anthropic's original");
        }
      } else {
        expect(record.license.startsWith('MIT-0')).toBe(true);
        expect(loaded.license).toBe('MIT-0');
      }
    });
  }
});
