/**
 * Post-build script: rewrites relative imports in dist/ for Node.js ESM.
 *
 * TypeScript source uses extensionless imports (for bundler compatibility),
 * but Node.js ESM requires explicit paths. This script patches the
 * compiled output after tsc:
 *
 *   - `./foo`  where foo.js exists       → `./foo.js`
 *   - `./foo`  where foo/index.js exists → `./foo/index.js`
 *   - `./foo`  with neither              → leaves unchanged (let Node throw a
 *                                          recognizable error)
 *
 * The directory case matters: when a folder is promoted from a single .ts
 * file to a folder with its own index.ts (e.g. `skills/messaging/shared`),
 * naive `.js` suffixing produces `./shared.js`, which Node ESM cannot
 * resolve to `./shared/index.js`.
 */
import { readdir, readFile, writeFile, stat } from 'fs/promises';
import { dirname, join, resolve } from 'path';

const distDir = resolve(new URL('..', import.meta.url).pathname, 'dist');

async function walk(dir) {
  const files = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const fullPath = join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...await walk(fullPath));
    } else if (
      entry.name.endsWith('.js') ||
      entry.name.endsWith('.d.ts') ||
      entry.name.endsWith('.d.ts.map')
    ) {
      files.push(fullPath);
    }
  }
  return files;
}

async function exists(p) {
  try { await stat(p); return true; } catch { return false; }
}

const relativeImportRe = /((?:from|import\s*\()\s*['"])(\.\.?\/[^'"]+?)(['"])/g;

async function resolveImport(fileDir, importPath, isDts) {
  // Already has an extension (e.g. .json, .css) — leave alone.
  if (/\.\w+$/.test(importPath)) return null;

  const candidateExt = isDts ? '.d.ts' : '.js';
  const fileCandidate = resolve(fileDir, `${importPath}${candidateExt}`);
  if (await exists(fileCandidate)) return `${importPath}${candidateExt}`;

  const dirCandidate = resolve(fileDir, importPath);
  if (await exists(dirCandidate)) {
    const dirIndex = join(dirCandidate, `index${candidateExt}`);
    if (await exists(dirIndex)) return `${importPath}/index${candidateExt}`;
  }

  return null;
}

const files = await walk(distDir);
let patched = 0;
let unresolved = 0;

for (const file of files) {
  const content = await readFile(file, 'utf-8');
  const fileDir = dirname(file);
  const isDts = file.endsWith('.d.ts') || file.endsWith('.d.ts.map');

  const matches = [];
  content.replace(relativeImportRe, (match, prefix, importPath, suffix, offset) => {
    matches.push({ match, prefix, importPath, suffix, offset });
    return match;
  });

  let updated = '';
  let cursor = 0;
  for (const m of matches) {
    updated += content.slice(cursor, m.offset);
    const resolvedPath = await resolveImport(fileDir, m.importPath, isDts);
    if (resolvedPath === null) {
      // Couldn't resolve to a file or dir-with-index — leave as-is so Node's
      // own error message points at the genuine problem.
      updated += m.match;
      unresolved++;
    } else {
      updated += `${m.prefix}${resolvedPath}${m.suffix}`;
    }
    cursor = m.offset + m.match.length;
  }
  updated += content.slice(cursor);

  if (updated !== content) {
    await writeFile(file, updated);
    patched++;
  }
}

console.log(
  `fix-extensions: patched ${patched}/${files.length} files in dist/` +
  (unresolved ? ` (${unresolved} unresolved imports left untouched)` : ''),
);
