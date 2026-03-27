/**
 * Post-build script: adds .js extensions to relative imports in dist/
 *
 * TypeScript source uses extensionless imports (for bundler compatibility),
 * but Node.js ESM requires explicit .js extensions. This script patches
 * the compiled output after tsc.
 */
import { readdir, readFile, writeFile, stat } from 'fs/promises';
import { join, resolve } from 'path';

const distDir = resolve(new URL('..', import.meta.url).pathname, 'dist');

async function walk(dir) {
  const files = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const fullPath = join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...await walk(fullPath));
    } else if (entry.name.endsWith('.js') || entry.name.endsWith('.d.ts') || entry.name.endsWith('.d.ts.map')) {
      files.push(fullPath);
    }
  }
  return files;
}

const relativeImportRe = /((?:from|import\s*\()\s*['"])(\.\.?\/[^'"]+?)(['"])/g;

function addJsExtension(match, prefix, importPath, suffix) {
  if (/\.\w+$/.test(importPath)) return match;
  return `${prefix}${importPath}.js${suffix}`;
}

const files = await walk(distDir);
let patched = 0;

for (const file of files) {
  const content = await readFile(file, 'utf-8');
  const updated = content.replace(relativeImportRe, addJsExtension);
  if (updated !== content) {
    await writeFile(file, updated);
    patched++;
  }
}

console.log(`fix-extensions: patched ${patched}/${files.length} files in dist/`);
