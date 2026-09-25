/**
 * Static relative imports under `src/core/` and `src/daemon/` carry NO file
 * extension (2026-09-23).
 *
 * WHY THIS IS A TEST AND NOT A STYLE NOTE. The portal bundles this package's
 * `src/core/**` through `instrumentation.ts`:
 *
 *   instrumentation.ts -> lib/agents/* -> webagents/typescript/src/core/index.ts
 *                                      -> src/core/extensions/local-dev.ts
 *
 * Turbopack resolves those files as TypeScript SOURCE, where `./x.js` does not
 * point at `./x.ts`. One import written as `'../../agents/index.js'` took the
 * portal's dev server down with `Module not found: Can't resolve
 * '../../agents/index.js'` until it was changed. `tsc` accepted it, this
 * package's own vitest accepted it, and the break only appeared in the
 * application that consumes the package.
 *
 * The package ships `.js` specifiers in `dist/` regardless: `scripts/
 * fix-extensions.mjs` adds them after `tsc`. So extensionless source is both
 * correct here and correct on disk after a build.
 *
 * `src/cli/**` is exempt: it is not in the portal's graph, it is loaded as
 * built ESM, and its DYNAMIC imports already use `.js` by convention.
 */

import { describe, expect, it } from 'vitest';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import * as path from 'node:path';

const SRC = path.resolve(__dirname, '../../../src');
const BUNDLED_BY_THE_PORTAL = ['core', 'daemon'];

function walk(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) walk(full, out);
    else if (full.endsWith('.ts') && !full.endsWith('.d.ts')) out.push(full);
  }
  return out;
}

describe('import extension convention', () => {
  it('no static relative import in core/ or daemon/ carries a .js extension', () => {
    const offenders: string[] = [];

    for (const area of BUNDLED_BY_THE_PORTAL) {
      for (const file of walk(path.join(SRC, area))) {
        readFileSync(file, 'utf8').split('\n').forEach((line, i) => {
          // Static `import ... from '../x.js'` / `export ... from './x.js'`.
          if (/^\s*(import|export)\b[^;]*\sfrom\s+['"]\.[^'"]*\.js['"]/.test(line)) {
            offenders.push(`${path.relative(SRC, file)}:${i + 1}: ${line.trim()}`);
          }
        });
      }
    }

    expect(
      offenders,
      'these resolve under tsc but break the portal build; drop the .js',
    ).toEqual([]);
  });

  it('the guard actually fires on the shape that broke it', () => {
    const pattern = /^\s*(import|export)\b[^;]*\sfrom\s+['"]\.[^'"]*\.js['"]/;

    for (const broken of [
      "import { parseAgentMarkdown } from '../../agents/index.js';",
      "export { thing } from './thing.js';",
      "import type { T } from '../types.js';",
    ]) {
      expect(pattern.test(broken), broken).toBe(true);
    }

    // ...and not on the shapes that are fine.
    for (const fine of [
      "import { parseAgentMarkdown } from '../../agents/index';",
      "import * as fs from 'fs';",
      "import { x } from 'node:path';",
      "    const { C } = await import('./daemon-client.js');", // dynamic, cli/
    ]) {
      expect(pattern.test(fine), fine).toBe(false);
    }
  });
});
