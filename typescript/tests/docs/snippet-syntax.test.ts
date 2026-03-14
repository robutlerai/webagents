/**
 * Validate that every TypeScript code block in the documentation compiles.
 *
 * Extracts fenced ```typescript blocks from all markdown files under
 * webagents/docs/ and runs ts.transpileModule() on each one.
 */

import { describe, it, expect } from 'vitest';
import * as ts from 'typescript';
import * as fs from 'node:fs';
import * as path from 'node:path';

const DOCS_ROOT = path.resolve(__dirname, '..', '..', '..', 'docs');

const FENCE_RE = /^```typescript\s*\n([\s\S]*?)^```/gm;

// Blocks that are intentionally pseudo-code (decorator signatures, API reference).
const PSEUDO_CODE_BLOCKS = new Set([
  'api/typescript.md#2',
  'api/typescript.md#3',
  'api/typescript.md#4',
  'api/typescript.md#5',
  'api/typescript.md#6',
  'api/typescript.md#7',
]);

interface Snippet {
  file: string;
  blockIdx: number;
  code: string;
}

function collectSnippets(): Snippet[] {
  const snippets: Snippet[] = [];

  function walk(dir: string) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(full);
      } else if (entry.name.endsWith('.md')) {
        const text = fs.readFileSync(full, 'utf-8');
        let match: RegExpExecArray | null;
        let idx = 0;
        const re = new RegExp(FENCE_RE.source, FENCE_RE.flags);
        while ((match = re.exec(text)) !== null) {
          const rel = path.relative(DOCS_ROOT, full);
          const key = `${rel}#${idx}`;
          if (!PSEUDO_CODE_BLOCKS.has(key)) {
            snippets.push({ file: rel, blockIdx: idx, code: match[1] });
          }
          idx++;
        }
      }
    }
  }

  walk(DOCS_ROOT);
  return snippets;
}

const SNIPPETS = collectSnippets();

const COMPILER_OPTIONS: ts.CompilerOptions = {
  target: ts.ScriptTarget.ES2022,
  module: ts.ModuleKind.ES2022,
  moduleResolution: ts.ModuleResolutionKind.Bundler,
  strict: false,
  noEmit: true,
  skipLibCheck: true,
  // Allow unresolved imports -- we only check syntax, not module resolution
  noResolve: true,
  isolatedModules: true,
};

describe('TypeScript doc snippets compile', () => {
  if (SNIPPETS.length === 0) {
    it('found at least one TypeScript snippet', () => {
      expect.fail('No TypeScript snippets found in docs');
    });
    return;
  }

  for (const snippet of SNIPPETS) {
    it(`${snippet.file}#${snippet.blockIdx}`, () => {
      const result = ts.transpileModule(snippet.code, {
        compilerOptions: COMPILER_OPTIONS,
        reportDiagnostics: true,
        fileName: `${snippet.file}#block${snippet.blockIdx}.ts`,
      });

      const errors = (result.diagnostics ?? []).filter(
        (d) => d.category === ts.DiagnosticCategory.Error
      );

      if (errors.length > 0) {
        const messages = errors.map((d) => {
          const msg = ts.flattenDiagnosticMessageText(d.messageText, '\n');
          const line = d.start !== undefined
            ? `:${ts.getLineAndCharacterOfPosition(
                ts.createSourceFile('', snippet.code, ts.ScriptTarget.ES2022),
                d.start
              ).line + 1}`
            : '';
          return `  ${msg}${line}`;
        });
        expect.fail(
          `TypeScript errors in ${snippet.file} block ${snippet.blockIdx}:\n` +
          messages.join('\n') +
          `\n\n${snippet.code}`
        );
      }
    });
  }
});
