// Flat config for ESLint 9. Before this file existed the `lint` script
// exited 2 on every run (eslint 9 requires flat config, and no TS parser
// was installed), so the CI lint step has never linted anything.
//
// Rule floor is the honest one: the set that is CLEAN on src today.
// Ratchet upward deliberately — never widen without making the tree pass.
import tseslint from 'typescript-eslint';

export default tseslint.config(
  {
    ignores: ['dist/**', 'node_modules/**', 'coverage/**'],
  },
  {
    files: ['src/**/*.ts'],
    languageOptions: {
      parser: tseslint.parser,
    },
    plugins: {
      '@typescript-eslint': tseslint.plugin,
    },
    rules: {
      // Correctness-class rules only, for now.
      'no-debugger': 'error',
      'no-dupe-else-if': 'error',
      'no-duplicate-case': 'error',
      'no-unreachable': 'error',
      'no-sparse-arrays': 'error',
      'use-isnan': 'error',
      'valid-typeof': 'error',
      'no-cond-assign': 'error',
      'no-compare-neg-zero': 'error',
    },
  },
);
