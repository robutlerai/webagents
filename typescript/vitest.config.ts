import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: [
      'tests/unit/**/*.test.ts',
      'tests/integration/**/*.test.ts',
      'tests/interop/**/*.test.ts',
      'tests/compliance/**/*.test.ts',
      'tests/docs/**/*.test.ts',
    ],
    exclude: [
      'tests/e2e/**',
    ],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      include: ['src/**/*.ts'],
      exclude: [
        'node_modules/',
        'tests/',
        'dist/',
        '**/*.d.ts',
        // `src/cli/**` was excluded here with the comment "CLI tested via
        // E2E". It was not: the only E2E file was `tests/e2e/cli.test.ts`,
        // which the `exclude` above kept out of vitest while playwright's
        // `testMatch: '**/*.spec.ts'` kept it out of playwright, so it ran
        // nowhere and the CLI had no coverage at all. The file now lives in
        // `tests/unit/cli/` and this exclusion is gone with it (2026-09-23).
      ],
    },
    testTimeout: 10000,
    hookTimeout: 10000,
  },
});
