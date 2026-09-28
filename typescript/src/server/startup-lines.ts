/**
 * Startup lines `serve()` prints beside the ones it always printed
 * (2026-09-26, the new-developer e2e run), the same in the Python `serve`
 * (`cli/startup_lines.py`), pinned by `python/tests/fixtures/cli/serve_startup.json`.
 */

/** An agent with `memory` and nothing that verifies a caller: memory is the owner's alone (the fixture's `memory_without_auth`). */
export const MEMORY_WITHOUT_AUTH =
  '[webagents] {name} has memory but no AuthSkill: nothing verifies a caller, so served callers read shared notes and cannot write; only the owner at this terminal can. Add AuthSkill, or an access: block, to identify callers.';

export function memoryWithoutAuthLine(name: string): string {
  return MEMORY_WITHOUT_AUTH.replace('{name}', name);
}
