/**
 * The frozen notes in the prompt (gap-closure plan item 2.1, principle 6,
 * 2026-09-26): a bounded rendering of the notes a caller may see, computed
 * once per session and never updated mid-session, so the provider's prompt
 * cache holds across the turns of a conversation. The Python twin is
 * `skills/local/memory/memory_notes.py`; both are pinned by the `notes` cases
 * of `python/tests/fixtures/memory_tool/definition.json`.
 *
 * The budget counts the entry lines only, so the structure is always whole;
 * entries come newest first and stop at the first that does not fit, and the
 * rest are counted rather than shown (the model has `memory_search`).
 */

export const NOTES_HEADING = '## Memory';
export const NOTES_TITLES = {
  owner: 'Your notes (owner)',
  shared: 'Shared notes',
  caller: 'Notes about this caller',
} as const;
export const DEFAULT_NOTES_BUDGET = 4000;

export interface NoteLine {
  key: string;
  content: string;
}

export interface NotesSection {
  title: string;
  entries: NoteLine[];
}

function oneLine(content: string): string {
  return content.replace(/\s*\n+\s*/g, ' ').trim();
}

/** The prompt text, or an empty string when there is nothing to show. */
export function renderNotes(sections: readonly NotesSection[], budget: number): string {
  const lines: string[] = [NOTES_HEADING];
  let used = 0;
  let shown = 0;
  let more = 0;
  let stopped = false;
  for (const section of sections) {
    const fitting: string[] = [];
    for (const entry of section.entries) {
      if (stopped) {
        more += 1;
        continue;
      }
      const line = `- ${entry.key}: ${oneLine(entry.content)}`;
      if (used + line.length > budget) {
        stopped = true;
        more += 1;
        continue;
      }
      used += line.length;
      fitting.push(line);
    }
    if (fitting.length) {
      lines.push(`${section.title}:`, ...fitting);
      shown += fitting.length;
    }
  }
  if (!shown) return '';
  if (more) lines.push(`(${more} more notes; use memory_search to find them.)`);
  return lines.join('\n');
}
