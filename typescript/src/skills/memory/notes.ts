/**
 * The memory index in the prompt (gap-closure plan item 2.1, principle 6,
 * 2026-09-26; an index since 2026-09-29): a bounded list of the notes a
 * caller may see, one line each, computed once per session and never updated
 * mid-session, so the provider's prompt cache holds across the turns of a
 * conversation. The Python twin is `skills/local/memory/memory_notes.py`;
 * both are pinned by the `notes` cases of
 * `python/tests/fixtures/memory_tool/definition.json`.
 *
 * AN INDEX, NOT THE NOTES (2026-09-29, the owner: "is it index based like in
 * claude?"). The block carried the newest notes' full text, one line each,
 * until 4,000 characters ran out, and the rest were only counted, so the model
 * did not know what else it had. It is now what Claude Code's `MEMORY.md` is:
 * every note's key and a one-line description (the note's own, or its first
 * line), newest first, so the model sees everything it remembers and reads a
 * note in full with `memory_read` when it needs it. The budget counts the
 * entry lines only, so the structure is always whole; the entries that do not
 * fit are counted (`memory_list` shows them all).
 */

export const NOTES_HEADING = '## Memory';
export const NOTES_GUIDE =
  'One line per note you keep, newest first. memory_read gives a note in full; memory_write keeps one, with a one-line description.';
export const NOTES_TITLES = {
  owner: 'Your notes (owner)',
  shared: 'Shared notes',
  caller: 'Notes about this caller',
} as const;
export const DEFAULT_NOTES_BUDGET = 4000;
/** The most of a description (or a first line) an index line shows. */
export const DESCRIPTION_CHARS = 120;

export interface NoteLine {
  key: string;
  description?: string;
  content: string;
}

export interface NotesSection {
  title: string;
  entries: NoteLine[];
}

function oneLine(content: string): string {
  return content.replace(/\s*\n+\s*/g, ' ').trim();
}

function cut(text: string, limit = DESCRIPTION_CHARS): string {
  return text.length <= limit ? text : `${text.slice(0, limit - 1).trimEnd()}…`;
}

/** A note's first line with text, without a heading's or a list's mark. */
export function firstLine(content: string): string {
  for (const line of String(content ?? '').split(/\r?\n/)) {
    const text = line.trim().replace(/^(?:#{1,6}\s+|[-*+]\s+|>\s*)/, '').trim();
    if (text) return text;
  }
  return '';
}

/** One note in the index: `- key: description`, the description the note's own or its first line, cut to `DESCRIPTION_CHARS`. */
export function noteLine(entry: NoteLine): string {
  const description = oneLine(String(entry.description ?? '')) || firstLine(entry.content);
  return description ? `- ${entry.key}: ${cut(description)}` : `- ${entry.key}`;
}

/** The prompt text, or an empty string when there is nothing to show. */
export function renderNotes(sections: readonly NotesSection[], budget: number): string {
  const lines: string[] = [NOTES_HEADING, NOTES_GUIDE];
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
      const line = noteLine(entry);
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
  if (more) lines.push(`(${more} more notes; memory_list shows them all.)`);
  return lines.join('\n');
}
