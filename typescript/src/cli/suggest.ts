/**
 * "(Did you mean ...?)" for a mistyped command (2026-09-24).
 *
 * Commander's own `suggestSimilar` (`commander/lib/suggestSimilar.js`, not an
 * exported path), ported so the unknown-command check below `program.parse()`
 * suggests exactly what commander suggests elsewhere, and what the Python CLI
 * suggests (`python/webagents/cli/help_format.py`).
 */

const MAX_DISTANCE = 3;

/** Optimal string alignment distance. */
function editDistance(a: string, b: string): number {
  if (Math.abs(a.length - b.length) > MAX_DISTANCE) return Math.max(a.length, b.length);
  const d: number[][] = [];
  for (let i = 0; i <= a.length; i++) d[i] = [i];
  for (let j = 0; j <= b.length; j++) d[0][j] = j;
  for (let j = 1; j <= b.length; j++) {
    for (let i = 1; i <= a.length; i++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) {
        d[i][j] = Math.min(d[i][j], d[i - 2][j - 2] + 1);
      }
    }
  }
  return d[a.length][b.length];
}

/** `\n(Did you mean x?)`, or `''`. */
export function suggestSimilar(word: string, candidates: string[]): string {
  let names = Array.from(new Set(candidates));
  if (names.length === 0) return '';
  const searchingOptions = word.startsWith('--');
  if (searchingOptions) {
    word = word.slice(2);
    names = names.map((candidate) => candidate.slice(2));
  }
  let similar: string[] = [];
  let bestDistance = MAX_DISTANCE;
  for (const candidate of names) {
    if (candidate.length <= 1) continue;
    const distance = editDistance(word, candidate);
    const length = Math.max(word.length, candidate.length);
    if ((length - distance) / length > 0.4) {
      if (distance < bestDistance) {
        bestDistance = distance;
        similar = [candidate];
      } else if (distance === bestDistance) {
        similar.push(candidate);
      }
    }
  }
  similar.sort((a, b) => a.localeCompare(b));
  if (searchingOptions) similar = similar.map((candidate) => `--${candidate}`);
  if (similar.length > 1) return `\n(Did you mean one of ${similar.join(', ')}?)`;
  if (similar.length === 1) return `\n(Did you mean ${similar[0]}?)`;
  return '';
}

/**
 * The position of the first word of `argv` that is not an option or an
 * option's value, given the flags there are and those that take a value; -1
 * when there is none, or when an unknown flag comes first (commander reports
 * that one itself, with its own suggestion).
 */
export function firstOperandIndex(argv: string[], knownFlags: Set<string>, valueFlags: Set<string>): number {
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--') return i + 1 < argv.length ? i + 1 : -1;
    if (arg.startsWith('-') && arg !== '-') {
      const flag = arg.split('=')[0];
      if (!knownFlags.has(flag)) return -1;
      if (!arg.includes('=') && valueFlags.has(flag)) i++;
      continue;
    }
    return i;
  }
  return -1;
}
