/**
 * Structured credential extraction, shared by file masking
 * (credential-mask-files.ts) and env-var masking (credential-mask-env.ts).
 *
 * `extract` is a regex whose capture group 1 picks the credential value(s)
 * out of a larger text (a config file, a connection-string env var). Only
 * the captured span(s) are replaced with sentinels; the rest of the text is
 * preserved byte-for-byte so parsers still succeed inside the sandbox.
 */
/**
 * Apply `pattern` globally to `content` and return `content` with each
 * matched capture-group-1 span replaced by `sentinelFor(capture, i)`,
 * where `i` is the zero-based index of the distinct captured value in
 * first-seen order.
 *
 * Offset-based: the regex `d` flag exposes capture-group offsets, so the
 * output is built by slicing between spans and splicing the sentinel in
 * at the exact `[start, end)` of group 1. By default only the
 * regex-matched span is replaced — a captured value that coincidentally
 * appears elsewhere in the content (outside any match) is left intact. No
 * placeholder pass, no substring-ordering concern.
 *
 * With `maskDuplicates` (see {@link ExtractOptions}), an indexOf scan
 * additionally collects every verbatim occurrence of each captured value
 * elsewhere in the content. All spans are computed against the ORIGINAL
 * content — never the partially substituted output — so an inserted
 * sentinel can never be re-matched and corrupted, even when a captured
 * value is a substring of the sentinel literal. Regex-match spans win
 * over verbatim ones, and verbatim scans run longest-capture-first so a
 * shorter capture that is a substring of a longer one cannot claim part
 * of the longer secret's occurrence. A capture the callback declined to
 * mask (returned unchanged — the decode-verification gate) is excluded
 * from the verbatim scan.
 *
 * Returns `null` when the pattern matches nothing — the caller routes
 * that per the entry's `onExtractNoMatch` option (warn / deny / error).
 *
 * Throws when a match has no group-1 capture. The schema already rejects
 * patterns with zero groups, so this only fires when group 1 is optional
 * and absent for some match (e.g. `"token: (\\S+)?"`); accepting that
 * would silently mask nothing for that occurrence.
 *
 * Pure on `content`/`pattern`; the callback may close over a registry.
 */
export function extractAndSubstitute(content, pattern, sentinelFor, options = {}) {
    // The schema validates `pattern` compiles; `g` makes matchAll iterate
    // every occurrence and `d` populates `m.indices` with group offsets.
    const re = new RegExp(pattern, 'gd');
    const indexByCapture = new Map();
    // First sentinel returned per capture — reused for verbatim spans so
    // the duplicate pass never re-invokes the callback.
    const sentinelByCapture = new Map();
    const spans = [];
    for (const m of content.matchAll(re)) {
        const cap = m[1];
        if (cap === undefined) {
            throw new Error(`extract pattern /${pattern}/ matched at offset ${m.index} but ` +
                `capture group 1 is undefined — group 1 must capture the ` +
                `credential value on every match.`);
        }
        // Empty captures are skipped: a zero-width span has nothing to mask.
        if (cap.length === 0)
            continue;
        let i = indexByCapture.get(cap);
        if (i === undefined)
            indexByCapture.set(cap, (i = indexByCapture.size));
        const [start, end] = m.indices[1];
        const sentinel = sentinelFor(cap, i);
        if (!sentinelByCapture.has(cap))
            sentinelByCapture.set(cap, sentinel);
        spans.push({ start, end, sentinel });
    }
    if (indexByCapture.size === 0)
        return null;
    if (options.maskDuplicates) {
        // Longest capture first: a shorter capture that is a substring of a
        // longer one would otherwise claim part of the longer secret's
        // occurrence and leave the remainder exposed.
        const byLength = [...indexByCapture.keys()].sort((a, b) => b.length - a.length);
        for (const cap of byLength) {
            const sentinel = sentinelByCapture.get(cap);
            // The callback declines masking by returning the capture itself
            // (decode verification failed). Replacing a value with itself is a
            // no-op, and its spans must not block other captures' duplicates —
            // skip declined captures entirely.
            if (sentinel === cap)
                continue;
            for (let start = content.indexOf(cap); start !== -1; start = content.indexOf(cap, start + 1)) {
                const end = start + cap.length;
                // Spans already collected (regex matches, then earlier — longer —
                // captures' verbatim occurrences) win over this occurrence.
                if (spans.some(s => start < s.end && s.start < end))
                    continue;
                spans.push({ start, end, sentinel });
            }
        }
        spans.sort((a, b) => a.start - b.start);
    }
    // Spans are non-overlapping and sorted by offset (matchAll yields
    // matches in order; the verbatim pass re-sorts), so a single
    // slice-and-concat pass over the original content is sound.
    let out = '';
    let pos = 0;
    for (const s of spans) {
        out += content.slice(pos, s.start) + s.sentinel;
        pos = s.end;
    }
    return {
        fakeContent: out + content.slice(pos),
        captures: [...indexByCapture.keys()],
    };
}
//# sourceMappingURL=credential-extract.js.map