/**
 * Linked AWS credential pairs for SigV4 re-signing.
 *
 * The sentinel registry maps independent name→value fakes, but SigV4
 * needs the access key id, secret access key, and optional session token
 * linked as ONE credential: the signature is an HMAC derived from the
 * secret, so a request signed with the fake secret must be re-signed with
 * the real one — and the only reliable trigger is the fake access key id
 * appearing in the signature's credential scope.
 *
 * Pairs come from two sources (see {@link registerAwsPairs}):
 * - an explicit `credentials.awsPairs` grouping, for non-standard
 *   variable names;
 * - auto-detection of the conventional AWS_ACCESS_KEY_ID /
 *   AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN trio when those vars are
 *   masked whole-value — the names the AWS SDKs and CLI actually read, so
 *   masking them is the opt-in.
 *
 * {@link createSigv4Planner} builds the per-request hook the
 * TLS-terminating proxy runs: exact-match the referenced access key id
 * against registered pair sentinels, classify the signature shape, and
 * either plan a re-sign (header-sigv4) or apply the configured
 * deny/passthrough policy (streaming, presigned, sigv4a).
 */
import { logForDebugging } from '../utils/debug.js';
import { detectSigv4, parseSigv4Authorization, signSigv4, singleHeader, UNSIGNED_PAYLOAD, } from './aws-sigv4.js';
/** Conventional env var names the AWS SDKs and CLI read credentials from. */
export const AWS_ACCESS_KEY_ID_VAR = 'AWS_ACCESS_KEY_ID';
export const AWS_SECRET_ACCESS_KEY_VAR = 'AWS_SECRET_ACCESS_KEY';
export const AWS_SESSION_TOKEN_VAR = 'AWS_SESSION_TOKEN';
/**
 * AWS pairs for one sandbox session, keyed by the access-key-id sentinel.
 * Like the sentinel registry, it lives only in process memory — never
 * written to disk, never logged.
 */
export class AwsPairRegistry {
    constructor() {
        this.bySentinel = new Map();
    }
    /** Insert or replace the pair keyed on its access-key-id sentinel. */
    register(pair) {
        this.bySentinel.set(pair.accessKeyIdSentinel, pair);
    }
    /** Exact-match lookup by the fake access key id; never pattern-based. */
    lookup(accessKeyIdSentinel) {
        return this.bySentinel.get(accessKeyIdSentinel);
    }
    get size() {
        return this.bySentinel.size;
    }
    /** Drop every pair. Called on session teardown. */
    clear() {
        this.bySentinel.clear();
    }
}
/**
 * Register AWS credential pairs from the masked env vars.
 *
 * Runs after `buildMaskedEnvVars`: `setEnvVars` maps each masked variable
 * to its fake in-sandbox value, which for whole-value masking IS the
 * sentinel. Pair specs are the explicit `credentials.awsPairs` entries
 * plus — when none of them claims a conventional name — an implicit
 * AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN spec.
 *
 * A spec registers only when both the access-key-id and secret variables
 * were actually masked (present in the host env, whole-value entries in
 * `setEnvVars`); the session token joins the pair when its variable was
 * masked too. A partially available spec is skipped — with one loud
 * warning for the dangerous half-masking: a masked secret without a
 * masked access key id means SigV4 requests carry a real key id (so
 * nothing triggers re-signing) over a fake-secret signature that can
 * never verify.
 */
export function registerAwsPairs(envVars, awsPairs, allowedDomains, setEnvVars, registry, env = process.env) {
    const specs = [...(awsPairs ?? [])];
    const explicitNames = new Set(specs.flatMap(p => [
        p.accessKeyIdVar,
        p.secretAccessKeyVar,
        ...(p.sessionTokenVar ? [p.sessionTokenVar] : []),
    ]));
    const conventional = [
        AWS_ACCESS_KEY_ID_VAR,
        AWS_SECRET_ACCESS_KEY_VAR,
        AWS_SESSION_TOKEN_VAR,
    ];
    if (!conventional.some(n => explicitNames.has(n))) {
        specs.push({
            accessKeyIdVar: AWS_ACCESS_KEY_ID_VAR,
            secretAccessKeyVar: AWS_SECRET_ACCESS_KEY_VAR,
            sessionTokenVar: AWS_SESSION_TOKEN_VAR,
        });
    }
    // A variable is usable as a pair member when it has a whole-value
    // mask entry AND was actually masked at runtime (set in the host env).
    const wholeValueEntry = (name) => envVars.find(v => v.name === name &&
        v.mode === 'mask' &&
        v.extract === undefined &&
        v.decode === undefined);
    const maskedSentinel = (name) => wholeValueEntry(name) !== undefined ? setEnvVars[name] : undefined;
    for (const spec of specs) {
        const akidSentinel = maskedSentinel(spec.accessKeyIdVar);
        const secretMasked = maskedSentinel(spec.secretAccessKeyVar) !== undefined;
        const realAkid = env[spec.accessKeyIdVar];
        const realSecret = env[spec.secretAccessKeyVar];
        if (akidSentinel === undefined ||
            !secretMasked ||
            realAkid === undefined ||
            realSecret === undefined) {
            if (secretMasked && akidSentinel === undefined) {
                const msg = `[sandbox-runtime] WARNING: "${spec.secretAccessKeyVar}" is ` +
                    `masked but "${spec.accessKeyIdVar}" is not, so no AWS ` +
                    `credential pair was registered. SigV4 requests will be signed ` +
                    `with the masked placeholder secret and the proxy cannot ` +
                    `detect or re-sign them — they will fail upstream. Mask both ` +
                    `variables (whole-value) to enable re-signing.`;
                console.warn(msg);
                logForDebugging(msg, { level: 'warn' });
            }
            continue;
        }
        const akidEntry = wholeValueEntry(spec.accessKeyIdVar);
        const sessionSentinel = spec.sessionTokenVar !== undefined
            ? maskedSentinel(spec.sessionTokenVar)
            : undefined;
        const realSessionToken = sessionSentinel !== undefined && spec.sessionTokenVar !== undefined
            ? env[spec.sessionTokenVar]
            : undefined;
        registry.register({
            accessKeyIdSentinel: akidSentinel,
            realAccessKeyId: realAkid,
            realSecretAccessKey: realSecret,
            ...(realSessionToken !== undefined ? { realSessionToken } : {}),
            injectHosts: akidEntry.injectHosts ?? allowedDomains,
        });
        logForDebugging(`[credential-aws] registered SigV4 pair for ` +
            `${spec.accessKeyIdVar}/${spec.secretAccessKeyVar}` +
            (realSessionToken !== undefined ? ` (+session token)` : ''));
    }
}
export function resolveSigv4Policies(config) {
    return {
        streaming: config?.streaming ?? 'deny',
        presigned: config?.presigned ?? 'deny',
        sigv4a: config?.sigv4a ?? 'deny',
    };
}
/**
 * Build the {@link PlanSigv4} hook from the session's pair registry and
 * configured policies. `matches` is the same host matcher used for
 * sentinel substitution, so injectHosts gating is consistent across both.
 */
export function createSigv4Planner(registry, config, matches) {
    const policies = resolveSigv4Policies(config);
    return (method, requestTarget, headers, destHost) => {
        if (registry.size === 0)
            return undefined;
        const detected = detectSigv4(headers, requestTarget);
        if (detected === null)
            return undefined;
        const pair = registry.lookup(detected.accessKeyId);
        // Unknown key id — including a real, unmasked AWS credential — is
        // never rewritten and never denied.
        if (pair === undefined)
            return undefined;
        if (!pair.injectHosts.some(p => matches(destHost, p)))
            return undefined;
        if (detected.kind !== 'header-sigv4') {
            if (policies[detected.kind] === 'passthrough') {
                logForDebugging(`[credential-aws] ${detected.kind} request with masked ` +
                    `credential passed through un-resigned per ` +
                    `credentials.sigv4.${detected.kind} (its signature covers ` +
                    `the placeholder secret and will fail upstream)`);
                return undefined;
            }
            return {
                action: 'deny',
                reason: `AWS SigV4 ${detected.kind} request uses a masked credential ` +
                    `but cannot be re-signed by the sandbox proxy; denied by ` +
                    `policy. Set credentials.sigv4.${detected.kind} to ` +
                    `"passthrough" to forward it unmodified (it will fail ` +
                    `upstream: the signature covers the masked placeholder).`,
            };
        }
        const auth = parseSigv4Authorization(singleHeader(headers.authorization));
        if (auth === null) {
            return {
                action: 'deny',
                reason: 'AWS SigV4 request uses a masked credential but its ' +
                    'Authorization header is malformed; the proxy cannot re-sign it.',
            };
        }
        const amzDate = singleHeader(headers['x-amz-date']);
        if (amzDate === undefined) {
            return {
                action: 'deny',
                reason: 'AWS SigV4 request uses a masked credential but carries no ' +
                    'x-amz-date header; the proxy cannot re-sign it.',
            };
        }
        // Presence check now so apply() cannot fail after the body was
        // consumed. host is synthesized; a session token the pair carries
        // is injected by apply().
        const signedHeaders = [...auth.signedHeaders];
        for (const name of signedHeaders) {
            if (name === 'host')
                continue;
            if (name === 'x-amz-security-token' &&
                pair.realSessionToken !== undefined) {
                continue;
            }
            if (headers[name] === undefined) {
                return {
                    action: 'deny',
                    reason: `AWS SigV4 request uses a masked credential but signed ` +
                        `header "${name}" is missing; the proxy cannot re-sign it.`,
                };
            }
        }
        if (pair.realSessionToken !== undefined &&
            !signedHeaders.includes('x-amz-security-token')) {
            signedHeaders.push('x-amz-security-token');
        }
        const contentSha = singleHeader(headers['x-amz-content-sha256']);
        return {
            action: 'resign',
            payloadHash: contentSha === UNSIGNED_PAYLOAD ? UNSIGNED_PAYLOAD : undefined,
            apply(fwdHeaders, hostHeader, payloadHash) {
                if (pair.realSessionToken !== undefined) {
                    fwdHeaders['x-amz-security-token'] = pair.realSessionToken;
                }
                if (fwdHeaders['x-amz-content-sha256'] !== undefined &&
                    contentSha !== UNSIGNED_PAYLOAD) {
                    // The recomputed hash is authoritative: the signature must
                    // cover the bytes forwarded upstream, not the client's claim.
                    fwdHeaders['x-amz-content-sha256'] = payloadHash;
                }
                const { authorization } = signSigv4({
                    method,
                    requestTarget,
                    headers: fwdHeaders,
                    hostHeader,
                    signedHeaders,
                    payloadHash,
                    amzDate,
                    scope: {
                        date: auth.date,
                        region: auth.region,
                        service: auth.service,
                    },
                    accessKeyId: pair.realAccessKeyId,
                    secretAccessKey: pair.realSecretAccessKey,
                });
                fwdHeaders.authorization = authorization;
            },
        };
    };
}
//# sourceMappingURL=credential-aws-pairs.js.map