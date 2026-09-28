/**
 * Resolved-address guard for hostname destinations.
 *
 * The domain allowlist decides by name, but whoever controls a permitted
 * name's DNS records (or any label under a permitted wildcard) decides what
 * that name resolves to. Without a check on the resolved address, an
 * allow-listed name can be pointed at the loopback interface, a link-local
 * address, or address space the embedder considers private, and the proxy
 * will dial it. This module wraps `dns.lookup` so a direct dial resolves
 * once, drops denied addresses, and connects to a surviving one — the
 * address that passed the check is the address dialed, with no second
 * resolution in between.
 *
 * Scope: hostnames only. An IP literal on the allowlist is an explicit
 * choice and is never re-judged here; conversely, a name may resolve to a
 * denied address only when that address (and port) is itself allow-listed,
 * so reaching it by name grants nothing the literal entry did not — and an
 * IP literal on the deny list is refused by name too, allowlist or not. The
 * reserved loopback names (`localhost` and anything under `.localhost`,
 * RFC 6761) resolve to loopback — that is what allow-listing them asks for —
 * or to an allow-listed literal, and to nothing else. Connections routed
 * through a parent proxy or a MITM socket are not resolved locally at all;
 * that hop resolves the name and is responsible for its own address policy.
 */
import { lookup as dnsLookup } from 'node:dns';
import { BlockList, isIP } from 'node:net';
import { networkInterfaces } from 'node:os';
import { logForDebugging } from '../utils/debug.js';
import { ipLiteralRules } from './domain-pattern.js';
import { addRange, addressInSet, embeddedIPv4, isLoopbackAddress, isLoopbackName, LOOPBACK_RANGES, } from './address.js';
/**
 * Cloud instance-metadata / platform endpoints that live outside link-local
 * space, so `169.254.0.0/16` / `fe80::/10` do not cover them. Single
 * addresses (or a provider-reserved block), so nothing else is caught.
 */
const CLOUD_METADATA_ADDRESSES = [
    '100.100.100.200', // Alibaba Cloud
    '168.63.129.16', // Azure WireServer / host agent endpoint
    '192.0.0.192', // Oracle Cloud Infrastructure Classic
    'fd00:ec2::/32', // AWS IPv6 service block: IMDS ::254, EKS Pod Identity ::23, DNS, NTP
    'fd20:ce::254', // Google Cloud, IPv6-only instances
    'fd00:c1::a9fe:a9fe', // Oracle Cloud Infrastructure IPv6
    'fd00:42::42', // Scaleway IPv6
    'fd00:a9fe:a9fe::1', // Akamai / Linode IPv6
    'fd00:100::100:200', // Alibaba Cloud IPv6
];
/**
 * Destinations an allow-listed hostname may not resolve to, grouped by the
 * class a refusal reports. Addresses assigned to this host's own interfaces
 * are denied too (see {@link localInterfaceAddresses}), since a service
 * bound to 0.0.0.0 answers on those exactly as on loopback. Private-use
 * ranges (RFC 1918, ULA, CGNAT) are deliberately absent: allow-listing an
 * intranet hostname is legitimate, so those are opt-in via
 * `network.deniedResolvedAddresses`. IPv4 rules also bind the IPv6 forms
 * that carry an IPv4 address (see `embeddedIPv4`).
 */
const DENIED_CLASSES = [
    ['a loopback address', LOOPBACK_RANGES],
    ['an unspecified address', ['0.0.0.0/8', '::']], // connects to the local host on common stacks
    ['a link-local address', ['169.254.0.0/16', 'fe80::/10']], // incl. most instance-metadata endpoints
    ['a multicast address', ['224.0.0.0/4', 'ff00::/8']],
    ['the broadcast address', ['255.255.255.255']],
    ['a cloud metadata address', CLOUD_METADATA_ADDRESSES],
];
/** Refusal classes in priority order, so a multi-address reason reads the same however the resolver ordered its answer. */
const REASON_ORDER = [
    'a deny-listed address',
    ...DENIED_CLASSES.map(([why]) => why),
    'a listed address',
    "one of this host's addresses",
    'a non-loopback address',
    'an unparsable address',
];
/** Unicast addresses currently assigned to this host's network interfaces. */
export function localInterfaceAddresses() {
    let byInterface;
    try {
        byInterface = networkInterfaces();
    }
    catch {
        // Interface enumeration is unavailable in some restricted environments;
        // the fixed denied set still applies.
        return [];
    }
    return Object.values(byInterface)
        .flat()
        .flatMap(i => (i ? [i.address] : []));
}
export class ResolvedAddressDeniedError extends Error {
    /**
     * @param reason parenthetical for the violation line and the client, e.g.
     *   `resolved to a loopback address` — the class, never the address, which
     *   the sandboxed client should not learn from a refusal.
     * @param addresses the refused addresses, for operator logs.
     */
    constructor(hostname, reason, addresses) {
        super(`Connection to ${hostname} blocked: ${reason}`);
        this.hostname = hostname;
        this.reason = reason;
        this.addresses = addresses;
        this.code = 'ERR_SRT_RESOLVED_ADDRESS_DENIED';
        this.name = 'ResolvedAddressDeniedError';
    }
}
export function isResolvedAddressDenied(err) {
    return err instanceof ResolvedAddressDeniedError;
}
function buildRuleSet(rules) {
    const set = { anyPort: new BlockList(), byPort: new Map() };
    for (const rule of rules) {
        const { range, port } = typeof rule === 'string' ? { range: rule, port: undefined } : rule;
        let list = port === undefined ? set.anyPort : set.byPort.get(port);
        if (!list)
            set.byPort.set(port, (list = new BlockList()));
        if (!addRange(list, range)) {
            throw new Error(`Invalid IP address or CIDR range: ${JSON.stringify(range)}`);
        }
    }
    return set;
}
function inRuleSet(set, address, port) {
    const forPort = set.byPort.get(port);
    return (addressInSet(set.anyPort, address) ||
        (forPort !== undefined && addressInSet(forPort, address)));
}
export function createResolvedAddressGuard(opts = {}) {
    const refused = buildRuleSet(ipLiteralRules(opts.deniedDomains ?? []));
    const allowed = buildRuleSet(ipLiteralRules(opts.allowedDomains ?? []));
    const denied = [
        ...DENIED_CLASSES.map(([why, ranges]) => [why, buildRuleSet(ranges)]),
        ['a listed address', buildRuleSet(opts.deniedResolvedAddresses ?? [])],
    ];
    const resolve = opts.resolve ?? dnsLookup;
    /** This host's addresses right now; a malformed entry from the seam is skipped. */
    const localSet = () => {
        const list = new BlockList();
        for (const a of (opts.localAddresses ?? localInterfaceAddresses)()) {
            addRange(list, a);
        }
        return list;
    };
    /** Why `hostname:port` may not use resolved `address`, or undefined if it may. */
    const denial = (hostname, address, port, local) => {
        if (isIP(hostname))
            return undefined;
        if (!isIP(address))
            return 'an unparsable address';
        // A NAT64 / 6to4 / IPv4-compatible answer is delivered to the IPv4
        // address it carries, so every deny is applied to that address too. The
        // carried form only ADDS denials: it never earns an allow-list carve-out
        // (that would let `::1`, whose IPv4-compatible form is `0.0.0.1`, ride a
        // carve-out for `0.0.0.1`) and does not stand in for the address in the
        // localhost rule. An IPv4-mapped answer needs no decode here — BlockList
        // matches it against the IPv4 rules in either spelling.
        const v4 = embeddedIPv4(address);
        const denies = (set) => inRuleSet(set, address, port) ||
            (v4 !== undefined && inRuleSet(set, v4, port));
        if (denies(refused))
            return 'a deny-listed address';
        if (inRuleSet(allowed, address, port))
            return undefined;
        if (isLoopbackName(hostname)) {
            return isLoopbackAddress(address) ? undefined : 'a non-loopback address';
        }
        const why = denied.find(([, set]) => denies(set))?.[0];
        if (why)
            return why;
        if (addressInSet(local, address) ||
            (v4 !== undefined && addressInSet(local, v4)))
            return "one of this host's addresses";
        return undefined;
    };
    const lookupFor = (port) => (hostname, options, callback) => {
        resolve(hostname, { ...options, all: true }, (err, addresses) => {
            if (err) {
                callback(err, []);
                return;
            }
            const local = localSet();
            const kept = [];
            const dropped = [];
            const whys = new Set();
            for (const a of addresses) {
                const why = denial(hostname, a.address, port, local);
                if (why === undefined)
                    kept.push(a);
                else {
                    dropped.push(a.address);
                    whys.add(why);
                }
            }
            // Order the classes by the built-in priority, not by the order the
            // resolver happened to return the addresses, so the reason (which
            // `ignoreViolations` matches by substring) is stable.
            const classes = REASON_ORDER.filter(w => whys.has(w));
            if (dropped.length) {
                logForDebugging(`Denied address(es) for ${hostname}:${port}: ${dropped.join(', ')} (${classes.join('; ')})`);
            }
            const first = kept[0];
            if (!first) {
                // An empty answer without an error is possible from some resolvers.
                const none = dropped.length
                    ? new ResolvedAddressDeniedError(hostname, `resolved to ${classes.join(' / ')}`, dropped)
                    : Object.assign(new Error(`getaddrinfo ENOTFOUND ${hostname}`), {
                        code: 'ENOTFOUND',
                    });
                callback(none, []);
                return;
            }
            if (options.all)
                callback(null, kept);
            else
                callback(null, first.address, first.family);
        });
    };
    return {
        permits: (hostname, address, port) => denial(hostname, address, port, localSet()) === undefined,
        lookupFor,
    };
}
//# sourceMappingURL=resolved-address-guard.js.map