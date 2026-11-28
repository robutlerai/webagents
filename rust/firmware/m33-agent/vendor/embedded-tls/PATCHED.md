# embedded-tls 0.19.0, patched

Vendored from crates.io `embedded-tls` 0.19.0 (Apache-2.0, see LICENSE) for the m33-agent firmware only.

Two changes, both marked `PATCH`. In `src/der_certificate.rs`, `AttributeTypeAndValue` derives `PartialEq` so names can be compared. In `src/pki.rs`: before walking the certificate chain, the chain is trimmed after the first certificate issued by the trust anchor. Without it, a server that sends extra certificates past the anchor (for example one that sends ISRG Root X2 cross-signed by Root X1 with RSA after Root YE) fails with `InvalidSignatureScheme`, because the walk starts from the last entry. Every certificate kept is still verified; extra ones are ignored, as path-building verifiers do.

To be offered upstream; remove this directory when a release includes the fix.
