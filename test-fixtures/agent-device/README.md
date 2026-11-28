# Agent sign-in by device authorization: shared vectors

One set of vectors for every SDK in this repository. A signer is correct when it reproduces each case byte for byte; a verifier is correct when it accepts every case and refuses every tampered variant.

| File | What it checks |
| --- | --- |
| `sign-in-vectors.json` | The device's Ed25519 key, its RFC 7638 thumbprint and fingerprint, the user-code rules, and RFC 9421 request signatures (Web Bot Auth profile): for each case the exact signature base and the `Signature-Input`, `Signature` and `Content-Digest` headers produced with the named key, clock and nonce. |
| `tool-description-vectors.json` | The signed tool description (`webagents-tool-description/1`): the exact message and signature for each case, the tampered variants a verifier must refuse, and a case signed with the wrong key. Keys come from `sign-in-vectors.json`. |

The keys in these files are test keys, published so that anyone can reproduce the signatures. Never use them for a real device.

Checked by `rust/crates/webagents/tests/auth_vectors.rs`.
