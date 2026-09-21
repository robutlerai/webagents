"""
Crypto tests: the Ed25519 identity key, the claim token, the Web Bot Auth
request signer and the cross-language signing vectors (ADR 0038 step 5, W2
design sections 9.2, 10.2 and 10.3, 2026-09-17).

A package, like `tests/server/`, so pytest can import the shared verifier in
`support.py` without a module-name collision.
"""
