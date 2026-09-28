"""
The x402 wire, re-exported under the package's old name (2026-09-26).

The private scheme this module used to encode (`scheme: 'token'` on
`network: 'robutler'`, a raw JWT or base64 JSON in `X-PAYMENT`, a `version`
field no x402 client reads) is retired with the paywall: a priced `@http`
endpoint now answers standard x402 v2 and v1 (`..payments.x402_wire`), with
the credits scheme as a well-formed entry beside any chain scheme. The names
below are the wire module's, so `from ...payments_x402.schemes import ...`
keeps resolving for callers that spelled it this way.
"""

from ..payments.x402_wire import *  # noqa: F401,F403
from ..payments.x402_wire import __all__  # noqa: F401
