"""
TrustFlow in the SDK (webagents gap-closure plan items 2.5 and 2.7, 2026-09-26).

TrustFlow is a platform service: Robutler computes an agent's score from
interactions on the platform. This package asks the platform for it
(`trust_lookup`) and verifies the signed record the platform exports for an
agent (`trust_record`). Nothing here computes a score. The TypeScript twin is
`typescript/src/trustflow/`; the name is `trustflow` because `webagents.trust`
already names the dot-namespace trust rules.
"""

from .trust_lookup import (  # noqa: F401
    NO_TRUST_CREDENTIAL,
    TRUST_CACHE_TTL_MS,
    TRUST_LOOKUP_PATH,
    TRUST_RECORD_PATH,
    PlatformCredential,
    TrustLookup,
    TrustLookupError,
    platform_credential_for,
    trust_failure_message,
)
from .trust_record import (  # noqa: F401
    TRUSTFLOW_RECORD_AUDIENCE,
    TRUSTFLOW_RECORD_EXTENSION_DESCRIPTION,
    TRUSTFLOW_RECORD_EXTENSION_URI,
    TRUSTFLOW_RECORD_TYP,
    VerifyTrustRecordResult,
    decode_trust_record,
    trust_record_extension,
    trust_record_from_card,
    verify_trust_record,
    with_trust_record_extension,
)
