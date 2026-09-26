"""
The platform's base URL, for every Python skill that calls the platform.

ONE ANSWER, IN THE TYPESCRIPT ORDER (2026-09-25): `robutler_api_url` (or
`webagents_api_url`) in the skill's config, else `ROBUTLER_API_URL`, else
`ROBUTLER_INTERNAL_API_URL`, else the CLI's `platform.url` (the portal
`webagents login` signed in to), else https://robutler.ai. The discovery skill
resolved it this way first (its module docstring says why the order is the
TypeScript one); the lookup moved here when the platform API client moved into
the SDK (`api/client.py`), so the client and the other platform skills stop
choosing their own.

Choosing their own is what S-252 was: the namespace, publish, CRM and message
history skills each fell back to https://webagents.ai, the project's old site,
which does not serve the platform, and sent the agent's platform key there on
every call when nothing named a URL.

The payment, auth and KV skills still build their own chain, with the
in-cluster variable first and http://localhost:3000 last, and pass the result
to the client explicitly; the TypeScript payment and auth skills do the same.
"""

import os
from typing import Any, Mapping, Optional

#: The platform when nothing names another.
DEFAULT_PLATFORM_URL = "https://robutler.ai"


def trimmed_url(value: Any) -> Optional[str]:
    """`value` as a base URL: stripped, no trailing slash, `None` when blank."""
    url = str(value or "").strip().rstrip("/")
    return url or None


def resolve_platform_url(config: Optional[Mapping[str, Any]] = None) -> str:
    """The platform's base URL (module docstring), the TypeScript order."""
    config = config or {}
    for value in (
        config.get("robutler_api_url"),
        config.get("webagents_api_url"),
        os.getenv("ROBUTLER_API_URL"),
        os.getenv("ROBUTLER_INTERNAL_API_URL"),
    ):
        url = trimmed_url(value)
        if url:
            return url
    try:
        from webagents.cli.config_store import platform_url

        return trimmed_url(platform_url()) or DEFAULT_PLATFORM_URL
    except Exception:
        # No CLI configuration to read: the default stands.
        return DEFAULT_PLATFORM_URL
