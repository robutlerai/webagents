"""
`timeout(seconds)`: `asyncio.timeout` on Python 3.11 and later, the
`async-timeout` backport it was made from on 3.10 (2026-09-25).

The package supports Python 3.10 (`requires-python`), and the NLI skill's UAMP
path used `asyncio.timeout` directly: on 3.10 that is an AttributeError, which
the skill's error handling reported as "UAMP transport failed", so every NLI
call over UAMP failed there. CI's 3.10 job was the only place that showed it.
"""

import sys

if sys.version_info >= (3, 11):
    from asyncio import timeout
else:  # pragma: no cover - exercised by the Python 3.10 CI job
    from async_timeout import timeout

__all__ = ["timeout"]
