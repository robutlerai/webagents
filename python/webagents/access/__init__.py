"""Who is calling an agent, and what they may use (ADR-0045)."""

from .caller import LOCAL_OWNER, CallerAuth, run_as_local_owner

__all__ = ["CallerAuth", "LOCAL_OWNER", "run_as_local_owner"]
