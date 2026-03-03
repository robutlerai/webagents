try:
    from .skill import LSPSkill
except ImportError:
    LSPSkill = None  # type: ignore[assignment,misc]

__all__ = ["LSPSkill"]
