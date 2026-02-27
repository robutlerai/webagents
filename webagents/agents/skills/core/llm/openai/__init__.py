# Native OpenAI skill - uses official openai SDK
try:
    from .skill import OpenAISkill
except ImportError:
    OpenAISkill = None

__all__ = ["OpenAISkill"]