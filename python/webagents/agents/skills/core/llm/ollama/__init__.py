# Local models through Ollama's OpenAI-compatible endpoint (plan item 2.8).
try:
    from .skill import OllamaSkill
except ImportError:
    OllamaSkill = None

from .probe import ollama_model_check, probe_ollama, serves_model

__all__ = ["OllamaSkill", "ollama_model_check", "probe_ollama", "serves_model"]
