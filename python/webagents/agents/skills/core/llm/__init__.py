"""
LLM Skills Package - WebAgents V2.0

Core LLM integration skills for various providers:
- OpenAI: Native OpenAI integration
- Anthropic: Native Anthropic Claude integration
- Google: Native Google Gemini integration
- XAI: Native X.AI Grok integration
- Fireworks: Native Fireworks AI integration (OSS models)
"""

# Import skills with graceful fallback
try:
    from .google import GoogleAISkill
except Exception:  # noqa: BLE001 - class-creation AttributeError/NameError, not just ImportError (F-040)
    GoogleAISkill = None

try:
    from .openai import OpenAISkill
except Exception:  # noqa: BLE001 - class-creation AttributeError/NameError, not just ImportError (F-040)
    OpenAISkill = None

try:
    from .anthropic import AnthropicSkill
except Exception:  # noqa: BLE001 - class-creation AttributeError/NameError, not just ImportError (F-040)
    AnthropicSkill = None

try:
    from .xai import XAISkill
except Exception:  # noqa: BLE001 - class-creation AttributeError/NameError, not just ImportError (F-040)
    XAISkill = None

try:
    from .fireworks import FireworksAISkill
except Exception:  # noqa: BLE001 - class-creation AttributeError/NameError, not just ImportError (F-040)
    FireworksAISkill = None

try:
    from .proxy import LLMProxySkill
except Exception:  # noqa: BLE001 - class-creation AttributeError/NameError, not just ImportError (F-040)
    LLMProxySkill = None

__all__ = [
    "GoogleAISkill",
    "OpenAISkill",
    "AnthropicSkill",
    "XAISkill",
    "FireworksAISkill",
    "LLMProxySkill",
]






