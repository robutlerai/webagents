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
except ImportError:
    GoogleAISkill = None

try:
    from .openai import OpenAISkill
except ImportError:
    OpenAISkill = None

try:
    from .anthropic import AnthropicSkill
except ImportError:
    AnthropicSkill = None

try:
    from .xai import XAISkill
except ImportError:
    XAISkill = None

try:
    from .fireworks import FireworksAISkill
except ImportError:
    FireworksAISkill = None

try:
    from .proxy import LLMProxySkill
except ImportError:
    LLMProxySkill = None

__all__ = [
    "GoogleAISkill",
    "OpenAISkill",
    "AnthropicSkill",
    "XAISkill",
    "FireworksAISkill",
    "LLMProxySkill",
]






