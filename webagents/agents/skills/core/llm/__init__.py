"""
LLM Skills Package - WebAgents V2.0

Core LLM integration skills for various providers:
- LiteLLM: Cross-provider routing (OpenAI, Anthropic, Google, XAI, etc.)
- Google: Native Google Gemini integration
- OpenAI: Native OpenAI integration
- Anthropic: Native Anthropic Claude integration
- XAI: Native X.AI Grok integration
"""

# Import skills with graceful fallback
try:
    from .litellm import LiteLLMSkill
except ImportError:
    LiteLLMSkill = None

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

__all__ = [
    "LiteLLMSkill",
    "GoogleAISkill", 
    "OpenAISkill",
    "AnthropicSkill",
    "XAISkill",
]






