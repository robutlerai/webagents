"""
Shared model catalog and auto-model mappings for BYOK resolution.
Must stay in sync with robutler/lib/models/catalog.ts.
"""

AUTO_MODEL_MAP = {
    'auto/fastest': {
        'openai': 'gpt-4o-mini',
        'anthropic': 'claude-3.5-haiku',
        'google': 'gemini-3.1-flash-lite',
        'xai': 'grok-3-mini',
        'fireworks': 'qwen3-8b',
    },
    'auto/smartest': {
        'openai': 'gpt-5.4',
        'anthropic': 'claude-opus-4-6',
        'google': 'gemini-3.1-pro',
        'xai': 'grok-4.20-beta',
        'fireworks': 'deepseek-r1',
    },
    'auto/balanced': {
        'openai': 'o4-mini',
        'anthropic': 'claude-sonnet-4-6',
        'google': 'gemini-2.5-flash',
        'xai': 'grok-4-0709',
        'fireworks': 'deepseek-v3p2',
    },
}

AUTO_PROVIDER_PRIORITY = {
    'auto/fastest': ['google', 'openai', 'anthropic', 'xai', 'fireworks'],
    'auto/smartest': ['google', 'anthropic', 'openai', 'xai', 'fireworks'],
    'auto/balanced': ['openai', 'google', 'anthropic', 'xai', 'fireworks'],
}

DEFAULT_PLATFORM_MODEL = 'google/gemini-3.1-flash-lite'


def resolve_auto_model(auto_tier: str, available_providers: list[str]) -> str | None:
    """Resolve an auto/* model to a concrete model ID given available BYOK providers."""
    mapping = AUTO_MODEL_MAP.get(auto_tier)
    if not mapping:
        return None

    priority = AUTO_PROVIDER_PRIORITY.get(auto_tier, list(mapping.keys()))
    for provider in priority:
        if provider in available_providers and provider in mapping:
            return mapping[provider]
    return None


def get_provider_from_model(model: str) -> str | None:
    """Extract the provider name from a model ID (e.g. 'openai/gpt-4o' -> 'openai')."""
    if '/' in model:
        return model.split('/')[0]
    model_lower = model.lower()
    if model_lower.startswith('gpt') or model_lower.startswith('text-embedding') or model_lower.startswith('o1') or model_lower.startswith('o3') or model_lower.startswith('o4'):
        return 'openai'
    elif model_lower.startswith('claude'):
        return 'anthropic'
    elif model_lower.startswith('gemini') or model_lower.startswith('vertex'):
        return 'google'
    elif model_lower.startswith('grok'):
        return 'xai'
    elif any(model_lower.startswith(p) for p in ['llama', 'deepseek', 'qwen', 'glm', 'kimi', 'minimax', 'cogito', 'gpt-oss']):
        return 'fireworks'
    return None
