"""
Shared model catalog and auto-model mappings for BYOK resolution.
Must stay in sync with roborum/lib/models/catalog.ts.
"""

AUTO_MODEL_MAP = {
    'auto/fastest': {
        'openai': 'gpt-4o-mini',
        'anthropic': 'claude-3-5-haiku',
        'google': 'gemini-2.5-flash',
    },
    'auto/smartest': {
        'openai': 'gpt-4.1',
        'anthropic': 'claude-3-5-sonnet',
        'google': 'gemini-2.5-pro',
    },
    'auto/balanced': {
        'openai': 'gpt-4o',
        'anthropic': 'claude-3-5-sonnet',
        'google': 'gemini-2.5-flash',
    },
}

# Priority order for provider selection per tier
AUTO_PROVIDER_PRIORITY = {
    'auto/fastest': ['google', 'openai', 'anthropic'],
    'auto/smartest': ['anthropic', 'openai', 'google'],
    'auto/balanced': ['openai', 'google', 'anthropic'],
}

DEFAULT_PLATFORM_MODEL = 'gpt-4o-mini'


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
    # Heuristic for models without a provider prefix
    model_lower = model.lower()
    if model_lower.startswith('gpt') or model_lower.startswith('text-embedding'):
        return 'openai'
    elif model_lower.startswith('claude'):
        return 'anthropic'
    elif model_lower.startswith('gemini') or model_lower.startswith('vertex'):
        return 'google'
    elif model_lower.startswith('grok') or model_lower.startswith('xai'):
        return 'xai'
    return None
