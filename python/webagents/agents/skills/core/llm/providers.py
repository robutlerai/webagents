"""
The LLM provider registry.

One table describing every provider this SDK can talk to: the skill names that
select it, the environment variables that carry its credential, and whether it
needs one at all.

WHY THIS EXISTS (2026-09-23). Nothing in the CLI checked for a credential before
starting a session. With no key set, `webagents connect` reached
`GoogleAISkill._get_client()`, which called `genai.Client()` with an empty key
and raised `ValueError: Missing key inputs argument!` deep inside daemon skill
initialisation. That was caught by a blanket handler, logged, and surfaced to the
user as a bare red `Error:` line in the TUI with no hint that an API key was the
problem. This table is what lets the CLI say the useful thing instead.

It is the counterpart of `typescript/src/skills/llm/providers.ts`. Keep the two
in step.

KEEPING IT HONEST: `env_vars` below were read out of each skill, not assumed. If
a skill changes how it resolves its key, this table is wrong and the preflight
lies, so change both together.

THE GOOGLE KEY (2026-09-25). This SDK's Google skill read `GOOGLE_GEMINI_API_KEY`
or `GEMINI_API_KEY` and the TypeScript one `GOOGLE_API_KEY`, so a developer who
exported the variable one SDK documented and ran the other got an authentication
failure. Both now read `GOOGLE_API_KEY`, then `GOOGLE_GEMINI_API_KEY`, then
`GEMINI_API_KEY`: the portal's own order (`lib/llm/provider-keys.ts`), and
Google's own client reads the first and the last.

`webagents models` names each provider's FIRST variable, as the TypeScript CLI
does; every variable listed is read.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class LLMProvider:
    """One provider, and what it needs to work."""

    # Canonical id, also the primary skill name.
    id: str
    # Every skill name that selects this provider, including the canonical id.
    aliases: Tuple[str, ...]
    # One line, for `webagents models`.
    description: str
    # "api_key", "local" (no credential) or "platform" (billed through Robutler).
    credential: str
    # Environment variables that carry the credential, in the order the skill
    # checks them. Empty for `local` providers.
    env_vars: Tuple[str, ...] = ()
    # The shape of a model string, e.g. "openai/<model>".
    model_format: str = ""
    # A model id known to exist. NOT a recommendation and NOT kept current
    # automatically; taken from the provider's `uamp_adapter.py` catalog, which
    # is the maintained model list in this repo. Check both when updating.
    default_model: Optional[str] = None


LLM_PROVIDERS: Tuple[LLMProvider, ...] = (
    LLMProvider(
        id="openai",
        aliases=("openai",),
        description="OpenAI hosted models",
        credential="api_key",
        env_vars=("OPENAI_API_KEY",),
        model_format="openai/<model>",
        default_model="gpt-4o-mini",
    ),
    LLMProvider(
        id="anthropic",
        aliases=("anthropic", "claude"),
        description="Anthropic hosted models",
        credential="api_key",
        env_vars=("ANTHROPIC_API_KEY",),
        model_format="anthropic/<model>",
        default_model="claude-haiku-4-5-20251001",
    ),
    LLMProvider(
        id="google",
        # `llm`: this SDK's old name for its Google skill; with the alias it
        # gets the agent's model, as `google` does.
        aliases=("google", "gemini", "llm"),
        description="Google hosted models",
        credential="api_key",
        # Order matters: this is what `_load_api_key` checks (module docstring).
        env_vars=("GOOGLE_API_KEY", "GOOGLE_GEMINI_API_KEY", "GEMINI_API_KEY"),
        model_format="google/<model>",
        default_model="gemini-2.5-flash",
    ),
    LLMProvider(
        id="xai",
        aliases=("xai", "grok"),
        description="xAI hosted models",
        credential="api_key",
        env_vars=("XAI_API_KEY",),
        model_format="xai/<model>",
        default_model="grok-3",
    ),
    LLMProvider(
        id="fireworks",
        aliases=("fireworks",),
        description="Fireworks hosted open models",
        credential="api_key",
        env_vars=("FIREWORKS_API_KEY",),
        model_format="fireworks/<model>",
        default_model="deepseek-v3p2",
    ),
    # Robutler's models, reached with a sign-in (`webagents login`) or an
    # explicit socket. Listed as the TypeScript registry lists it; not a key
    # provider, so nothing that offers or checks a key includes it.
    LLMProvider(
        id="proxy",
        aliases=("proxy",),
        description="Platform-hosted inference, billed to your Robutler account",
        credential="platform",
        env_vars=("ROBUTLER_LLM_PROXY_URL",),
        model_format="proxy/<model>",
        default_model="gpt-4o-mini",
    ),
)

# Variables a user plausibly exports that this SDK does NOT read, mapped to the
# ones it does. Purely for diagnostics: naming the near-miss turns a silent
# auth failure into a one-line fix.
NEAR_MISS_ENV_VARS: Dict[str, str] = {
    "ANTHROPIC_KEY": "ANTHROPIC_API_KEY",
    "OPENAI_KEY": "OPENAI_API_KEY",
}


def find_provider(skill_name: str) -> Optional[LLMProvider]:
    """The provider a skill name selects, or None."""
    lower = (skill_name or "").lower()
    for provider in LLM_PROVIDERS:
        if lower in provider.aliases:
            return provider
    return None


def configured_providers(env: Optional[Dict[str, str]] = None) -> List[LLMProvider]:
    """Providers whose credential is present in this environment."""
    env = os.environ if env is None else env
    return [p for p in LLM_PROVIDERS if any(env.get(v) for v in p.env_vars)]


def near_miss_env_vars(env: Optional[Dict[str, str]] = None) -> List[Tuple[str, str]]:
    """`(set_but_unused, what_to_set_instead)` pairs present in this environment.

    Only reports a near miss when the variable the SDK actually reads is ABSENT,
    because otherwise there is nothing wrong to report.
    """
    env = os.environ if env is None else env
    out = []
    for wrong, right in NEAR_MISS_ENV_VARS.items():
        if env.get(wrong) and not env.get(right):
            out.append((wrong, right))
    return out


def provider_for_model(model: Optional[str]) -> Optional[LLMProvider]:
    """The provider a `provider/model` string runs on, or None when unknown.

    So key advice can name the ONE variable a given agent needs (2026-09-24).
    `doctor` and the `connect` preflight both told a first-time developer to
    set `GOOGLE_GEMINI_API_KEY`, because an agent with no model defaults to
    Google; but `webagents init` writes `model: openai/gpt-4o-mini`, so the
    agent they had just created needed `OPENAI_API_KEY`, and following the
    advice changed nothing.
    """
    if not model or "/" not in model:
        return None
    return find_provider(model.split("/", 1)[0])


def missing_key_for_model(
    model: Optional[str], env: Optional[Dict[str, str]] = None
) -> Optional[LLMProvider]:
    """The provider whose key `model` needs and this environment lacks, or None."""
    provider = provider_for_model(model)
    if provider is None or provider.credential != "api_key" or not provider.env_vars:
        return None
    env = os.environ if env is None else env
    return None if any(env.get(v) for v in provider.env_vars) else provider
