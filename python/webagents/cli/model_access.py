"""
Which model a CLI agent runs on, and how it gets there (2026-09-24).

THE PROBLEM. Every agent the daemon built without an explicit LLM skill got a
Google model: `GoogleAISkill` auto-attached and `model` defaulted to
`google/gemini-2.5-flash`, whatever keys the developer had. With no Google key,
or without `google-genai` installed, the chat answered "Internal Server Error"
(the ImportError escaped the daemon). The TypeScript chat had the mirror image,
OpenAI only. And a developer signed in to Robutler, which serves models, was
told to go and find a provider key.

ONE DECISION, shared by the daemon (which builds the agent) and the chats
(which tell the person what is happening before anything is sent):

  1. The agent names a model (`model: openai/gpt-4o`): its provider's key and
     client library, when both are here. Otherwise, when signed in, the SAME
     model through Robutler's LLM proxy, paid from the person's credits.
  2. The agent names none: the first provider that has a key and a client
     here, at its default model; otherwise, when signed in, Robutler's
     default (`auto/balanced`, which the platform maps to its current model).
  3. Neither: `none`, with the reason, and the chat offers to sign in or to
     take a key there and then (`cli/commands/agent.py`).

An agent that lists a provider's skill itself (`skills: [openai]`) keeps it
while it can run. Without that provider's key it gets the same treatment as
point 1, for that provider's model: listing the skill says which provider,
not "fail unless its key is exported" (`named_provider_blocked`).

The proxy gets the login token through `platform_token` (read per request, so
a fresh `webagents login` is picked up by a running daemon) and never through
the environment: the shell skill hands its environment to every command the
agent runs.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from ..agents.skills.core.llm.providers import LLM_PROVIDERS, LLMProvider, find_provider, provider_for_model
from .config_store import cli_command

#: What each provider's skill imports; without it the skill raises on construction.
_CLIENT_MODULES: Dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google.genai",
    "xai": "openai",
    "fireworks": "openai",
}

_logger = logging.getLogger("webagents.cli.model_access")

#: The platform's own choice of model, resolved server-side.
PROXY_DEFAULT_MODEL = "auto/balanced"

#: Skill names that ARE an LLM: an agent listing one in `skills:` chose its model itself.
LLM_SKILL_NAMES = frozenset({"llm", "google", "openai", "anthropic", "xai", "fireworks", "primary_llm", "proxy"})


def _client_installed(provider: LLMProvider) -> bool:
    module = _CLIENT_MODULES.get(provider.id)
    if not module:
        return True
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def _has_key(provider: LLMProvider, env: Optional[Dict[str, str]] = None) -> bool:
    env = os.environ if env is None else env
    return any(env.get(name) for name in provider.env_vars)


def usable_directly(provider: LLMProvider, env: Optional[Dict[str, str]] = None) -> bool:
    """A key AND the client library: what calling the provider directly needs."""
    return provider.credential != "api_key" or (_has_key(provider, env) and _client_installed(provider))


@dataclass(frozen=True)
class ModelAccess:
    #: "direct" (the provider, with the developer's key), "proxy" (Robutler) or "none".
    kind: str
    #: The model string, e.g. "openai/gpt-4o" or "auto/balanced"; None when kind is "none".
    model: Optional[str]
    #: The provider the model belongs to, when known.
    provider: Optional[LLMProvider] = None
    #: Why it is not "direct", in one sentence, when it is not.
    reason: str = ""

    def describe(self) -> str:
        """For the welcome card: the model, and how it is reached when that is not obvious."""
        if self.kind == "proxy":
            return f"{self.model} via Robutler"
        return self.model or ""


class ModelUnavailable(RuntimeError):
    """No model this agent can use here: the reason, and both ways out."""

    def __init__(self, access: ModelAccess) -> None:
        self.access = access
        # The TypeScript chat's words (`typescript/src/cli/model-access.ts`
        # `unavailableMessage`), with this SDK's key names.
        provider = access.provider
        sign_in = f"Sign in with `{cli_command('login')}` to use Robutler's models"
        if (access.reason or "").endswith("is served by Robutler"):
            message = f"No model for this agent: {access.reason}. {sign_in}."
        elif provider is not None and provider.env_vars:
            message = (
                f"No model for this agent: {access.reason}. {sign_in}, "
                f"or add a key with `{cli_command(f'secrets set {provider.env_vars[0]}')}`."
            )
        else:
            names = ", ".join(p.env_vars[0] for p in LLM_PROVIDERS if p.credential == "api_key" and p.env_vars)
            message = (
                f"No model for this agent: {access.reason}. {sign_in}, "
                f"or add a key with `{cli_command('secrets set <NAME>')}` ({names})."
            )
        super().__init__(message)


def _why_not_direct(provider: LLMProvider, env: Optional[Dict[str, str]]) -> str:
    if not _has_key(provider, env):
        return f"{provider.env_vars[0]} is not set" if provider.env_vars else f"{provider.id} needs a key"
    return (
        f"the {provider.id} client library is not installed "
        f"(pip install 'webagents[llm]')"
    )


def resolve_model_access(
    declared_model: Optional[str],
    *,
    signed_in: Callable[[], bool],
    env: Optional[Dict[str, str]] = None,
) -> ModelAccess:
    """The decision described in the module docstring. `signed_in` is asked only when needed."""
    if declared_model:
        prefix, _, rest = declared_model.partition("/")
        if prefix in ("proxy", "robutler", "auto"):
            # Models only Robutler serves: `auto/*` is the platform's own
            # choice; `proxy/` and `robutler/` name "through Robutler" and are
            # not something the platform strips, so they go without it.
            model = declared_model if prefix == "auto" else (rest or PROXY_DEFAULT_MODEL)
            if signed_in():
                return ModelAccess("proxy", model, None)
            return ModelAccess("none", None, None, f"{declared_model} is served by Robutler")
        provider = provider_for_model(declared_model)
        if provider is None or usable_directly(provider, env):
            # A provider this table does not know is the skill's own business.
            return ModelAccess("direct", declared_model, provider)
        reason = _why_not_direct(provider, env)
        if signed_in():
            return ModelAccess("proxy", declared_model, provider, reason)
        return ModelAccess("none", None, provider, reason)

    for provider in LLM_PROVIDERS:
        if provider.credential == "api_key" and provider.default_model and usable_directly(provider, env):
            return ModelAccess("direct", f"{provider.id}/{provider.default_model}", provider)
    if signed_in():
        return ModelAccess("proxy", PROXY_DEFAULT_MODEL, None, "no provider key is set")
    return ModelAccess("none", None, None, "no provider key is set")


def model_for_named_provider(provider: LLMProvider, declared_model: Optional[str]) -> Optional[str]:
    """The model an agent that lists `provider`'s skill runs on.

    Its own `model:` when that is this provider's (a bare id counts), else the
    provider's default: what runs in place of the skill is the same model,
    not whatever the platform would pick.
    """
    if declared_model:
        if "/" not in declared_model:
            return f"{provider.id}/{declared_model}"
        owner = provider_for_model(declared_model)
        if owner is not None and owner.id == provider.id:
            return declared_model
    return f"{provider.id}/{provider.default_model}" if provider.default_model else None


def named_provider_blocked(skill_name: str, env: Optional[Dict[str, str]] = None) -> Optional[LLMProvider]:
    """The provider a listed LLM skill names, when that provider cannot run here; else None.

    None for a skill this table cannot place (`llm`, `primary_llm`, `proxy`)
    or a provider that needs no key: those keep their own behaviour.
    """
    provider = find_provider(skill_name)
    if provider is None or provider.credential != "api_key" or usable_directly(provider, env):
        return None
    return provider


def choose_model_access(
    declared_model: Optional[str], skills: Dict[str, object], agent_name: str
) -> Optional[ModelAccess]:
    """How an agent reaches its model, adding the proxy skill when that is the way.

    One function for every place that builds a CLI agent: the daemon, the chat
    (`cli/agent_builder.py`) and `webagents run -p`. `skills` is the agent's
    instantiated skills and is changed in place.

    Returns None when the agent lists its own LLM skill and that skill can run:
    its choice stands, with its own model. Otherwise the decision in this
    module's docstring: `direct` (build with `access.model`), `proxy` (the
    proxy skill was added), or `ModelUnavailable` when nothing works.

    An agent whose listed provider skill found no key (in its config or the
    environment) could only fail, so that skill gives way to the decision for
    the same model: a key stored since, or Robutler when signed in. Listing
    `openai` says which provider, not "fail unless OPENAI_API_KEY is exported".
    """
    named = next((name for name in skills if name in LLM_SKILL_NAMES), None)
    if named == "proxy":
        # The file names Robutler's models: that is the way, as in the
        # TypeScript chat, and the card says so ("... via Robutler").
        skill = skills[named]
        return ModelAccess("proxy", getattr(skill, "model", None) or declared_model or PROXY_DEFAULT_MODEL, find_provider("proxy"))
    if named is not None:
        provider = find_provider(named)
        skill = skills[named]
        if (
            provider is None
            or provider.credential != "api_key"
            or not hasattr(skill, "api_key")
            or getattr(skill, "api_key", None)
        ):
            return None  # the agent picked its LLM skill, and it can run; its choice stands
        del skills[named]
        declared_model = model_for_named_provider(provider, declared_model)
        _logger.info(f"{agent_name}: the {named} skill has no key; choosing how {declared_model} runs")

    from .commands.secrets import load_into_environment

    # Keys stored with `webagents secrets set` after this process started.
    load_into_environment()
    access = resolve_model_access(declared_model, signed_in=is_signed_in)
    if access.kind == "direct":
        _logger.info(f"{agent_name}: {access.model} with this machine's key")
        return access
    if access.kind == "proxy":
        skills["llm"] = proxy_skill_for(access.model)
        _logger.info(f"{agent_name}: {access.model} through Robutler ({access.reason})")
        return access
    raise ModelUnavailable(access)


def choose_model(declared_model: Optional[str], skills: Dict[str, object], agent_name: str) -> Optional[str]:
    """The `model` to build an agent with (see `choose_model_access`): the
    model for `direct`, and None when a skill carries it (the file's own LLM
    skill, or the proxy). Raises `ModelUnavailable` when nothing works."""
    access = choose_model_access(declared_model, skills, agent_name)
    if access is None:
        # The file's own LLM skill stands, and carries the model already
        # (`agent_builder.load_skills`); another would be a second LLM skill.
        return None
    return access.model if access.kind == "direct" else None


def is_signed_in() -> bool:
    """Whether a Robutler login token is stored (or passed) for this profile."""
    try:
        from .credentials import get_token

        return bool(get_token())
    except Exception:
        return False


def platform_llm_url() -> str:
    """The `/llm` socket of the platform this profile signs in to."""
    explicit = os.environ.get("ROBUTLER_LLM_PROXY_URL")
    if explicit:
        return explicit
    from .config_store import platform_url

    base = platform_url().rstrip("/")
    if base.startswith("https://"):
        return "wss://" + base[len("https://"):] + "/llm"
    if base.startswith("http://"):
        return "ws://" + base[len("http://"):] + "/llm"
    return base + "/llm"


def proxy_skill_for(model: str):
    """An `LLMProxySkill` for `model`, paid by the signed-in person."""
    from ..agents.skills.core.llm.proxy.skill import LLMProxySkill
    from .credentials import get_token

    return LLMProxySkill({"model": model, "proxy_url": platform_llm_url(), "platform_token": get_token})


def provider_names() -> List[str]:
    """Every provider that takes a key, for the chat's prompt."""
    return [p.id for p in LLM_PROVIDERS if p.credential == "api_key"]


def welcome_model(declared_model: Optional[str], skills: Optional[List[object]] = None) -> "tuple[str, Optional[str]]":
    """What a chat's welcome card shows as the model, and its warning when there is no way to run one.

    Both chats call this, so the card says what the daemon will actually do:
    `openai/gpt-4o via Robutler` rather than "OPENAI_API_KEY is not set" for
    a person who is signed in, and one sentence naming both ways out when
    nothing works. An agent that lists a provider's skill it cannot run here
    gets the same decision for that provider's model; a skill this table
    cannot place (`llm`, `primary_llm`) keeps the old key check.
    """
    names = [s if isinstance(s, str) else next(iter(s), None) if isinstance(s, dict) else None for s in (skills or [])]
    named = next((name for name in names if name in LLM_SKILL_NAMES), None)
    blocked = named_provider_blocked(named) if named else None
    if blocked is not None:
        # Its provider cannot run here: the decision below, for that model.
        declared_model = model_for_named_provider(blocked, declared_model)
    elif named is not None:
        missing = None
        if declared_model:
            from ..agents.skills.core.llm.providers import missing_key_for_model

            missing = missing_key_for_model(declared_model)
        if missing is not None and missing.env_vars:
            return declared_model or "", (
                f"{missing.env_vars[0]} is not set, and this agent's model needs it. "
                f"Run `{cli_command(f'secrets set {missing.env_vars[0]}')}`."
            )
        return declared_model or "", None
    access = resolve_model_access(declared_model, signed_in=is_signed_in)
    if access.kind == "none":
        return declared_model or "", str(ModelUnavailable(access))
    return access.describe(), None
