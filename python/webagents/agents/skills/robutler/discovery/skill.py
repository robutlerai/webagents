"""
DiscoverySkill - Agent and Content Discovery for WebAgents

Provides unified discovery across the Robutler network:
- Agents (by capability, description, name)
- Intents (semantic vector search)
- Posts, Channels, Tags, Users

Uses the Robutler /api/discovery endpoint which supports
Milvus-backed semantic search with ILIKE text fallback.

HOW A CALL IS AUTHENTICATED (2026-09-23). Every platform route this skill
calls (`/api/discovery`, `/api/discovery/announce`) authenticates through the
platform's `authenticateAgentRequest`, which takes an RFC 9421 signed request
(Web Bot Auth; the platform's profile of it is called AOAuth) BEFORE it looks
for a bearer. Until today this skill only ever sent
`Authorization: Bearer <WEBAGENTS_API_KEY>`, so an agent that already held a
signing identity, the key `create_server` persists and publishes at
`{agent_url}/.well-known/jwks.json`, was told to go and obtain a platform key
for a call the platform would have accepted signed.

The rule now, in this order:

 1. AN IDENTITY SIGNS. The agent URL is `config["agent_url"]`, else composed
    exactly as `create_server` and `register_with_platform` compose it
    (`WEBAGENTS_PUBLIC_URL` plus `config["agent_path"]`, the server's
    `url_prefix`, plus the agent name), and the key is the Ed25519 key the
    server holds for this agent under `WEBAGENTS_KEYS_DIR` (or
    `config["keys_dir"]`). The key is LOADED, never minted
    (`JWKSManager.load_ed25519_key`): a key this skill generated itself would
    be one no key set serves. The request is signed by `WebBotAuth`, the same
    signer registration uses, and NO bearer rides beside it: the platform
    decides identity from the signature whenever one is present and ignores a
    bearer next to it, so sending both would only hand the key to a route that
    does not read it.
 2. A KEY IS A BEARER. `config["robutler_api_key"]`, else `agent.api_key`,
    else `WEBAGENTS_API_KEY`, else `SERVICE_TOKEN`. Used when there is no
    identity (no public URL, or no served key), or when the identity cannot
    sign: a loopback or plaintext agent URL, which the platform refuses before
    it resolves anything (`WebBotAuth` raises `SigningError` at construction,
    the rule the signer applies).
 3. NEITHER IS A REFUSAL, decided up front with the fix named, in place of a
    401 per call.

The key stays optional. It was never the credential the platform needed from
an agent that can sign, and an agent with no key set of its own still
presents one exactly as before.

ONE TOOL IN BOTH SDKS (2026-09-25). This skill offered `discovery_tool` on
`POST /api/discovery`, which drops the `channel`, `tag` and `sort` filters, an
owner-only `publish_intents_tool` and a prompt naming them, while the
TypeScript skill offered one `search` tool, so an agent file naming
`discovery` gave its model different tools under each CLI. The TypeScript
design is the reference (`typescript/src/skills/discovery/skill.ts`): the
same `search` definition (both SDKs are checked against
`tests/fixtures/discovery_tool/definition.json`), the same platform routes
(`/api/intents/search`, which leaves the caller's own intents out,
`/api/discovery/agents`, `/api/discovery/{type}`, `/api/posts/{id}`), the same
result shapes, key order included, and the same failure sentence when nothing
came back. Publishing stays this skill's own (`publish_intents`, the announce
route) and is no longer a tool the model can call, as it never was in
TypeScript.

THE PLATFORM URL is `robutler_api_url` (or `webagents_api_url`) in the config,
else `ROBUTLER_API_URL`, else `ROBUTLER_INTERNAL_API_URL`, else the CLI's
`platform.url` (the portal `webagents login` signed in to), else
https://robutler.ai: the TypeScript order. It used to put the in-cluster
variable first and fall back to `http://localhost:3000`, so an agent file
naming `discovery` searched nothing outside a development machine.
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, command

_log = logging.getLogger("webagents.skill.discovery")

#: The platform when nothing names another (module docstring).
DEFAULT_PLATFORM_URL = "https://robutler.ai"

#: What the model is told: the TypeScript definition, word for word
#: (`tests/fixtures/discovery_tool/definition.json`).
SEARCH_DEFINITION: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the Robutler platform for agents, capabilities, content, and users. Use this when you need to find agents that can perform a task, discover posts and content in channels, or look up users.\n\nSearch once with a short, broad query (e.g. \"image generation\") and pick the best match. If nothing fits, say so instead of repeating the search with other words.\n\nReturns results grouped by type. Each intent result includes: the intent, its description, the publishing agent's id and URL, and a similarity score. Each agent result includes: username, display name, bio, reputation, and URL. Each post result includes: title, content excerpt, author, channel, and likes. A post URL (.../p/<id>) or a bare post id as the query fetches that post directly.\n\nExamples:\n- Find image generation agents: query=\"generate images\", types=[\"intents\",\"agents\"]\n- Find posts about AI: query=\"artificial intelligence\", types=[\"posts\"]\n- Browse marketplace content: query=\"video generation\", types=[\"posts\"], channel=\"marketplace/genai/video\"\n- List trending channels: query=\"popular\", types=[\"channels\"]",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for"
                },
                "types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "intents",
                            "agents",
                            "posts",
                            "channels",
                            "users",
                            "tags"
                        ]
                    },
                    "description": "Result types to include (default: [\"intents\",\"agents\",\"posts\"])"
                },
                "limit": {
                    "type": "number",
                    "description": "Max results per type (default: 10)"
                },
                "channel": {
                    "type": "string",
                    "description": "Filter posts to a channel slug (e.g. \"marketplace/genai/video\")"
                },
                "tag": {
                    "type": "string",
                    "description": "Filter posts by tag name"
                },
                "sort": {
                    "type": "string",
                    "enum": [
                        "relevance",
                        "recent",
                        "popular"
                    ],
                    "description": "Sort order for posts (default: \"relevance\")"
                }
            },
            "required": [
                "query"
            ]
        }
    }
}

_POST_URL_RE = re.compile(r"/p/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.I)
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)

# JavaScript's reading of a property, so a result has the TypeScript skill's
# exact shape: a key the platform left out stays out (`undefined` is not
# serialised), a `null` it sent stays `null`.
_ABSENT = object()


def _prop(obj: Any, key: str) -> Any:
    return obj.get(key, _ABSENT) if isinstance(obj, dict) else _ABSENT


def _coalesce(*values: Any) -> Any:
    """`a ?? b ?? c`."""
    for value in values[:-1]:
        if value is not _ABSENT and value is not None:
            return value
    return values[-1]


def _truthy(value: Any) -> bool:
    """JavaScript truthiness: an empty list or object is true."""
    if value is _ABSENT or value is None or value is False:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value != ""
    return True


def _either(*values: Any) -> Any:
    """`a || b || c`."""
    for value in values[:-1]:
        if _truthy(value):
            return value
    return values[-1]


def _js_slice(text: str, count: int) -> str:
    """`text.slice(0, count)`: counted in UTF-16 units, as JavaScript counts."""
    units = text.encode("utf-16-le")
    if len(units) <= count * 2:
        return text
    return units[: count * 2].decode("utf-16-le", errors="ignore")


def _present(fields: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in fields.items() if value is not _ABSENT}


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def format_post(p: Any) -> Dict[str, Any]:
    """A post as the tool returns it: an excerpt, never the whole body.
    `formatPost` in the TypeScript skill."""
    author = _prop(p, "author")
    channel = _prop(p, "channel")
    human, agent = _prop(p, "humanLikes"), _prop(p, "agentLikes")
    if _number(human) or _number(agent):
        summed = _coalesce(human, 0) + _coalesce(agent, 0)
    else:
        summed = 0
    content = _prop(p, "content")
    return _present({
        "id": _prop(p, "id"),
        "title": _prop(p, "title"),
        "content": _js_slice(content, 300) if isinstance(content, str) else _ABSENT,
        "author": _coalesce(_prop(author, "username"), _prop(p, "authorUsername")),
        "channel": _coalesce(_prop(channel, "slug"), _prop(p, "channelSlug")),
        "likes": _coalesce(_prop(p, "totalLikes"), summed),
    })


def format_agent(a: Any) -> Dict[str, Any]:
    """An agent from `/api/discovery/agents` as the tool returns it.
    `formatAgent` in the TypeScript skill."""
    bio = _prop(a, "bio")
    return _present({
        "username": _prop(a, "username"),
        "display_name": _either(_prop(a, "displayName"), _prop(a, "display_name")),
        "bio": _js_slice(bio, 200) if isinstance(bio, str) else _ABSENT,
        "url": _coalesce(_prop(a, "agentUrl"), _prop(a, "agent_url")),
        "reputation": _coalesce(_prop(a, "reputationScore"), _prop(a, "reputation"), 0),
        "trust_level": _coalesce(_prop(a, "trustLevel"), _prop(a, "trust_level"), "standard"),
        "tier": _prop(a, "tier"),
        "is_online": _coalesce(_prop(a, "isOnline"), _prop(a, "is_online")),
    })


def _trimmed_url(value: Any) -> Optional[str]:
    url = str(value or "").strip().rstrip("/")
    return url or None


def _query_string(pairs: List[Tuple[str, str]]) -> str:
    """`new URLSearchParams(pairs).toString()`, byte for byte: letters,
    digits and `*-._` as they are, a space as `+`, every other byte of the
    UTF-8 as `%XX`. (`urlencode` leaves `~` alone and escapes `*`.)"""
    def encode(text: str) -> str:
        out = []
        for byte in text.encode("utf-8"):
            char = chr(byte)
            if char.isascii() and (char.isalnum() or char in "*-._"):
                out.append(char)
            elif char == " ":
                out.append("+")
            else:
                out.append(f"%{byte:02X}")
        return "".join(out)

    return "&".join(f"{encode(k)}={encode(v)}" for k, v in pairs)


def _js_string(value: Any) -> str:
    """`String(value)` for what a model passes as `limit`."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def resolve_platform_url(config: Dict[str, Any]) -> str:
    """The platform's base URL (module docstring), the TypeScript order."""
    for value in (
        config.get("robutler_api_url"),
        config.get("webagents_api_url"),
        os.getenv("ROBUTLER_API_URL"),
        os.getenv("ROBUTLER_INTERNAL_API_URL"),
    ):
        url = _trimmed_url(value)
        if url:
            return url
    try:
        from webagents.cli.config_store import platform_url

        return _trimmed_url(platform_url()) or DEFAULT_PLATFORM_URL
    except Exception:
        # No CLI configuration to read: the default stands.
        return DEFAULT_PLATFORM_URL


@dataclass
class PlatformCredential:
    """How one platform call is authenticated (module docstring). Exactly one
    of the three is set."""

    #: A `WebBotAuth`: the request is signed with the agent's identity.
    auth: Optional[Any] = None
    #: A platform key, presented as `Authorization: Bearer`.
    bearer: Optional[str] = None
    #: Neither: the sentence naming the fix.
    refusal: Optional[str] = None

    @property
    def kind(self) -> str:
        if self.auth is not None:
            return "signature"
        if self.bearer:
            return "bearer"
        return "none"

    def headers(self, **fixed: str) -> Dict[str, str]:
        """The request headers: `fixed` plus the bearer when that is the
        credential. A signed request carries no `Authorization` at all."""
        headers = dict(fixed)
        if self.auth is None and self.bearer:
            headers["Authorization"] = f"Bearer {self.bearer}"
        return headers


#: The sentence for an agent with neither credential, the TypeScript one
#: word for word (`tests/fixtures/discovery_tool/definition.json`,
#: `no_credential`). It names the ways a person at the CLI has; it named
#: `create_server` and a config key, which only code can use.
NO_DISCOVERY_CREDENTIAL = (
    "No credential for the platform: this agent has no signing identity and no platform key. "
    "Publish it with `webagents publish` (the chat and `serve` then use the key it stores for this folder), "
    "serve it at a public https URL (WEBAGENTS_PUBLIC_URL) so its requests are signed, "
    "or set WEBAGENTS_AGENT_TOKEN to the agent's key."
)


class DiscoverySkill(Skill):
    """
    Unified discovery skill for the Robutler / WebAgents network.

    Searches across agents, intents, posts, channels, tags, and users
    via the Robutler API.  Results include @username references for agents
    so they can be passed directly to the NLI tool.
    """

    #: Seconds the auto-publish waits after `initialize` for the server to
    #: finish starting. A class attribute so a test can set it to zero.
    AUTO_PUBLISH_DELAY_S = 5.0

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all")

        self.config = config or {}
        self.enable_discovery = self.config.get('enable_discovery', True)

        # The platform's base URL (module docstring).
        self.robutler_api_url = resolve_platform_url(self.config)
        # Per platform call, in milliseconds: the TypeScript skill's `timeout`.
        self.timeout_ms = int(self.config.get('timeout') or 8000)

        # API key (resolved in initialize). OPTIONAL since 2026-09-23: an
        # agent with a signing identity needs none (module docstring).
        self.robutler_api_key = self.config.get('robutler_api_key')

        # The signer, built once the identity has been resolved (module
        # docstring, rule 1). `None` until then; rebuilt never, because the
        # key it holds is the key the server publishes for the whole process.
        self._web_bot_auth: Optional[Any] = None

        # The scheduled auto-publish, held so the garbage collector cannot
        # take the task mid-flight: `asyncio.create_task` returns the only
        # strong reference there is.
        self._auto_publish_task: Optional[Any] = None

        # Intents published so far by this process, intent -> description.
        # `POST /api/discovery/announce` REPLACES the caller's whole set, so
        # the union is re-sent on every call to keep `publish_intents`
        # additive (see its docstring).
        self._published_intents: Dict[str, str] = {}

    async def initialize(self, agent) -> None:
        """Initialize DiscoverySkill"""
        import asyncio
        from webagents.utils.logging import get_logger, log_skill_event

        self.agent = agent
        self.logger = get_logger('skill.webagents.discovery', self.agent.name)

        # Resolve API key: config -> agent -> env
        if not self.robutler_api_key:
            if hasattr(self.agent, 'api_key') and self.agent.api_key:
                self.robutler_api_key = self.agent.api_key
            elif os.getenv('WEBAGENTS_API_KEY'):
                self.robutler_api_key = os.getenv('WEBAGENTS_API_KEY')
            elif os.getenv('SERVICE_TOKEN'):
                self.robutler_api_key = os.getenv('SERVICE_TOKEN')

        # Say what the calls will carry, once, at start: the refusal sentence
        # names the fix, where the old "No API key configured" told an agent
        # with a perfectly good signing identity to go and get a key.
        credential = self._credential()
        if credential.refusal:
            self.logger.warning(credential.refusal)

        log_skill_event(self.agent.name, 'discovery', 'initialized', {
            'enable_discovery': self.enable_discovery,
            'robutler_api_url': self.robutler_api_url,
            'has_api_key': bool(self.robutler_api_key),
            'credential': credential.kind,
        })

        # Auto-publish intents on startup (best-effort, non-blocking)
        if self._configured_intents() and self.enable_discovery and not credential.refusal:
            self._auto_publish_task = asyncio.create_task(self._auto_publish_intents())

    # ===== CREDENTIAL (module docstring) =====

    def _configured_intents(self) -> List[str]:
        """The intents this agent publishes: `config["intents"]`, else an
        `intents` attribute on the agent (the shape the `/intent publish`
        command has always read)."""
        intents = self.config.get('intents') or getattr(self.agent, 'intents', None) or []
        return list(intents) if isinstance(intents, (list, tuple)) else [intents]

    def _configured_description(self) -> str:
        return (
            self.config.get('description')
            or getattr(self.agent, 'description', None)
            or 'An AI agent'
        )

    def _signing_principal(self) -> Tuple[Optional[str], Optional[str]]:
        """The agent URL a signature names, or `(None, why)`.

        `config["agent_url"]` when given; otherwise composed the way
        `create_server` composes the URL it serves the key set under
        (`compose_principal`: `WEBAGENTS_PUBLIC_URL` plus the server's
        `url_prefix`, passed as `config["agent_path"]`, plus the agent name),
        so the key set the signature names is the one the platform can fetch.
        """
        from webagents.crypto.http_signature import canonical_agent_url
        from webagents.server.core.registration import compose_principal, resolve_public_base_url

        explicit = (self.config.get('agent_url') or '').strip()
        if explicit:
            return canonical_agent_url(explicit), None
        name = getattr(self.agent, 'name', None)
        if not name:
            return None, "the skill is not attached to an agent yet"
        base = resolve_public_base_url(self.config.get('public_url'), name)
        if not (base.startswith('http://') or base.startswith('https://')):
            return None, "WEBAGENTS_PUBLIC_URL is not set to the https base URL this agent is served at"
        return compose_principal(base, name, self.config.get('agent_path')), None

    def _signer(self) -> Tuple[Optional[Any], Optional[str]]:
        """The `WebBotAuth` for this agent, or `(None, why)`. Built once."""
        from webagents.crypto.http_signature import SigningError, WebBotAuth
        from webagents.crypto.jwks import JWKSManager

        if self._web_bot_auth is not None:
            return self._web_bot_auth, None
        principal, why = self._signing_principal()
        if principal is None:
            return None, why
        name = getattr(self.agent, 'name', None)
        if not name:
            return None, "the skill is not attached to an agent yet"
        keys_dir = self.config.get('keys_dir')
        manager = JWKSManager({'keys_dir': keys_dir} if keys_dir else {})
        # LOADED, never minted (module docstring). An unusable key file is
        # reported as-is: it is the server's identity and the server stops on
        # it too, so a bearer beside it is the fallback and silence is not.
        try:
            thumbprint = manager.load_ed25519_key(name)
        except RuntimeError as e:
            return None, str(e)
        if thumbprint is None:
            return None, (
                f"no Ed25519 key for '{name}' under {manager.keys_dir}; create_server writes one there "
                "at start, and WEBAGENTS_KEYS_DIR (or config['keys_dir']) must point at the same directory"
            )
        try:
            self._web_bot_auth = WebBotAuth(
                manager.held_ed25519_keys(),
                principal,
                allow_http=self.config.get('allow_http'),
            )
        except SigningError as e:
            return None, f"this agent's identity cannot sign: {e}"
        return self._web_bot_auth, None

    def _credential(self) -> PlatformCredential:
        """Which credential the next platform call carries (module docstring)."""
        bearer = (self.robutler_api_key or '').strip() or None
        auth, why = self._signer()
        if auth is not None:
            return PlatformCredential(auth=auth)
        if bearer:
            return PlatformCredential(bearer=bearer)
        if why and why.startswith("this agent's identity cannot sign"):
            # The signer's own sentence names the variable to set.
            return PlatformCredential(refusal=why)
        if why and why.startswith("no Ed25519 key"):
            # A public URL is set, so signing was meant; say what is missing.
            return PlatformCredential(refusal=f"This agent cannot sign its platform calls: {why}.")
        return PlatformCredential(refusal=NO_DISCOVERY_CREDENTIAL)

    def credential(self) -> PlatformCredential:
        """How this skill will authenticate its next platform call."""
        return self._credential()

    async def _auto_publish_intents(self) -> None:
        """Auto-publish agent intents on startup (best-effort)."""
        import asyncio
        # Small delay to let the server finish starting
        await asyncio.sleep(self.AUTO_PUBLISH_DELAY_S)
        try:
            intents = self._configured_intents()
            if not intents:
                return
            result = await self.publish_intents(
                intents=intents,
                description=self._configured_description(),
            )
            if result.get('success'):
                self.logger.info(f"Auto-published {len(intents)} intents for agent '{self.agent.name}'")
            else:
                self.logger.warning(f"Auto-publish intents failed: {result.get('error', 'unknown')}")
        except Exception as e:
            self.logger.warning(f"Auto-publish intents error (non-fatal): {e}")

    # ===== THE SEARCH TOOL (module docstring: the same in both SDKs) =====

    @tool(name="search", description=SEARCH_DEFINITION["function"]["description"], scope="all")
    async def search(self,
                     query: str,
                     types: Optional[List[str]] = None,
                     limit: Optional[int] = None,
                     channel: Optional[str] = None,
                     tag: Optional[str] = None,
                     sort: Optional[str] = None,
                     context=None) -> Dict[str, Any]:
        """Search the platform for agents, intents, posts, channels, users and
        tags. `PortalDiscoverySkill.search` in TypeScript, call for call."""
        if not self.enable_discovery:
            return {'error': 'Discovery is disabled'}

        # Refused before anything is dialled: the reason names the fix, where
        # a 401 per type would only say it failed.
        credential = self._credential()
        if credential.refusal:
            return {'error': credential.refusal}

        query = "" if query is None else str(query)
        if isinstance(types, (list, tuple)):
            type_list = [str(t) for t in types]
        elif isinstance(types, str):
            type_list = [t.strip() for t in types.split(',') if t.strip()]
        else:
            type_list = ['intents', 'agents', 'posts']
        limit = 10 if limit is None else limit
        base = self.robutler_api_url.rstrip('/')

        results: Dict[str, Any] = {}
        # Each call's label in the order the calls start, for the answer's
        # order and the failure sentence; `failures` holds why a call gave
        # nothing.
        started: List[str] = []
        failures: Dict[str, str] = {}
        direct: Dict[str, Any] = {}

        match = _POST_URL_RE.search(query)
        direct_id = match.group(1) if match else (query.strip() if _UUID_RE.match(query.strip()) else None)

        import httpx

        async with httpx.AsyncClient(timeout=self.timeout_ms / 1000) as client:

            def call(label: str, method: str, url: str, **kwargs: Any):
                if label not in started:
                    started.append(label)
                return self._get_json(client, credential, failures, label, method, url, **kwargs)

            async def direct_post() -> None:
                post = await call('post', 'GET', f"{base}/api/posts/{direct_id}")
                if post is not None and _truthy(_prop(post, 'id')):
                    direct['post'] = format_post(post)

            async def intents() -> None:
                data = await call('intents', 'POST', f"{base}/api/intents/search",
                                  json_body={'query': query, 'limit': limit})
                if data is not None:
                    rows = data.get('results')
                    results['intents'] = rows if isinstance(rows, list) else []

            async def agents() -> None:
                data = await call('agents', 'GET', f"{base}/api/discovery/agents",
                                  params=[('search', query), ('type', 'agent'), ('limit', _js_string(limit))])
                if data is not None:
                    rows = _either(data.get('agents', _ABSENT), [])
                    results['agents'] = [format_agent(a) for a in rows] if isinstance(rows, list) else []

            async def content(kind: str) -> None:
                params = [('q', query), ('limit', _js_string(limit))]
                if kind == 'posts':
                    if channel:
                        params.append(('channel', channel))
                    if tag:
                        params.append(('tag', tag))
                    if sort:
                        params.append(('sort', 'trending' if sort == 'relevance' else 'top' if sort == 'popular' else sort))
                data = await call(kind, 'GET', f"{base}/api/discovery/{kind}", params=params)
                if data is None:
                    return
                rows = _either(data.get(kind, _ABSENT), data.get('results', _ABSENT), [])
                if kind == 'posts' and isinstance(rows, list):
                    rows = [format_post(row) for row in rows]
                results[kind] = rows

            jobs = []
            if direct_id:
                jobs.append(direct_post())
            for kind in type_list:
                if kind == 'intents':
                    jobs.append(intents())
                elif kind == 'agents':
                    jobs.append(agents())
                else:
                    jobs.append(content(kind))
            await asyncio.gather(*jobs)

        if 'post' in direct:
            found = direct['post']
            posts = results.get('posts') if _truthy(results.get('posts', _ABSENT)) else []
            if not any(isinstance(p, dict) and p.get('id', _ABSENT) == found.get('id', _ABSENT) for p in posts):
                results['posts'] = [found, *posts]

        # In the order asked for, not the order the answers came back, so the
        # same search reads the same way twice (and under either SDK).
        ordered: Dict[str, Any] = {}
        for key in [*started, *results.keys()]:
            if key in results and key not in ordered:
                ordered[key] = results[key]
        if not ordered and failures:
            # Nothing came back and something failed: say what, rather than
            # an empty answer the model would read as "nothing found".
            failed = [failures[label] for label in started if label in failures]
            return {'error': f"Search failed: {', '.join(failed)}."}
        return ordered

    search._webagents_tool_definition = SEARCH_DEFINITION

    async def _get_json(self, client: Any, credential: 'PlatformCredential', failures: Dict[str, str],
                        label: str, method: str, url: str,
                        params: Optional[List[Tuple[str, str]]] = None,
                        json_body: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """One platform call's JSON body, or None when the call failed, with
        the reason left in `failures`: a failed type is left out of the
        results rather than failing the search. The log line names the type,
        the status and the time, never the query."""
        import httpx

        started = time.monotonic()
        headers = credential.headers(**({'Content-Type': 'application/json'} if json_body is not None else {}))
        if params:
            url = f"{url}?{_query_string(params)}"
        try:
            response = await client.request(
                method,
                url,
                headers=headers,
                # Serialised as `JSON.stringify` does, so the body a signature
                # covers is the one the TypeScript skill sends.
                content=json.dumps(json_body, separators=(',', ':'), ensure_ascii=False).encode() if json_body is not None else None,
                auth=credential.auth if credential.auth is not None else httpx.USE_CLIENT_DEFAULT,
            )
        except httpx.TimeoutException:
            _log.debug("[search] %s timed out after %dms", label, (time.monotonic() - started) * 1000)
            failures[label] = f"{label} timed out"
            return None
        except httpx.HTTPError:
            _log.debug("[search] %s unreachable after %dms", label, (time.monotonic() - started) * 1000)
            failures[label] = f"{label} unreachable"
            return None
        _log.debug("[search] %s %s in %dms", label, response.status_code, (time.monotonic() - started) * 1000)
        if not response.is_success:
            failures[label] = f"{label} {response.status_code}"
            return None
        try:
            data = response.json()
        except ValueError:
            data = None
        if isinstance(data, dict):
            return data
        failures[label] = f"{label} unreadable"
        return None

    # ===== PUBLISHING (this SDK's own; not a tool, module docstring) =====

    async def publish_intents(self,
                              intents: List[str],
                              description: str,
                              replace: bool = False) -> Dict[str, Any]:
        """Publish agent intents to the WebAgents platform.

        A method for code (and the owner's `/intent/publish` command), not a
        tool the model can call: the TypeScript skill never offered one
        (module docstring). `publish_intents_tool` is the old name.

        ADDITIVE BY DEFAULT, on purpose. `POST /api/discovery/announce`
        REPLACES the caller's whole intent set in one transaction
        (app/api/discovery/announce/route.ts), while the tool it replaced
        (`/api/intents/create`) was additive. Two calls would therefore have
        left only the second call's intents — a silent semantic change for
        anyone publishing in batches. This skill keeps the published set and
        re-sends the union, so the tool behaves the way its callers expect
        while the wire call stays a single atomic replace.

        Pass `replace=True` for the platform's raw semantics (and
        `intents=[]` with it to de-list entirely).
        """
        if not self.enable_discovery:
            return {'success': False, 'error': 'Discovery disabled'}
        credential = self._credential()
        if credential.refusal:
            return {'success': False, 'error': credential.refusal}

        try:
            import httpx
            from urllib.parse import urlparse

            # The endpoint being announced MUST be this agent's own URL.
            # The old default advertised the PORTAL's `/u/{name}` URL for
            # every unconfigured on-prem agent — a listing that routes every
            # caller back at the platform instead of at the agent.
            #
            # SIGNED, the default is the principal the signature names
            # (2026-09-23): the platform requires the announced endpoint to
            # share its origin with the signed agent URL, and that URL is by
            # construction where this agent is served, because it is where
            # the platform just fetched the key set from. `config["agent_url"]`
            # still overrides. With a bearer, `WEBAGENTS_PUBLIC_URL` is read
            # as before.
            agent_url = (self.config.get('agent_url') or '').strip()
            if not agent_url:
                if credential.auth is not None:
                    agent_url = credential.auth.agent_url
                else:
                    agent_url = os.getenv('WEBAGENTS_PUBLIC_URL') or ''
            if not agent_url:
                return {
                    'success': False,
                    'error': (
                        "No public URL configured for this agent. Set "
                        "config['agent_url'] or WEBAGENTS_PUBLIC_URL to the "
                        "URL this agent actually serves — never the portal's."
                    ),
                }
            portal_host = urlparse(self.robutler_api_url).hostname
            if portal_host and urlparse(agent_url).hostname == portal_host:
                return {
                    'success': False,
                    'error': (
                        f"Refusing to publish the portal's own host ({portal_host}) "
                        "as this agent's endpoint. Set WEBAGENTS_PUBLIC_URL to the "
                        "agent's own URL."
                    ),
                }

            # The union this skill will announce. `_published_intents` is the
            # accumulator that keeps the tool additive over a REPLACING
            # endpoint (see the docstring); `replace=True` resets it.
            if replace:
                self._published_intents = {}
            for intent in intents:
                self._published_intents[intent] = description
            announced = [
                {'intent': intent, 'description': desc}
                for intent, desc in self._published_intents.items()
            ]

            # POST /api/discovery/announce: the target agent comes from the
            # credential (the signed identity IS the agent row; a per-agent
            # key names it); the endpoint and the intent set are replaced in
            # one transaction. (The old /api/intents/create call sent an agent
            # NAME where the schema requires a UUID, so it 400'd every time.)
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.robutler_api_url.rstrip('/')}/api/discovery/announce",
                    headers=credential.headers(**{'Content-Type': 'application/json'}),
                    json={
                        'url': agent_url,
                        'intents': announced,
                    },
                    auth=credential.auth,
                )

                if response.status_code != 200:
                    # Roll the accumulator back: nothing was published, and a
                    # phantom entry would be re-sent on the next call.
                    for intent in intents:
                        self._published_intents.pop(intent, None)
                    raise Exception(
                        f"Announce API error: HTTP {response.status_code} "
                        f"POST /api/discovery/announce - {response.text[:300]}"
                    )

                return {
                    'success': True,
                    'agent_url': agent_url,
                    'published_intents': [i['intent'] for i in announced],
                    'newly_published': intents,
                    'replaced': replace,
                }

        except Exception as e:
            self.logger.error(f"Intent publishing failed: {e}")
            return {'success': False, 'error': str(e)}

    #: The old name, for code written against it.
    publish_intents_tool = publish_intents

    def get_dependencies(self) -> List[str]:
        return ['httpx']

    # ===== SLASH COMMANDS =====

    @command("/discover", description="Search for agents and content across the network")
    async def cmd_discover(self, query: str, types: str = None) -> Dict[str, Any]:
        """Search for agents and content.

        Usage: /discover <query> [types]

        Examples:
          /discover image generation
          /discover python tutorials types=posts,channels
        """
        if not query:
            return {"error": "Please provide a search query", "usage": "/discover <query>"}

        return await self.search(query=query, types=types)

    @command("/intent/discover", description="Discover agents by intent (legacy)")
    async def cmd_intent_discover(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Discover agents by intent (legacy command, uses new discovery API).

        Usage: /intent discover <query>
        """
        if not query:
            return {"error": "Please provide a search query"}

        return await self.search(query=query, types=["agents", "intents"], limit=top_k)

    @command("/intent/publish", description="Publish agent intents to the platform", scope="owner")
    async def cmd_intent_publish(self) -> Dict[str, Any]:
        """Publish agent's intents to the platform.

        Usage: /intent publish
        """
        intents = self._configured_intents()
        if not intents:
            return {"error": "No intents defined in agent configuration"}

        return await self.publish_intents(intents=intents, description=self._configured_description())

    @command("/intent/list", description="List current agent intents")
    async def cmd_intent_list(self) -> Dict[str, Any]:
        """List the current agent's configured intents."""
        intents = self._configured_intents()
        return {
            "agent": getattr(self.agent, 'name', 'unknown'),
            "intents": intents,
            "count": len(intents)
        }
