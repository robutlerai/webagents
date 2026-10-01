"""
YAML Schema Definitions

Pydantic models for AGENT.md and WEBAGENTS.md YAML frontmatter.

UNKNOWN KEYS ARE REJECTED (2026-09-23). Both models used to set
`extra = "allow"`, so `skils: [mcp]` parsed cleanly, was stored on the model,
and was never looked at again. The agent then ran with no skills and nothing
anywhere said why. For a file a person hand-writes, silently keeping a typo is
the worst option available: it cannot be acted on and it cannot be seen.

The rejection carries a suggestion, because "unknown key" on its own sends you
to the documentation for something you can see on screen.
"""

import difflib
from typing import List, Optional, Dict, Any, Union
from pydantic import field_validator, BaseModel, ConfigDict, Field, model_validator


class AgentFormatError(Exception):
    """A frontmatter file that a person needs to fix by hand.

    Separate from `pydantic.ValidationError` so callers can tell "this file is
    wrong" from "this program passed the wrong type", and can print the message
    as-is: it is already written for the author of the file.

    DELIBERATELY NOT A `ValueError`. Pydantic catches `ValueError` and
    `AssertionError` raised inside a validator and re-wraps them in a
    `ValidationError`, which buries the message under type/url/input noise and
    makes the class uncatchable by callers. Anything else propagates untouched.
    """


def _reject_unknown_keys(values: Any, known: set, what: str) -> Any:
    """Refuse keys the schema does not define, naming the nearest match."""
    if not isinstance(values, dict):
        return values

    unknown = [key for key in values if key not in known]
    if not unknown:
        return values

    problems = []
    for key in unknown:
        close = difflib.get_close_matches(str(key), sorted(known), n=1, cutoff=0.6)
        if close:
            problems.append(f"unknown key '{key}' (did you mean '{close[0]}'?)")
        else:
            problems.append(f"unknown key '{key}'")

    raise AgentFormatError(
        f"{what}: " + "; ".join(problems)
        + f". Known keys: {', '.join(sorted(known))}"
    )


#: Keys the schema accepts and validates but which change NOTHING about how an
#: agent runs (audited 2026-09-23 across both SDKs and the daemon).
#:
#: Most are never read at all. `watch` is the exception and is listed anyway,
#: because the distinction is invisible from outside: it IS read, into
#: `DaemonAgent.watch_patterns` (`cli/daemon/registry.py:64`), and then nothing
#: consumes it. `FileWatchTrigger.start()` was a bare `pass`, nothing
#: constructs the trigger, and `registry.get_agents_with_watch()` has no
#: caller. To the person who wrote `watch: ["*.csv"]`, "stored in a struct" and
#: "ignored" are the same thing.
#:
#: `sandbox` WAS in this set and is not any more (2026-09-23). It used to read
#: as a security control and enforce nothing, which is S-217; it is now applied
#: at the OS level by `webagents/sandbox/` and consumed by `ShellSkill`. Its
#: `allowed_imports` field is still unenforceable and is reported separately,
#: because an OS sandbox confines file and socket access, not `import`
#: statements.
#:
#: They are kept rather than dropped because `webagents init` has always
#: written `visibility: local`, so removing the key would turn every agent file
#: the CLI ever generated into a hard error under the stricter parsing above.
#: Keeping them declared costs nothing; pretending they work costs a user an
#: afternoon. `webagents doctor` and the validator report them, so a file that
#: sets one is told it is inert instead of quietly ignored.
#:
#: `mcp_servers` is the odd one out and the likeliest to be dropped: it is
#: `List[str]`, which cannot express a server's command or environment, and the
#: working way to configure MCP is the `mcp` skill's dict form
#: (`skills: [{mcp: {servers: [...]}}]`).
INERT_FIELDS = frozenset({
    "tools",
    "visibility",
    "version",
    "author",
    "tags",
    "mcp_servers",
    "watch",
})

#: Sub-keys of `sandbox:` that are parsed but cannot be enforced by an OS
#: sandbox. Reported rather than silently ignored, for the same reason the set
#: above exists.
INERT_SANDBOX_FIELDS = frozenset({"allowed_imports"})


class SandboxFilesConfig(BaseModel):
    """`sandbox: files:` (the sandbox-default lane, 2026-09-27): the folders
    commands may write, what they may read (`all`, or a list that scopes
    reads), and more paths they may never read."""
    write: List[str] = Field(default_factory=lambda: ["."])
    read: Union[str, List[str]] = "all"
    deny: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class SandboxNetworkConfig(BaseModel):
    """`sandbox: network:`: the hosts (or groups) commands may reach, whether
    local servers and listening ports are allowed, and unix sockets by path."""
    hosts: List[str] = Field(default_factory=list)
    local: bool = False
    sockets: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class SandboxConfig(BaseModel):
    """Sandbox configuration in YAML frontmatter, normalised.

    UNKNOWN KEYS ARE REJECTED HERE TOO (2026-09-26, S-270). This model had no
    `model_config`, so pydantic's default `extra="ignore"` applied while the
    two file models around it forbid unknown keys: `presets: strict` (one
    letter off) loaded without a word and ran with `preset: development`,
    the looser default. A mistyped STRICTER setting silently running looser
    is the worst outcome a sandbox schema can have. The rejection carries the
    same did-you-mean as the file-level one, and the TypeScript loader says
    the same sentence (`tests/fixtures/sandbox/srt.json`, `unknown_key`).

    THE SHAPE IS GRANULAR (2026-09-27, the sandbox-default lane; the words
    are `webagents/sandbox/policy.py`'s). The fields here are the normalised
    ones; `normalize_declaration` runs first and folds the flat aliases in
    (`allowed_folders` is `files.write`, a bare `network:` list is
    `network.hosts`, `env_passthrough` is `env`), takes `sandbox: off` (the
    boolean False once PyYAML has read it) as `preset: unrestricted`, and
    refuses an unknown key at any level with the did-you-mean.
    """
    preset: str = "development"  # strict, development, unrestricted (`off` arrives as unrestricted)
    files: SandboxFilesConfig = Field(default_factory=SandboxFilesConfig)
    network: SandboxNetworkConfig = Field(default_factory=SandboxNetworkConfig)
    # Environment variables a sandboxed command may see despite looking like a
    # secret. Everything whose name matches KEY, SECRET, TOKEN, PASSWORD,
    # CREDENTIAL, PRIVATE or AUTH is scrubbed otherwise (S-220).
    env: List[str] = Field(default_factory=list)
    allowed_commands: List[str] = Field(default_factory=list)
    allowed_imports: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _check_keys(cls, values):
        from webagents.sandbox.policy import SandboxDeclarationError, normalize_declaration

        try:
            return normalize_declaration(values)
        except SandboxDeclarationError as error:
            raise AgentFormatError(str(error)) from None

    # The old spellings, for readers that still ask by them.
    @property
    def allowed_folders(self) -> List[str]:
        return list(self.files.write)

    @property
    def env_passthrough(self) -> List[str]:
        return list(self.env)


class AgentMetadata(BaseModel):
    """YAML frontmatter schema for AGENT.md and AGENT-*.md files."""
    
    # Identity (required)
    name: str = "assistant"
    # Empty when the file says nothing (2026-09-25): a made-up "An AI
    # assistant" showed on the chat's card for every agent without one, where
    # the TypeScript card shows none.
    description: str = ""
    
    # Namespace
    namespace: str = "local"
    
    # Discovery - intents (unified discovery mechanism)
    intents: List[str] = Field(default_factory=list)
    # Example: ["summarize documents", "create weekly reports", "parse PDFs"]
    
    # Configuration
    model: Optional[str] = None
    skills: List[Union[str, Dict[str, Any]]] = Field(default_factory=list)
    # SKILL.md skills kept outside `.agents/skills` (plan item 1.4,
    # 2026-09-26): a list of folders, each a skill folder or a folder of
    # skill folders, relative to the agent file. `skills:` names CODED skills
    # (`shell`, `openai`); this key names the other kind, so neither list can
    # be mistaken for the other. `.agents/skills/*` beside the file is found
    # without being named (`agents/skills/local/skillmd/`).
    agent_skills: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    scopes: List[str] = Field(default_factory=lambda: ["all"])  # Permissions/scopes

    # Triggers. `cron:` is a list of schedules (`cli/loader/schedules.py`,
    # plan item 1.7): each names what to run and where the result goes. The
    # old string form is refused there with the shape that works.
    cron: Optional[List[Dict[str, Any]]] = None
    watch: Optional[List[str]] = None  # File patterns to watch
    
    # Visibility
    visibility: str = "local"  # local | namespace | public
    
    # Sandbox overrides
    sandbox: Optional[SandboxConfig] = None
    
    # Additional metadata
    version: str = "1.0.0"
    author: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    # MCP servers
    mcp_servers: List[str] = Field(default_factory=list)

    # Who may call the agent and what each group gets (ADR-0045). Kept as
    # written; `webagents.access.policy.parse_access` is the one reader, run
    # here so a malformed block stops the load with its own sentence.
    access: Optional[Dict[str, Any]] = None

    # The models to try, in order, when `model` fails with a provider error
    # (a 5xx, a 429, no answer), each `provider/model` (plan item 2.8,
    # 2026-09-26; `agents/skills/core/llm/failover.py`). The TypeScript
    # loader reads the same key with the same sentence, pinned by
    # `tests/fixtures/w2ops/models.json` (`failover`).
    fallback_models: List[str] = Field(default_factory=list)

    # `{otel: true}` records the run as OpenTelemetry spans (plan item 2.4,
    # `webagents/observability/otel.py`); a bare boolean is the short form.
    # Kept as the parsed block; an unknown key stops the load with the
    # shared sentence (`tests/fixtures/w2ops/otel.json`, `config`).
    observability: Optional[Union[Dict[str, Any], bool]] = None

    # The tool rounds one turn may run before its last, tool-less answer
    # (2026-09-28, `agents/core/tool_budget.py`): a whole number from 1 to
    # 1000, refused with the shared sentence otherwise. `--max-tool-rounds`
    # and the chat's `/rounds` take precedence; the TypeScript loader reads
    # the same key.
    max_tool_rounds: Optional[int] = None

    # How the conversation is compacted when it fills the model's context
    # (2026-09-29, `agents/core/context_compaction.py`): `auto`, `at`, `keep`,
    # `hard`, `clear_tool_results`, `model`, `instructions`, `window`, each
    # with a default; an unknown key or a bad value stops the load with the
    # shared sentence (`tests/fixtures/context/compaction.json`, `policies`).
    # The TypeScript loader reads the same key.
    compaction: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("compaction", mode="before")
    @classmethod
    def _check_compaction(cls, value):
        if value is None:
            return None
        from webagents.agents.core.context_compaction import CompactionPolicyError, parse_policy

        try:
            parse_policy(value)
        except CompactionPolicyError as e:
            raise AgentFormatError(str(e)) from None
        return value

    @field_validator("max_tool_rounds", mode="before")
    @classmethod
    def _check_max_tool_rounds(cls, value):
        if value is None:
            return None
        from webagents.agents.core.tool_budget import parse_max_tool_rounds

        try:
            return parse_max_tool_rounds(value)
        except ValueError as e:
            # Not re-raised as a ValueError: pydantic would bury it (see AgentFormatError).
            raise AgentFormatError(str(e)) from None

    @field_validator("fallback_models", mode="before")
    @classmethod
    def _check_fallback_models(cls, value):
        if value is None:
            return []
        if not isinstance(value, list) or not all(isinstance(entry, str) and entry.strip() for entry in value):
            raise AgentFormatError("fallback_models: must be a list of provider/model strings")
        return [entry.strip() for entry in value]

    @field_validator("observability", mode="before")
    @classmethod
    def _check_observability(cls, value):
        if value is None:
            return None
        from webagents.observability.otel import ObservabilityConfigError, parse_observability

        try:
            return parse_observability(value)
        except ObservabilityConfigError as e:
            # Not re-raised as a ValueError: pydantic would bury it (see AgentFormatError).
            raise AgentFormatError(str(e)) from None

    @field_validator("access", mode="before")
    @classmethod
    def _check_access(cls, value):
        if value is None:
            return value
        from webagents.access.policy import AccessConfigError, parse_access

        try:
            parse_access(value)
        except AccessConfigError as e:
            # Not re-raised as a ValueError: pydantic would bury it (see AgentFormatError).
            raise AgentFormatError(str(e)) from None
        return value

    @field_validator("agent_skills", mode="before")
    @classmethod
    def _check_agent_skills(cls, value):
        if value is None:
            return []
        if not isinstance(value, list) or not all(isinstance(entry, str) and entry.strip() for entry in value):
            # The TypeScript loader's sentence (`agents/index.ts`), pinned by
            # `tests/fixtures/skillmd/skillmd.json`.
            raise AgentFormatError("agent_skills: must be a list of folder paths")
        return value

    @field_validator("cron", mode="before")
    @classmethod
    def _check_cron(cls, value):
        if value is None:
            return value
        from .schedules import parse_cron_block

        # Kept as written; the daemon parses it again when it runs. Raises the
        # fixture's own sentence, an AgentFormatError pydantic leaves alone.
        parse_cron_block(value)
        return value

    @model_validator(mode="before")
    @classmethod
    def _check_keys(cls, values):
        return _reject_unknown_keys(values, set(cls.model_fields), "AGENT.md frontmatter")


class ContextMetadata(BaseModel):
    """YAML frontmatter schema for WEBAGENTS.md context files."""
    
    # Namespace for all agents in this directory
    namespace: str = "local"
    
    # Default configuration for agents
    model: Optional[str] = None
    skills: List[Union[str, Dict[str, Any]]] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    
    # Sandbox defaults
    sandbox: Optional[SandboxConfig] = None
    
    # Default visibility
    visibility: str = "local"
    
    # MCP servers available to all agents
    mcp_servers: List[str] = Field(default_factory=list)
    
    # Additional context
    guidelines: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _check_keys(cls, values):
        return _reject_unknown_keys(values, set(cls.model_fields), "WEBAGENTS.md frontmatter")


class TemplateMetadata(BaseModel):
    """YAML frontmatter schema for TEMPLATE.md files."""
    
    name: str
    description: str
    
    # Template configuration
    output_name: Optional[str] = None  # Default output filename
    keep_template: bool = False  # Keep TEMPLATE.md after applying
    
    # Default values for generated agent
    defaults: Dict[str, Any] = Field(default_factory=dict)
    
    # Template variables
    variables: List[str] = Field(default_factory=list)

    # Templates stay permissive on purpose: `defaults` is a free-form mapping
    # and a template may carry rendering hints this schema does not model.
    model_config = ConfigDict(extra="allow")
