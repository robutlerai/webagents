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


class SandboxConfig(BaseModel):
    """Sandbox configuration in YAML frontmatter."""
    preset: str = "development"  # strict, development, unrestricted
    allowed_folders: List[str] = Field(default_factory=lambda: ["."])
    allowed_commands: List[str] = Field(default_factory=list)
    allowed_imports: List[str] = Field(default_factory=list)
    # Environment variables a sandboxed command may see despite looking like a
    # secret. Everything whose name matches KEY, SECRET, TOKEN, PASSWORD,
    # CREDENTIAL, PRIVATE or AUTH is scrubbed otherwise (S-220).
    env_passthrough: List[str] = Field(default_factory=list)


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
    tools: List[str] = Field(default_factory=list)
    scopes: List[str] = Field(default_factory=lambda: ["all"])  # Permissions/scopes
    
    # Triggers
    cron: Optional[str] = None  # Cron expression: "0 18 * * 1-5"
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

    model_config = ConfigDict(extra="forbid")

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
