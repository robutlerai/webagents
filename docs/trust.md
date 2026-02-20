# Trust Zones

The `webagents.trust` module provides trust evaluation for agent-to-agent communication based on dot-namespace naming.

## Quick Start

```python
from webagents.trust import evaluate_trust_rules, extract_trust_labels

# Simple preset: allow only family (same namespace)
rules = ["family"]
assert evaluate_trust_rules("alice.bot1", "alice.bot2", rules) == True
assert evaluate_trust_rules("bob.bot1", "alice.bot2", rules) == False

# Pattern matching
rules = ["@alice.*", "#verified"]
labels = {"trust:verified"}
assert evaluate_trust_rules("alice.helper", "my-agent", rules) == True
assert evaluate_trust_rules("bob.bot", "my-agent", rules, labels) == True

# Allow/deny with deny taking precedence
rules = {"allow": ["everyone"], "deny": ["@spammer"]}
assert evaluate_trust_rules("spammer", "my-agent", rules) == False
assert evaluate_trust_rules("alice", "my-agent", rules) == True
```

## Presets

| Preset | Description |
|:-------|:------------|
| `"everyone"` | Any agent (platform + external) |
| `"platform"` | Platform-native agents (non-TLD root segment) |
| `"family"` | Same namespace (alice.* are family, com.example.* are family) |
| `"nobody"` | Deny all |

## Patterns

Prefix patterns with `@`:

- `@alice.*` — matches `alice.bot1`, `alice.bot2` (one level)
- `@alice.**` — matches `alice.bot1`, `alice.bot1.helper` (any depth)
- `@com.example.*` — direct children of `com.example`
- `@com.example.**` — all descendants under `com.example`

## Trust Labels

Prefix labels with `#`. These match against JWT `trust:*` scopes:

- `#verified` matches `trust:verified` in JWT scope
- `#x-linked` matches `trust:x-linked`
- `#reputation:100` matches any `trust:reputation-N` where N ≥ 100

Extract labels from a JWT scope string:

```python
from webagents.trust import extract_trust_labels

labels = extract_trust_labels(
    "read write trust:verified trust:reputation-500",
    issuer="https://robutler.ai",
)
# {"trust:verified", "trust:reputation-500"}
```

Labels from untrusted issuers are ignored (default trusted: `robutler.ai`).

## Namespace Derivation

```python
from webagents.trust import derive_namespace, is_same_namespace

derive_namespace("alice.my-bot")         # "alice"
derive_namespace("com.example.bot")      # "com.example"
is_same_namespace("alice.a", "alice.b")  # True
is_same_namespace("alice.a", "bob.a")    # False
```

## External Agent URL Mapping

```python
from webagents.trust import url_to_reversed_domain_name

url_to_reversed_domain_name("https://agents.example.com/my-bot")
# "com.example.agents.my-bot"
```

## Integration

### Inbound (AuthSkill)

The `AuthSkill.validate_request` hook automatically checks `accept_from` rules from the agent's config against incoming agent tokens.

### Outbound (NLI Skill)

The NLI skill checks `talk_to` rules from the agent's config before making outbound calls.

### Configuration

Set trust rules in your agent's config:

```python
config = {
    "accept_from": ["family", "#verified"],
    "talk_to": ["everyone"],
}
```

Or use allow/deny:

```python
config = {
    "accept_from": {
        "allow": ["everyone"],
        "deny": ["@spammer", "@com.evil.**"],
    },
}
```
