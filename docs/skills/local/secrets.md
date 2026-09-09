---
title: Secrets Skill
description: Named credentials in the operating system keystore, with a fallback that tells you when it is not one.
---

# Secrets Skill

Named credentials in the operating system's own keystore, so an agent has
somewhere to put a token that is not a dotfile and not an environment
variable.

| Platform | Backend |
|---|---|
| macOS | Keychain |
| Linux | Secret Service (libsecret over DBus) |
| Windows | Credential Manager |
| Anywhere else, or a machine with no keystore | An owner-only file, announced loudly |

TypeScript reaches all three through [`@napi-rs/keyring`](https://www.npmjs.com/package/@napi-rs/keyring),
an optional dependency. Python reaches them through
[`keyring`](https://pypi.org/project/keyring/), the optional `keyring` extra.
Neither is required to run the skill. Without it you get the file backend,
and you are told.

## Install the keystore backend

```bash
npm install @napi-rs/keyring     # TypeScript
pip install 'webagents[keyring]' # Python
```

## Usage

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';
import { SecretsSkill } from 'webagents/skills/secrets';

const secrets = new SecretsSkill({ namespace: 'my-agent' });

const agent = new BaseAgent({
  name: 'my-agent',
  model: 'openai/gpt-4o-mini',
  skills: [secrets],
});

// Code reads the credential directly. This is the normal path.
const store = await secrets.getStore();
const key = await store.get('openai_api_key');
```

```python tab="Python"
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.local.secrets import SecretsSkill

secrets = SecretsSkill({"namespace": "my-agent"})

agent = BaseAgent(
    name="my-agent",
    model="openai/gpt-4o-mini",
    skills={"secrets": secrets},
)

# Code reads the credential directly. This is the normal path.
key = secrets.get_store().get("openai_api_key")
```

You can also open a store on its own, with no agent involved:

```typescript tab="TypeScript"
import { openSecretStore } from 'webagents/skills/secrets';

const store = await openSecretStore({ namespace: 'my-agent' });
await store.set('openai_api_key', process.env.OPENAI_API_KEY ?? '');
```

```python tab="Python"
import os

from webagents.agents.skills.local.secrets import open_secret_store

store = open_secret_store(namespace="my-agent")
store.set("openai_api_key", os.environ["OPENAI_API_KEY"])
```

## Namespacing

The keystore is machine-wide, so every secret is filed under
`webagents:<namespace>`. The namespace defaults to the agent's name, then to
`WEBAGENTS_SECRETS_NAMESPACE`, then to `webagents`. Two agents with different
names on one machine cannot read each other's secrets. Two agents that share
a name share secrets, which is a property of the name.

## The fallback

There is no keystore in most containers, on most CI runners, and on a
headless Linux host with no DBus session. The skill still works there: it
writes `~/.webagents/secrets/<namespace>.json` with mode 0600 inside a 0700
directory.

That file is **plaintext**. Anything that can read it, including a backup and
any process running as the same user, can read the credentials. So the skill
says so three separate times, because a fallback nobody noticed is the failure
this design is built around:

1. **At startup**, once, in the log. The message names why there is no
   keystore, where the file is, and how to refuse the fallback.
2. **On every write**, naming the secret. Names are logged, values never are.
3. **In every tool result**, as `backend: "file"` plus a `warning` string. This
   is the copy the model reads, so the agent itself knows it is not on a
   keystore and can tell its owner.

```
[webagents] secrets for "my-agent" are NOT in an OS keystore: optional package
@napi-rs/keyring is not installed (npm install @napi-rs/keyring). They are
stored as PLAINTEXT in /home/agent/.webagents/secrets/my-agent.json (0600).
Anything that can read that file, including a backup and any process running
as this user, can read the secrets. Set WEBAGENTS_SECRETS_REQUIRE_KEYSTORE=1
to refuse this fallback.
```

Check it from code with `secrets_status`, or:

```typescript tab="TypeScript"
const status = (await secrets.getStore()).status();
if (!status.keystore) {
  console.warn(status.warning);
}
```

```python tab="Python"
status = secrets.get_store().status()
if not status["keystore"]:
    print(status["warning"])
```

### Refusing it

Set `WEBAGENTS_SECRETS_REQUIRE_KEYSTORE=1` and opening a store throws instead
of falling back. Use it in a deployment that would rather fail to start than
write a bearer to disk.

## Reading a secret back

`secrets_get` reports whether a secret exists. It does not return the value.

That is deliberate. The consumer of a credential is code, and code calls
`store.get(name)` directly. Returning the value from a tool puts it in the
conversation, in whatever the conversation is persisted to, and in the
inference provider's records, which are three copies in places nobody thinks
to rotate. `reveal` exists for the exception and is refused unless the
developer passed `allowReveal` / `allow_reveal` when constructing the skill,
so revealing is a decision made in code rather than one the model makes.

## Deleting

`secrets_delete` removes the secret from every backend, including a plaintext
copy left in the file from a machine that had no keystore at the time. A
delete that cleared only the active backend would leave exactly the copy the
caller believed they had removed.

## Persisting the platform bearer

`registerWithPlatform` / `register_with_platform` return a platform bearer for
the agent. Hand them a store and they keep it, then read it back on the next
start instead of registering again.

```typescript tab="TypeScript"
import { serve, registerWithPlatform } from 'webagents';
import { openSecretStore } from 'webagents/skills/secrets';

const server = await serve(agent, { port: 8000, basePath: '/agents/my-agent' });
const secretStore = await openSecretStore({ namespace: 'my-agent' });

const registration = await registerWithPlatform(server.identity, {
  secrets: secretStore,
});

console.log(registration.reused ? 'reused a stored bearer' : 'registered');
```

```python tab="Python"
from webagents.agents.skills.local.secrets import open_secret_store
from webagents.server.core.registration import register_with_platform


async def register():
    secret_store = open_secret_store(namespace="my-agent")
    result = await register_with_platform("my-agent", secrets=secret_store)
    print("reused a stored bearer" if result["reused"] else "registered")
```

The store is optional at every step. With no store, registration behaves as it
always has: it mints a fresh bearer each time and hands it back. Nothing about
the keystore is a prerequisite for running an agent.

> [!IMPORTANT]
> That bearer is worth protecting more than an ordinary API key. It is valid
> for seven days, carries `agents:own` on the agent account, and is issued
> without a `jti`, so there is no revocation lever for it: rotating the key
> published on your agent card does not invalidate a bearer already minted,
> because nothing on the bearer's validation path reads that key. Deleting or
> suspending the agent account does still take effect. Treat a leak as good
> for a week, keep the bearer in a keystore, and prefer reusing a stored one
> over minting a new week-long credential on every restart.

A stored bearer is reused only while its `exp` is more than five minutes away.
Past that it is deleted and a new one is minted. The check reads `exp` locally
and cannot see a suspended principal, so pass `refresh` when a stored bearer
has stopped working.

## Tool Reference

All five tools are `owner` scope. A secret store readable by a counterparty
agent is not a secret store.

Every result carries `backend` (`keystore` or `file`), `keystore` (boolean),
and, on the file backend, `warning`.

### `secrets_set`

Store a named secret.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | str | Yes | — | 1 to 128 characters from A-Z a-z 0-9 `.` `_` `-` |
| `value` | str | Yes | — | The secret value. Empty values are refused. |

### `secrets_get`

Report whether a named secret exists.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | str | Yes | — | Secret name |
| `reveal` | bool | No | `false` | Return the value too. Refused unless the skill was constructed with `allowReveal` / `allow_reveal`. |

### `secrets_list`

Names of stored secrets. Never values. On the keystore backend the result
carries `complete: false`, because a keystore cannot be enumerated portably
and the skill lists only the names it recorded itself.

### `secrets_delete`

Remove a named secret from every backend.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | str | Yes | — | Secret name |

### `secrets_status`

Where secrets are being stored and why. Returns `backend`, `keystore`,
`namespace`, and on the fallback `reason`, `path` and `warning`.

## Configuration

| Option | Environment variable | Default | Description |
|---|---|---|---|
| `namespace` | `WEBAGENTS_SECRETS_NAMESPACE` | agent name, then `webagents` | Collision boundary between agents on one machine |
| `requireKeystore` / `require_keystore` | `WEBAGENTS_SECRETS_REQUIRE_KEYSTORE` | `false` | Throw instead of falling back to a file |
| `secretsDir` / `secrets_dir` | `WEBAGENTS_SECRETS_DIR` | `~/.webagents/secrets` | Where the fallback file lives |
| `backend` | `WEBAGENTS_SECRETS_BACKEND` | `auto` | `file` skips the keystore probe and uses the file deliberately |
| `allowReveal` / `allow_reveal` | — | `false` | Permit `secrets_get` to return a value |
| `quiet` | — | `false` | Suppress the log warnings. Does not suppress the `warning` field on results. |

## See also

- [Self-Registration](../../guides/self-registration.md) for the registration
  flow this skill persists the bearer for
- [AOAuth](../../protocols/aoauth.md) for how the agent's signing key works,
  and where it is kept
