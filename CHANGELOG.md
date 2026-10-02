# Changelog

Both packages are versioned independently (see [RELEASE.md](RELEASE.md)), so
every entry is tagged **[py]**, **[ts]** or **[both]**.

Entries describe each change in terms of what a developer has to edit, because
most of them are breaking.

## 0.3.9 (2026-10-02)

### Fixed

- **Several piped lines reach the chat** [py]. A pipe delivers its lines in
  one read, and the chat kept the first and dropped the rest:
  `printf '/status\n/context\n/exit\n' | webagents` ran `/status` alone.
  Every line is read now, as in the TypeScript chat.

## 0.3.8 (2026-10-02)

Many changes below are breaking: public APIs are removed, and defaults that
used to be open are now closed. Each breaking entry says what changed and what
to edit.

**This version needs the matching Robutler platform.** Several features call
platform routes, or read fields the platform sends, that ship with the same
round of platform changes. Against a platform without them the SDK fails
closed:

- A Portal Connect turn runs as the caller the platform asserts. Without the
  assertion every relayed turn runs anonymous, so an owner's own shell and
  filesystem tools are absent on relayed turns.
- Every settle carries an `Idempotency-Key`. A platform without the settle-key
  store answers a keyed settle `500` rather than charging unguarded.
- Caller-scoped memory on Robutler, x402 credit payments on priced endpoints,
  channel senders, TrustFlow lookups and the budget tree each need their
  platform routes, and fail with a sentence naming what is missing.

### Breaking

#### Agent files and the CLI

- **A new agent names no model when you have no provider key** [both].
  `init` and the chat's `/agent new` wrote `model: openai/gpt-4o-mini` and
  listed `openai` whatever the machine had; with no OpenAI key that model ran
  through Robutler, and when that failed `/model` refused every other
  provider. With no key the file now names no model and no provider skill,
  and says so in two comment lines: it runs as the chat does, on Robutler's
  choice (`auto/balanced`) until a key is set, then on that key's default
  model. With a key, `init` names that provider's default model as before.
  `init --json` answers `"model": null` for a file that names none, and its
  last line no longer tells a signed-in person to sign in.

- **Agent files are checked strictly** [both]. A file whose front matter is
  not valid YAML (reported with its line and column), has an unknown top-level
  key, uses the old string form of `cron:`, or is a symbolic link is refused
  when it loads, in the chat, `serve`, the daemon and every other command. The
  TypeScript loader used to drop an unknown key or a string `cron:` without a
  word, and read a CRLF file as an agent named `default` with no tools; it now
  reads CRLF, and refuses the rest with the Python loader's sentences. The keys
  that take effect are `name`, `description`, `namespace`, `intents`, `model`,
  `fallback_models`, `skills`, `agent_skills`, `access`, `sandbox`, `cron`,
  `observability` and `scopes`; `tools`, `visibility`, `version`, `author`,
  `tags`, `mcp_servers` and `watch` still load and do nothing, and
  `webagents doctor` says so.

- **`cron:` is a list of named schedules** [both]. The string form named no
  prompt and no target, and neither daemon ran it; it is refused now, with a
  sentence showing the list form:

  ```diff
  - cron: "0 9 * * *"
  + cron:
  +   - name: daily
  +     schedule: "0 9 * * *"
  +     prompt: Summarize yesterday's activity.
  +     deliver:
  +       file: reports/daily.md
  ```

  Each schedule takes `name`, `schedule` (five cron fields) or `every` (`30m`,
  `2h`, at least one minute), `prompt` or `heartbeat: true`, an optional
  `timezone` and `enabled`, and exactly one `deliver` target: `file`,
  `webhook` or `chat: owner`. The daemon's HTTP routes that added and removed
  cron jobs are gone, so schedules come only from agent files, and
  `GET /agents/cron` answers `{"schedules": [...]}`. In TypeScript,
  `DaemonClient.addCronJob` is removed, `listCronJobs` is now `listSchedules`,
  and the `CronScheduler` and `ScheduledJob` exports are removed.

- **`webagents secrets`** [both]. `set` never takes the value as an argument:
  it asks with echo off, or reads a pipe (`echo "$KEY" | webagents secrets set
  NAME`). `unset` is now `remove`; the old name still works. The same store
  holds the secrets MCP servers name as `${secret:NAME}`. In Python,
  `load_into_environment` exports provider keys only, so a stored MCP secret
  never becomes a variable the shell tool could read.

- **`webagents doctor --json` has ten rows** [both]: the `skills`, `mcp` and
  `keychain` checks joined the seven. A script that reads rows by position sees
  three more; read them by `name`.

- **Each CLI keeps its own keychain items** [both]. The sign-in, the keys
  `webagents secrets set` stores and the Secrets skill's secrets are filed
  under `webagents (Python) <namespace>` and `webagents (TypeScript)
  <namespace>` instead of the shared `webagents:<namespace>`, so macOS never
  asks one SDK about an item the other created. What to do: sign in, and store
  each key, once in each CLI you use. An item under the old name is copied to
  the CLI's own name the first time that CLI finds none of its own (on macOS,
  only in a terminal) and is left in place; `logout` and `secrets remove`
  retire it, and name it when removing it would make macOS ask. The owner-only
  file fallback is still shared by both. `service_key` / `serviceKey` return
  the new names, and `legacy_service_key` / `legacyServiceKey` the old one. In
  keystore mode each SDK keeps its own name index
  (`<namespace>.python.index.json`, `<namespace>.typescript.index.json`).

- **The chat's `/publish` asks before it updates a linked agent** [both]:
  `Update <agent> on <host> from <file>? [y/N]`. From a pipe it never updates.
  `webagents publish` on the command line behaves as before.

#### Access and the sandbox

- **The sandbox is on by default** [both]. An agent with `shell` (or SKILL.md
  scripts) and no `sandbox:` block now runs its commands confined, as if it
  declared `preset: development`. Those commands have no network, cannot reach
  local servers or listen on a port, and write only in the agent's folder and a
  private scratch folder. Commands that used to fetch packages, clone over
  https, run a dev server or talk to Docker are refused until you open them in
  the agent file: `network: hosts:` (or a group: `npm`, `pypi`, `github`),
  `network: local: true`, `network: sockets:`, or `files: write:`. To opt out,
  write `sandbox: off` (`preset: unrestricted` still works), or pass
  `--no-sandbox` for one run; either applies to the owner's commands only.
  Where srt cannot run, commands are refused with a sentence naming the
  opt-out, instead of running unconfined.
- **`shell` and `filesystem` are owner-only by default** [both]. The shell
  tool and the six filesystem tools were offered to every caller an agent
  answered, so anyone the platform let message a Portal Connect agent, or
  anyone who reached a served agent with any bearer, could run commands and
  read and write files as the developer's user. They are offered to the owner
  and admins now, in agent files and in code-built agents alike. To open them
  to other callers, name the callers in an `access:` group and grant the group
  the skills; to restore the old behaviour for every caller, anonymous ones
  included, grant the default group:

  ```yaml
  access:
    tools:
      everyone: [filesystem, shell]
  ```

  `webagents init --template tool-agent` writes an `access:` block with an
  empty `trusted` group granted both skills, so the file shows where a caller
  goes. The other host-reaching tools (`rag`, `todo`, `web`) stay open to
  other callers; the todo list is kept per caller (below).

- **The sandbox is srt, in both SDKs, and `sandbox:` is strict** [both].
  `sandbox:` is enforced by `@anthropic-ai/sandbox-runtime` 0.0.77 (Seatbelt
  on macOS, bubblewrap on Linux, a proxy for the network), run per command
  with a private settings file, a root-owned PATH, secrets withheld and the
  process group killed on timeout. TypeScript enforces the block for the
  first time (it was dropped without a word); Python's hand-written Seatbelt and bubblewrap runner is
  replaced. New `network:` list of hosts the command may reach (empty means
  none). `preset: unrestricted` is an explicit opt-out that confines nothing
  and is reported as such. A misspelled `sandbox:` key is an error with a
  "did you mean". A caller other than the owner is refused a command unless
  it runs confined. Both packages carry srt: TypeScript as a dependency,
  Python inside the package (below, "The sandbox engine ships with the Python
  package"); `WEBAGENTS_SRT_CLI` names another install. `runSandboxed` /
  `run_sandboxed(command, policy)` are exported for code that runs scripts of
  its own. The TypeScript Docker `SandboxSkill`'s tools are owner-only and
  run the configured image only; the Python one mounts the agent folder, not
  its parent.

  What to check: a TypeScript agent file with `sandbox:` is now confined, so a
  command that reached outside `allowed_folders` or the network is refused;
  on Linux both SDKs need bubblewrap, socat and ripgrep (`webagents sandbox
  setup` names what is missing). `unrestricted` used to mean files confined
  and network open; srt cannot express that, so it now confines nothing. Use
  `development` or `strict` with a `network:` list instead. `git init` and
  `git clone` fail inside an allowed folder, because `.git/config` stays
  read-only.

- **An agent's own files are read-only inside the sandbox** [both]. A confined
  command can no longer write `AGENT.md`, any `AGENT-<name>.md`,
  `WEBAGENTS.md`, `mcp.json` or anything under `.agents/skills`, beside the
  paths that were already denied (`.git/hooks`, `.webagents`, shell rc files,
  `.env` and others). On Linux, srt denies only files that exist when the
  command starts, so an `AGENT-<name>.md` a command creates is not covered
  there; on macOS the pattern is denied. A SKILL.md script sees every skill
  folder as read-only, its own included. A command that edited an agent's
  file on purpose now fails; edit the file yourself.

- **`plugin_load` loads only from the plugin directories** [ts]. `PluginSkill`
  imported any path it was given, and every caller could ask it to. Its three
  tools are owner-only, and a module loads only when its real path lies inside
  a configured `pluginDirs` entry: `..`, a symlink pointing outside, a missing
  path and the directory itself are refused, with the reason, before anything
  is imported.

- **A Portal Connect turn runs as the caller the platform asserts** [both].
  Every relayed turn ran anonymous, so an agent could not tell its owner from a
  stranger and offered every open tool to both. The platform now decides the
  caller against the agent's owner and puts it on the `input.text` frame as
  `caller: {user_id, tier, username?}`; both SDKs read that field only, never
  message content, and build the run's auth from it (`owner` for the owner,
  `user` with `user:<id>` and `user:@<handle>` principals for everyone else,
  which an `access:` block can place in groups). A frame without the field
  runs anonymous.

  **Rollout order matters.** A new SDK against a portal that does not yet
  send the assertion runs every relayed turn anonymous, so owners lose their
  shell and filesystem tools through the platform until the portal ships the
  caller assertion; nothing is opened by the mismatch. Deploy the portal
  first, or expect the owner-only tools to be absent on relayed turns in
  between.

- **Portal Connect callers with no `access:` block keep their identity**
  [py]. `portal_caller_auth` now leaves `principals` unset (`None`) rather
  than `[]` when no access block ran, as TypeScript does, so such a caller's
  memory, conversations and todo list are keyed `caller:user:<id>`. Code that
  tested `principals == []` tests `principals is None`.

- **A relayed channel sender is its own caller** [both]. A turn that reached
  the agent through a channel its owner connected on Robutler (Telegram,
  Slack, email and others) carries `caller.channel = {type, sender_id}` on the
  Portal Connect frame, for user-tier callers only. The SDKs list
  `channel:<type>:<sender id>` first among the caller's identities, so its
  memory namespace, conversations and todo list change from
  `caller:user:<id>` to `caller:channel:<type>:<sender id>`. An `access:`
  block can place senders in groups (`channel:telegram:8842`,
  `channel:slack:*`).

- **A served run no longer inherits the base context's payment or auth**
  [ts]. `serve()` built each run's context by spreading the agent's base
  context, where the UAMP transport had written one caller's payment token, so
  another caller's A2A run could be billed to it. A run's auth and payment now
  come from its own request and the configured defaults only; the A2A skill
  passes the caller's token explicitly. Code that put `auth` or `payment` on
  the base context for every run to see passes them per run instead. Python
  already built a fresh context per request.

- **The todo list is per caller** [both]. It was one list, shared by every
  caller of a served agent. The owner's list stays where it was; each verified
  caller gets its own, under `.webagents/todos/callers/`; a caller nothing
  verified is refused with `todo: nothing is kept for a caller nothing
  verified.`

#### MCP

- **Every MCP tool is named `<server>__<tool>`** [py]. The Python skill used
  the bare tool name unless it collided with one already registered, so a
  tool's name depended on which other servers the file named and on the order
  they connected in; TypeScript always qualified it. An `access.tools` rule, a
  prompt or a test that named a bare MCP tool (`query`) must name it as
  `<server>__query`.

- **The `mcp` skill fails to load, loudly, when the MCP SDK is missing** [both].
  Constructing `LocalMcpSkill` without the `mcp` package raises [py], and the
  TypeScript loader rejects the `mcp` entry with the reason [ts]; the agent
  file's loader reports it as a failed skill. Before, both started the agent
  with none of its MCP tools and a log line nobody read.

- **A stdio MCP server sees only its own environment** [both]. It starts with
  the MCP SDK's default environment (such as `PATH` and `HOME`) plus its
  resolved `env`, not the agent's whole environment, so a provider key in the
  agent's process no longer reaches every server it starts. A server that
  relied on inheriting a variable names it:

  ```yaml
  - mcp:
      search:
        command: npx
        args: [-y, some-mcp-server]
        env:
          API_KEY: ${env:API_KEY}
  ```

  The Python Docker sandbox passes `-e NAME` rather than the value.

- **`${` in MCP server settings is a reference** [both]. `${secret:NAME}` reads
  the store `webagents secrets set` writes and `${env:NAME}` the environment,
  in `env`, `headers` and the server's address, when the server connects. A
  literal `${` is written `$${`, and any other `${...}` is refused with the
  forms it takes. A reference in `command` or `args` is refused when the file
  loads, because other local accounts can read a command line; pass it
  through `env`. References resolve only in agent files you run (the `mcp:`
  entry and the `mcp.json` beside the file): an `MCPSkill` built in code
  resolves nothing unless the code passes `references`, and sends `${...}` as
  written.

- **The `notify` tool policy is never skipped** [both]. A `toolPolicies`
  entry of `notify` asks before each call through a host's approval hook. In
  TypeScript a `notify` tool with no hook is refused rather than run; Python,
  which has no approval hook, refuses a server that names `notify` when it
  loads. Python now also applies `block` and `enabledTools` when it discovers
  a server's tools, as TypeScript did, so a blocked tool is never
  registered. Use `allow` or `block` for a server both SDKs load.

#### Protocols and servers

- **A2A is v1.0, in both SDKs** [both]. The Python 0.2.1 REST binding (the
  `/tasks` routes) and `A2AUAMPAdapter` are removed, and so is the TypeScript
  `tasks/send` method. Both SDKs serve JSON-RPC at `POST /a2a` (the v1.0
  method names, `SendMessage` and the rest, and the dotted names of earlier
  drafts) and the HTTP+JSON binding (`/a2a/message:send`,
  `/a2a/message:stream`, `/a2a/tasks/...`), and a signed card at
  `/.well-known/agent-card.json` beside `agent.json`. The task routes need a
  credential and answer only the caller that created the task. A client that
  called the 0.2.1 routes sends `SendMessage` instead. New `a2a` settings:
  `peers`, `task_ttl_seconds`, `blocking_timeout_seconds`, `version`,
  `provider`, `documentation_url`, `icon_url`, `public_url` and
  `trust_record`.

- **ACP is served over stdio only** [both]. The HTTP and WebSocket endpoints
  (`/acp`, `/acp/stream`) and the Python ACP UAMP adapter are removed;
  `webagents acp [path]` serves the agent to a code editor over the Agent
  Client Protocol on stdin and stdout. The TypeScript "Agent Commerce
  Protocol" stub is retired, and the same import path
  (`webagents/skills/transport/acp`) now holds the Agent Client Protocol
  skill. `RESOURCE_NOT_FOUND` is `-32002`, as the protocol defines it. `acp`
  in an agent file no longer makes the daemon serve an HTTP endpoint.

- **A served Python agent shows its description, not its instructions** [py].
  The agent listing, `/health`, `/info` and the registration card carried the
  whole system prompt; they carry `description` now, and
  `AgentInfoResponse.description` replaces `instructions`. Read the
  instructions from the agent file, not from the server.

- **The TypeScript daemon binds loopback, and registering or removing an agent
  needs a credential** [ts]. It bound `0.0.0.0` by default, and
  `POST /agents/register` and `DELETE /agents/:name` took no credential. Pass
  `hostname` (`--host` on the CLI) to listen elsewhere, and send a credential
  to the two routes.

- **The legacy Python daemon class is closed off loopback** [py]. The
  `WebAgentsDaemon` library class's `/scan` and `POST /agents/` need a
  credential when it listens on anything but loopback. Its `AgentManager`
  loads exactly the skills a file declares, with no default shell and no
  default Google model. `webagents daemon` is unaffected.

- **`?token=` on a Python `@http` route no longer authenticates** [py]. The
  dynamic route turned a `?token=` query parameter into a bearer header for
  loopback hosts. Send the credential in `Authorization`. WebSocket
  handshakes still accept `?token=`, `?access_token=` and `?api_key=`, since a
  browser cannot set headers there.

- **`serve` and the daemon never run a caller's turn on your sign-in** [both].
  With no provider key, a served Python agent ran every caller's turn on the
  signed-in owner's Robutler credits, for any bearer string. `serve`,
  `mcp serve --http` and `webagents daemon` now run Robutler's models only
  when the agent has its own platform credential (`webagents publish`, or
  `WEBAGENTS_AGENT_TOKEN`), each call paid by the payment token the caller's
  request carries; a call with none is refused with 402. With neither a key
  nor that credential, `serve` refuses to start with a sentence naming both,
  and the daemon does not serve that agent. ACP and `mcp serve` over stdio,
  which your own editor or client starts, keep the chat's rule. To serve on
  your own key, `webagents secrets set <NAME>`.

- **`WEBAGENTS_PUBLIC_URL` alone keeps `serve` on loopback** [both]. Without
  an AuthSkill, `serve` and `mcp serve --http` listen on 127.0.0.1 whatever the
  public URL says, and say so. Put a tunnel or proxy on this machine in front
  of it, or pass `--host 0.0.0.0` to listen everywhere on purpose.

#### Payments

- **Priced `@http` endpoints answer standard x402, and the Robutler-token x402
  scheme is gone** [both]. `@pricing` stacked on `@http` now answers an unpaid
  request with an x402 v2 challenge (`PAYMENT-REQUIRED`, with the v1 body
  beside it), verifies the payment before the handler, and settles once after
  it. The private `scheme: "token"` 402 is removed from both SDKs, with
  Python's facilitator call. The TypeScript `PaymentSkill` loses
  `acceptedSchemes`, `facilitatorUrl`, `maxPayment`, `createX402Requirements()`
  and `verifyX402Payment()`, and takes `x402: {...}` instead;
  `X402_MAX_PAYMENT` is no longer read. A priced endpoint on an agent with no
  payment skill answers `503 payment_not_configured` rather than being served
  free.

  ```diff
  - new PaymentSkill({ acceptedSchemes: [{ scheme: 'token', network: 'robutler' }], maxPayment: 10 })
  + new PaymentSkill({ x402: { nonceSecret: process.env.X402_NONCE_SECRET } })
  ```

  In Python the seller is `PaymentSkillX402`, configured under `x402`. Set
  the same `X402_NONCE_SECRET` on every process that serves one agent.

- **`delegate` takes a `budget`** [both]: the credits one hop may spend, 0.1
  unless given and at most 5, with `authorized_amount` accepted as another
  name for it. A self-hosted TypeScript agent with no platform key refuses a
  paid hop, as does one whose budget the platform refuses; Python refuses
  rather than delegate without one. The result carries a receipt line and
  `data.receipt`, so TypeScript may return a structured result where it
  returned a string. `createDelegateToken` takes the budget as a fifth
  argument.

#### Removed

- **`webagentsd` is gone** [py]. It was a Python-only second name for the
  daemon. The daemon is `webagents daemon` in both packages, with the same
  flags, so a service definition that ran `webagentsd --port 8766` runs
  `webagents daemon --port 8766`. Both packages now install the same commands:
  `webagents` and `robutler`.

- **The `robutler` package is no longer a dependency** [py]. The SDK used one
  module of it, `robutler.api`, the platform API client, which now lives in
  the SDK under the same names: `from robutler.api import RobutlerClient`
  becomes `from webagents.agents.skills.robutler.api import RobutlerClient`,
  and `robutler.api.client` and `robutler.api.types` become `.api.client` and
  `.api.types` there. Code that imports `robutler` itself, counting on this
  package to install it, installs it separately now. A bare install carries
  16 fewer packages, `litellm` among them, and the `robutler` command is always
  this package's: both packages declared one, and whichever an installer wrote
  last owned it. An upgrade leaves the old package installed. `pip uninstall
  robutler` removes it, and, because both packages recorded the `robutler`
  command, removes the command too, so reinstall afterwards:

  ```bash
  pip install --force-reinstall --no-deps webagents
  ```

- **The CRM skill is gone** [py]. `CRMAnalyticsSkill` (registered as `crm`
  and `analytics`) called `/api/crm/*` routes the platform never had: writes
  came back 404, and reads with the default config failed before they were
  sent. It could not be named in an agent file. Its import from
  `webagents.agents.skills` and `webagents.agents.skills.robutler` no longer
  works. The platform's contact book is the Reach app, whose signed intake is
  the way in from outside; this SDK has no client for it.

- **The platform client's old content methods are gone** [py]:
  `list_content`, `upload_content`, `get_content`, `delete_content`,
  `update_content`, `get_agent_content` and the `content` resource. They
  called routes an API key cannot use (they take a browser session) or that do
  not exist, so none of them worked with a key. Use `upload_file`, which posts
  to `/api/content/upload`, and `list_agent_content`, `read_agent_content`
  and `delete_agent_content`, which call `/api/agents/{agent_id}/content`.

- **`PaymentX402Skill.lockPayment` is gone** [ts]. It posted `{ amount,
  audience }` to the route that locks against an existing token, so it could
  never create one and always returned `null`. A payer's token comes from a
  signed-in session, or for an agent from `/api/payments/delegate`, which the
  NLI skill uses.

- **`session` is one skill in both SDKs, and the old ones are gone** [both].
  The TypeScript `SessionSkill` was a key-value scratchpad for the model
  (`session_get`, `session_set`, `session_delete`, `session_list`,
  `session_clear`, `session_get_all`) keyed on a chat id the caller chose, so
  one caller could read another's entries; no agent file could name it. The
  Python `SessionManagerSkill` recorded every caller into one shared session
  in the project folder and served `/sessions` routes that listed, read,
  created and deleted conversations for anyone who reached the port, plus
  `/session/*` commands. Both are replaced by `SessionSkill`, the same in both:
  it keeps a served caller's conversation when the request names it (see
  Added). Code that used the scratchpad keeps that state in its own skill.

  ```diff
  - from webagents.agents.skills.local.session import SessionManagerSkill
  + from webagents.agents.skills.local.session import SessionSkill
  ```

- **The `checkpoint` skills are gone** [both]. The Python one could not be
  reached from the chat, its restore brought deleted files back and could
  write through symlinks, and it kept a git repository inside the agent's
  folder; the TypeScript one could not be named in an agent file and gave
  every caller of a served agent unscoped tools that restored any readable
  folder over the working one. Undo is the chat's now (see Added): drop
  `checkpoint` from an agent file's `skills:`, which refuses it.

- **The Python `message_history` skill is gone** [py]. It called
  `/api/chat`, which does not exist, and nothing could name it. On Robutler a
  conversation is a chat; see `- session: {backend: robutler}`.

#### Other behaviour

- **The discovery `search` tool fences text other people wrote** [both]. The
  `intent`, `description`, `bio`, `title` and `content` fields of its rows
  come back as `<untrusted>...</untrusted>`, screened (invisible characters
  stripped, links to private hosts withheld, chat-template markers removed,
  instruction-shaped text marked in a `screen` field), and the answer carries
  a `notice` key saying what the fence means. Code that read those fields as
  bare strings sees the fence.

- **`robutler -p "..." --json` prints what `webagents` prints** [ts].
  `robutler` now runs through the main CLI, as the Python one does, so its
  `--json` reply is the one `webagents -a robutler -p "..." --output-format json`
  prints, not the REPL's internal response object, and a failed request gets
  the main CLI's message and hint rather than a bare `Error:` line.

- **The namespace and publish skills find the platform as the discovery
  skill does** [py]:
  `robutler_api_url` or `webagents_api_url` in the skill's config, then
  `ROBUTLER_API_URL`, then `ROBUTLER_INTERNAL_API_URL`, then the CLI's
  `platform.url`, then `https://robutler.ai`, the TypeScript order. They fell
  back to `https://webagents.ai`, an old address that does not serve the
  platform, and sent the agent's platform key there. The namespace and publish skills read
  `ROBUTLER_API_URL` before their own config; the config now wins. The
  platform API client, given no URL, asks the same
  lookup; it tried the in-cluster variable first. The payment, auth and
  storage skills still build their own, with `http://localhost:3000` last, and
  pass it to the client.

- **`@name` in the NLI skill is an agent on the platform** [both]:
  `{platform}/agents/{name}`, the platform found by the same lookup as the
  discovery skill, after `baseUrl` [ts] or `AGENTS_BASE_URL` and
  `agent_base_url` [py]. TypeScript defaulted to `https://portal.webagents.ai`,
  which does not resolve, and Python to `http://localhost:2224`, which nothing
  serves, so `@name` reached nothing unless a base was set.

- **The UCP skill publishes and sends only addresses that serve it** [py].
  A merchant's checkout endpoint is where the agent is served: `base_url`, else
  `public_url` or `WEBAGENTS_PUBLIC_URL` plus `agent_path` plus the name, else
  that path, which a buyer resolves against the merchant. A buyer's `UCP-Agent`
  profile is its own `/.well-known/ucp` when it has a public URL, and the
  header is left out when it has none. A Robutler token is checked on the
  platform, and that handler's `spec` is its published page; it names no
  schema documents, as none exist. Each of these was an address on
  `webagents.ai`, the project's old site, and buyers following a merchant's
  profile sent their checkouts and payment tokens there.

- **The JSON storage skill works, and its documents are private** [py]. It
  stores through `/api/content/upload` and finds, reads and removes through
  the agent's own content routes, with the agent's own key and the agent id
  that key carries (`api_key` and `agent_id` in the config come first).
  Reading, updating and deleting never worked before, and every document was
  stored public, because the skill read the caller's scope from a context key
  nothing sets. A name can hold several documents on the platform: reading
  takes the newest, and an update stores the new one before removing the
  older ones. The tools no longer take a `description`, which the platform has
  nowhere to keep.

### Added

- **`webagents mcp list` and `mcp add`: the MCP servers other apps use**
  [both]. `mcp list` reads the settings of Claude Desktop, Claude Code,
  Cursor, VS Code and Windsurf (never writing to them) and shows each server's
  command or address, naming its variables and headers but never their
  values; a key-looking argument, the value after a flag such as `--api-key`,
  and a key in an address's query show as `***`. `mcp add <name> [path]
  [--from <app>]` copies one into the agent: into the agent file's own `mcp`
  block when it has one, edited line by line and read back before it is
  written, else into `mcp.json` next to it, adding `mcp` to the skills when it
  is missing. A key in the server's `env`, `headers` or address goes into this
  profile's secrets and the entry reads it as `${secret:NAME}`; VS Code's
  `${workspaceFolder}` becomes the folder and its `${input:NAME}` becomes a
  secret to set; a key on the command line is refused, since other local
  accounts can read it from the process list. `mcp remove <name> [path]`
  takes a server out of the agent file's `mcp` block or `mcp.json`, with the
  comment lines above it, and with the last one the `- mcp` entry itself; the
  secrets it read stay stored, and the command names them. A layout it cannot
  edit safely is refused with the file to edit by hand. The chat has the same
  as `/mcp list`, `/mcp add <name>` and `/mcp remove <name>`, with the names
  completed.

- **Commands in groups** [both]. `webagents --help` lists the commands under
  Chat, Build, Run, Robutler and This machine, and the chat's `/help` under
  Conversation, Agent, Limits, Account and Chat, in the same order in both
  CLIs. A chat command takes at most one word naming what to do (`/mcp add`,
  `/resume delete`), never a second level. `connect` (the old name of `chat`)
  and `templates list` still work and are no longer listed; `init --list`
  shows the templates. `init` and `/agent new` refuse the names `new` and
  `edit`, which `/agent` keeps for itself (`--json` code `reserved_name`).
  New descriptions for `chat`, `serve`, `daemon`, `mcp`, `login`, `models`,
  `skills`, `config`, `init`, `publish` and `budget`.

- **Continuing, deleting and pruning conversations** [both]. `webagents -c`
  opens the chat in the last conversation in this folder, and `-r [number]`
  in an earlier one, or with the list (both refused with `-p`, and together).
  A chat that starts new says under the banner when the last conversation
  here was used in the past day and how to continue it. `/resume delete
  <number>` deletes one after asking. `webagents conversations list [--all]`,
  `delete <id>` (the start of its id) and `prune --older-than 30d [--dry-run]`
  do the same outside the chat, asking first and needing `--yes` from a
  script, with `--json`. A conversation also kept on Robutler keeps that copy.

- **The chat's menu searches, and picks** [both]. What is typed narrows the
  menu by name and, word by word, by what each row says: `/sign` finds
  `/login`, `/resume budget sheet` a conversation, `/model 4.1` a model,
  `/help sign in` a command (each word matches the start of a word, in any
  order). `/resume` lists this folder's conversations in the menu, numbered
  as before, with when, how many messages and how each began; `↑` `↓` and
  `enter` continue one, and `/resume delete` chooses the same way. `/rewind`
  lists the snapshots, and `/model` the models the agent can switch to (its
  provider's known models, or with no provider skill every known model and the
  `auto/` tiers). `enter` on a row that completes the command runs it (none of
  those changes a file or what is stored without asking first); `enter` on a
  form, a skill, or
  a row of `/mcp add`, `/mcp remove` or `/keys remove` puts it in the box, and
  `tab` always does. `/resume` and `/rewind` chosen in the menu open their list
  instead of printing it. A command's completer is now told everything typed
  after it and answers the rows and the text they match (`Slot`). A `/resume`
  pick of 8 or more digits past the list's length is taken as the start of an
  id (the picker inserts an id's first 8 characters); a shorter number is a
  position only.

- **`/keys remove NAME`** [both]. The chat's `/keys` takes `set` and
  `remove`, as `/secrets` does; `/keys unset NAME` still works.

- **`pip install 'webagents[uv]'` brings `uvx`** [py]. The extra installs uv
  from PyPI. A stdio server whose command is `uvx` runs with that copy (the
  one next to the Python running the agent, else the `uv` package's own) when
  no `uvx` is on the `PATH`.

- **The TypeScript chat suggests from history** [ts]. As in the Python chat,
  the box shows in grey the rest of the newest line from history that starts
  with what was typed; `→`, `tab`, `ctrl+e` and `ctrl+f` take it, and `alt+f`
  or `alt+→` take one word of it.

- **Ready-made skills in `skills/`** [both]. Thirty-three SKILL.md skills:
  thirty published by their authors on ClawHub under MIT-0 (among them
  architecture decision records, API design and contract reviews, bug
  reports, changelogs, code and SQL review, commit messages, data analysis
  and experiment design, database schemas, diagrams with draw.io and
  Mermaid, document conversion with pandoc, slides with Marp, accessibility
  audits, literature reviews, editing, observability, Kubernetes triage,
  dependency upgrades, a security review and Word documents), each read in
  full and checked for copied text before it was taken; and three from Anthropic's skills repository that keep its Apache
  License 2.0 (frontend design, building MCP servers, testing web apps with
  Playwright). Install one with `webagents skills add robutlerai/webagents
  --skill <name>`. The folder is MIT No Attribution except those three
  (`skills/NOTICE`); `skills/PROVENANCE.json` names each author and source,
  and both SDKs' tests hold every file to its recorded checksum.

- **The tool-round cap is never a silent stop, and it can be set** [both].
  A turn that reaches its limit makes one more model call with tools off and
  a wrap-up message, so the model answers from what it gathered, and the turn
  ends with the finish reason `tool_round_limit` in every mode: `-p` prints
  the answer, says the cap on stderr and exits 1 (`json` and `stream-json`
  carry `finish`), `serve` responses carry `webagents_finish`, ACP answers
  `stopReason: max_turn_requests` with the reason in `_meta`, and the
  daemon's run record keeps `finish`. When that last call brings no answer
  the chat says "The agent stopped after N tool rounds without an answer."
  The interactive chat then asks "Used N tool rounds. Keep going? [Y/n]":
  yes goes on from the conversation with a fresh budget, no ends the turn;
  nothing else ever asks. The limit is, most specific first, the chat's
  `/rounds <n>` (`--save` keeps it in the agent file, bare `/rounds` shows
  it and where it came from), the `--max-tool-rounds <n>` flag (the chat,
  `-p`, `serve` and the daemon, hoisted like `--profile`), the agent file's
  new `max_tool_rounds` key, and 50; a whole number from 1 to 1000, refused
  with a sentence otherwise, and never raised by a served agent's callers.
  `/status` shows it. A turn that makes the same tool call with the same
  arguments three times stops early the same way, with the reason
  `tool_loop` and a sentence naming the tool. In TypeScript the reason
  travels on `response.done` (`finish_reason`, `finish_rounds`,
  `finish_tool`) and `RunResponse.finish`; only a last call that brings no
  answer still ends with the `max_iterations` error, its finish in
  `details.finish`.

- **Keychain dialogs on macOS are explained, and never waited on** [both].
  Before a read that may make macOS ask (the interpreter changed since the
  item was last used, or an item from an earlier version), the CLI prints four
  lines saying what the dialog is and what to click, once per run and only in
  a terminal. `serve`, the daemon, `mcp serve --http`, a pipe and CI never make
  such a read: they use `WEBAGENTS_TOKEN` or the owner-only file, or stop with
  one sentence naming `webagents whoami`, which, run in a terminal, reads what
  they could not so macOS asks there, once. Python makes every keychain call
  with macOS's dialogs switched off first (`SecKeychainSetUserInteractionAllowed`,
  per process), so a call that would ask returns an error instead. TypeScript
  cannot switch them off, so it decides from `keychain.json`, a record of
  which program last used each item (0600, names only, in the profile's
  folder), and an attribute-only lookup, and bounds a read with nobody to
  answer by a timeout. Where macOS has no default keychain for the current
  HOME, the store uses the owner-only file instead of raising macOS's
  "keychain cannot be found" prompt. `doctor` has a `keychain` line and the
  chat's `/status` a `Keychain` row: where the sign-in and keys live, which
  program last used them, and whether the next use will ask. New page:
  Keychain dialogs on macOS.

- **The sandbox engine ships with the Python package** [py]. srt 0.0.77 and
  its four dependencies come inside the wheel, the files npm publishes,
  unmodified (`webagents/sandbox/sandbox_engine/`, with each file's sha256 and
  the licenses). Node is `node` from PATH when it is 20.11 or later, otherwise
  the one the new `nodejs-wheel-binaries` dependency installs on macOS and
  Linux (arm64 and x86_64), so `pip install webagents` alone confines commands
  on a machine with no Node. That dependency is a 56 MB (macOS) to 61 MB
  (Linux) download. Nothing is downloaded when a command runs.
  `WEBAGENTS_SRT_CLI` and `WEBAGENTS_SRT_NODE` still override. What to check:
  a global `npm install -g @anthropic-ai/sandbox-runtime` is no longer looked
  for; set `WEBAGENTS_SRT_CLI` to keep using one.
- **`webagents sandbox setup`** [both]. A check, not an installer: it runs one
  confined `true` through the engine and says what this machine lacks, with
  the fix: the missing Linux programs with the distribution's install line,
  what a container must allow bubblewrap, WSL 2 on native Windows, a macOS
  `TMPDIR` too long for srt's socket. It exits 1 when shell commands would be
  refused, and `--json` gives the checks as one document.
- **Granular `sandbox:` settings** [both]. `files: {write, read, deny}`,
  `network: {hosts, local, sockets}` and `env` say exactly what confined
  commands may touch. The flat keys stay as aliases (`allowed_folders`, a bare
  `network:` list, `env_passthrough`), and unknown keys are refused with a
  did-you-mean. Host groups `npm`, `pypi` and `github` expand to the hosts
  each needs.
- **The chat asks about a refused host** [both]. When a confined command is
  refused a host, the interactive chat asks the owner: allow once, always for
  this agent (written into the agent file), or no. Other modes refuse, and the
  shell tool adds one sentence naming the setting that opens what was refused.
- **`--profile` and `--no-sandbox` work after the subcommand** [both], so
  `webagents login --profile local` is the same as
  `webagents --profile local login`.
- **SKILL.md skills** [both]. Skills in the Agent Skills format (a folder with a
  `SKILL.md` and, optionally, scripts and reference files) load and run the
  same way in both SDKs. `.agents/skills/*/SKILL.md` beside an agent file is
  found without being named, and a new top-level `agent_skills:` lists other
  folders; `skills:` keeps meaning coded skills. The model sees each skill's
  name and description and loads one when it needs it (`activate_skill`,
  `read_skill_file`, `run_skill_script`). The three tools are owner-only until
  `access: tools:` names `agent_skills`. Scripts always run in the sandbox,
  under the agent's own `sandbox:` or `strict`, never `unrestricted`.
  `webagents doctor` has a `skills` check, and `skills list` shows both kinds.

- **`webagents skills add <source>`** [both] installs SKILL.md skills from
  `owner/repo`, a git URL, a `tree` or `blob` URL, a `file://` URL or a
  folder, with `--skill <name>` to pick one. It clones at one commit (at most
  10 MiB fetched, 25 MiB installed, 1,000 files), refuses symbolic links,
  lists every file with scripts and binaries flagged, and asks (`--yes` where
  nothing can answer). The skill goes into `.agents/skills/<name>/`, named by
  its own `name:`, and `.webagents/skills.lock` records the source, commit and
  a SHA-256 digest. `skills remove` takes out only what the lock recorded.
  `skills add` and `skills remove` never write through a symbolic link, into a
  file outside the folder or into another user's file.

- **`webagents skills add <names...>` and `webagents skills remove
  <names...>`** [both]. They change the `skills:` list of this folder's agent
  file (`-a <agent>` for another) and nothing else: comments, the other keys,
  a skill's own settings, the instructions and the file's line ends stay as
  written. A name `skills list` does not show is refused with a suggestion and
  nothing is written; `remove` also takes a name the file lists that this SDK
  cannot load. After an add, the command says what a skill still needs here: a
  provider's key, or a sign-in. The same edits and words in both CLIs, pinned
  by `python/tests/fixtures/cli/skills_edit.json`.

  ```bash
  webagents skills add discovery shell
  webagents skills remove shell
  ```

- **The chat can make and change an agent** [both]. `/agent new <name>
  [chatbot|tool-agent]` makes an agent file in the chat's folder and switches
  to it; `/agent edit [name]` opens a file in `$VISUAL` or `$EDITOR`, then uses
  it; `/skills` lists, adds and removes skills, SKILL.md sources included;
  `/model <provider/model> --save` keeps a model in the file; `/reload` reads
  the file again. Each change shows what it will do and asks, runs only at a
  terminal, is snapshotted so `/undo` takes it back, and never writes through
  a link. The chat notices when the file changes outside it and says so once;
  it never reloads unasked, and asks before using a new `access:`,
  `sandbox:`, `cron:`, skill list or `mcp.json`.

- **Chat views and help** [both]. `/tools` shows who may use each tool (`only
  you`, `every caller`, `you and <groups>`); `/access` reads the `access:`
  block; `/mcp` lists the MCP servers with their tools or why they did not
  connect; `/cron` lists the agent's schedules and `/cron run <name>` runs
  one; `/memory` shows the agent's notes and `/memory forget <key>` removes
  one; `/secrets` stores and removes MCP secrets with echo off; `/status`
  shows the profile and whether the agent is published; `/publish --dry-run`
  sends nothing. `/help` is grouped, `/help <command>` shows a command's
  forms, a mistyped command gets a "did you mean", and a failing command says
  why without ending the chat. The `/` menu completes arguments (agents,
  skills, key names, schedules, notes); `enter` inserts a value and never runs
  the command.

- **`/undo` and `/rewind` in the chat** [both]. When the agent can change
  files (`filesystem` or `shell`), the chat snapshots its folder before each
  message; `/undo` puts back what the last message changed (edited and deleted
  files come back, new ones are removed) after showing the list and asking,
  and `/rewind` lists the folder's snapshots and puts one back. A restore
  snapshots first, so it can be taken back too. Snapshots live under the
  profile (`~/.webagents/checkpoints/<folder>/`), never in the folder; they
  skip `.git`, `node_modules` and the like, files over 10 MB, and keep
  symlinks as links. Off in the home folder and above. Both CLIs read and
  write the same snapshots.

- **Conversations on Robutler** [both]. With `- session: {backend: robutler}`
  in the agent file's `skills:`, the chat also keeps each conversation as your chat with
  the agent on Robutler (your messages as you, its replies as the agent,
  recorded without waking it or notifying you), so it shows in your chat list
  and `/resume` on another machine lists and continues it, as it does a chat
  started on the web. Needs `webagents login` and a published agent
  (`webagents publish`); otherwise conversations stay on this machine and the
  chat says which is missing. Uses the portal's
  `/api/agents/{id}/conversations` routes, which ship with this change.

- **A served agent keeps its callers' conversations** [both]. With `session`
  in the agent file, `serve` and the daemon keep each verified caller's
  conversation when the request names it with `metadata.session_id`: the
  owner's beside the chat's (so `/resume` finds them), anyone else's under
  `callers/<hash>/`, one namespace per caller. Anonymous callers are not kept.
  `session` is now in both SDKs' `skills list`.

- **`discovery` searches as you in the chat** [both]. In the chat and with
  `-p`, an agent with no platform credential of its own (no signing identity
  the platform can check, no key from `webagents publish`) searches with your
  `webagents login` sign-in, so a first search no longer waits on a publish.
  `serve` and `webagents daemon` never use it: everyone who calls a served
  agent would search as you. Publishing intents always speaks for the agent.
  Signed out, the search says to sign in or publish.

- **Schedules that run and deliver** [both]. `webagents daemon` runs the
  `cron:` schedules of the agents it serves, as their owner, and delivers each
  result to a file in the agent's folder, a webhook (retried with backoff, and
  signed with the agent's Web Bot Auth key when it has one) or your chat with
  the agent on Robutler. `heartbeat: true` runs the agent's standing
  instructions and delivers nothing when there is nothing to report. A
  schedule never runs twice at once, a restart does not repeat a slot, and a
  missed slot runs once. `webagents cron list` shows every schedule under a
  folder, and `webagents cron run <agent> <name>` runs one now. The Python
  daemon runs real agent turns for schedules.

- **An A2A client, and signed cards** [both]. The A2A skill's `callPeer(url,
  input)` / `call_peer(url, input)` calls another A2A agent: it reads the
  peer's card, sends `SendMessage`, falls back to `message/send`, polls until
  the task settles, and refuses redirects. The bearer for a peer comes from
  the `peers` setting, by the longest matching URL. The served card is
  signed: a JWS (JSON Web Signature) over the canonical card, made with the
  agent's Ed25519 signing key. A TypeScript agent and a Python agent call each
  other and verify each other's cards. A Python server serving one agent at the
  origin (`create_server(root_agent=...)`) serves its card and `POST /a2a` at
  the root.

- **`webagents acp [path]`** [both]: the agent in a code editor's agent
  panel, over the Agent Client Protocol on stdio. It answers `initialize`,
  `authenticate` (a `terminal` login that runs `webagents login`),
  `session/new` (attaching the MCP servers the editor names), `session/prompt`,
  `session/cancel`, `session/load` and `session/list`; sessions are kept under
  the profile. The agent runs as its owner, streams text, thinking, tool calls
  and plans, and asks the editor before a tool edits, deletes, moves or runs
  anything. See the Transports page for the Zed and JetBrains settings.

- **`completions`, `a2a`, `realtime` and `acp` in TypeScript agent files**
  [ts], with the same names and settings as Python, and **Realtime in
  `serve()`** [ts]: a TypeScript agent with the Realtime skill answers a voice
  session at `/realtime`, behind the credential floor and the origin check.

- **`mcp` in TypeScript agent files** [ts]. `skills: [- mcp: {...}]` loads the
  same two shapes Python accepts (servers at the top level, or under
  `mcpServers`), with stdio, Streamable HTTP and SSE servers; a bare `- mcp`
  reads `mcp.json` next to the agent file. `@modelcontextprotocol/sdk` is a
  declared dependency, and a load that fails names what failed instead of
  quietly disabling the skill. Both SDKs run
  `python/tests/fixtures/mcp_tool/config_shapes.json`.

- **Streamable HTTP for MCP servers** [py]. `transport: http` on a remote
  server, or `auto` (unset) to try it before SSE, as TypeScript does.

- **`webagents mcp serve [path]`** [both]. The agent's tools to an MCP client:
  over stdio by default (how Claude Code, Codex and OpenCode start a local
  server), or over stateless Streamable HTTP at `/mcp` with `--http <port>`
  (`--host` to accept other machines). Over stdio the caller is the owner; over
  HTTP the caller needs a credential, the agent's auth skills and `access:`
  block identify it, and it lists and calls only the tools it may use, through
  the agent's own tool path (hooks, pricing). Library entry points:
  `serveMcpStdio` / `serveMcpHttp` from `webagents/server` [ts],
  `webagents.server.mcp_server` [py]. Pinned across SDKs by
  `python/tests/fixtures/mcp_tool/serve.json`.

- **Secrets for MCP servers** [both]. `${secret:NAME}` and `${env:NAME}` in a
  server's `env`, `headers` and address (see Breaking). A value that looks
  like a key written into a server's `env` or `headers` (`sk-`, `ghp_`,
  `github_pat_`, `xox`, `AKIA`, a long `Bearer`) is warned about when the file
  loads, with the `${secret:...}` name and the `secrets set` command to move
  it. A reference that cannot be resolved stops that one server with a
  sentence naming it; values are masked in every message. `serverReport()` /
  `server_report()` describe each server, masked, and `webagents doctor` has
  an `mcp` check that connects each one and names what is missing.

- **The `memory` skill** [both]. `- memory` (or `- memory: {local, portal,
  notes_budget}`) gives an agent notes that last between conversations, kept
  per verified caller: the owner reads everything, a verified caller reads its
  own notes and the shared ones, and a caller nothing verified reads the
  shared notes and writes nothing. No tool argument can widen that. The tools
  are `memory_search`, `memory_read`, `memory_write`, `memory_forget` and
  `memory_list`; `memory_write` takes a one-line `description`, and
  `memory_read` gives a note in full. Each conversation starts with a
  `## Memory` block that is an index of the notes, as Claude Code's
  `MEMORY.md` is: one line per note, its key and its description (or its
  first line), newest first, within `notes_budget` characters, the rest
  counted; it is fixed for the conversation, so the provider's prompt cache
  holds. On this machine notes are Markdown files under `.webagents/memory/`,
  owner-only, the description in the front matter, with a full-text index
  that also searches descriptions; with `portal: true` they are also kept on
  Robutler, synced by entry, descriptions included. An unknown setting is
  refused. `webagents serve` warns when an agent has memory and nothing to
  verify its callers. The summary of each compacted conversation is kept as a
  searchable `episode-...` entry in the caller's namespace (`on_compaction`),
  and left out of the index.

- **Context compaction** [both]. Every agent keeps a conversation within its
  model's context, with or without the `memory` skill, cheapest step first:
  the long outputs of earlier tool calls are cleared (a one-line note stays);
  if that is not enough, everything before the recent part becomes one
  summary by the model, an earlier summary rolled in; and if no summary can be
  made, the oldest whole turns are dropped with a note. A tool call is never
  separated from its results and the system prompt is never touched. The
  agent file's `compaction:` block sets it: `auto` (true), `at` (0.8 of the
  context window, or a number of tokens), `keep` (0.25), `hard` (0.95),
  `clear_tool_results` (true), `model`, `instructions` and `window`; an
  unknown key or a bad value is refused. The chat compacts before a message
  would pass `at`, says so, and keeps the whole conversation in its file as
  `transcript`; `/compact [focus]` compacts now, `/context` says how full the
  context is, and the footer shows `context N%` past half. `/resume`,
  `conversations list`, `/status` and the start hint count and preview a
  compacted conversation from its `transcript`, so it keeps its size and its
  first line. A run compacts by
  itself past `hard` inside one long turn, leaving the turn in progress
  whole. For other hosts, `compact` and `compactIfNeeded` (Python
  `compact_if_needed`) return the outcome without changing their input, the
  policy is `compactionPolicy` (`compaction_policy`), and a skill's
  `onCompaction` (`on_compaction`) hears of each compaction once. The memory
  skill's `compaction: {threshold}` setting is still read, as `at`, when the
  agent file has no `compaction:` block; its `keep` and `summarizer` are no
  longer used.

- **TrustFlow lookups** [both]. TrustFlow is a platform service: Robutler
  computes an agent's score, and the SDK looks it up. `- trust` gives the model
  a `trust` tool; an `access:` group can admit agents by score
  (`trust: {min: 0.6, topic: billing}`); discovery results carry `trustflow`;
  and `- a2a: {trust_record: true}` adds the agent's signed TrustFlow record to
  its A2A card.

- **Priced endpoints over x402** [both]. A priced `@http` endpoint offers
  Robutler credits (`robutler-credits`, with Robutler as the seller of record)
  and, on a self-hosted agent that names a receiving address (`x402: {chain}`
  or `X402_PAY_TO`), the x402 chain schemes `exact` and, for a metered
  endpoint, `upto`, through a configured facilitator. A challenge is bound to
  the request it answered, and a payment pays for one request, across
  processes. A metered endpoint (`lock` above the per-call price) settles the
  amount its handler names. The challenge carries the endpoint's description
  and, when declared, its Bazaar discovery entry. MPP (Machine Payments
  Protocol) challenges can ride on the same `402`.

- **Every settle carries an `Idempotency-Key`** [both]. The platform records
  a settle under the key (`POST /api/payments/settle`, header or the
  `idempotencyKey` body field) and answers a repeat with the same key and the
  same lock from that record, charging nothing and saying so
  (`Idempotent-Replayed: true`, `replayed: true` in the body, kept by
  `readSettleResult` / `read_settle_result`). The payment skills derive
  `settle:<lockId>:<purpose>` for the settles their lifecycle names (`usage`,
  `agent_fee`, `release`), so a second finalize on one context is a replay;
  a priced tool's per-call settle adds the tool call's id
  (`settle:<lockId>:<purpose>:<callId>`), and a settle nothing names (the
  legacy `redeem`, a raw client call) gets a key minted for that call. The
  platform answers `422` when a key comes back with a different request, and
  `409` when it names another lock. The Python client sends a
  settle again after a 5xx or a dropped connection only because it carries
  the key, and never resends a POST without one. The derivation is pinned by
  `python/tests/fixtures/payments/settle_idempotency.json`, which both suites
  and the portal read. A platform without the settle-key store answers a
  keyed settle `500` by name rather than charging unguarded.

- **`webagents budget <token_id>`** [both] shows the budget tree of a run: a
  payment token and every hop's share of it.

- **OpenTelemetry** [both]. `observability: {otel: true}` in an agent file, or
  `WEBAGENTS_OTEL=1`, records each run as spans that follow the GenAI
  semantic conventions (`invoke_agent`, `chat <model>`, `execute_tool`,
  `settle_payment`) and the `gen_ai.client.token.usage` and
  `gen_ai.client.operation.duration` metrics. It adds no dependency: spans are
  recorded when the OpenTelemetry API is installed, and nothing happens
  otherwise. No message text, tool arguments or results are put in an
  attribute.

- **Cost in the chat** [both]. The footer, `/status` and the line printed on
  leaving show what a conversation cost in credits: the platform's number when
  it reports one, or an estimate from the provider's list price, marked `~`,
  with your own key. Local models and models the price table does not know
  show tokens alone. A resumed conversation keeps its cost.

- **Local models through Ollama** [both]. `ollama/<model>` and `- ollama` run
  a model on an Ollama server (`OLLAMA_BASE_URL`, `http://localhost:11434/v1`
  unless set), with no key. `webagents models` marks it ready when the server
  answers, and `webagents doctor` says when nothing answers or the model is
  not pulled.

- **`fallback_models`** [both]: models to try, in order, when the agent's
  `model` does not answer (a timeout, no answer, or 408, 429 or a 5xx), before
  it has started to answer. Each switch leaves a note in the transcript; a
  served agent's callers are never told the provider's address. A fallback
  that cannot be built here is left out and reported. Any other failure is
  reported as it is.

- **Smaller CLI additions** [both]. `webagents doctor -a <name>` checks
  another agent in the folder. `--json` covers `secrets list` and the refusal
  of an unknown agent (`agent_not_found`). The `tool-agent` template writes a
  description and a `sandbox:` block. The chat's typed-line history is kept
  per profile and shared by both CLIs.

### Changed

- **A confined command is never refused for its name** [both]. The command
  list refused `sleep` and `mkdir` in both CLIs, `python3` in TypeScript and
  `rg` in Python, before the sandbox was reached. When a command runs
  confined, the kernel decides what it may touch, and only a name the shell
  skill's `blocked_commands` lists is refused. With `sandbox: off` or
  `--no-sandbox` one list and one wording hold in both CLIs; `./script`
  names are refused there in Python as they always were in TypeScript (list
  them under the shell skill's `allowed_commands`).
- **The opt-out is said one way** [both]. The load line for `sandbox: off`
  and `--no-sandbox` is now `/sandbox`'s own headline and fix ("Remove
  `sandbox: off` ..."), where it said "Use `development` or `strict`". In the
  chat it is a notice, wrapped at words; outside the chat it stays on
  standard error; `doctor` says it once, in its sandbox line.
- **A refused command says what this machine lacks** [both]. When the sandbox
  cannot run, the refusal names the missing piece and its fix on this
  machine, then `webagents sandbox setup` and the opt-out (`--no-sandbox`,
  `sandbox: off`), instead of advising `npm install -g`. `doctor`'s `sandbox`
  line and the chat's `/sandbox` give the same fix.
- **Completions responses have the OpenAI shape** [both]. Python answers
  carry `object: chat.completion` and a timestamp in `created`, and a
  handed-back tool call has no `delta`; TypeScript stream chunks carry `id`,
  `object`, `created` and `model`. `usage` adds up every model call of a turn,
  on the served path too.

- **Clearer failures** [both]. A port already in use is one sentence and exit
  1, with no stack trace. `webagents -p` in Python reports an MCP server that
  failed to connect on standard error, as TypeScript does, and neither
  `doctor` prints raw log lines above its report (the Python chat logs to
  `~/.webagents/logs/repl.log` unless `WEBAGENTS_DEBUG` is set).
  `webagents cron list` exits 1 when an agent file is refused. The Python
  daemon's log lines carry no emoji.

### Fixed

- **Confined commands run on Linux under the default and `strict` sandbox**
  [both]. With reads scoped to the declared folders (`strict`, and an agent
  with no `sandbox:` block), an SDK installed outside them (a virtualenv, a
  project's `node_modules`) hid srt's own seccomp helper from the sandbox, and
  every command failed with `apply-seccomp: No such file or directory`. The
  helper's folder is now readable there.
- **`network: local: true` reaches local servers on Linux** [both]. It opened
  nothing there: a command has its own network namespace on Linux. It now
  lists `localhost` and `127.0.0.1` for the sandbox's proxy, so curl, pip, npm,
  Node and Python reach a server on this machine; a raw socket to 127.0.0.1
  still does not.
- **The `.env` hint appears on Linux** [both]. A confined `cat .env` answers
  "Permission denied" there, which the hint about `sandbox: env:` did not
  recognise.

- **The chat's history is this folder's** [both]. `↑` and the grey
  suggestion offered every line typed in any folder, so another project's
  prompts (and those of scripted runs sharing the profile) came back in this
  one. Each line is now kept with the folder it was typed in (a
  `# folder <path>` line under its time stamp in the `history` file) and a
  chat recalls only its own folder's lines. Lines from before this change
  carry no folder and are not recalled.

- **A recalled command does not open the menu** [both]. `↑` onto a line from
  history that starts with `/` opened the command menu, which then took `↑`
  and `↓` for itself, so walking the history stopped at the first command.
  A recalled line now keeps the menu closed until it is edited or the cursor
  moves; `↑` and `↓` carry on through the history.

- **An MCP server whose command is not installed says what to install**
  [both]. A server whose `command` is missing showed the operating system's
  words (`spawn uvx ENOENT`, `[Errno 2] No such file or directory: 'uvx'`).
  `/mcp`, `doctor` and `list_mcp_servers` now say `uvx is not installed or not
  on PATH:` with where it comes from (uv for `uvx`, Node.js for `npx`, and the
  like), and `doctor`'s fix line says the same.

- **An MCP server that stops at start says why** [both]. A stdio server that
  ended before the MCP handshake was shown in `/mcp` and `doctor` as
  `not connected: Connection closed` (TypeScript: `MCP error -32000:
  Connection closed`), and the reason was only in the server's own stderr log,
  which nothing named. The row now says `the server stopped before it
  answered:` with the last error line the server wrote (a Python traceback's
  exception line, Node's `Error...` line, uv's resolution failure) and that
  log's path, `logs/mcp-<name>.log` in the profile's folder. The agent's
  `list_mcp_servers` tool names every server that did not connect, with the
  same reason (Python: a "Not connected:" list; TypeScript: `not_connected`),
  where it said only that none were connected. The sqlite examples in the docs
  now run `uvx --with "mcp<2" mcp-server-sqlite`: the server was written for
  `mcp` 1 and stops at start under `mcp` 2 (and one example passed `--db`
  where the server takes `--db-path`).

- **Tab and → take the grey suggestion from history** [py]. The chat's box
  showed a suggestion from history after what was typed, and no key took it:
  tab was prompt_toolkit's `menu-complete`, which does nothing without a
  completer, and the keys that take a suggestion (→, ctrl+e, ctrl+f, and
  alt+f for one word) are loaded by `PromptSession`, never by the box's own
  application. Tab (with the command menu closed) and those keys now take it.

- **No blank rows under the prompt after a reply with a code block** [py].
  rich's live region is redrawn inside every print from the view it built at
  its last refresh, so a finished code block was followed, for one frame, by
  the region as it had last looked, still holding that block's last lines; the
  next refresh drew the region at its real height, and the rows between stayed
  erased. At the bottom of the terminal they showed as blank rows under the
  prompt box (six in a replay of a streamed code block). The chat now records a
  block as printed and refreshes the region before the block prints. The
  TypeScript chat draws its own region and never left them.

- **An edit-and-rerun cycle is no longer a tool loop** [both]. A turn that
  ran `python3 analyze.py`, rewrote the script with a file tool and ran it
  again, twice, was stopped with `finish: tool_loop` at the third run, and
  `-p` exited 1 under a complete answer: the repeat detector counted only the
  tool's name and arguments. A repeat now counts only when nothing changed in
  between: the same call in a row, with the same result. Any other call
  between two repeats (an edit, a write, another command) or a different
  result starts the count over; a true loop, the same call over and over with
  nothing new coming back, still stops after three. The wrap-up message and
  the chat's sentence say so ("3 times in a row with the same arguments and
  got the same result each time").

- **The MCP client says when a server wants a credential** [both]. Pointed
  at a Streamable HTTP server that answers 401, the Python `doctor` printed
  `unhandled errors in a TaskGroup (1 sub-exception)` (the transport's
  exception group, never unwrapped) and the TypeScript one printed the
  transport's raw line; both fix lines said to fix the server's entry. An
  exception group is now unwrapped to its root cause everywhere the skill
  reports a connection failure, and a 401 or 403 is said plainly: the server
  needs a credential, a bearer token goes in the entry's `headers` as
  `Authorization: Bearer ${secret:NAME}`, and OAuth sign-in to MCP servers is
  not supported yet. `doctor`'s fix line for that row is the recipe with the
  `webagents secrets set` command. With `transport: auto` the Python client
  now reports the Streamable HTTP attempt's error, as the TypeScript client
  does, not the SSE fallback's.

- **Every 401 carries a `WWW-Authenticate` challenge** [both]. `webagents
  mcp serve --http` answered its 401 with no challenge at all, and once it
  had one every other 401 was found bare: the credential floor's own refusal
  on `chat/completions`, the A2A routes and the `command` paths, a scoped
  endpoint's "needs a caller" refusal, an auth skill's refusal of a token on
  `chat/completions`, a scoped websocket upgrade, the daemons' register and
  command routes, and the raw 401 a `serve()` or `WebAgentsServer` upgrade
  on `/uamp` got. RFC 7235 makes the header a MUST on every 401. All of them
  now carry the RFC 6750 challenge under one rule: `Bearer realm="webagents"`
  when the request carried no credential, and `Bearer realm="webagents",
  error="invalid_token"` when it carried one (in any of the floor's
  credential headers) that was refused or that no skill verified. In Python,
  `unauthorized_response()` carries the plain challenge by default and a gate
  refusal is answered through `endpoint_gate.refusal_response(request,
  refusal)`; in TypeScript, `unauthorizedResponse()` does the same and a
  `GateRefusal` carries its `headers`, which every route answers. A server
  that adds a 401 has to list it in the shared fixture
  (`python/tests/fixtures/credential_floor/www_authenticate.json`) before
  either suite goes green.

- **One repeated-call detector** [ts]. The TypeScript agent kept an older
  detector beside the fixed one: keyed per whole round on the arguments alone
  (with `delegate` messages and `text_editor` paths normalised), it rewrote
  the third identical round's tool result into "You have called this tool 3
  times with the same arguments. The result is unlikely to change" (the
  second's for `delegate`). The two disagreed on what a repeat is (the older
  one stopped the edit-and-rerun cycle above), and on a true loop the model
  got a rewritten third result under a wrap-up message saying all three
  results were the same. The older one is gone. In both SDKs a repeat is
  counted once, after each result; every tool result reaches the model and
  the client as the tool returned it; and the wrap-up system message with
  tools off is the only thing a loop adds. If you relied on the rewritten
  result to steer a model away from re-delegating a rephrased request, that
  nudge no longer exists: a delegate whose reply differs each time is not a
  loop, and the tool-round budget is what ends the turn.

- **The Python MCP client uses mcp's current Streamable HTTP client** [py].
  With mcp 1.26 every connect over Streamable HTTP printed `DeprecationWarning:
  Use streamable_http_client instead`: the skill called
  `streamablehttp_client`, which mcp 1.24 deprecated in favour of
  `streamable_http_client` (an `httpx.AsyncClient` in place of `headers` and
  the timeouts) and mcp 2 removes. The skill now binds whichever name the
  installed mcp has, so the floor stays `mcp>=1.0.0`: the current name with a
  client built from the entry's `headers` and the old defaults (30 s to
  connect, 300 s for a silent stream), else the old name as before. A 401 or
  403 is still unwrapped to the "needs a credential" row through the new
  transport.

- **`/skills` shows where an installed skill came from** [both]. The docs
  said so; both chats showed only `in .agents/skills/<name>`. For a skill the
  lock records, the line is its source and short commit (`from
  robutlerai/webagents at 0123456`), or `from local folder <path>` for a
  local install; a folder the lock does not know is listed as your own.

- **The chat footer shows credits only for turns that ran through Robutler**
  [both]. It showed `~<0.0001 credits` for an `openai/` turn on the person's
  own `OPENAI_API_KEY`, where Robutler spends nothing. A turn on the person's
  own key now shows tokens alone; the estimate (with its tilde) stands in only
  for a Robutler turn whose usage carried no reported cost. A failover member
  says its own route, so a fallback on a key under a Robutler primary, or the
  reverse, is counted right.

- **The chat's command menu always opens above the input box** [both]. It
  opened under the box whenever it fitted there, so the direction depended on
  the room left: at the bottom of the terminal `/` opened above while
  `/agent `'s four values opened below, and a box half way down the screen
  opened everything below. It now opens above whenever there are at least
  three known rows there, showing fewer entries (and "N more") when the rows
  are few. It opens under the box only at the top of a cleared screen, or
  when the chat cannot tell what the rows above hold.

- **`FORCE_COLOR` is a floor, not the colour depth** [ts]. `FORCE_COLOR=1`
  held a truecolour terminal to 16 colours, and below 256 colours the chat
  drew no idle sparkles in the input box, no shimmer on the status line and
  no shaded bands. The Python chat, which reads its depth from `TERM` and
  `COLORTERM`, drew them all. `FORCE_COLOR` now turns colour on (into a pipe
  too) at least at the depth it names; `supports-color` also reads it as a
  minimum.

- **A key typed straight after `esc` is that key** [both]. The TypeScript
  chat waited half a second after an `esc` for the rest of a key sequence, so
  a key typed within that time arrived as alt+key: a `/` typed after `esc`
  closed the menu was lost. It now waits 0.1 s, as the Python chat does, and
  both chats take `esc` then `/` (alt+/) as the `/`, which opens the menu.
  Python's emacs keys took it for "complete".

- **The menu finds a command by a word inside its name in both chats** [py].
  `/mo` offered `/model` and `/memory` in the TypeScript chat and only
  `/model` in the Python one, which matched on the text with its `/`. An
  argument value's column is now as wide in the TypeScript menu as in the
  Python one [ts].

- **A turn that spends its tool rounds says so, and the Python budget is the
  TypeScript one** [both]. The Python agent stopped after five tool rounds,
  hard-coded, without calling the model again, and the chat then blamed the
  provider ("the provider reported STOP", the finish reason of the model's
  last tool call). Python now takes `max_tool_iterations` (default 50, the
  TypeScript `maxToolIterations`), and both agents send the model the same
  warning at the same round: half way through a budget of 20 or less, 80% of
  a larger one. An answer in the last allowed round is no longer followed by
  a `max_iterations` error [ts], and a non-streaming Python turn no longer
  answers an apology about "technical difficulties" [py].

- **The built-in agent answers small talk without tools** [both]. Its
  instructions (`ROBUTLER.md`) now say to answer greetings, thanks and small
  talk directly, to use a tool only when the request needs one, and to explore
  the folder only when asked. It listed the folder before saying hello.

- **Small things from the CLI e2e pass** [both unless marked]:
  - the control-file diff shows the removed line before the added one [ts];
  - a tool line is cut at a word boundary, and a file tool's refusal is shown
    whole;
  - `cron list` says a refused file in its own words, and no longer claims no
    file declares `cron:` when the refused one does;
  - `secrets set NAME VALUE` says the value comes from a prompt or a pipe,
    and stores nothing;
  - `-p --output-format json` answers a failure as one JSON document, and
    `-p` names `webagents login` where the chat says `/login`;
  - `-p --output-format stream-json` carries the model failover's note [py];
  - the Python server prints no emoji, and `webagents daemon` prints
    warnings, errors and its own lines (each schedule's outcome as
    `[daemon] <agent>/<schedule>: ...`) rather than every skill's INFO line
    [py].

- **`/memory` claims Robutler only with the agent's key** [both]. With the
  Robutler tier on and no agent key, it said "and on Robutler" while nothing
  reached Robutler; it now says the tier waits for the agent's own key
  (`webagents publish`).

- **A Python agent's A2A card and listing carry its description** [py]. The
  file's `description:` never reached the agent, so the card said `""`.

- **An MCP server's stderr no longer draws over the chat** [both]. A stdio
  server's own output goes to `logs/mcp-<name>.log` in the profile's folder.

- **`--json` works after the subcommand** [both], like `--profile`:
  `doctor --json` and `sandbox setup --json` answered "unknown option".

- **The chat shows what Robutler charged, not an estimate** [both]. For
  Robutler's models the footer showed a list-price estimate with a tilde,
  because the platform sent no charge. Both chats now read `usage.total_cost`
  from the platform's `response.done` (the credits its settle deducted, sent
  by the matching platform change) and show it as it is, summed over a
  turn's model calls; the older `usage.cost` is still read.

- **`/model` names a way out that works** [both]. An agent naming `openai`
  answered a request for another provider's model with "Pick a openai/ model,
  or /skills add google first", and adding the skill changed nothing, since
  the file still named `openai` first. It now says to name a model that starts
  with `openai/`, or `/skills remove openai` to run another provider's.

- **`skills add google` on a Python without the Google SDK is refused with
  the install command** [py], before the file changes. It wrote the name, and
  every start then said the skill failed to load.

- **The quickstart names the same Google key for both SDKs** [both]:
  `GOOGLE_API_KEY` (or `GEMINI_API_KEY`), which both read first.

- **A served stream that fails says so** [ts]. A run that failed after its
  stream began (a provider's 402 or 429, a refused credential) dropped the
  error and answered 200 with an empty `stop`, and the server logged
  nothing. The stream now ends with OpenAI's in-stream error,
  `{"error": {"message", "type", "code"}}`, logged under a reference; the
  Python server sends the same shape in place of its bare string.

- **`serve` answers a provider's refusal with JSON** [py]. A non-streaming
  call whose model provider refused (the agent's own key out of credit)
  answered a text 500 and printed the traceback over the terminal; it is a
  JSON 500 with a reference now, as in TypeScript.

- **TypeScript `serve` and the daemon run the model the file names** [ts]. An
  agent file that listed no LLM skill served with no model, silently, even
  with its key exported, and answered every request 500. They now decide the
  model as the chat does (for callers, see Breaking), and `serve` refuses to
  start, with one sentence, when there is none.

- **An unknown skill in `serve` is one sentence** [ts], not a stack trace,
  and exit 1, as in Python.

- **A served agent's callers never read the platform's protocol wording**
  [py]. A refused credential answered them `session.create requires
  X-Payment-Token ...`; they read a plain sentence now.

- **A priced endpoint never names localhost as the platform** [both]. With
  no `ROBUTLER_API_URL`, the payment skill fell back to
  `http://localhost:3000` and published it in every 402; it now uses the
  CLI's `platform.url`, else https://robutler.ai. A paid request whose token
  the platform cannot be reached to verify answers 503 with a sentence and
  `Retry-After`, runs nothing, and settles nothing; it answered 500.

- **Three testing pages render their examples** [both]. `test-format.md`,
  `writing-tests.md` and `multi-agent.md` under `docs/testing/` nested a
  fenced block inside a fence of the same length, so each example closed early
  and the rest of the page rendered wrongly. The outer fences are four
  backticks, and a docs test holds every page to it.

- **Esc and Ctrl+C stop the command, not only the reply** [both]. An
  interrupted turn left its shell command or SKILL.md script running until
  its own timeout (13 to 18 seconds in the test pass). The interrupt now
  kills the command's whole process group, as a timeout does, and the tool
  answers "Interrupted". In Python the turn's cancellation also reaches the
  tool's own task, and a SKILL.md script no longer blocks the chat while it
  runs.
- **`webagents sandbox setup` passes on a stock Mac** [both]. Its socket-path
  check added a `/private` and a 7-digit pid that srt never uses, and failed
  every default `TMPDIR` at "110 bytes" while confined commands ran. It now
  measures the path srt builds, and fails only above 104 bytes, where srt
  itself fails.
- **Refusals look like refusals** [both]. A refused `.env` read showed a green
  "Read 1 lines", and "The owner declined ..." was green; refused, declined,
  interrupted and timed-out calls now show as failures. A command's sandbox
  hint, which named the switch to change, was hidden behind "(+N lines)"; it
  is now its own line under the tool, whole.
- **Answered questions stay on the screen** [both]. In the Python chat the
  live display redrew over the question, the typed answer and the last line
  of the diff. Both chats now keep them, add one line saying what was
  decided ("Allowed example.com for this command."), show a control file's
  path as this folder knows it (`AGENT.md`), and leave the time spent
  answering out of the tool's timer. The "Added ... to network.hosts" notice
  in the TypeScript chat is no longer erased by the next redraw.
- **Ctrl+C after a question says "Interrupted"** [py]. A question took over
  the turn's Ctrl+C handler and removed it, so a later Ctrl+C in the same
  turn read "The model returned no answer".
- **"always" is not "a change during the last reply"** [both]. The chat's own
  write of a host into the agent file made the next prompt report a change,
  and `/sandbox` said "(default)" until `/reload`. It now says "(agent file)"
  at once, and a change someone else made is still reported.
- **`serve` answers a refused completion with its status** [ts]. A payment
  refusal in a non-streaming `chat/completions` call, or before a stream's
  first chunk, came back as 500 `completions_error`; it is now the 402 with a
  JSON error, as the Python server answers.
- **`-p` says when the model returned nothing** [both]. An empty reply printed
  an empty line; the chat's one truthful line now goes to standard error.
- **`-p` and `doctor` log where the profile says** [py]. They wrote
  `~/.webagents/logs/repl.log` whatever `--profile` named; they now use the
  profile's folder, as the chat does.
- **Listed hosts work from Node and npm inside the sandbox** [both]. Confined
  commands get `NODE_USE_ENV_PROXY=1`, so Node's `fetch` uses the sandbox proxy
  (Node 24 and later ignore the proxy variables without it), and npm's cache
  moves to the scratch folder, because npm refused to run with `~/.npm` read-only.
- **Python agents on Robutler's models send their tools** [py]. The proxy
  skill put them in `session.tools`, which the platform never read, so every
  agent ran with no tools: the model guessed names and arguments, and Gemini
  ended most tool turns with `MALFORMED_FUNCTION_CALL`. They now travel in
  `response.create` (`response.tools`), as the TypeScript client has always
  sent them, which every deployed platform reads.

- **Parallel tool calls keep their own index** [py]. Every call from the
  proxy carried `index: 0`, and the agent joins argument fragments by index,
  so two parallel calls became one with both argument strings glued together.

- **An array tool parameter declares its items** [py]. `@tool` mapped
  `List[str]` to a bare `array`, which Google refuses ("items: missing
  field"); the built-in agent's `list_directory` declares one, so its first
  turn with tools on a Gemini model was a 400. `items` now follows the
  annotation, as the TypeScript decorators already declared it.

- **An empty reply says why** [both]. An empty completion whose finish
  reason is `MALFORMED_FUNCTION_CALL` or `UNEXPECTED_TOOL_CALL` is sent once
  more; after that, and for every other empty reply, the chat prints one line
  naming the provider's reason (or that the model only produced thinking) and
  points to `/model`. The Python agent's canned "content filtering" apology is
  gone and nothing is added to the conversation in its place. The goodbye line
  counts only turns that answered, and a turn that said nothing leaves no
  unanswered message behind. The ACP transport no longer sends a literal
  `<think></think>`.

- **`/skills` lists the built-in agent's skills** [both], then says how to get
  an agent file to add more, instead of refusing.

- **A key hint names the model's provider** [both]. The out-of-credits and
  sign-in hints said `OPENAI_API_KEY` on every model; they name the key of the
  model the turn ran on.

- **A refused platform key reads as "This model is unavailable right now"**
  [both], with `/model` as the way out; the provider's own text stays in the
  platform's log.

- **Robutler's credit refusals are shown as Robutler words them** [both]. A
  call on Robutler's models that cannot be funded says what it needs reserved
  while it runs, or that credits are held by unfinished model calls and when
  they return; the chat and `-p` used to print "Not enough credits to start a
  model call." over every refusal. `-p --output-format stream-json` reports
  `credits_held` for the second case and `insufficient_credits` for the first.

- **`models` shows Robutler's models ready when signed in** [both].

- **Enter on a fully typed argument sends the line** [both]. `/agent edit`
  followed by enter put a space after `edit` instead of sending.

- **`list_directory` needs no argument** [both]. Its `path` defaults to the
  agent's folder; a model that called it with none got a raw TypeError back.

- **`search_file_content` searches the folder's top-level files** [ts]. With
  no `include`, its `**/*` pattern demanded a directory, so only files in
  subfolders were ever read.

- **`serve` answers a refused non-streaming call with its status** [py]: a
  402, 401 or 429 with a JSON error, never a 500 with a traceback.

- **`repl.log` follows `--profile`** [py]. It went to `~/.webagents/logs`
  whatever the profile. An agent's signing keys stay in `~/.webagents/keys`,
  and the configuration docs say why.

- **`-m bedrock/...` runs through Robutler** [py]. A model whose provider this
  SDK has no client for failed with "No handoff registered"; signed in, it
  now runs through Robutler as the TypeScript chat does, and signed out the
  chat says how to name it.

- **Line breaks in an answer are kept** [py]. An answer asked for "one per
  line" drew as one line.

- **A hand-written memory note is loaded** [both]. A plain `.md` file with no
  front matter under `owner/` or `shared/` was ignored; it is now a note keyed
  by its file name.

- **The TypeScript daemon uses stored keys** [ts]. A key kept with
  `webagents secrets set` reached the chat and `serve`, but not the daemon; it
  now reaches the daemon and `webagents cron run` too.

- **Rebuilding an agent with a stdio MCP server no longer ends the chat**
  [py]. Switching agents, `/model` and `/keys` rebuilt the agent and crashed
  with `CancelledError`; each MCP connection now lives in one task, and
  `/reload` takes the same path.

- **`/model` runs the model it names, or refuses** [both]. TypeScript said
  "Model set to anthropic/..." and sent the turn to the agent's OpenAI model;
  Python ignored the request. An agent that names one provider's skill now
  refuses another provider's model and says which skill to add.

- **`/sandbox` and `/status` tell the truth about the sandbox** [both]. They
  said `On` when srt could not run, in which case every command is refused;
  they now check the engine as `doctor` does. Python showed `Off` for an
  invalid declaration and now shows why it is invalid.

- **Ctrl+C at the first-run offer continues without a model** [both]. Python
  hung.

- **`addSkill` registers a decorated `@prompt` once** [ts]. It was added
  twice.

- **`base_url` in an agent file reaches the OpenAI skill** [ts]. It was mapped
  to a key the skill does not read, so only `OPENAI_BASE_URL` worked.

- **Rebuilding an agent closes the old one's MCP connections** [both]. Every
  `/model`, `/login` or `/keys` in the chat left the previous connections
  open.

- **`- mcp: {mcpServers: {...}}` connected to nothing** [py]. The agent builder
  handed that shape to the skill unwrapped and the skill looked for servers
  one level too high. Both shapes now go through one normalizer.

- **`webagents daemon` serves this folder's agents without `-w`** [ts]. It
  served nothing without `-w`, and with it only files changed after it
  started: the watcher's first scan registered nothing. It also watched the
  top level only, took any `AGENT*.md` in any case, and kept serving a deleted
  file. It now serves the working directory by default, as the Python daemon
  does, with the Python rules: `AGENT.md` and `AGENT-<name>.md` by exact name,
  anywhere under the folder, never inside `.git`, `node_modules`, `.venv`,
  `.webagents` and the other tool directories; a changed file reloads its
  agent (under its new name when it renamed it), a deleted one is let go, and
  two files declaring one name are said out loud. The daemon no longer prints
  a line per agent it registers (`WEBAGENTS_DEBUG` still shows them).

- **The `/` menu leaves no gap in the chat's history** [both]. With the
  input box at the bottom of the terminal, the menu had no room under it, so
  opening it scrolled conversation lines into the scrollback. The box then
  scrolled the screen back down when the menu closed, which left blank rows in
  the history where those lines had been. The menu now opens over the last
  rows of the conversation, above the box, and draws them again when it
  closes: nothing scrolls, and the box does not move. The chat keeps a record
  of what the terminal shows so it can draw those rows exactly, and checks the
  record against the terminal's cursor position report at every prompt. When
  the record cannot vouch for those rows, or the terminal is too short, the
  menu opens under the box as before. The box then stays a little higher until
  the next message, but no blank rows are left.

- **The Python chat's input box is as wide as the TypeScript one** [py]. It
  took the terminal's full width, one column more than the TypeScript box,
  which leaves the last column empty. Both chats now draw the box, its status
  line and the menu identically.

- **`webagents daemon` reads files by their names on disk** [py]. Its scan
  globbed `**/AGENT.md`, and on a case-insensitive disk (macOS) a glob
  reports a file named `agent.md` as `AGENT.md`, so the daemon served a file
  neither the chat nor the TypeScript daemon counts as an agent; the glob also
  walked all of `node_modules` before filtering it out. It now walks the
  folder, skipping the ignored directories, as the TypeScript daemon does.

- **`robutler` reads its arguments** [py]. It started the chat whatever it
  was given, so `robutler --help` opened a chat and `robutler -p "..."` ignored
  the prompt. It now takes the TypeScript command's options (`-p`, `-m`, `-a`,
  `--json`, `-h`), refuses anything else with its help (exit 2), and runs
  through the main CLI: `robutler -p "..."` answers exactly as
  `webagents -a robutler -p "..."` does, failure hints included.

- **Paid requests work on a pip install** [py]. The payment skill verifies,
  locks, settles and extends locks through the platform API client, and the
  `robutler` package on PyPI (0.2.0) predates all four calls, so every request
  carrying a payment token was refused as an invalid token. The client in the
  SDK is the one the skill was written against, and it calls the platform's
  payment routes with the field names they read.

- **The platform API client no longer prints part of the API key** [py].
  `agent_access()` printed the first 20 characters of the key, all of a
  shorter one, to stdout on every call. It went with the other content
  methods, and the client prints nothing.

- **A payment settle is never sent twice** [py]. The platform client retried
  every request after a 5xx or a network error, and the platform charges a
  repeated settle again while the lock still holds more than the charge, so a
  lost answer could charge a payer up to four times. A POST or PATCH is now
  sent again only when the connection was never made; reads are still
  retried. The TypeScript SDK does not retry.

- **Priced tools run when their lock needs extending** [ts]. Extending a
  lock sent `amount` where the platform reads `additionalAmount`, so the
  extension was refused and the tool was blocked as "spending limit
  exceeded".

- **LLM usage is charged** [ts]. Usage records went to the settle route in
  camelCase (`promptTokens`, `completionTokens`); the route reads snake_case
  and drops keys it does not know, so every LLM record priced at 0. They are
  sent as `prompt_tokens`, `completion_tokens`, `cached_read_tokens` and
  `tool_name` now, as the Python skill sends them.

- **Tool fees are charged, once** [ts]. A priced tool's fee was settled on its
  own with `chargeType: 'tool_fee'`, which the settle route does not accept,
  so no tool fee was charged; the record kept after it would have charged the
  fee again at finalize. The fee is now a usage record, charged with the rest
  of the run's usage in one settle at finalize, as in Python, and credited to
  the agent the way `agent_fee` is.

- **A payment token that cannot back its first lock gets a 402** [both]. The
  skills retried with a zero-amount lock, which the platform never accepts,
  and then served the request with nothing charged. They now ask for payment,
  as they already did for a token below the minimum balance.

- **`PaymentX402Skill.settlePayment` sends the agent's key** [ts]. The settle
  route charges for an authenticated agent and this sent no credential, so
  every settle was refused with a 401. The key is `apiKey`, else the agent's
  own; the platform is `facilitatorUrl`, else the SDK's platform lookup.

- **The input box goes back down when the command menu closes** [both]. A box
  near the bottom of the terminal rose to make room for the `/` menu and stayed
  there, with empty rows under it. It now returns to where it was, and the
  conversation above it with it. Esc also closes the menu at once in Python,
  where it waited a second.

### Security

What this release enforces that earlier releases did not. Most are also
listed above, where the change is breaking.

- **The default sandbox keeps far more of your credentials unreadable** [both].
  Under `development`, a command could read everything but eleven credential
  folders: shell history, git's credential store, the GitHub CLI's folder,
  other agents' logins, browser profiles and every other project's `.env`
  were readable. The command had no network, but what it printed went back to
  the model, which a skill's instructions or a fetched page could steer into
  sending it on. The deny list now also covers those and the other common
  credential files (the full list is `CREDENTIAL_DIRS`), and every `.env` and
  `.env.*` under your home folder is unreadable, not only the agent folder's
  own. This is a change for a command that read a nested `.env` inside the
  agent's folder (`packages/api/.env`): pass the variables it needs with
  `sandbox: env:` instead. On Linux the `.env` files are found by a short walk
  of the home folder, four levels deep, and one created after the walk stays
  readable until the next one.
- **A served agent's callers never spend your credits** [both]. `serve` and
  the daemon ran every caller's turn, for any bearer string, on the signed-in
  owner's Robutler credits when the agent had no provider key, and
  `WEBAGENTS_PUBLIC_URL` alone opened `serve` to the whole network with no
  AuthSkill. The model a served agent's callers reach is now the agent's own
  key, or Robutler's models paid by each caller's own payment token when the
  agent has its own platform credential; the proxy skill built for them
  carries no sign-in and refuses a call with no token before dialling
  anything. Without an AuthSkill, `serve` stays on loopback. See Breaking,
  Protocols and servers.
- **`credentials.json` is 0600** [py]. The profile's account metadata beside
  the keychain token (username, user id and expiry, never the token) was
  written 0644. It is written 0600 now, and a looser file is repaired when it
  is read. The TypeScript CLI writes no such file.
- **A command never holds your terminal** [py]. Every command the Python
  CLI ran, confined or not, inherited the terminal as its standard input, so
  a confined command could read what you typed into the chat and write to
  your screen, fake prompts included, and after an interrupted command Enter
  stopped sending. Commands now start with standard input at `/dev/null`, as
  they always did in TypeScript.
- **A command cannot rewrite webagents itself** [both]. With webagents
  installed inside the agent's folder (a project `.venv` or `node_modules`),
  the default `development` preset let a confined command rewrite the SDK's
  own code and the sandbox engine, which runs unconfined, so the next command
  or the next start ran what it planted. That install, the engine and the
  node that runs it are now write-denied whenever they sit in a folder
  commands may write; another venv or `node_modules` there stays writable.
  `doctor` and `webagents sandbox setup` show an `install` line when this
  applies.
- **Confined commands cannot read the agent's own secrets** [both]. Every
  confined preset now denies reads of the keychain files, the per-profile
  folders `~/.webagents-<profile>`, and `.env`, `.env.*` and `.webagents/` in
  the agent's folder. A confined command that ran the same interpreter as the
  CLI could read the CLI's platform token and stored secrets from the keychain
  with no dialog, and any profile's history and, with the file backend, its
  secrets. A command that needs one variable gets it by name through
  `sandbox: env:`.
- **The delegate fallback sends the platform credential to the platform
  only** [both]. `delegate` set `Authorization: Bearer <platform token>`
  (Python also `X-API-Key`, TypeScript also the caller's forwarded token) on
  whatever URL the model named, so a prompt-injected hosted agent could hand
  a 24-hour token acting as its owner to any https server that serves no A2A
  card. Only the origin of the configured platform base URL gets them now;
  every other origin gets none. The payment token and the chat id go as
  before. To edit: nothing, unless you relied on a non-platform server
  reading the agent's platform token, which it must not.
- **The file tools never touch the agent's secrets** [both]. `read_file`,
  `write_file`, `replace` and `search_file_content` refuse `.env`, `.env.*`
  and anything under `.webagents/`, with a sentence, for every caller, after
  resolving symbolic links; `list_directory` and `glob` still show the names.
  Those files hold the provider keys the CLI loads and a served agent's
  schedule state, and the default agent has `rest_request` beside these
  tools. To edit: keep secrets in `webagents secrets set`, and read them with
  `${secret:NAME}` rather than through a file tool.
- **The file tools change the agent's control files only with the owner's
  yes in the chat** [both]. A write to the agent's own file, any
  `AGENT-*.md`, `WEBAGENTS.md`, `mcp.json` and `.mcp.json`, `.agents/skills/`,
  `.git/hooks` and `.git/config`, `.env*`, `.webagents/` and the rest of the
  sandbox's write-deny set (matched on any component of the path, links
  resolved) shows the diff in the interactive chat and asks; under `serve`,
  the daemon, `-p`, and for any caller who is not the owner, it is refused
  with a sentence. The real-model pass had rewritten `AGENT.md` to
  `preset: unrestricted` and planted `AGENT-evil.md`, `mcp.json` and a git
  hook through `write_file`. The built-in agent's job of writing agent files
  keeps working, behind the prompt.
- **A daemon-served agent's signing key lives in the key store, not in the
  agent folder** [both]. For one uncommitted day the daemon wrote
  `<folder>/.webagents/keys/<name>.ed25519.jwk.json`, where `git add -A`
  commits it, copies of the folder carry it and the file tools look. It now
  uses the store `serve()` uses (`WEBAGENTS_KEYS_DIR`, else
  `~/.webagents/keys`); a key found at the old place is moved on load with
  its thumbprint unchanged and one line says so, plus a second line to rotate
  it when git ever tracked the file. A non-secret `<name>.ed25519.origin.json`
  beside the key records the folder it was made for, and a daemon loading the
  same agent name from another folder is told, once, that it shares the key.
  In a container, point `WEBAGENTS_KEYS_DIR` at a mounted secret.
- **An MCP server entry that asks for a sandbox it cannot get does not
  start** [both]. `sandbox: true` (any value but `false` and `null`) made
  Python print "Running locally" and run the server with the owner's
  permissions when no Docker `sandbox` skill was loaded, and TypeScript
  ignored the key. Both now refuse the entry by name with a sentence
  (TypeScript at load; Python when the skill initializes, unless the
  `sandbox` skill is loaded), and the other servers still load.
- **Host tools are the owner's** [both]: shell and filesystem were offered to
  every caller a served or Portal Connect agent answered (see Breaking).
- **The agent's own files are write-denied in the sandbox** [both], so a
  confined command cannot widen its own policy or plant an agent for the
  daemon (see Breaking).
- **A misspelled `sandbox:` key is an error** [both]; `presets: strict` used
  to load and run under the looser default. A caller other than the owner
  only runs commands confined.
- **The Python ACP endpoint ran any tool for any credentialed caller** [py].
  It is removed with HTTP ACP.
- **The TypeScript daemon's cron routes ran any caller's prompt as the owner**
  [ts]. They are removed.
- **The TypeScript daemon listened on every interface with unauthenticated
  register and remove routes** [ts] (see Breaking).
- **One caller's payment token could pay for another caller's run on a
  served TypeScript agent** [ts] (see Breaking).
- **Portal Connect callers are what the platform asserts, never what a client
  claims** [both]: the SDKs read only the platform's `caller` field, and
  Python refuses a turn whose caller context cannot be set up.
- **`plugin_load` imported any path any caller named** [ts]; it is owner-only
  and confined to the plugin directories.
- **Tools a skill registered after start escaped `access.tools`** [ts]; the
  block's grant now covers them.
- **Discovery results are fenced as untrusted text** [both], so text another
  agent's owner wrote reaches the model marked as such.
- **Stdio MCP servers inherited the agent's whole environment** [both],
  provider keys included (see Breaking). Values in MCP errors and debug lines
  are masked, and Python no longer logs header and environment values.
- **The `notify` MCP policy failed open** [both] (see Breaking).
- **A served Python agent showed its instructions to anyone** [py] (see
  Breaking).
- **The todo list was shared by every caller** [both] (see Breaking).
- **The Python chat's history was readable by other local accounts and
  ignored `--profile`** [py]. It is kept per profile, 0600 in a 0700 folder,
  and an old `~/.webagents/history` is secured on first run.
- **`?token=` became a bearer on Python `@http` routes** [py] (see Breaking).
- **The legacy Python daemon class registered any agent file a caller named**
  [py] (see Breaking).
- **The TypeScript `delegate` tool logged the start of every delegated
  message** [ts]. It logs lengths, only behind the trace switch.
- **The Python examples listened on every interface** [py]; they bind
  `127.0.0.1`, and the Portal Connect example opens no port.
- **Every served run sees the caller's credential** [both]. `WebAgentsServer`'s
  built-in `chat/completions`, `uamp` and `uamp/stream` routes, the built-in
  `uamp` routes of `createFetchHandler` and `serve()`, and the TypeScript
  daemon's `chat/completions` ran the agent without the request's credential
  headers on the run's context, so an `AuthSkill` on the agent could neither
  verify nor refuse a token there: behind the floor's presence check, any
  non-empty credential header ran the model on the owner's key, and a
  refusal that did surface was a 500. The legacy Python daemon class
  (`webagents.cli.daemon.WebAgentsDaemon`) ran its completions with no
  request on the context at all. Every served run now carries the same
  request metadata and inbound request `serve()`'s `chat/completions`
  carries, and a refused credential answers 401 with the bearer challenge on
  each of those routes, before any stream. A credential named in a request
  body's `metadata` no longer reaches the run's metadata; a credential is a
  header. To edit: nothing, unless a caller relied on a body `metadata` key
  named like a credential header.

## 0.3.6 (2026-09-25)

### Removed (breaking)

- **`connect()` and `host()` are gone** [both]. They were one-word wrappers
  around "attach a transport skill and run a server", and hiding the lifecycle
  is what made them wrong: a bridged agent cannot run until it is connected,
  skills initialise lazily on first run, and the wrapper papered over that
  deadlock for exactly one caller. Everything they added moved into
  `create_server` / `serve()` and `PortalConnectSkill`, so it now applies to
  every entry point instead of the one the docs named.

  ```diff
  - const server = await connect(agent);            // TypeScript
  + const agent = new BaseAgent({ ..., skills: [new PortalConnectSkill()] });
  + const server = await serve(agent, { port: 8000 });
  ```

  ```diff
  - server = webagents.connect(agent)               # Python
  + agent = BaseAgent(..., skills={"portal": PortalConnectSkill()})
  + server = create_server(agents=[agent])
  ```

  `host(agent)` becomes `serve(agent, {...})` [ts] / `create_server(agents=[agent])`
  [py]. The agent card (at the ORIGIN and under the agent prefix, carrying
  `metadata.publicKey`), the JWKS endpoints and the 60s presence heartbeat are
  all served by the plain server now. Before, only the wrapper added them,
  which meant the documented server could not complete platform registration.

  The Python `webagents.portal` module is deleted; `webagents.connect` and
  `webagents.host` no longer exist as attributes.

- **`PortalWSSkill` is dropped** [both], along with `python/.../robutler/portal_ws/`
  and its root export. It spoke a `register` protocol the platform's `/ws`
  never handled: its frames were dropped on the floor. Replace it with
  `PortalConnectSkill`, which speaks the real contract
  (`session.create` → `session.created`, `input.text` in, `response.delta` /
  `response.done` out).

### Changed (breaking)

- **`PortalConnectSkill` is a different skill at the same import path** [ts].
  It used to be a REST register/heartbeat/deregister skill exported from
  `skills/social`; it is now the reverse-WebSocket bridge, exported from
  `skills/transport/portal-connect`. The name and the root import specifier are
  unchanged, so code that imported the old one compiles and then behaves
  completely differently. If you depended on the REST registration behaviour,
  that job now belongs to the server's own registration surface plus the
  heartbeat, not to a skill.

- **`serve()` returns a handle instead of `void`** [ts]. It now resolves to a
  `ServeHandle`: `{ fetch, identity, port, close }`; `port` is the port really
  bound (meaningful with `port: 0`), and `close()` stops the HTTP server, the
  heartbeat and any portal bridge the agent opened. Callers that ignored the
  return value are unaffected; callers that annotated it as `Promise<void>`
  need to update the type.

- **`POST /{agent}/chat/completions` refuses a request that presents no
  credential, with `401`** [py]. This endpoint runs the model on the OWNER's
  credit, and the Python route had no auth check at all: an unauthenticated
  POST reached the model provider and spent the owner's quota, while
  `docs/quickstart.md` stated the opposite. The floor now matches the
  TypeScript one exactly: a non-blank value in `Authorization`, `X-Api-Key` or
  `X-Owner-Assertion` (a bare `Bearer` does not count). It is a floor, not
  authentication: add an `AuthSkill` to have the credential actually verified.
  Any client that posted to completions anonymously must now send a credential.

  The floor covers BOTH doors to that endpoint: the statically registered
  route and the dynamic-agent catch-all HTTP dispatch, where a transport
  skill's `@http("/chat/completions")` handler serves the same path. It is
  deliberately scoped to completions: an agent's other `@http` handlers may
  be public by design and are not gated.

- **A `PortalConnectSkill` that cannot work now FAILS server startup** [py].
  `WebAgentsServer`'s startup event used to log `PortalCredentialError` /
  `PortalConnectConfigError` and continue, which produced a process that was
  up, answered `/health` with 200, and had no socket, indistinguishable from a
  healthy agent, and the exact failure the credential guard exists to prevent.
  Those two errors now propagate out of startup. Transient I/O failures are
  still logged and survived; the bridge owns its own reconnect loop.

### Fixed

- **The agent card publishes the configured public URL** [ts]. `serve()`
  documented `publicUrl` (and `WEBAGENTS_PUBLIC_URL`) as the card's `url` but
  used it only as the identity issuer, so the card was built from the REQUEST
  HOST: an agent behind a proxy, tunnel or container published an address
  nothing outside the process could dial, and platform registration stores that
  as the agent's callable URL. `publicUrl` is now threaded into the card
  builder and preferred over the request origin, matching Python's
  `resolve_public_base_url`.

- The CLI no longer calls `agent.initialize()` before `serve()` [ts]; `serve()`
  owns that. Double initialisation is idempotent for the stock skills but is
  not part of the `Skill` contract.

### Added

- **`registerWithPlatform()`** [ts] / **`register_with_platform()`** [py]:
  the call that turns a served agent card into a platform account. Serving the
  card was only half of joining: the platform registers an agent on the first
  request that verifies, and neither SDK ever presented such a token, so the
  key was persisted and the card was correct and nothing had read it.

  ```typescript
  const server = await serve(agent, { port: 8000, basePath: '/agents/mini' });
  const registration = await registerWithPlatform(server.identity);
  ```

  ```python
  result = await register_with_platform(agent.name)
  ```

  Set `WEBAGENTS_PUBLIC_URL` (an address the PLATFORM can fetch, so not
  loopback and not a `100.64.0.0/10` overlay address) and `ROBUTLER_API_URL`.
  The `aud` claim is the platform base URL and never the agent's own URL, and
  the helpers set it. See `docs/guides/self-registration.md`.

- **`agent_path` on minted AOAuth tokens** [both]. `serve()` derives it from
  `basePath`, and `mint_aoauth_token` takes it as an argument. The platform
  keys a registration on `iss + agent_path + "/" + sub` and falls back to the
  bare `iss` without the claim, and that column is unique, so several agents
  on one origin with no `agent_path` collide, the first registering and the
  rest verifying against its key.

- **`PortalConnectSkill.stop()`** [py]: the cross-SDK name for
  `disconnect()`, which stays. The two SDKs' socket-only examples sit in the
  same section of the same doc page and now use the same verbs
  (`initialize` / `stop`).

### Docs

- `docs/guides/self-registration.md`: how an agent you host yourself joins
  Robutler, and which URLs the platform can actually fetch a card from. The
  section on failure modes is the point: every one of them surfaces as a bare
  401 on your call, so the distinguishing information is not in the response.
- `docs/protocols/aoauth.md` sections 1.2, 6.2 and 8.1 now describe the card
  shape and the registration rate limit as they are.
- `docs/skills/platform/portal-connect.md`'s connection-flow diagram showed
  `input.text { session_id: "sess_..." }`. The platform actually sends a
  PER-REQUEST `req_...` id plus an `agent` field; the ACKed `sess_...` id is
  not the turn id. Resolving turns by the ACKed session id drops every real
  turn on the floor.
- `scripts/removed-api-guard.json`: shared config for the anti-regression
  guard both suites run over docs, `python/webagents/**`, `typescript/src/**`
  and both examples trees. It matches call shapes, not import lines, and
  exempts deliberate narrative by exact line.
