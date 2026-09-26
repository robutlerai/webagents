# Changelog

Both packages are versioned independently (see [RELEASE.md](RELEASE.md)), so
every entry is tagged **[py]**, **[ts]** or **[both]**.

Entries under Unreleased describe the change in terms of what a developer has
to edit, because most of them are breaking.

## Unreleased

### Removed — breaking

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
  all served by the plain server now — they used to be added only by `host()`,
  which meant the documented server could not complete platform registration.

  The Python `webagents.portal` module is deleted; `webagents.connect` and
  `webagents.host` no longer exist as attributes.

- **`PortalWSSkill` is dropped** [both], along with `python/.../robutler/portal_ws/`
  and its root export. It spoke a `register` protocol the platform's `/ws`
  never handled: its frames were dropped on the floor. Replace it with
  `PortalConnectSkill`, which speaks the real contract
  (`session.create` → `session.created`, `input.text` in, `response.delta` /
  `response.done` out).

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

### Changed — breaking

- **`PortalConnectSkill` is a different skill at the same import path** [ts].
  It used to be a REST register/heartbeat/deregister skill exported from
  `skills/social`; it is now the reverse-WebSocket bridge, exported from
  `skills/transport/portal-connect`. The name and the root import specifier are
  unchanged, so code that imported the old one compiles and then behaves
  completely differently. If you depended on the REST registration behaviour,
  that job now belongs to the server's own registration surface plus the
  heartbeat, not to a skill.

- **`serve()` returns a handle instead of `void`** [ts]. It now resolves to a
  `ServeHandle`: `{ fetch, identity, port, close }` — `port` is the port really
  bound (meaningful with `port: 0`), and `close()` stops the HTTP server, the
  heartbeat and any portal bridge the agent opened. Callers that ignored the
  return value are unaffected; callers that annotated it as `Promise<void>`
  need to update the type.

- **`POST /{agent}/chat/completions` refuses a request that presents no
  credential, with `401`** [py]. This endpoint runs the model on the OWNER's
  credit, and the Python route had no auth check at all: an unauthenticated
  POST reached the model provider and spent the owner's quota, while
  `docs/quickstart.md` stated the opposite. The floor now matches the
  TypeScript one exactly — a non-blank value in `Authorization`, `X-Api-Key` or
  `X-Owner-Assertion` (a bare `Bearer` does not count). It is a floor, not
  authentication: add an `AuthSkill` to have the credential actually verified.
  Any client that posted to completions anonymously must now send a credential.

  The floor covers BOTH doors to that endpoint: the statically registered
  route and the dynamic-agent catch-all HTTP dispatch, where a transport
  skill's `@http("/chat/completions")` handler serves the same path. It is
  deliberately scoped to completions — an agent's other `@http` handlers may
  be public by design and are not gated.

- **A `PortalConnectSkill` that cannot work now FAILS server startup** [py].
  `WebAgentsServer`'s startup event used to log `PortalCredentialError` /
  `PortalConnectConfigError` and continue, which produced a process that was
  up, answered `/health` with 200, and had no socket — indistinguishable from a
  healthy agent, and the exact failure the credential guard exists to prevent.
  Those two errors now propagate out of startup. Transient I/O failures are
  still logged and survived; the bridge owns its own reconnect loop.

- **`robutler -p "..." --json` prints what `webagents` prints** [ts].
  `robutler` now runs through the main CLI, as the Python one does, so its
  `--json` reply is the one `webagents -a robutler -p "..." --output-format json`
  prints, not the REPL's internal response object, and a failed request gets
  the main CLI's message and hint rather than a bare `Error:` line.

- **The namespace, publish and message history skills find the platform as
  the discovery skill does** [py]:
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

### Fixed

- **The agent card publishes the configured public URL** [ts]. `serve()`
  documented `publicUrl` (and `WEBAGENTS_PUBLIC_URL`) as the card's `url` but
  used it only as the identity issuer, so the card was built from the REQUEST
  HOST — an agent behind a proxy, tunnel or container published an address
  nothing outside the process could dial, and platform registration stores that
  as the agent's callable URL. `publicUrl` is now threaded into the card
  builder and preferred over the request origin, matching Python's
  `resolve_public_base_url`.

- The CLI no longer calls `agent.initialize()` before `serve()` [ts]; `serve()`
  owns that. Double initialisation is idempotent for the stock skills but is
  not part of the `Skill` contract.

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

### Added

- **`registerWithPlatform()`** [ts] / **`register_with_platform()`** [py] —
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
  bare `iss` without the claim, and that column is unique — so several agents
  on one origin with no `agent_path` collide, the first registering and the
  rest verifying against its key.

- **`PortalConnectSkill.stop()`** [py] — the cross-SDK name for
  `disconnect()`, which stays. The two SDKs' socket-only examples sit in the
  same section of the same doc page and now use the same verbs
  (`initialize` / `stop`).

### Docs

- `docs/guides/self-registration.md` — how an agent you host yourself joins
  Robutler, and which URLs the platform can actually fetch a card from. The
  section on failure modes is the point: every one of them surfaces as a bare
  401 on your call, so the distinguishing information is not in the response.
- `docs/protocols/aoauth.md` sections 1.2, 6.2 and 8.1 now describe the card
  shape and the registration rate limit as they are.
- `docs/skills/platform/portal-connect.md`'s connection-flow diagram showed
  `input.text { session_id: "sess_..." }`. The platform actually sends a
  PER-REQUEST `req_...` id plus an `agent` field; the ACKed `sess_...` id is
  not the turn id. Resolving turns by the ACKed session id drops every real
  turn on the floor (F-043).
- `scripts/removed-api-guard.json` — shared config for the anti-regression
  guard both suites run over docs, `python/webagents/**`, `typescript/src/**`
  and both examples trees. It matches call shapes, not import lines, and
  exempts deliberate narrative by exact line.
