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
