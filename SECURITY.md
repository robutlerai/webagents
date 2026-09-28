# Security

## Reporting a vulnerability

Email **security@robutler.ai**. Please do not open a public issue, pull request
or discussion for a vulnerability.

Include what you can of:

- the package (`webagents` on PyPI or npm) and its version;
- what an attacker can do, and what they need first (a network position, a
  credential, a malicious repository, a crafted message);
- the steps or a small proof of concept;
- whether the issue is already public.

We confirm that a report arrived and keep you informed while we work on it.
Please give us a reasonable time to ship a fix before you disclose it.

The same address takes reports about the Robutler platform. Its contact is
also published at `https://robutler.ai/.well-known/security.txt`.

## Supported versions

The two packages are versioned independently (see [RELEASE.md](RELEASE.md)).

| Version | Security fixes |
| --- | --- |
| The newest release of each package | Yes |
| Any earlier release | No. Upgrade to the newest release |
| npm `webagents` 1.0.0 and 1.0.1 | No. They are not part of the release line; use the newest release |

A fix ships in a new release, and the [CHANGELOG](CHANGELOG.md) lists it under
Security. A release with a serious hole is deprecated on npm and yanked on
PyPI once its fix is out.

## Trust model

What the SDK treats as a security boundary, and what it does not. A way around
anything in the first list is a vulnerability; please report it.

### Boundaries

- **Who a caller is.** An identity comes only from a credential the agent
  verified: a Web Bot Auth signature checked against the caller's key set, a
  Robutler credential an auth skill verified, or the caller the Robutler
  platform asserts on a Portal Connect turn. Never from a request body,
  message text, a `Host` header or where a request came from. A `caller` a
  client writes into its own frame is dropped by the platform and never read
  by the SDK.
- **What a caller may use.** Tool and prompt scopes, and the `access:` block's
  `deny`, groups, default group and `tools:` grants, decide what each caller
  sees and may call, in both SDKs. That includes groups that admit other
  agents by their TrustFlow score, which the platform computes.
- **Host tools are the owner's.** `shell`, `filesystem`, `rest`,
  `plugin_load` and the SKILL.md skill tools are offered to the owner only,
  until an `access:` block names a group for them.
- **The sandbox, for everyone but the owner.** A caller other than the owner
  runs commands only inside srt (`@anthropic-ai/sandbox-runtime`): writes only
  to the allowed folders, reads limited under `strict`, network only to the
  hosts in `network:` through srt's proxy, secret-looking variables withheld,
  and the process group killed on timeout. When the agent declares no sandbox,
  declares `unrestricted`, or srt cannot run, that caller's command is
  refused.
- **An agent's control files, inside the sandbox.** `AGENT.md`, every
  `AGENT-<name>.md`, `WEBAGENTS.md`, `mcp.json`, `.agents/skills`,
  `.webagents`, `.git/hooks`, `.git/config`, shell rc files and `.env` are
  write-denied, so a confined command cannot widen its own policy. On Linux,
  srt can deny only files that exist when the command starts, so a new
  `AGENT-<name>.md` created by a command is not covered there.
- **SKILL.md scripts.** They always run inside the sandbox, under the agent's
  own policy or `strict`, never `unrestricted`, and every skill folder is
  read-only to them.
- **Per-caller state.** Memory, conversations and todo lists are kept per
  verified caller. One caller cannot read or change another's, and no
  argument the model passes can widen that.
- **Secrets for MCP servers.** `${secret:NAME}` and `${env:NAME}` resolve only
  in agent files you run (the `mcp:` entry and the `mcp.json` beside it). An
  `MCPSkill` built in code from someone else's settings resolves nothing. A
  stdio MCP server receives only the variables its entry names, and a
  reference is never put on a command line. Resolved values are masked in
  messages and reports.
- **The credential floor.** On a served agent, every route that runs the
  model, and the A2A task routes, refuse a request that carries no
  credential. A served agent with neither an auth skill nor a public URL
  listens on loopback only, unless you pass `--host`.
- **The local daemon.** It listens on loopback unless told otherwise on the
  command line; a configured address that is not local is refused. Routes
  that register or remove an agent need a credential.
- **Payments.** A payment pays for one request: an x402 challenge is bound to
  the request it answered, a payment cannot be spent twice, and every settle
  carries an idempotency key, so a retried settle never charges twice. A
  priced endpoint on an agent with no payment skill is refused, never served
  free.
- **Peers.** A peer's A2A card is trusted only when its signature verifies
  against the key set on the peer's own origin, and the A2A client refuses
  redirects.
- **Agent files and skill installs.** An agent file that is a symbolic link is
  refused. `webagents skills add` fetches one commit, refuses symbolic links,
  lists every file before installing, and never replaces a folder its lock
  does not know; the skill editors never write through a link or into another
  user's file.

### Not boundaries

- **The owner.** You, at your terminal (the chat, `webagents -p`,
  `webagents acp`, `webagents mcp serve` over stdio), and the owner the
  platform asserts on a relayed turn, have every tool. Your own commands run
  with your permissions unless the agent file declares a `sandbox:`, and
  `unrestricted` is not a sandbox.
- **What the model decides.** Text from tools, web pages, MCP servers, other
  agents, discovery results, apps and SKILL.md skills can steer the model.
  Discovery results arrive fenced as untrusted text, but the protection is
  what the caller is allowed to do, not what the model is told.
- **Code you choose to run.** Coded skills, plugins and MCP servers you list
  run in or beside the agent process, with your permissions, outside the
  sandbox. A SKILL.md skill from someone else's repository is someone else's
  code and instructions: read the listing `skills add` shows before you
  install it.
- **Apps and SKILL.md scripts are untrusted.** An app (an interactive view an
  agent renders; the SDK calls it a widget) is someone's code, rendered by the
  host in a sandboxed frame, and every message it sends is input like any
  caller's. SKILL.md scripts are untrusted too, which is why they only ever
  run confined.
- **A credential that is merely present.** The credential floor checks that a
  credential is there, not that it is valid. Add an auth skill to verify it.
- **`allowed_commands`.** It decides what runs without asking, by reading the
  command's text; it is not enforcement. The sandbox is.
- **Other processes on your machine.** Loopback is reachable by every local
  process and account.
- **TrustFlow scores.** A score is a reputation signal computed by the
  platform, not a proof about a peer's behaviour.
- **Windows.** The sandbox has no backend there, so a command that needs one
  is refused.

An agent hosted on Robutler runs under the platform's own protections, and its
configuration is set on Robutler rather than in an agent file.
