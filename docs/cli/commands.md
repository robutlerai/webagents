# CLI Commands Reference

Complete reference for all WebAgents CLI commands.

## Global Options

```bash
webagents [OPTIONS] COMMAND [ARGS]
```

| Option | Description |
|--------|-------------|
| `--help` | Show help message |
| `--install-completion` | Install shell completion |
| `--show-completion` | Show completion script |

## Agent Commands

### webagents (no command)

Launch interactive REPL session.

```bash
webagents              # Connect to default agent
```

### webagents connect

Start interactive REPL with an agent.

```bash
webagents connect [AGENT]

# Examples
webagents connect              # Default agent
webagents connect planner      # Named agent
webagents connect ./AGENT.md   # Specific file
```

### webagents run

Execute agent in headless mode.

```bash
webagents run [AGENT] [OPTIONS]

Options:
  -p, --prompt TEXT    Single prompt, exit after response
  --all                Run all agents in directory

# Examples
webagents run -p "Generate a report"
webagents run planner -p "Create weekly plan"
webagents run --all
```

### webagents list

List registered agents.

```bash
webagents list [OPTIONS]

Options:
  -l, --local          Local agents only
  -r, --remote         Remote agents only
  --running            Currently running agents
  -n, --namespace NS   Filter by namespace
```

### webagents init

Initialize a new agent.

```bash
webagents init [NAME] [OPTIONS]

Options:
  -t, --template NAME  Use template
  -c, --context        Create AGENTS.md context file

# Examples
webagents init                  # Create AGENT.md
webagents init writer           # Create AGENT-writer.md
webagents init --template planning
webagents init --context        # Create AGENTS.md
```

### webagents register

Register agents with daemon.

```bash
webagents register [PATH] [OPTIONS]

Options:
  -r, --recursive      Scan subdirectories
  -w, --watch          Watch for changes
```

### webagents scan

Scan for agent files.

```bash
webagents scan [PATH]
```

## Daemon Commands

### webagents daemon start

Start the webagentsd daemon.

```bash
webagents daemon start [OPTIONS]

Options:
  -b, --background     Run in background
  -p, --port PORT      Daemon port (default: 8765)
  -w, --watch DIR      Directories to watch
```

### webagents daemon stop

Stop the daemon.

```bash
webagents daemon stop
```

### webagents daemon status

Show daemon status.

```bash
webagents daemon status
```

### webagents daemon install

Install as system service.

```bash
webagents daemon install [OPTIONS]

Options:
  --systemd            Linux systemd service
  --launchd            macOS launchd service
```

### webagents daemon expose

Expose agent via HTTP.

```bash
webagents daemon expose AGENT [OPTIONS]

Options:
  -p, --port PORT      Custom port
```

## Discovery Commands

### webagents discover

Discover agents by intent.

```bash
webagents discover INTENT [OPTIONS]

Options:
  -l, --local          Local agents only
  -n, --namespace NS   Specific namespace
  -g, --global         Global platform discovery
  --limit N            Max results

# Example
webagents discover "summarize documents"
```

### webagents search

Full-text search.

```bash
webagents search QUERY
```

### webagents browse

Interactive agent browser.

```bash
webagents browse [OPTIONS]

Options:
  -n, --namespace NS   Filter by namespace
```

## Skill Commands

### webagents skill list

List available skills.

```bash
webagents skill list [OPTIONS]

Options:
  -i, --installed      Installed skills only
```

### webagents skill install

Install skills to agent.

```bash
webagents skill install SKILL... [OPTIONS]

Options:
  -a, --agent NAME     Target agent

# Examples
webagents skill install cron
webagents skill install cron folder-index
webagents skill install mcp --agent planner
```

## Template Commands

### webagents template list

List available templates.

```bash
webagents template list [OPTIONS]

Options:
  -r, --remote         Include remote templates
```

### webagents template use

Apply a template.

```bash
webagents template use NAME [OPTIONS]

Options:
  -n, --name NAME      Output filename (AGENT-<name>.md)
  -k, --keep           Keep TEMPLATE.md after applying

# Examples
webagents template use planning
webagents template use assistant --name helper
```

### webagents template pull

Pull template from GitHub.

```bash
webagents template pull URL [OPTIONS]

Options:
  -b, --branch NAME    Branch name
  -p, --path PATH      Path in repo
  -a, --apply          Apply immediately

# Examples
webagents template pull user/repo
webagents template pull https://github.com/user/repo
```

## Intent Commands

### webagents intent publish

Publish intents.

```bash
webagents intent publish [AGENT] [OPTIONS]

Options:
  -i, --intent TEXT    Ad-hoc intent
  -l, --local          Local discovery only
  -n, --namespace NS   Target namespace
  -p, --public         Global public discovery
```

### webagents intent list

List published intents.

```bash
webagents intent list [AGENT]
```

### webagents intent subscribe

Subscribe to intent notifications.

```bash
webagents intent subscribe INTENT [OPTIONS]

Options:
  -a, --agent NAME     Route to agent
```

## Namespace Commands

### webagents namespace list

List namespaces.

```bash
webagents namespace list [OPTIONS]

Options:
  -a, --all            All accessible namespaces
```

### webagents namespace create

Create a namespace.

```bash
webagents namespace create NAME [OPTIONS]

Options:
  -t, --type TYPE      user, reversedomain, or global
```

### webagents namespace auth

Configure namespace authentication.

```bash
webagents namespace auth NS [OPTIONS]

Options:
  -s, --secret SECRET  Set shared secret
  --verify-domain      Verify domain ownership
  --token              Generate access token
```

### webagents namespace invite

Generate invite code.

```bash
webagents namespace invite NS
```

### webagents namespace join

Join namespace via invite.

```bash
webagents namespace join INVITE_CODE
```

## Cron Commands

### webagents cron list

List scheduled jobs.

```bash
webagents cron list
```

### webagents cron add

Schedule agent execution.

```bash
webagents cron add AGENT SCHEDULE

# Examples
webagents cron add planner "0 9 * * *"
webagents cron add reporter "@daily"
```

### webagents cron remove

Remove a job.

```bash
webagents cron remove JOB_ID
```

## Auth Commands

### webagents login

Login to robutler.ai.

```bash
webagents login [OPTIONS]

Options:
  --api-key KEY        Use API key instead of OAuth
```

### webagents logout

Logout and clear credentials.

```bash
webagents logout
```

### webagents whoami

Show current user.

```bash
webagents whoami
```

## Sync Commands

### webagents sync

Sync with remote registry.

```bash
webagents sync [AGENT] [OPTIONS]

Options:
  --auto               Enable auto-sync
```

### webagents publish

Publish agent to registry.

```bash
webagents publish [AGENT] [OPTIONS]

Options:
  --internal           Internal namespace only
  --public             Public discovery
  -n, --namespace NS   Target namespace
```

## Config Commands

### webagents config

Show configuration.

```bash
webagents config [OPTIONS]
```

### webagents config get/set

Get or set config values.

```bash
webagents config get KEY
webagents config set KEY VALUE
```

### webagents config sandbox

Sandbox configuration.

```bash
# Show status
webagents config sandbox

# Allowlists
webagents config sandbox allow-folder PATH
webagents config sandbox allow-command CMD
webagents config sandbox allow-import MODULE

# Presets
webagents config sandbox preset development
webagents config sandbox preset strict
```
