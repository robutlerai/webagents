# WebAgents Manual Testing Guide

This guide provides step-by-step instructions for manually verifying all WebAgents skills and webagentsd functionality.

## Prerequisites

```bash
cd /Users/vs/dev/webagents
source .venv/bin/activate

# Verify dependencies
pip show multilspy litellm fastapi uvicorn

# Set API keys (optional, for LLM-powered features)
export OPENAI_API_KEY="your-key-here"
```

---

## Part 1: Core Skills Testing

### 1.1 AuthSkill (AOAuth)

The AuthSkill provides agent-to-agent authentication using OAuth 2.0 with JWT tokens.

#### Step 1: Start an Auth-Enabled Agent

```bash
python examples/skills/run_skill_demos.py
```

#### Step 2: Verify JWKS Endpoint

```bash
# Get the public keys for token verification
curl http://localhost:8000/auth-demo/.well-known/jwks.json
```

**Expected**: JSON with `keys` array containing RSA public key(s).

#### Step 3: Test Token Generation (via REPL)

```bash
# Start REPL with auth-demo agent
webagents repl auth-demo

# In REPL:
/auth status
/auth token http://localhost:9000/other-agent
```

**Expected**: JWT token string and status information.

---

### 1.2 LSPSkill (Code Intelligence)

The LSPSkill provides code navigation using Language Server Protocol.

#### Step 1: Start LSP-Enabled Agent

```bash
python examples/skills/run_skill_demos.py
```

#### Step 2: Test LSP Commands (via REPL)

```bash
webagents repl lsp-demo

# In REPL:
/lsp status
```

#### Step 3: Test Code Navigation via API

```bash
# Test goto_definition
curl -X POST http://localhost:8000/lsp-demo/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Find the definition of BaseAgent in webagents/agents/core/base_agent.py"}]
  }'
```

**Expected**: Agent uses `goto_definition` tool and returns file location.

---

### 1.3 PluginSkill (Marketplace)

The PluginSkill provides plugin discovery and installation.

#### Step 1: Start Plugin-Enabled Agent

```bash
python examples/skills/run_skill_demos.py
```

#### Step 2: Test Plugin Commands

```bash
webagents repl plugin-demo

# In REPL:
/plugin list
/plugin search calculator
```

**Expected**: Empty list (no plugins installed) or search results.

---

### 1.4 WebUISkill (Browser Interface)

The WebUISkill serves a React-based chat interface.

#### Step 1: Build the WebUI (First Time Only)

```bash
cd webagents/cli/webui
pnpm install
pnpm build
cd ../../..
```

#### Step 2: Start WebUI Agent

```bash
python examples/skills/run_skill_demos.py
```

#### Step 3: Open in Browser

```bash
open http://localhost:8000/webui-demo/ui
```

**Expected**: React chat interface loads.

#### Step 4: Test Chat

Type a message in the UI and send.

**Expected**: Agent responds (requires OPENAI_API_KEY).

---

### 1.5 UCPSkill (Commerce)

The UCPSkill enables agent-to-agent commerce using Universal Commerce Protocol.

#### Step 1: Start Commerce Demo

```bash
python examples/ucp_commerce/run_commerce_demo.py
```

#### Step 2: Test Merchant Discovery

```bash
# Get merchant's UCP profile
curl http://localhost:8000/ucp-merchant/.well-known/ucp
```

**Expected**: JSON with UCP version, capabilities, payment handlers.

#### Step 3: Test Service Catalog

```bash
curl http://localhost:8000/ucp-merchant/ucp/services
```

**Expected**: List of services with IDs, titles, and prices.

#### Step 4: Test Checkout Flow

```bash
# Create checkout session
curl -X POST http://localhost:8000/ucp-merchant/checkout-sessions \
  -H "Content-Type: application/json" \
  -d '{
    "line_items": [{"item": {"id": "quick_analysis"}, "quantity": 1}],
    "buyer": {"email": "test@example.com", "full_name": "Test User"}
  }'
```

**Expected**: Checkout session with ID and `ready_for_complete` status.

---

## Part 2: Webagentsd Testing

### 2.1 Start the Daemon

```bash
# Create example agents directory
mkdir -p ~/.webagents/agents
cp examples/skills/AGENT-*.md ~/.webagents/agents/

# Start daemon
webagentsd start --port 8765 --watch ~/.webagents/agents
```

**Expected**: Daemon starts, discovers agents, logs agent names.

### 2.2 List Running Agents

```bash
webagents list
```

**Expected**: List of discovered agents with names and paths.

### 2.3 Test Agent Endpoints

```bash
# Health check
curl http://localhost:8765/health

# Agent info
curl http://localhost:8765/agents/auth-demo

# Chat completion
curl -X POST http://localhost:8765/agents/auth-demo/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

### 2.4 Test Hot Reload

```bash
# Modify an agent file
echo "# Updated" >> ~/.webagents/agents/AGENT-auth-demo.md
```

**Expected**: Daemon logs file change and reloads agent.

### 2.5 Stop the Daemon

```bash
webagentsd stop
```

---

## Part 3: REPL Testing

### 3.1 Start REPL

```bash
webagents repl
```

### 3.2 Test Slash Commands

```
/help                    # Show available commands
/agents                  # List available agents
/switch auth-demo        # Switch to specific agent
/skills                  # List loaded skills
/tools                   # List available tools
/clear                   # Clear conversation
/quit                    # Exit REPL
```

### 3.3 Test Conversation

```
> Hello, what can you do?
> /switch lsp-demo
> Find the definition of UCPSkill
```

---

## Part 4: Integration Scenarios

### 4.1 Agent-to-Agent Commerce

**Scenario**: Client agent discovers and purchases from merchant agent.

```bash
# Terminal 1: Start commerce demo
python examples/ucp_commerce/run_commerce_demo.py

# Terminal 2: Use client to discover merchant
curl -X POST http://localhost:8000/ucp-client/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Discover services at http://localhost:8000/ucp-merchant"}]
  }'
```

### 4.2 Authenticated Agent Communication

**Scenario**: Agent A generates token to call Agent B.

```bash
# Start two agents
python examples/skills/run_skill_demos.py

# REPL: Generate token
webagents repl auth-demo
/auth token http://localhost:8000/another-agent
```

### 4.3 Code Analysis with LSP

**Scenario**: Ask agent to analyze code structure.

```bash
webagents repl lsp-demo
> List all classes in webagents/agents/core/base_agent.py
> Find where UCPSkill is defined
> Show the hover documentation for the BaseAgent class
```

---

## Part 5: Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `OPENAI_API_KEY not set` | Export your API key: `export OPENAI_API_KEY=...` |
| `WebUI dist not found` | Build the UI: `cd webagents/cli/webui && pnpm build` |
| `multilspy not installed` | Install: `.venv/bin/pip install multilspy` |
| `Port already in use` | Kill existing: `pkill -f webagentsd` or use different port |
| `Agent not found` | Check agent file is in watched directory |

### Logs

```bash
# Daemon logs
tail -f ~/.webagents/logs/daemon.log

# Enable debug logging
WEBAGENTS_DEBUG=1 webagentsd start
```

---

## Test Checklist

Use this checklist to verify functionality:

- [ ] **AuthSkill**: JWKS endpoint returns valid keys
- [ ] **AuthSkill**: Token generation works
- [ ] **LSPSkill**: Language detection works for .py, .ts, .rs files
- [ ] **LSPSkill**: goto_definition returns valid locations
- [ ] **PluginSkill**: /plugin list command works
- [ ] **PluginSkill**: Marketplace search returns results
- [ ] **WebUISkill**: UI loads in browser at /ui
- [ ] **WebUISkill**: Chat messages send and receive
- [ ] **UCPSkill (Client)**: Merchant discovery works
- [ ] **UCPSkill (Client)**: Checkout creation works
- [ ] **UCPSkill (Server)**: /.well-known/ucp returns profile
- [ ] **UCPSkill (Server)**: Services catalog available
- [ ] **Webagentsd**: Daemon starts and discovers agents
- [ ] **Webagentsd**: Hot reload on file change
- [ ] **Webagentsd**: Agent endpoints accessible
- [ ] **REPL**: Slash commands work
- [ ] **REPL**: Agent switching works
- [ ] **REPL**: Conversation history maintained

---

## Automated Tests

After manual verification, run the automated test suite:

```bash
cd /Users/vs/dev/webagents

# Core skill tests (fast)
.venv/bin/pytest tests/test_local_auth_skill.py tests/test_lsp_skill.py \
  tests/test_plugin_skill.py tests/test_webui_skill.py \
  tests/test_skill_integration.py -v

# UCP tests
.venv/bin/pytest tests/test_ucp_skill.py tests/test_ucp_commerce.py -v

# Full test suite
.venv/bin/pytest tests/ --ignore=tests/integration -q
```

---

*Last Updated: 2026-01-21*
