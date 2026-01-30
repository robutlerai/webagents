# CLI Documentation

The webagents CLI provides an interactive interface for chatting with AI agents and managing agent services.

## Installation

The CLI is included with the webagents package:

```bash
npm install webagents
```

Or run directly with npx:

```bash
npx webagents --help
```

## Commands

### chat

Start an interactive chat session.

```bash
webagents chat [options]
```

**Options:**

| Option | Description |
|--------|-------------|
| `-m, --model <model>` | Model to use (e.g., `gpt-4o`, `claude-3-5-sonnet`) |
| `-p, --provider <provider>` | LLM provider (`openai`, `anthropic`, `google`, `xai`) |
| `-s, --system <prompt>` | System prompt |
| `--no-stream` | Disable streaming output |
| `-h, --headless` | Non-interactive mode (pipe input) |

**Examples:**

```bash
# Interactive chat with default settings
webagents chat

# Use specific model
webagents chat --model gpt-4o

# Use Anthropic Claude
ANTHROPIC_API_KEY=sk-... webagents chat --provider anthropic

# With system prompt
webagents chat --system "You are a helpful coding assistant"

# Non-interactive (pipe input)
echo "What is 2+2?" | webagents chat --headless
```

### models

List available models.

```bash
webagents models [options]
```

**Options:**

| Option | Description |
|--------|-------------|
| `-p, --provider <provider>` | Filter by provider |

**Example:**

```bash
webagents models
webagents models --provider openai
```

### info

Show agent and SDK information.

```bash
webagents info
```

### daemon

Start the agent daemon (webagentsd).

```bash
webagents daemon [options]
```

**Options:**

| Option | Description |
|--------|-------------|
| `-p, --port <port>` | Server port (default: 3000) |
| `-w, --watch <dir>` | Watch directory for agent files |
| `-c, --config <file>` | Config file path |

**Example:**

```bash
webagents daemon --port 8080 --watch ./agents
```

## Interactive Mode

When running `webagents chat`, you enter an interactive REPL with special commands.

### Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/chat [name]` | Start new chat or switch to named chat |
| `/model <name>` | Switch to a different model |
| `/history` | Show conversation history |
| `/clear` | Clear current conversation |
| `/save <file>` | Save conversation to file |
| `/load <file>` | Load conversation from file |
| `/tools` | List available tools |
| `/exit` or `/quit` | Exit the CLI |

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Cancel current response |
| `Ctrl+D` | Exit |
| `Up/Down` | Navigate history |
| `Tab` | Auto-complete commands |

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GOOGLE_API_KEY` | Google AI API key |
| `XAI_API_KEY` | xAI API key |
| `WEBAGENTS_MODEL` | Default model |
| `WEBAGENTS_PROVIDER` | Default provider |

### Config File

Create `~/.webagents/config.json`:

```json
{
  "default_model": "gpt-4o",
  "default_provider": "openai",
  "system_prompt": "You are a helpful assistant.",
  "stream": true,
  "history_size": 100
}
```

## Usage Examples

### Basic Chat

```text
$ webagents chat
Welcome to webagents CLI!
Type /help for commands, /exit to quit.

You: Hello!
Assistant: Hello! How can I help you today?

You: What's the capital of France?
Assistant: The capital of France is Paris.

You: /exit
Goodbye!
```

### Piped Input

```bash
# Single question
echo "Explain async/await in JavaScript" | webagents chat --headless

# From file
cat questions.txt | webagents chat --headless

# In a script
result=$(echo "What is 2+2?" | webagents chat --headless)
echo "Answer: $result"
```

### Switching Models

```text
You: /model gpt-4o
Switched to model: gpt-4o

You: /model claude-3-5-sonnet
Switched to model: claude-3-5-sonnet
```

### Managing Conversations

```text
You: /clear
Conversation cleared.

You: /history
[1] User: Hello
[2] Assistant: Hello! How can I help?
[3] User: What's 2+2?
[4] Assistant: 4

You: /save chat.json
Saved conversation to chat.json

You: /load chat.json
Loaded conversation from chat.json
```

### Using Tools

When the agent has tools available:

```text
You: What's the weather in Tokyo?

[Tool Call] get_weather: {"city": "Tokyo"}
[Tool Result] {"temp": 22, "conditions": "sunny"}