---
title: Accelerator backend
description: Run the gateway's model on the board's own AI accelerator - the configuration, the token link protocol, the tokenizer and the chat template.
---

# Accelerator backend

By default a gateway asks the platform for its model. With `kind =
"accelerator"` the model runs on the board's own AI accelerator instead: the
gateway builds the prompt, turns it into token ids, sends the ids over a link
to the accelerator and turns the ids that come back into text and tool calls.
No model call leaves the board.

Everything else stays as it is. The agent is still reached through the
platform's agent socket, the tool loop is the same, and every tool call goes
through the same envelope with the same refusals. Only the question "given
these messages and tools, what next?" is answered locally.

## What runs where

An accelerator computes token ids from token ids. Everything textual stays on
the host that drives it.

```text
                    gateway (host)                                 accelerator
 messages, tools --> chat template --> tokenizer --> prompt ids --> [ link ] --> model
 text, tool calls <-- tool-call parse <-- tokenizer <-- generated ids <-- [ link ] <-- sampler
```

| Step | Where | Module |
| --- | --- | --- |
| Render the conversation and the tool list as one prompt | host | `webagents::model::chat` |
| Text to token ids, and back | host | `webagents::model::bpe` |
| Carry ids to the accelerator and back | host side of the link | `webagents::model::accel` |
| Compute the next token, sample it, decide when to stop | accelerator | (the chip) |
| Find the tool calls in the answer | host | `webagents::model::chat` |
| Join the steps into a model backend for the gateway | host | `webagents::gateway::accel` |

## Configuration

```toml
[portal]
url = "wss://robutler.ai/ws"
agent = "bench"

[llm]
kind = "accelerator"
link = "127.0.0.1:7100"              # the accelerator link, host:port
tokenizer = "qwen3/tokenizer.json"   # the model's own tokenizer file
max_new = 512                        # most tokens generated per model call
token_timeout_s = 600                # longest wait for the next token

[[device]]
spec = "pwm1.regspec.json"
transport = { kind = "uart", path = "/dev/ttyUSB0" }
```

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `kind` | no | `"portal"` | `"accelerator"` selects this backend; `"portal"` (or no `kind`) is the platform's model |
| `link` | yes | | Where the accelerator link is served, as `host:port`. The gateway opens one TCP connection per model call |
| `tokenizer` | yes | | Path to the model's `tokenizer.json`, relative to the configuration file |
| `max_new` | no | `512` | The most tokens one model call may generate |
| `token_timeout_s` | no | `600` | The longest wait for the next token, in seconds. A software model of the accelerator needs far longer per token than silicon |

`url` and `model` in `[llm]` belong to the platform backend and are not used
here. An unknown `kind`, or an accelerator backend without `link` or
`tokenizer`, is a configuration error and the gateway does not start.

`robutler-device gateway check` loads the configuration and opens the devices.
It does not load the tokenizer or connect to the link; `gateway run` loads the
tokenizer at start-up and connects to the link at the first model call.

## One model call

For each round of a turn the backend:

1. renders the messages and the tools with the model's chat template, ending
   with the opening of the assistant's answer;
2. encodes the prompt to token ids;
3. opens the link, sends the ids and the stop ids, and reads generated ids
   until the accelerator says it is done;
4. drops the stop id, decodes the ids to text and splits the text into what is
   for the person and the `<tool_call>` blocks;
5. returns the text and the calls to the tool loop.

Calls are made one at a time. An accelerator holds one sequence (one key-value
cache), so a second turn waits for the first turn's call to finish.

Generation stops at the first of the model's end tokens, `<|im_end|>` or
`<|endoftext|>`. A tokenizer that has neither is refused at start-up, because
every answer would then run to `max_new`. An answer that does reach `max_new`
without a stop is passed on with `[stopped at 512 tokens]` appended, so a
cut-off answer is never taken for a whole one.

Each call logs one line to standard error:

```text
[webagents-gateway] accelerator: 1843 prompt tokens, 41 generated, 1 tool call(s)
```

Turns served by this backend use no payment token: the compute is the board's
own.

## The token link

The link is a line protocol over a byte stream. The gateway speaks it over
TCP; what carries it on the board (a serial port bridged to TCP, a driver that
serves a local port) is up to the board.

```text
host -> chip   R                         reset: a new sequence (the cache is cleared)
host -> chip   P <n> <id> ... <id>       up to 64 prompt ids, appended in order
host -> chip   G <max_new> <stop>[,<stop>...]
chip -> host   T <id>                    one per generated token (a stop id is sent, then D)
chip -> host   D <eos|max> <n>           done, n tokens generated
chip -> host   E <reason>                refused (the sequence is gone; send R)
```

Ids are decimal. One request is `R`, then the prompt in `P` lines of at most 64
ids each (a prompt of 1843 tokens is 29 lines), then one `G`. The accelerator
answers with one `T` line per generated token, as it is produced, and ends with
`D eos <n>` when a stop id ended the run or `D max <n>` when `max_new` did.

The host checks what it receives: more than `max_new` tokens, a count in `D`
that does not match the tokens received, or a line that is none of `T`, `D`
and `E` ends the call with an error, and the turn ends with `model_error`.

In Rust, the host side is one function over any `Read + Write` stream:

```rust
use webagents::model::accel::generate;

let mut link = std::net::TcpStream::connect("127.0.0.1:7100")?;
let generation = generate(&mut link, &prompt_ids, 512, &stop_ids, |id| {
    // called for each token as it arrives
})?;
// generation.tokens: the ids; generation.stopped: true when a stop id ended the run
```

The chip side exists too, for a software stand-in in tests:
`webagents::model::accel::serve_one` reads one request from a stream and
answers it with a closure you supply as the model.

## The tokenizer

`webagents::model::bpe::Tokenizer` reads a Hugging Face `tokenizer.json` of the
byte-level BPE (byte pair encoding) family. It is plain Rust with no native
library, so it builds for any host the gateway runs on.

It follows the file's own pipeline:

1. added tokens (`<|im_start|>`, `<tool_call>` and the like) are matched
   literally, longest first, and become their own ids;
2. the rest is NFC-normalized (Unicode Normalization Form C) when the file says so;
3. the pre-tokenizer's regular expression, taken from the file, cuts it into
   words;
4. each word's UTF-8 bytes are mapped to the byte-level alphabet and merged by
   rank.

Only what this family uses is accepted: a `BPE` model, no normalizer or `NFC`,
and a pre-tokenizer that is `ByteLevel` with its own regex or a sequence of an
isolated `Split` and a `ByteLevel`. Anything else is refused by name when the
file is loaded. It is never approximated, because a tokenizer that is nearly
right produces ids the model was not trained on, and the model degrades
without any error.

```rust
use webagents::model::bpe::Tokenizer;

let tokenizer = Tokenizer::from_file(std::path::Path::new("qwen3/tokenizer.json"))?;
let ids = tokenizer.encode("<|im_start|>user\nRead the status.<|im_end|>\n");
assert_eq!(tokenizer.decode(&ids), "<|im_start|>user\nRead the status.<|im_end|>\n");
let im_end = tokenizer.special("<|im_end|>");   // Some(id)
```

## The chat template

`webagents::model::chat::render_qwen3` renders the Qwen3 chat template with
tools: the system message, the tool list inside `<tools>`, the instruction to
answer with `<tool_call>` blocks, the conversation, tool results as
`<tool_response>` blocks, and the opening of the assistant's answer with an
empty thinking block (thinking off).

```text
<|im_start|>system
You operate the following hardware through its tools: ...

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"function": {"description": "Read CTRL (Control) of the example three-phase PWM driver.", "name": "pwm1_read_ctrl", "parameters": {...}}, "type": "function"}
...
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
Set the dead time to 1.5 us.<|im_end|>
<|im_start|>assistant
<think>

</think>

```

The backend uses this template for every model, so the model on the
accelerator must be one that was trained with it.

The model answers with blocks like this one, which
`webagents::model::chat::parse_tool_calls` turns into calls:

```text
<tool_call>
{"name": "pwm1_set_deadtime", "arguments": {"deadtime": 1.5}}
</tool_call>
```

A block whose JSON does not parse, or that has no `name`, is not run. It stays
in the text, so the person sees what the model wrote.

### Why byte for byte

A model on an accelerator sees only token ids, so the host builds the prompt
itself. A prompt that differs from the one the model was trained on by one byte
(a separator in the tools JSON, a newline) does not fail; it quietly makes tool
use worse. So the rendering and the tokenizer are pinned to test vectors
produced by Hugging Face `transformers` and `tokenizers` themselves, for
`Qwen/Qwen3-0.6B`: the rendered prompts and the token ids must match byte for
byte. The tools JSON is printed the way Python's `json.dumps` prints it,
separators and number formatting included, because that is what the template
does.

## Try it without an accelerator

The example in
[`rust/examples/accelerator`](https://github.com/robutlerai/webagents/tree/main/rust/examples/accelerator)
runs the whole loop on one machine: a stand-in for the accelerator that serves
the link, a mock of the platform, a simulated part, and the gateway between
them. It shows a tool call parsed from token ids, refused by the part's
envelope, and the refusal reaching the model in the next prompt.

## Writing another backend

The accelerator backend is one implementation of a small trait. A backend for
another local runtime implements the same one and is passed to
`webagents::gateway::run_portal` in place of the configured one:

```rust
use webagents::gateway::llm::BoxFuture;
use webagents::gateway::{GatewayError, ModelBackend, ModelReply, ModelRequest};

struct MyRuntime;

impl ModelBackend for MyRuntime {
    fn complete(&self, req: ModelRequest) -> BoxFuture<'_, Result<ModelReply, GatewayError>> {
        Box::pin(async move {
            // req.messages and req.tools are OpenAI-style JSON values.
            // Return text, tool calls, or both. Never run a tool here:
            // the gateway runs them through each device's envelope.
            Ok(ModelReply { text: "...".into(), tool_calls: vec![] })
        })
    }
}
```
