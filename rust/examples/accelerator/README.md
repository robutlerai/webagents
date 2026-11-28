# The model on the board's AI accelerator

With `kind = "accelerator"` in `[llm]`, the gateway's model runs on the board's
own AI accelerator. The gateway renders the conversation with the model's chat
template, turns it into token ids with the model's tokenizer, sends the ids
over a link to the accelerator and parses the tool calls out of the ids that
come back. No model call leaves the board, and every tool call still goes
through the part's envelope.

| File | What it is |
| --- | --- |
| [`gateway.toml`](gateway.toml) | A gateway configuration with the accelerator backend |
| [`tokenizer.json`](tokenizer.json) | A byte-level tokenizer for the local run below: one token per byte, plus the chat template's special tokens. It is not a model's tokenizer |
| [`accel_stand_in.rs`](../../crates/webagents/examples/accel_stand_in.rs) | A stand-in for the accelerator: it serves the token link with a script in place of a model |

## Run the whole loop on one machine

Three processes, started in this order from `rust/`. Nothing leaves the
machine, and no account is needed.

```sh
# 1. The accelerator stand-in, serving the token link.
cargo run -p webagents --features model --example accel_stand_in -- \
    127.0.0.1:7100 examples/accelerator/tokenizer.json

# 2. A mock of the platform. It sends one message from a person and prints
#    every tool call, result and reply the agent sends back.
cargo run -p webagents --features gateway --example mock_platform -- \
    127.0.0.1:18080 accel-bench local-key --say 'Set the dead time to 0.3 us.'

# 3. The gateway, with a simulated part.
WEBAGENTS_AGENT_TOKEN=local-key cargo run -p robutler -- gateway run examples/accelerator/gateway.toml
```

The mock prints the turn as the person's chat would show it:

```text
SAY Set the dead time to 0.3 us.
CALL pwm1_set_deadtime {"deadtime":0.3}
RESULT error {"ok":false,"error":"out_of_range","message":"DEADTIME = 0.3 is outside its limits 0.5 to 25.5 us. Nothing was written."}
TEXT The part refused: DEADTIME = 0.3 is outside its limits 0.5 to 25.5 us. Nothing was written.
MOCK PASS: 0 scripted turn(s), 1 spoken turn(s), 0 platform model calls, frames and tool results checked
```

What happened:

1. The gateway rendered the person's message and the part's 20 tools into one
   prompt and sent it to the stand-in as token ids.
2. The stand-in answered with the ids of a `<tool_call>` block asking for a
   dead time of 0.3 us.
3. The gateway parsed the call and ran it through the part's envelope, which
   refused it: the dead time may not go below 0.5 us. The register was not
   written.
4. The refusal went back in the next prompt as a tool response, and the
   stand-in answered in text.

The stand-in prints what crossed the link, which is token ids only (the
tokenizer here makes one token per byte, so the prompt with its 20 tools is
about 10,000 tokens):

```text
accelerator stand-in on 127.0.0.1:7100
9935 prompt tokens in, 66 tokens out: <tool_call> {"name": "pwm1_set_deadtime", "arguments": {"deadtime": 0.3}} </tool_call>
10174 prompt tokens in, 92 tokens out: The part refused: DEADTIME = 0.3 is outside its limits 0.5 to 25.5 us. Nothing was written.
```

The mock fails the run if the gateway calls the platform's model at all: with
this backend the platform is never asked for a completion.

Stop the gateway and the stand-in with Ctrl-C when the mock has exited.

## On a real board

Change three things in `gateway.toml`:

- `[portal]`: the platform (`url = "wss://robutler.ai/ws"`) and the agent's
  credential;
- `link`: where the board serves the accelerator's token link, as `host:port`;
- `tokenizer`: the `tokenizer.json` that belongs to the model on the
  accelerator.

The backend renders the Qwen3 chat template, so the model must be one trained
with it. The link protocol, the tokenizer and the template are described on the
[Accelerator backend](../../../docs/rust/accelerator-backend.md) page.
