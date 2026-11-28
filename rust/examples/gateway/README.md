# Gateway configurations

A gateway puts parts on the agent network as one agent. These two
configurations are the smallest starting points.

| File | What it is |
| --- | --- |
| [`sim.toml`](sim.toml) | One simulated part (`kind = "sim"`): an in-memory copy of the part at its reset values |
| [`serial-and-can.toml`](serial-and-can.toml) | Two parts behind one gateway, one on a serial line and one on CAN |
| [`pwm2.regspec.json`](pwm2.regspec.json) | The spec of the second part |

```sh
robutler-device gateway check examples/gateway/sim.toml    # load, open every device, connect to nothing
```

To run it, the agent needs a credential: either `auth = "device"` in `[portal]`
after `robutler-device agent login`, or a per-agent key in
`WEBAGENTS_AGENT_TOKEN`.

```sh
WEBAGENTS_AGENT_TOKEN=... robutler-device gateway run examples/gateway/sim.toml
```

`serial-and-can.toml` runs without hardware against simulated parts on real
transports; the commands are in the comment at its top.
