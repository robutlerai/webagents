# Examples

One folder per use case. Each has a short README with the commands to run.
Commands are written to be run from the `rust/` directory, with `robutler-device`
installed (`cargo install robutler`) or replaced by `cargo run -p robutler --`.

| Folder | What it shows | Needs hardware |
| --- | --- | --- |
| [`regspec-tools`](regspec-tools) | A register spec turned into typed tools, called from Rust and from the command line, with one refusal of each kind | no |
| [`gateway`](gateway) | The smallest gateway configuration: one simulated part. Also two parts on a serial line and CAN | no |
| [`gateway-raspberry-pi`](gateway-raspberry-pi) | A gateway on a Raspberry Pi with a part on the serial port and a part on I2C, device sign-in, and a systemd unit | a Raspberry Pi and the parts |
| [`gateway-can`](gateway-can) | A gateway with a part on a CAN bus, tried on a virtual CAN interface | no (Linux with `vcan`) |
| [`accelerator`](accelerator) | The gateway's model on the board's own AI accelerator, run locally with a stand-in for the accelerator | no |
| [`firmware`](firmware) | Where the microcontroller (Cortex-M33) firmware is and how to run it in emulation | no (QEMU) |

The Rust programs among them are cargo examples of the `webagents` crate, in
[`crates/webagents/examples`](../crates/webagents/examples), so they are built
and linted with the crate:

| Example | Run it with |
| --- | --- |
| `regspec_tools` | `cargo run -p webagents --example regspec_tools` |
| `accel_stand_in` | `cargo run -p webagents --features model --example accel_stand_in -- <listen addr> <tokenizer.json>` |
| `mock_platform` | `cargo run -p webagents --features gateway --example mock_platform -- <listen addr> <agent> <key>` |

The documentation for all of this is in the Rust section of the
[docs site](../../docs/rust/index.md).
