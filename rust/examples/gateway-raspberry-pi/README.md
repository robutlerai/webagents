# A gateway on a Raspberry Pi (serial + I2C)

Two parts behind a Raspberry Pi, offered as one agent: a motor driver whose
firmware answers on the Pi's serial port, and a heater controller on I2C.

| File | What it is |
| --- | --- |
| [`gateway.toml`](gateway.toml) | The gateway configuration |
| [`robutler-device.service`](robutler-device.service) | A systemd unit that keeps the gateway running |

On the Pi:

```sh
sudo apt install build-essential
cargo install robutler
cd examples/gateway-raspberry-pi                                     # gateway.toml is here

# The buses: enable the serial port and I2C in `sudo raspi-config`, then
sudo usermod -aG dialout,i2c $USER        # log out and in again

robutler-device gateway check gateway.toml                           # opens both parts
robutler-device agent login @bench --portal https://robutler.ai      # allow it on your phone
robutler-device gateway run gateway.toml
```

The configuration refers to the two specs in this repository by relative path.
On a Pi without a checkout, copy the two `.regspec.json` files next to
`gateway.toml` and shorten the `spec` paths.

To try the configuration before the parts are wired, change both transports to
`{ kind = "sim" }`.

The full walk-through, with every configuration key and transport, is the
[Gateway on a Raspberry Pi](../../../docs/rust/gateway-raspberry-pi.md) page.
