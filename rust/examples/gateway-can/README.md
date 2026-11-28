# A gateway with a part on CAN

One part on a CAN bus, reached through SocketCAN. Each register access is one
request frame and one answer frame:

```text
request  id 0x600 + node   [op, off_lo, off_hi, v0, v1, v2, v3]   op 0x40 read, 0x23 write
answer   id 0x580 + node   [st, off_lo, off_hi, v0, v1, v2, v3]   st 0x43 read ok, 0x60 write ok, 0x80 refused
```

Classic CAN frames with 11-bit identifiers, values little-endian, register
offsets up to `0xffff`, node ids 0 to 127.

## Without hardware

A virtual CAN interface and a simulated part are enough to run the whole path:

```sh
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0

robutler-device twin examples/gateway/pwm2.regspec.json can vcan0 5 &     # the part, as node 5
robutler-device gateway check examples/gateway-can/gateway.toml           # the gateway opens it
```

## With a CAN interface

Bring the interface up with the bus's bit rate, then set `iface = "can0"` and
the part's node id in [`gateway.toml`](gateway.toml):

```sh
sudo ip link set can0 up type can bitrate 500000
robutler-device gateway check examples/gateway-can/gateway.toml
WEBAGENTS_AGENT_TOKEN=... robutler-device gateway run examples/gateway-can/gateway.toml
```

A part's firmware serves the two frames above from its own register table. If
it runs the `webagents` engine over the same spec, it refuses a write outside
its limits itself (status `0x80`), whatever sent the frame.
