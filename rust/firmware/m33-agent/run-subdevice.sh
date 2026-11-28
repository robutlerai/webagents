#!/bin/bash
# The signed tool description, end to end in emulation: the M33 firmware in the
# sub-device role (built with AGENT_ROLE=subdevice and AGENT_DEVICE_SEED_FILE) on
# QEMU mps2-an505, its UART1 a TCP server; the gateway connects as the bus master.
#   ROBUTLER=<robutler binary> QEMU_SYSTEM_ARM=<qemu> ./run-subdevice.sh <firmware ELF>
# Checks: (1) the gateway verifies the description with the key the part printed;
# (2) a wrong key is refused and no tool is offered; (3) a raw write below the
# signed limit is refused by the part itself and the register keeps its value.
set -u
ELF=$1
PORT=${PORT:-18310}
HERE=$(cd "$(dirname "$0")" && pwd)
SPEC=$HERE/../../fixtures/pwm1.regspec.json
W=$(mktemp -d); trap 'kill $QP 2>/dev/null; rm -f "$W"/*.toml "$W"/console.out; rmdir "$W"' EXIT
LINK_LISTEN=127.0.0.1:$PORT timeout 300 "$HERE/run-qemu.sh" "$ELF" > "$W/console.out" 2>&1 & QP=$!
for _ in $(seq 100); do grep -q "public key x" "$W/console.out" && break; sleep 0.2; done
X=$(sed -n 's/.*public key x \([A-Za-z0-9_-]*\).*/\1/p' "$W/console.out")
[ -n "$X" ] || { echo "FAIL: the part printed no public key"; cat "$W/console.out"; exit 1; }
conf() { printf '[portal]\nurl = "ws://127.0.0.1:1/ws"\nagent = "pwm-m33"\n\n[[device]]\nspec = "%s"\ntransport = { kind = "tcp", addr = "127.0.0.1:%s", timeout_ms = 20000 }\ndevice_key = "%s"\n' "$SPEC" "$PORT" "$1" > "$W/$2.toml"; }
conf "$X" good
# A key that is not the part's: the same length, one character changed.
BAD=$(python3 -c "import sys;x=sys.argv[1];print(('B' if x[0]!='B' else 'C')+x[1:])" "$X"); conf "$BAD" bad
"$ROBUTLER" gateway check "$W/good.toml" > good.out 2>&1; G=$?
"$ROBUTLER" gateway check "$W/bad.toml" > bad.out 2>&1; B=$?
python3 - "$PORT" > raw.out <<'PY'
import socket,sys
s=socket.create_connection(("127.0.0.1",int(sys.argv[1])),timeout=20); f=s.makefile("rwb")
def ask(l):
    f.write((l+"\n").encode()); f.flush()
    while True:
        a=f.readline().decode().strip()
        if a[:1] in "VKE": return a
for l in ["R 10","W 10 00000003","R 10","W 10 00000014","R 10"]: print(l,"->",ask(l))
f.write(b"Q\n"); f.flush()
PY
R=$?
wait $QP; QE=$?
echo "--- part console"; cat "$W/console.out"
echo "--- gateway, right key (exit $G)"; cat good.out
echo "--- gateway, wrong key (exit $B)"; cat bad.out
echo "--- raw serial (exit $R)"; cat raw.out
ok=1
[ $G -eq 0 ] && grep -q verified good.out || { echo "FAIL: right key not verified"; ok=0; }
[ $B -ne 0 ] && grep -q "does not verify" bad.out || { echo "FAIL: wrong key not refused"; ok=0; }
grep -q "^W 10 00000003 -> E" raw.out && [ "$(sed -n 3p raw.out)" = "R 10 -> V 0000000a" ] || { echo "FAIL: low write not refused by the part"; ok=0; }
grep -q "^W 10 00000014 -> K" raw.out && grep -q "^R 10 -> V 00000014" raw.out || { echo "FAIL: in-limit write"; ok=0; }
[ $ok -eq 1 ] && echo "SUBDEVICE PASS" || { echo "SUBDEVICE FAIL"; exit 1; }
