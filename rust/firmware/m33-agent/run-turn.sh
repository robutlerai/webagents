#!/bin/bash
# The emulated end-to-end turn: mock platform <- ESP-AT twin <- QEMU mps2-an505 firmware.
#   ROBUTLER=<robutler binary> MOCK=<mock_platform binary> QEMU_SYSTEM_ARM=<qemu> ./run-turn.sh <firmware ELF>
# The firmware must be built with AGENT_HOST=127.0.0.1 AGENT_PORT=$MOCK_PORT.
set -u
ELF=$1
MOCK_PORT=${MOCK_PORT:-18080}
TWIN_PORT=${TWIN_PORT:-18081}
KEY=${AGENT_TOKEN:-placeholder-not-a-real-key}
"$MOCK" 127.0.0.1:$MOCK_PORT pwm-m33 "$KEY" > mock.out 2> mock.err & MP=$!
"$ROBUTLER" twin esp-at --listen 127.0.0.1:$TWIN_PORT > twin.out 2>&1 & TP=$!
sleep 1
LINK=127.0.0.1:$TWIN_PORT timeout 300 "$(dirname "$0")/run-qemu.sh" "$ELF"; QE=$?
wait $MP; ME=$?
wait $TP 2>/dev/null
echo "--- twin"; cat twin.out; echo "--- mock"; cat mock.err mock.out
echo "qemu exit=$QE mock exit=$ME"
[ $QE -eq 0 ] && [ $ME -eq 0 ]
