#!/bin/bash
# Runs a firmware ELF on QEMU mps2-an505. UART0 -> stdout; UART1 -> TCP client to
# $LINK (host:port, the network co-processor twin) when set.
# QEMU comes from $QEMU_SYSTEM_ARM (a qemu-system-arm with mps2-an505).
set -e
ELF=$1; shift
Q=${QEMU_SYSTEM_ARM:-qemu-system-arm}
LINK=${LINK:-}; LINK_LISTEN=${LINK_LISTEN:-}
LINKARG=(-serial null)
[ -n "$LINK" ] && LINKARG=(-serial "tcp:$LINK")
# LINK_LISTEN=host:port: UART1 is a TCP server instead (a gateway connects to the part).
[ -n "$LINK_LISTEN" ] && LINKARG=(-serial "tcp:$LINK_LISTEN,server=on,wait=off")
exec "$Q" -machine mps2-an505 -cpu cortex-m33 -nographic -monitor none \
  -semihosting-config enable=on,target=native \
  -serial stdio "${LINKARG[@]}" -kernel "$ELF" "$@"
