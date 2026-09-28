"""Fixture script that probes the sandbox: it tries to write a file and to
reach a URL, and reports each result on its own line. Both SDKs' confinement
tests run it through `run_skill_script` and expect every probe to be refused
outside the agent's allowed folders and the agent's `network:` list.

    probe.py PROBE-OK
    probe.py --write /path/to/file
    probe.py --fetch http://127.0.0.1:PORT/

Fetches go through the sandbox's proxy when it names one (HTTP_PROXY), never
direct: srt sets NO_PROXY for loopback, and a direct connection from inside
the sandbox is refused whatever the allow-list says, which is what the
tests' local server would otherwise measure.
"""

import os
import sys
import urllib.request

print("PROBE-OK")
proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
# urllib bypasses the proxy for every host NO_PROXY names, loopback included.
for name in ("NO_PROXY", "no_proxy"):
    os.environ.pop(name, None)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy} if proxy else {}))
args = sys.argv[1:]
while args:
    verb = args.pop(0)
    target = args.pop(0) if args else ""
    if verb == "--write":
        try:
            with open(target, "w", encoding="utf-8") as handle:
                handle.write("written by probe\n")
            print("write ok " + target)
        except OSError as error:
            print("write refused " + error.strerror)
    elif verb == "--fetch":
        try:
            with opener.open(target, timeout=5) as response:
                print("fetch ok " + response.read().decode("utf-8", "replace").strip())
        except Exception as error:  # noqa: BLE001 - the refusal is the point
            print("fetch refused " + type(error).__name__)
