"""``webagents cas`` / ``robutler cas``: CASFS projects from the command line.

::

    cas ls <project>[:<path>]
    cas cat <project>:<path>
    cas mounts <project>
    cas mount <project> <path> <source> [--root <sha>]   source: project:<id> | <id> | pdk:<pkg>[@<version>]
    cas mount <project> --pdk <pkg>[@<version>]           lands at pdk/<pkg>@<version>
    cas unmount <project> <path>
    cas update <project> <path> [--root <sha>]

Every command goes through ``CasFs`` under the CLI's login. Reads are verified
chunk by chunk and cached on this machine (``~/.cache/robutler/casfs``).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from ..casfs import CasError, CasFs

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def parse_target(arg: str) -> Dict[str, str]:
    project, _, path = arg.partition(":")
    if not _UUID.match(project):
        raise CasError("Cas:PathInvalid", f"expected <project id>[:<path>], got {arg}")
    return {"project": project.lower(), "path": path}


def parse_source(arg: str) -> Dict[str, str]:
    if arg.startswith("pdk:"):
        pkg, _, version = arg[4:].partition("@")
        if not pkg:
            raise CasError("Cas:MountSource", "pdk:<pkg>[@<version>]")
        return {"kind": "pdk", "pkg": pkg, **({"version": version} if version else {})}
    pid = arg[8:] if arg.startswith("project:") else arg
    if not _UUID.match(pid):
        raise CasError("Cas:MountSource", "a source is project:<id>, <id> or pdk:<pkg>[@<version>]")
    return {"kind": "project", "id": pid.lower()}


async def run_cas(fs: CasFs, sub: str, args: List[str], root: Optional[str], pdk: Optional[str], out: Callable[[str], None], write: Callable[[bytes], None]) -> Any:
    first = args[0] if args else ""
    if sub == "ls":
        t = parse_target(first)
        rows = await (await fs.open(t["project"])).list(t["path"])
        for r in rows:
            if r["type"] == "mount":
                s = r["source"]
                kind = f"mount {s['pkg'] + '@' + s['version'] if s['kind'] == 'pdk' else s['id']} {r['root'][:12]}"
            else:
                kind = r["type"]
            size = str(r["size"]).rjust(12) if r["type"] == "file" else " " * 12
            out(f"{size}  {kind.ljust(6)}  {r['name']}{'/' if r['type'] in ('dir', 'mount') else ''}")
        return rows
    if sub == "cat":
        t = parse_target(first)
        if not t["path"]:
            raise CasError("Cas:PathInvalid", "cat <project>:<path>")
        h = await fs.open(t["project"])
        async for part in h.read_stream(t["path"]):
            write(part)
        return {"path": t["path"]}
    if sub == "mounts":
        t = parse_target(first)
        ms = await (await fs.open(t["project"])).mounts()
        for m in ms:
            s = m["source"]
            out(f"{m['path']}  {s['pkg'] + '@' + s['version'] if s['kind'] == 'pdk' else s['id']}  {m['root'][:12]}")
        return ms
    if sub == "mount":
        t = parse_target(first)
        h = await fs.open(t["project"])
        if pdk:
            pkg, _, version = pdk.partition("@")
            r = await h.ensure_pdk_mount(pkg, version or None, commit=True)
            out(f"{'Mounted' if r.get('added') else 'Already mounted'} {r['path']}")
            return r
        if len(args) < 3:
            raise CasError("Cas:PathInvalid", "mount <project> <path> <source>, or mount <project> --pdk <pkg>[@<version>]")
        r = await h.mount(args[1], parse_source(args[2]), root)
        out(f"Mounted {r['path']} at {r['entry']['root'][:12]}")
        return r
    if sub == "unmount":
        t = parse_target(first)
        r = await (await fs.open(t["project"])).unmount(args[1] if len(args) > 1 else "")
        out(f"Unmounted {r['path']}")
        return r
    if sub == "update":
        t = parse_target(first)
        r = await (await fs.open(t["project"])).update_mount(args[1] if len(args) > 1 else "", root)
        out(f"{r['path']}: {r['from'][:12]} -> {r['entry']['root'][:12]}")
        return r
    raise CasError("Cas:PathInvalid", f"unknown cas command {sub}: ls, cat, mounts, mount, unmount, update")
