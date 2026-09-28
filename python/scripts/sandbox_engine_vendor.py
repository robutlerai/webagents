#!/usr/bin/env python3
"""
Vendor srt, `@anthropic-ai/sandbox-runtime`, into the Python package (the
sandbox-engine lane, 2026-09-27).

WHY. The sandbox is on by default and fails closed. A pip package cannot
declare an npm dependency, so until this lane a Python user without Node and
a separately installed srt had every shell command refused. The engine now
ships inside the wheel, at `webagents/sandbox/sandbox_engine/node_modules/`,
laid out as npm lays out a package and its dependencies, so
`node .../@anthropic-ai/sandbox-runtime/dist/cli.js` resolves `commander`,
`zod`, `node-forge` and `@pondwader/socks5-server` the ordinary way.

THE BYTES ARE THE PUBLISHED ONES. Each package is the npm registry's tarball
for the exact version `typescript/pnpm-lock.yaml` pins, checked against the
lockfile's sha512 integrity before anything is extracted, so the Python
engine is byte for byte what the TypeScript package installs. Nothing is
rebuilt or bundled into one file. Left out, because Node never loads them
here: TypeScript sources and declarations (zod's `src/` included), source
maps, Markdown, a CI workflow, node-forge's Flash socket policy file and its
browser bundles (`dist/`; Node loads `lib/`), and from srt the Windows helper
(`vendor/srt-win`, 5.8 MB; this SDK has no Windows sandbox) and the Bun build
scripts (`vendor/*/build.ts`).
Kept from srt's `vendor/`: the Linux seccomp helpers
(`vendor/seccomp/{x64,arm64}/apply-seccomp`) and the Java proxy agent
(`vendor/java-proxy-agent/srt-proxy-agent.jar`), both loaded at run time.

`sandbox_engine_provenance.json` beside the tree records, per package, the
tarball URL, its integrity, its license, the sha256 of every file kept, and
the files left out. `NOTICE` names each component and its license; every
package keeps its own LICENSE file.

Usage (from `webagents/python`):
    .venv/bin/python scripts/sandbox_engine_vendor.py          fetch, verify, extract, record (network)
    .venv/bin/python scripts/sandbox_engine_vendor.py --check  verify the tree against the record and the lockfile

`--check` needs no network and is what `tests/sandbox/sandbox_engine_bundle_test.py`
runs, so a lockfile bump without a re-vendor fails the Python suite.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

PY_ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = PY_ROOT / "webagents" / "sandbox" / "sandbox_engine"
MODULES_DIR = ENGINE_DIR / "node_modules"
RECORD = ENGINE_DIR / "sandbox_engine_provenance.json"
NOTICE = ENGINE_DIR / "NOTICE"
LOCKFILE = PY_ROOT.parent / "typescript" / "pnpm-lock.yaml"
TS_PACKAGE = PY_ROOT.parent / "typescript" / "package.json"
SRT_MODULE = PY_ROOT / "webagents" / "sandbox" / "srt.py"

SRT = "@anthropic-ai/sandbox-runtime"
REGISTRY = "https://registry.npmjs.org"

#: Suffixes Node never loads here: TypeScript (sources and declarations),
#: source maps, Markdown, CI workflows, Flash.
OMIT_SUFFIXES = (".ts", ".cts", ".mts", ".map", ".md", ".yml", ".yaml", ".swf")
#: Paths left out per package: srt's Windows helper (its Bun build scripts
#: are `.ts`), and node-forge's browser bundles.
OMIT_PATHS = {
    SRT: re.compile(r"^vendor/srt-win/"),
    "node-forge": re.compile(r"^dist/"),
}

#: Who holds each component, for NOTICE (from each package's own LICENSE or package.json).
HOLDERS = {
    SRT: "Anthropic PBC",
    "@pondwader/socks5-server": "PondWader",
    "commander": "TJ Holowaychuk",
    "node-forge": "Digital Bazaar, Inc.",
    "zod": "Colin McDonnell",
}


def srt_version() -> str:
    """`SRT_VERSION` as `webagents/sandbox/srt.py` pins it (read, not imported)."""
    match = re.search(r'^SRT_VERSION = "([^"]+)"', SRT_MODULE.read_text(encoding="utf-8"), re.M)
    if not match:
        raise SystemExit(f"SRT_VERSION not found in {SRT_MODULE}")
    return match.group(1)


def lockfile_closure(version: str) -> List[Tuple[str, str, str]]:
    """srt at `version` and every package it depends on, transitively, as
    `(name, version, integrity)` from the lockfile, srt first."""
    lock = yaml.safe_load(LOCKFILE.read_text(encoding="utf-8"))
    packages = lock.get("packages") or {}
    snapshots = lock.get("snapshots") or {}
    out: List[Tuple[str, str, str]] = []
    pending = [(SRT, version)]
    seen = set()
    while pending:
        name, ver = pending.pop(0)
        key = f"{name}@{ver}"
        if key in seen:
            continue
        seen.add(key)
        entry = packages.get(key)
        if not entry or "integrity" not in (entry.get("resolution") or {}):
            raise SystemExit(f"{key} has no integrity in {LOCKFILE}")
        out.append((name, ver, entry["resolution"]["integrity"]))
        for dep, dep_ver in sorted(((snapshots.get(key) or {}).get("dependencies") or {}).items()):
            pending.append((dep, str(dep_ver).split("(")[0]))
    return out


def tarball_url(name: str, version: str) -> str:
    return f"{REGISTRY}/{name}/-/{name.split('/')[-1]}-{version}.tgz"


def integrity_of(data: bytes) -> str:
    return "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode("ascii")


def omitted(name: str, relative: str) -> bool:
    if relative.lower().endswith(OMIT_SUFFIXES):
        return True
    pattern = OMIT_PATHS.get(name)
    return bool(pattern and pattern.search(relative))


def extract(name: str, data: bytes) -> Tuple[Dict[str, str], List[str]]:
    """Write the kept files of one tarball under `node_modules/<name>/`; return
    `({relative: sha256}, [omitted relative paths])`. Only regular files are
    taken, and a path that would leave the package folder is refused."""
    target = MODULES_DIR / name
    kept: Dict[str, str] = {}
    left_out: List[str] = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            relative = "/".join(parts[1:])  # npm tarballs nest everything under one folder, `package/`
            if not relative or relative.startswith("/") or ".." in parts:
                raise SystemExit(f"{name}: refusing tarball entry {member.name!r}")
            if omitted(name, relative):
                left_out.append(relative)
                continue
            handle = archive.extractfile(member)
            assert handle is not None
            content = handle.read()
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            path.chmod(0o755 if member.mode & 0o111 else 0o644)
            kept[relative] = hashlib.sha256(content).hexdigest()
    return dict(sorted(kept.items())), sorted(left_out)


def package_license(name: str) -> str:
    data = json.loads((MODULES_DIR / name / "package.json").read_text(encoding="utf-8"))
    return str(data.get("license") or "")


def write_notice(entries: List[Dict[str, object]]) -> None:
    lines = [
        "The sandbox engine that ships with webagents",
        "",
        "This folder carries srt, the sandbox runtime, and the packages it depends on,",
        "as published to the npm registry and unmodified (files never loaded at run time",
        "are left out; sandbox_engine_provenance.json lists every file and its sha256).",
        "Each package keeps its own LICENSE file in its folder under node_modules/.",
        "",
    ]
    for entry in entries:
        name = str(entry["name"])
        license_text = str(entry["license"])
        if name == "node-forge":
            license_text += ", used under BSD-3-Clause"
        lines += [
            f"{name} {entry['version']}",
            f"  Copyright {HOLDERS.get(name, 'its authors')}",
            f"  License: {license_text} (node_modules/{name}/LICENSE)",
            f"  Source: {entry['tarball']}",
            "",
        ]
    NOTICE.write_text("\n".join(lines), encoding="utf-8")


def vendor() -> None:
    version = srt_version()
    closure = lockfile_closure(version)
    if MODULES_DIR.exists():
        shutil.rmtree(MODULES_DIR)
    entries: List[Dict[str, object]] = []
    for name, ver, integrity in closure:
        url = tarball_url(name, ver)
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - a fixed https registry URL
            data = response.read()
        got = integrity_of(data)
        if got != integrity:
            raise SystemExit(f"{name}@{ver}: tarball integrity {got} does not match the lockfile's {integrity}")
        kept, left_out = extract(name, data)
        entries.append(
            {
                "name": name,
                "version": ver,
                "license": package_license(name),
                "tarball": url,
                "integrity": integrity,
                "files": kept,
                "omitted": left_out,
            }
        )
        print(f"{name}@{ver}: {len(kept)} files kept, {len(left_out)} left out", file=sys.stderr)
    record = {
        "about": (
            "srt and its dependencies as vendored by scripts/sandbox_engine_vendor.py: each tarball is the "
            "npm registry's for the version typescript/pnpm-lock.yaml pins, checked against the lockfile's "
            "integrity, and every kept file is listed with its sha256. `--check` verifies the tree against this."
        ),
        "engine": SRT,
        "version": version,
        "lockfile": "typescript/pnpm-lock.yaml",
        "packages": entries,
    }
    RECORD.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    write_notice(entries)


def check() -> List[str]:
    """What is wrong with the vendored tree, or nothing. No network."""
    problems: List[str] = []
    if not RECORD.is_file():
        return [f"{RECORD} is missing; run scripts/sandbox_engine_vendor.py"]
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    version = srt_version()
    if record.get("version") != version:
        problems.append(f"the record vendors {SRT} {record.get('version')}, srt.py pins {version}")
    pinned = json.loads(TS_PACKAGE.read_text(encoding="utf-8")).get("dependencies", {}).get(SRT)
    if pinned != version:
        problems.append(f"typescript/package.json pins {SRT} {pinned}, srt.py pins {version}")
    expected = {(name, ver): integrity for name, ver, integrity in lockfile_closure(version)}
    recorded = {(p["name"], p["version"]): p["integrity"] for p in record.get("packages", [])}
    if recorded != expected:
        problems.append(f"the record's packages {sorted(recorded)} differ from the lockfile's {sorted(expected)}")
    listed = set()
    for package in record.get("packages", []):
        for relative, digest in package["files"].items():
            path = MODULES_DIR / package["name"] / relative
            listed.add(path)
            if not path.is_file():
                problems.append(f"missing: {path.relative_to(ENGINE_DIR)}")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                problems.append(f"changed: {path.relative_to(ENGINE_DIR)}")
    present = {path for path in MODULES_DIR.rglob("*") if path.is_file()} if MODULES_DIR.is_dir() else set()
    for extra in sorted(present - listed):
        problems.append(f"not in the record: {extra.relative_to(ENGINE_DIR)}")
    return problems


def main(argv: List[str]) -> int:
    if argv[1:] == ["--check"]:
        problems = check()
        for problem in problems:
            print(problem, file=sys.stderr)
        print("sandbox engine: " + ("does not match its record" if problems else "matches its record and the lockfile"))
        return 1 if problems else 0
    if argv[1:]:
        print(__doc__, file=sys.stderr)
        return 2
    vendor()
    problems = check()
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
