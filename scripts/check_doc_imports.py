#!/usr/bin/env python3
"""Check that every `webagents` import in the docs actually resolves.

WHY THIS EXISTS. A documentation review in 2026-09 found that the docs named
classes that do not exist (`OpenAILLMSkill` for `OpenAISkill`), imported from
subpaths the package does not export (`webagents/server/node`), and imported
symbols from the package root that live elsewhere (`pricing` in Python). Every
one of those is mechanically checkable, and none of them were caught, because
nothing read the docs against the packages.

WHAT IT CHECKS. Only imports of this project's own packages, in fenced code
blocks:

  TypeScript   import { A, B } from 'webagents'
               import { A } from 'webagents/skills/llm'
  Python       from webagents.x.y import A, B
               import webagents.x.y

For TypeScript it resolves the built `dist/` entry point and asserts each named
symbol is actually exported. For Python it imports the module in the repo venv
and asserts each name exists on it.

WHAT IT DOES NOT CHECK. That an example RUNS. Examples needing a model
provider, a served port or a platform account cannot run here. Import
resolution is the layer where the documented errors actually were, and it is
cheap enough to run on every change.

Exit 1 on any unresolved import, so it can gate a build.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
DIST = REPO / "typescript" / "dist" / "index.js"
VENV = REPO / "python" / ".venv" / "bin" / "python"

FENCE_RE = re.compile(r"```(?P<info>[^\n]*)\n(?P<body>.*?)```", re.DOTALL)
# `import type { ... }` is erased at compile time, so there is nothing to find
# at runtime and nothing to check. Only value imports are collected.
TS_IMPORT_RE = re.compile(
    r"import\s+\{(?P<names>[^}]*)\}\s*from\s*['\"](?P<mod>webagents[^'\"]*)['\"]"
)
PY_FROM_RE = re.compile(
    r"^\s*from\s+(?P<mod>webagents[\w.]*)\s+import\s+(?P<names>[^\n#]+)", re.MULTILINE
)


def code_blocks(text: str):
    for m in FENCE_RE.finditer(text):
        info = m.group("info").lower()
        lang = info.split()[0] if info.split() else ""
        yield lang, m.group("body")


def collect():
    """(file, lang, module, [names]) for every webagents import in the docs."""
    ts, py = [], []
    for f in sorted(list(DOCS.rglob("*.md")) + list(DOCS.rglob("*.mdx"))):
        rel = str(f.relative_to(REPO))
        for lang, body in code_blocks(f.read_text()):
            if lang in ("typescript", "ts", "javascript", "js"):
                for m in TS_IMPORT_RE.finditer(body):
                    blob = re.sub(r"//[^\n]*", "", m.group("names"))  # trailing comments
                    names = [
                        n.strip().split(" as ")[0].strip()
                        for n in blob.split(",")
                        if n.strip() and not n.strip().startswith("type ")
                    ]
                    ts.append((rel, m.group("mod"), names))
            elif lang in ("python", "py"):
                for m in PY_FROM_RE.finditer(body):
                    raw = m.group("names").strip()
                    if raw.startswith("("):
                        raw = raw[1:].rstrip(")")
                    names = [
                        n.strip().split(" as ")[0].strip()
                        for n in raw.split(",")
                        if n.strip() and n.strip() != "*"
                    ]
                    py.append((rel, m.group("mod"), names))
    return ts, py


def check_ts(entries):
    if not DIST.exists():
        return [("(build)", "typescript/dist", ["run `npm run build` in typescript/ first"])]
    wanted = {}
    for rel, mod, names in entries:
        wanted.setdefault(mod, set()).update(names)
    script = [
        "const out = {};",
        f"const mods = {json.dumps({k: sorted(v) for k, v in wanted.items()})};",
        "for (const [mod, names] of Object.entries(mods)) {",
        "  let ns;",
        "  try {",
        f"    const spec = mod === 'webagents' ? {json.dumps(str(DIST))}",
        f"      : {json.dumps(str(DIST.parent))} + '/' + mod.replace(/^webagents\\//, '') + '/index.js';",
        "    ns = await import(spec);",
        "  } catch (e) { out[mod] = { error: String(e.message).split('\\n')[0] }; continue; }",
        "  out[mod] = { missing: names.filter((n) => ns[n] === undefined) };",
        "}",
        "console.log(JSON.stringify(out));",
    ]
    res = subprocess.run(
        ["node", "--input-type=module", "-e", "\n".join(script)],
        capture_output=True, text=True, cwd=REPO,
    )
    if res.returncode != 0:
        return [("(node)", "?", [res.stderr.strip()[:200]])]
    report = json.loads(res.stdout.strip().splitlines()[-1])
    failures = []
    for rel, mod, names in entries:
        r = report.get(mod, {})
        if "error" in r:
            failures.append((rel, mod, [f"module does not resolve: {r['error']}"]))
        else:
            bad = [n for n in names if n in r.get("missing", [])]
            bad = [n for n in bad if not declared_in_types(mod, n)]
            if bad:
                failures.append((rel, mod, [f"not exported: {', '.join(bad)}"]))
    return failures


_DTS_CACHE: dict[str, str] = {}


def declared_in_types(mod: str, name: str) -> bool:
    """True when the name is exported as a TYPE.

    Types vanish at runtime, so the runtime probe cannot see them, and
    `import { Context }` without the `type` keyword is legal TypeScript. Read
    the declaration file so a type import is not reported as a missing export.
    """
    if mod not in _DTS_CACHE:
        sub = mod[len("webagents"):].lstrip("/")
        dts = DIST.parent / (f"{sub}/index.d.ts" if sub else "index.d.ts")
        _DTS_CACHE[mod] = dts.read_text() if dts.exists() else ""
    text = _DTS_CACHE[mod]
    if not text:
        return False
    return bool(re.search(rf"\b{re.escape(name)}\b", text))


def check_py(entries):
    if not VENV.exists():
        return [("(venv)", "python/.venv", ["create the venv first"])]
    wanted = {}
    for rel, mod, names in entries:
        wanted.setdefault(mod, set()).update(names)
    prog = (
        "import json, importlib\n"
        f"mods = {json.dumps({k: sorted(v) for k, v in wanted.items()})}\n"
        "out = {}\n"
        "for mod, names in mods.items():\n"
        "    try:\n"
        "        m = importlib.import_module(mod)\n"
        "    except Exception as e:\n"
        "        out[mod] = {'error': f'{type(e).__name__}: {e}'}\n"
        "        continue\n"
        "    out[mod] = {'missing': [n for n in names if not hasattr(m, n)]}\n"
        "print(json.dumps(out))\n"
    )
    res = subprocess.run([str(VENV), "-c", prog], capture_output=True, text=True, cwd=REPO / "python")
    if res.returncode != 0:
        return [("(python)", "?", [res.stderr.strip()[:200]])]
    report = json.loads(res.stdout.strip().splitlines()[-1])
    failures = []
    for rel, mod, names in entries:
        r = report.get(mod, {})
        if "error" in r:
            failures.append((rel, mod, [f"module does not import: {r['error']}"]))
        else:
            bad = [n for n in names if n in r.get("missing", [])]
            if bad:
                failures.append((rel, mod, [f"not found in module: {', '.join(bad)}"]))
    return failures


def main() -> int:
    ts, py = collect()
    print(f"docs import checks: {len(ts)} TypeScript, {len(py)} Python")
    failures = check_ts(ts) + check_py(py)
    if not failures:
        print("OK: every webagents import in the docs resolves.")
        return 0
    seen = set()
    print(f"\n{len(failures)} unresolved import(s):\n")
    for rel, mod, msgs in failures:
        key = (rel, mod, tuple(msgs))
        if key in seen:
            continue
        seen.add(key)
        print(f"  {rel}\n    {mod}: {'; '.join(msgs)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
