#!/usr/bin/env python3
"""Generate the portal route manifest fixture from the portal's app router.

Walks `app/api/**` (plus `app/.well-known/**`) in the portal repo for
`route.ts` files and writes `python/tests/fixtures/portal_routes.json`:

    {
      "routes": [
        {"path": "/api/content/upload", "methods": ["POST"], "catchAll": false},
        {"path": "/api/agents/[id]/[...skillPath]", "methods": [...], "catchAll": true},
        ...
      ]
    }

Dynamic segments keep their bracket spelling so the contract test can flag a
catch-all match DISTINCTLY: `app/api/agents/[id]/[...skillPath]` must never
silently satisfy a typo like `/api/agents/{id}/deregister`.

The manifest source is the route tree itself — `openapi.json` is
hand-maintained (23 paths, stale) and cannot be trusted for this.

Usage: python3 scripts/generate_route_manifest.py [portal_root]
Default portal_root: ../../ relative to this script (the vendored layout).
"""

import json
import re
import subprocess
import sys
from pathlib import Path

METHOD_RE = re.compile(
    r"export\s+(?:async\s+)?(?:function\s+(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b"
    r"|const\s+(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s*=)"
)


def collect_routes(portal_root: Path):
    app_dir = portal_root / "app"
    routes = []
    for sub in ("api", ".well-known"):
        base = app_dir / sub
        if not base.is_dir():
            continue
        for route_file in sorted(base.rglob("route.ts")):
            rel = route_file.parent.relative_to(app_dir)
            # Next.js route groups `(group)` do not appear in the URL.
            segments = [s for s in rel.parts if not (s.startswith("(") and s.endswith(")"))]
            path = "/" + "/".join(segments)
            text = route_file.read_text(errors="replace")
            methods = sorted({m for pair in METHOD_RE.findall(text) for m in pair if m})
            routes.append({
                "path": path,
                "methods": methods,
                "catchAll": any(s.startswith("[...") or s.startswith("[[...") for s in segments),
            })
    return routes


def uncommitted_route_files(portal_root: Path):
    """Route files that exist only in the WORKING TREE (untracked or
    modified), as URL paths.

    PROVENANCE MATTERS HERE. The manifest is generated from whatever is on
    disk, so a route added by an unlanded workstream is indistinguishable
    from a shipped one: the SDK could dial it, the contract test would pass,
    and the released SDK would 404 in production. Recording the set makes
    that dependency reviewable — `test_manifest_declares_unlanded_routes`
    fails when the fixture's list and the tree disagree.
    """
    try:
        proc = subprocess.run(
            # --untracked-files=all: without it git collapses a brand-new
            # directory to `?? app/api/discovery/announce/`, and a NEW route
            # (exactly the case this exists to surface) would be missed.
            ["git", "status", "--porcelain", "--untracked-files=all",
             "--", "app/api", "app/.well-known"],
            cwd=portal_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None  # not a git checkout: provenance unknown, not empty

    app_dir = portal_root / "app"
    paths = set()
    for line in proc.stdout.splitlines():
        status, name = line[:2], line[3:].strip().strip('"')
        # Only NEW files matter: a modified existing route still exists in a
        # clean checkout, so the SDK dialling it is not a landing dependency.
        if status not in ("??", "A ", "AM"):
            continue
        if " -> " in name:  # rename
            name = name.split(" -> ", 1)[1]
        if not name.endswith("route.ts"):
            continue
        rel = Path(name).parent
        try:
            rel = rel.relative_to("app")
        except ValueError:
            continue
        segments = [seg for seg in rel.parts if not (seg.startswith("(") and seg.endswith(")"))]
        candidate = "/" + "/".join(segments)
        if (app_dir / rel / "route.ts").is_file():
            paths.add(candidate)
    return sorted(paths)


def git_commit(portal_root: Path):
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=portal_root, capture_output=True, text=True, check=True,
        )
        return proc.stdout.strip()
    except Exception:
        return None


def main() -> int:
    here = Path(__file__).resolve()
    portal_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else here.parents[3]
    if not (portal_root / "app" / "api").is_dir():
        print(f"portal app router not found under {portal_root}", file=sys.stderr)
        return 2
    routes = collect_routes(portal_root)
    unlanded = uncommitted_route_files(portal_root)
    manifest = {
        "routes": routes,
        "provenance": {
            "commit": git_commit(portal_root),
            # Routes present only in the working tree at generation time. An
            # SDK path that resolves ONLY through one of these is a path that
            # 404s until that workstream lands.
            "unlandedRoutes": unlanded,
        },
    }
    out = here.parents[1] / "tests" / "fixtures" / "portal_routes.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out} ({len(routes)} routes)")
    if unlanded:
        print(f"  NOTE: {len(unlanded)} route(s) exist only in the working tree: {', '.join(unlanded)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
