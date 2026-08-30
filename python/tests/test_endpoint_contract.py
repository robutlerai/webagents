"""T2 — endpoint contract: every literal API path the SDK dials must exist
in the portal route manifest, and a catch-all route must never silently
satisfy a typo.

The manifest (`tests/fixtures/portal_routes.json`) is generated from the
portal's app router by `scripts/generate_route_manifest.py` — NOT from
openapi.json, which is hand-maintained and stale. When the portal source
tree is present (the vendored layout), a freshness test regenerates and
compares, so the fixture cannot rot either.

SCOPE — read this before trusting it. The general check
(`test_every_sdk_literal_resolves_to_a_real_route`) is PATH-ONLY: it catches
a path that exists nowhere, and a path that "matches" only via the portal's
`/api/[...unmatched]` catch-all — which is how the SDK's dead
`/api/agents/publish`, `/api/content/agent` and `/api/auth/jwks` shipped
looking alive. It does NOT check the HTTP METHOD, because the verb lives at
the call site and the URL is usually assembled from a base plus a constant,
so the two cannot be paired statically with any reliability.

F-041 (`POST /api/content` — the route exists for GET only) is therefore a
METHOD bug that the general check cannot see; it is pinned individually by
`test_upload_path_is_the_working_one` below. Any new method-sensitive
constant needs its own pin like that one.
"""

import json
import re
from pathlib import Path

import pytest

PY_ROOT = Path(__file__).resolve().parents[1]          # webagents/python
SDK_ROOT = PY_ROOT / "webagents"
TS_SRC = PY_ROOT.parent / "typescript" / "src"
PORTAL_ROOT = PY_ROOT.parents[1]                        # portal repo (vendored layout)
FIXTURE = PY_ROOT / "tests" / "fixtures" / "portal_routes.json"

# A literal API path inside a Python or TS string: capture from /api/ (or
# /.well-known/) to the closing quote/backtick/whitespace.
LITERAL_RE = re.compile(
    r"""["'`f]?(/(?:api|\.well-known)/[A-Za-z0-9_\-./{}$\[\]]*[A-Za-z0-9_\-}\]])"""
)

# Paths that are NOT requests this SDK makes against the portal API:
# URL prefixes used for string building/rewriting, and platform-agnostic
# well-known paths served BY the agent itself.
NOT_A_REQUEST_PREFIXES = (
    "/api/content/public",          # public-content URL prefix (and example URLs in tool descriptions)
    "/api/llm/mock/v1",             # mock-provider base URL used by adapter tests
    "/.well-known/agent.json",      # served BY the agent (and fetched from AGENTS, not the portal)
    "/.well-known/openid-configuration",  # served BY the agent
)
NOT_A_REQUEST = {
    "/api/payments",                # x402 base PREFIX; requests are /lock, /verify, /settle
}

# Single dynamic-subpath dispatchers the SDK drives with a variable segment;
# unverifiable statically.
VARIABLE_SUBPATH = {
    "/api/discovery/:param",
}

# Documented-but-external endpoints: dialled against OTHER hosts (never the
# portal), so the portal manifest is not their contract.
EXTERNAL_PREFIXES = (
    "/api/v1",        # third-party providers (fal, n8n, ...)
    "/api/v10",       # discord
)

# Source subtrees that integrate EXTERNAL services (their /api/... literals
# go to those services' own hosts, not the portal).
EXTERNAL_SERVICE_DIRS = (
    "typescript/src/skills/messaging/",   # slack, discord, reddit, x
    "typescript/src/cli/",                # talks to the LOCAL daemon's API
    "typescript/src/transport/terminal/", # talks to the local Tauri host
    "python/webagents/agents/skills/ecosystem/",  # n8n, zapier, google, fal...
    "python/webagents/agents/skills/examples/",
)

# Catch-all routes that ARE the intended public contract (a generic path
# space, not a 404 sink). A match through one of these counts as real;
# a match through /api/[...unmatched] or the agent skill-path catch-all
# stays a failure.
ALLOWED_CATCHALL_ROUTES = {
    "/api/storage/memory/[[...path]]",   # scoped KV (ADR-0023)
    "/api/proxy/[...path]",
    "/api/channels/[...slug]",           # the channels API keys on slug paths
}

# QUARANTINE — known-dead portal paths still present in SDK source, each the
# same class as F-041. Every entry must eventually be repointed or deleted;
# NEVER add to this list to make the test pass for new code.
KNOWN_DEAD = {
    "/api/rag/ingest",            # TS RAGSkill portal backend: no /api/rag/* routes exist
    "/api/rag/search",
    "/api/rag/delete",
    "/api/rag/stats",
    "/api/storage/files",         # TS storage skill: no /api/storage/files|json routes exist
    "/api/storage/files/:param",
    "/api/storage/json/:param",
    "/api/chats",                 # TS social chats tools: the portal chat surface is /api/messages/[chatId]
    "/api/chats/:param/messages",
    "/api/chats/:param/completions",  # py chats skill advertises this dead URL in its listing payload
    "/api/chat",                  # py message_history: no such route
    "/api/namespaces",            # py namespace skill: no /api/namespaces* routes exist
    "/api/namespaces/:param",
    "/api/namespaces/:param/join",
    "/api/namespaces/:param/leave",
    "/api/auth/owner-assertion",  # py nli: mint endpoint does not exist
    "/api/auth/jwks",             # local/auth robutler provider: portal serves /.well-known/jwks.json
    "/api/auth/token",            # local/auth robutler provider: no such route
    "/api/auth/userinfo",         # local/auth robutler provider: no such route
    "/api/marketplaces",          # local/plugin marketplace: the real route is /api/marketplace (singular)
}


def _placeholder_segments(path: str) -> str:
    """Normalize f-string / template placeholders to ':param' segments."""
    path = re.sub(r"\$\{[^}]*\}", ":param", path)     # ${expr} first
    path = re.sub(r"\$\{.*$", ":param", path)          # ${expr with (parens) — truncated by the capture
    path = re.sub(r"\{[^/}]*\}", ":param", path)       # {expr}
    path = re.sub(r"\$[A-Za-z0-9_.]+", ":param", path)  # $var
    path = path.replace("::param", ":param")
    return path


def _is_comment_line(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith(("#", "//", "*", "/*"))


def _py_code_strings(text: str):
    """String constants in a .py file, EXCLUDING docstrings (module/class/
    function first-statement strings), which are prose, not requests."""
    import ast as _ast

    try:
        tree = _ast.parse(text)
    except SyntaxError:
        yield text
        return
    doc_ids = set()
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.Module, _ast.ClassDef, _ast.FunctionDef, _ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], _ast.Expr) and isinstance(body[0].value, _ast.Constant)                     and isinstance(body[0].value.value, str):
                doc_ids.add(id(body[0].value))
    joined_children = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.JoinedStr):
            parts = []
            for v in node.values:
                if isinstance(v, _ast.Constant) and isinstance(v.value, str):
                    parts.append(v.value)
                    joined_children.add(id(v))
                else:
                    parts.append("{p}")
            yield "".join(parts)
    for node in _ast.walk(tree):
        if (
            isinstance(node, _ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in doc_ids
            and id(node) not in joined_children
        ):
            yield node.value


def _candidate_strings(f: Path, text: str):
    if f.suffix == ".py":
        yield from _py_code_strings(text)
    else:
        for line in text.splitlines():
            if _is_comment_line(line):
                continue
            yield line


def collect_sdk_literals():
    """(path, source_file) for every literal API path in the two SDKs."""
    found = []
    files = list(SDK_ROOT.rglob("*.py"))
    if TS_SRC.is_dir():
        files += list(TS_SRC.rglob("*.ts"))
    for f in sorted(files):
        rel = str(f.relative_to(PY_ROOT.parent))
        if any(rel.startswith(d) for d in EXTERNAL_SERVICE_DIRS):
            continue
        text = f.read_text(errors="replace")
        for chunk in _candidate_strings(f, text):
            for m in LITERAL_RE.finditer(chunk):
                raw = m.group(1)
                path = _placeholder_segments(raw.split("?")[0].rstrip("/"))
                if not path or path in NOT_A_REQUEST or path in KNOWN_DEAD or path in VARIABLE_SUBPATH:
                    continue
                if any(path == p or path.startswith(p + "/") for p in NOT_A_REQUEST_PREFIXES):
                    continue
                if any(path == p or path.startswith(p + "/") for p in EXTERNAL_PREFIXES):
                    continue
                found.append((path, rel))
    return found


def load_manifest():
    return json.loads(FIXTURE.read_text())["routes"]


def match_route(route_path: str, sdk_path: str) -> str:
    """'exact' | 'catchall' | '' — how this route satisfies the SDK path."""
    r_segs = [s for s in route_path.split("/") if s]
    s_segs = [s for s in sdk_path.split("/") if s]
    used_catchall = False
    i = 0
    for j, r in enumerate(r_segs):
        if r.startswith("[[..."):
            used_catchall = True
            i = len(s_segs)
            break
        if r.startswith("[..."):
            if i >= len(s_segs):
                return ""
            used_catchall = True
            i = len(s_segs)
            break
        if i >= len(s_segs):
            return ""
        s = s_segs[i]
        if r.startswith("["):
            i += 1
            continue
        if r != s:
            return ""
        i += 1
    else:
        if i != len(s_segs):
            return ""
    return "catchall" if used_catchall else "exact"


def classify(sdk_path: str, routes) -> str:
    """'exact' when a non-catch-all route (or an intended catch-all API from
    ALLOWED_CATCHALL_ROUTES) matches; 'catchall' when ONLY a 404-sink
    catch-all does; 'none' otherwise."""
    best = "none"
    for r in routes:
        kind = match_route(r["path"], sdk_path)
        if kind == "exact":
            return "exact"
        if kind == "catchall":
            if r["path"] in ALLOWED_CATCHALL_ROUTES:
                return "exact"
            best = "catchall"
    return best


class TestEndpointContract:
    def test_every_sdk_literal_resolves_to_a_real_route(self):
        routes = load_manifest()
        failures = []
        for path, src in sorted(set(collect_sdk_literals())):
            kind = classify(path, routes)
            if kind == "exact":
                continue
            if kind == "catchall":
                failures.append(
                    f"{src}: {path} is satisfied ONLY by a catch-all route — "
                    "that is how /api/content (405) and /api/agents/publish "
                    "(nonexistent) shipped looking alive"
                )
            else:
                failures.append(f"{src}: {path} matches no portal route at all")
        assert not failures, "\n".join(failures)

    def test_upload_path_is_the_working_one(self):
        """F-041 pin: the files skill posts /api/content/upload (POST exists),
        never bare /api/content (whose route has no POST)."""
        from webagents.agents.skills.robutler.storage.files import skill as files_skill

        assert files_skill.UPLOAD_PATH == "/api/content/upload"
        routes = {r["path"]: r for r in load_manifest()}
        assert "POST" in routes["/api/content/upload"]["methods"]
        assert "POST" not in routes.get("/api/content", {}).get("methods", [])

    def test_catch_all_is_flagged_distinctly(self):
        """The portal has a top-level /api/[...unmatched] catch-all, so an
        undistinguished existence check can never fail. Pin the classifier."""
        routes = load_manifest()
        assert classify("/api/agents/:param/definitely-not-a-route", routes) == "catchall"
        assert classify("/api/content/upload", routes) == "exact"

    def test_sdk_paths_that_depend_on_unlanded_routes_are_declared(self):
        """The manifest is generated from the WORKING TREE, so a route added
        by an unlanded workstream is indistinguishable from a shipped one:
        the SDK dials it, this suite passes, and the released SDK 404s. The
        generator therefore records `provenance.unlandedRoutes`, and any SDK
        literal that resolves ONLY through one of those must appear here —
        a deliberate, reviewable list, not a silent dependency."""
        manifest = json.loads(FIXTURE.read_text())
        unlanded = set((manifest.get("provenance") or {}).get("unlandedRoutes") or [])
        if not unlanded:
            pytest.skip("manifest records no unlanded routes")

        landed = [r for r in manifest["routes"] if r["path"] not in unlanded]
        depends = sorted({
            f"{path} ({src})"
            for path, src in set(collect_sdk_literals())
            if classify(path, manifest["routes"]) == "exact"
            and classify(path, landed) != "exact"
        })
        # Known and accepted: the discovery/announce surface ships with M3.
        expected_prefixes = ("/api/discovery/announce",)
        unexpected = [
            d for d in depends if not d.startswith(expected_prefixes)
        ]
        assert not unexpected, (
            "SDK paths resolve only through routes that exist nowhere but this "
            "working tree:\n" + "\n".join(unexpected)
        )

    @pytest.mark.skipif(
        not (PORTAL_ROOT / "app" / "api").is_dir(),
        reason="portal source tree not present (standalone SDK checkout)",
    )
    def test_manifest_is_fresh(self):
        """Regenerate from the live route tree and compare — the committed
        fixture must never drift from the portal it describes."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "generate_route_manifest", PY_ROOT / "scripts" / "generate_route_manifest.py"
        )
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
        fresh = gen.collect_routes(PORTAL_ROOT)
        committed = load_manifest()
        assert fresh == committed, (
            "portal_routes.json is stale; run python3 scripts/generate_route_manifest.py"
        )
