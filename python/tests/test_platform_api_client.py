"""
The platform API client is part of the SDK, and every platform skill asks one
lookup for the platform's URL (2026-09-25).

The client was `robutler.api` from the `robutler` package, a dependency that
installed a whole earlier copy of this framework, fought this package for the
`robutler` command, and on PyPI predated the payment calls the payment skill
makes, so a paid request on a pip install was refused. It now lives at
`webagents/agents/skills/robutler/api/` (its header says what differs from the
source). Pinned here:

  - nothing imports the `robutler` package and nothing declares it;
  - the payment calls reach the platform's routes with the field names the
    portal validates (`app/api/payments/{verify,lock,settle}/route.ts` and
    `lock/[id]/route.ts`, which reads `additionalAmount`), and `extend_lock`
    exists once;
  - `agent_access()` prints nothing (S-253: it printed part of the API key);
  - with no URL named, the client and the namespace, publish and message
    history skills call https://robutler.ai, and no code in the SDK names
    webagents.ai, the project's old site (S-252).
"""

import ast
import asyncio
import base64
import importlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest

from webagents.agents.skills.robutler.api import RobutlerClient
from webagents.agents.skills.robutler.api.client import TokensResource
from webagents.agents.skills.robutler.api.types import ApiResponse
from webagents.agents.skills.robutler.platform_url import DEFAULT_PLATFORM_URL

PYTHON_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = PYTHON_ROOT / "webagents"


@pytest.fixture
def no_platform_named(monkeypatch, tmp_path):
    """No variable and no CLI configuration names a platform."""
    for name in ("ROBUTLER_API_URL", "ROBUTLER_INTERNAL_API_URL", "WEBAGENTS_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def _client(**kwargs) -> RobutlerClient:
    return RobutlerClient(api_key="rok_test", **kwargs)


def _answering(client: RobutlerClient, data: dict) -> AsyncMock:
    client._make_request = AsyncMock(return_value=ApiResponse(success=True, data=data))
    return client._make_request


# ---------------------------------------------------------------------------
# The package is gone
# ---------------------------------------------------------------------------


def _imported_modules(path: Path):
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


def test_nothing_imports_the_robutler_package():
    offenders = [
        f"{path.relative_to(PYTHON_ROOT)}: {name}"
        for root in (PACKAGE, PYTHON_ROOT / "tests")
        for path in root.rglob("*.py")
        for name in _imported_modules(path)
        if name == "robutler" or name.startswith("robutler.")
    ]
    assert offenders == []


def test_the_robutler_package_is_not_a_dependency():
    tomllib = pytest.importorskip("tomllib")  # 3.11+; the other runs read the same file
    from packaging.requirements import Requirement

    project = tomllib.loads((PYTHON_ROOT / "pyproject.toml").read_text())["project"]
    extras = project["optional-dependencies"]
    declared = list(project["dependencies"]) + [r for group in extras.values() for r in group]
    assert "robutler" not in {Requirement(r).name.lower() for r in declared}
    # The documented `webagents[robutler]` still resolves: the extra stays, empty.
    assert extras["robutler"] == []

    lines = [line.split("#")[0].strip() for line in (PYTHON_ROOT / "requirements.txt").read_text().splitlines()]
    assert "robutler" not in {Requirement(line).name.lower() for line in lines if line and not line.startswith("-")}


# ---------------------------------------------------------------------------
# The payment calls, as the platform reads them
# ---------------------------------------------------------------------------


async def test_verify_posts_the_token():
    client = _client(base_url="https://platform.example")
    request = _answering(client, {"valid": True, "balanceDollars": 1.5})
    assert await client.tokens.validate_with_balance("tok") == {"valid": True, "balance": 1.5}
    assert request.await_args.args == ("POST", "/payments/verify")
    assert request.await_args.kwargs["data"] == {"token": "tok"}


async def test_lock_posts_the_token_and_the_amount():
    client = _client(base_url="https://platform.example")
    request = _answering(client, {"lockId": "L1", "lockedAmountDollars": 0.005})
    assert await client.tokens.lock("tok", 0.005) == {"lockId": "L1", "lockedAmountDollars": 0.005}
    assert request.await_args.args == ("POST", "/payments/lock")
    assert request.await_args.kwargs["data"] == {"token": "tok", "amount": 0.005}


async def test_settle_posts_the_lock_with_the_platforms_field_names():
    client = _client(base_url="https://platform.example")
    request = _answering(client, {"success": True, "chargedDollars": 0.003, "remainingDollars": 0.002})
    result = await client.tokens.settle(
        lock_id="L1", amount=0.003, description="one reply", charge_type="platform_fee", release=True
    )
    assert result == {"success": True, "chargedDollars": 0.003, "remainingDollars": 0.002}
    assert request.await_args.args == ("POST", "/payments/settle")
    assert request.await_args.kwargs["data"] == {
        "lockId": "L1",
        "amount": 0.003,
        "description": "one reply",
        "chargeType": "platform_fee",
        "release": True,
    }


async def test_extend_lock_patches_the_lock_with_additional_amount():
    client = _client(base_url="https://platform.example")
    request = _answering(client, {"success": True, "newAmountDollars": 0.01})
    assert await client.tokens.extend_lock("L1", 0.005) == {"success": True, "newAmountDollars": 0.01}
    assert request.await_args.args == ("PATCH", "/payments/lock/L1")
    assert request.await_args.kwargs["data"] == {"additionalAmount": 0.005}


def test_extend_lock_is_defined_once():
    """The source defined it twice and the second won silently; the first sent
    `amount`, which the platform's route does not read."""
    assert inspect.getsource(TokensResource).count("def extend_lock") == 1


# ---------------------------------------------------------------------------
# S-254: a settle is never sent twice
# ---------------------------------------------------------------------------


class _Session:
    """Answers each request with the next outcome: a status, or an exception."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def request(self, method, url, **_kwargs):
        self.calls.append(method)
        outcome = self.outcomes.pop(0)

        class _Exchange:
            async def __aenter__(self):
                if isinstance(outcome, BaseException):
                    raise outcome
                return SimpleNamespace(status=outcome, text=AsyncMock(return_value="{}"))

            async def __aexit__(self, *exc):
                return False

        return _Exchange()


class _Refused(aiohttp.ClientConnectorError):
    """A connection that was never made (the constructor needs a live key)."""

    def __init__(self):
        pass

    def __str__(self):
        return "connection refused"


@pytest.fixture
def exchanges(monkeypatch):
    async def no_wait(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_wait)

    def attach(*outcomes):
        session = _Session(*outcomes)
        client = _client(base_url="https://platform.example")
        client._get_session = AsyncMock(return_value=session)
        return client, session

    return attach


@pytest.mark.parametrize("method, endpoint", [("POST", "/payments/settle"), ("POST", "/payments/lock"), ("PATCH", "/payments/lock/L1")])
async def test_a_payment_write_is_sent_once_after_a_5xx(exchanges, method, endpoint):
    client, session = exchanges(502, 200)
    response = await client._make_request(method, endpoint, data={"amount": 0.003})
    assert (response.success, response.status_code) == (False, 502)
    assert session.calls == [method]


async def test_a_payment_write_is_sent_once_after_a_dropped_connection(exchanges):
    client, session = exchanges(aiohttp.ServerDisconnectedError(), 200)
    response = await client._make_request("POST", "/payments/settle", data={"lockId": "L1", "amount": 0.003})
    assert (response.success, response.error) == (False, "Network error")
    assert session.calls == ["POST"]


async def test_a_payment_write_whose_connection_was_never_made_is_sent_again(exchanges):
    client, session = exchanges(_Refused(), 200)
    response = await client._make_request("POST", "/payments/settle", data={"lockId": "L1", "amount": 0.003})
    assert response.success is True
    assert session.calls == ["POST", "POST"]


async def test_a_read_is_still_retried_after_a_5xx(exchanges):
    client, session = exchanges(503, 200)
    response = await client._make_request("GET", "/balance")
    assert response.success is True
    assert session.calls == ["GET", "GET"]


# ---------------------------------------------------------------------------
# S-253: no key on stdout
# ---------------------------------------------------------------------------


def test_the_client_prints_nothing():
    """`agent_access()` printed part of the key; it went with the other dead
    content methods, and nothing in the module prints (docstring examples are
    strings, not calls)."""
    from webagents.agents.skills.robutler.api import client as module

    calls = [
        node.lineno
        for node in ast.walk(ast.parse(inspect.getsource(module)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"
    ]
    assert calls == []


# ---------------------------------------------------------------------------
# Content through the routes an API key can use (S-253, S-255)
# ---------------------------------------------------------------------------

AGENT = "11111111-2222-4333-8444-555555555555"
FILE = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"


async def test_content_calls_are_the_base_url_and_a_fixed_path():
    client = _client(base_url="https://platform.example")
    request = _answering(client, {"items": []})
    await client.list_agent_content(AGENT, query="notes.json")
    assert request.await_args.args == ("GET", f"/agents/{AGENT}/content")
    assert request.await_args.kwargs["params"] == {"limit": 100, "page": 1, "q": "notes.json"}
    await client.read_agent_content(AGENT, FILE)
    assert request.await_args.args == ("GET", f"/agents/{AGENT}/content/{FILE}")
    assert request.await_args.kwargs["params"] == {"format": "text"}
    await client.delete_agent_content(AGENT, FILE)
    assert request.await_args.args == ("DELETE", f"/agents/{AGENT}/content/{FILE}")


@pytest.mark.parametrize("bad", ["../../api/admin", "https://elsewhere.example/x", "", None, "not-a-uuid"])
async def test_an_id_that_is_not_one_never_becomes_a_path(bad):
    client = _client(base_url="https://platform.example")
    request = _answering(client, {})
    with pytest.raises(ValueError):
        await client.read_agent_content(AGENT, bad)
    with pytest.raises(ValueError):
        await client.list_agent_content(bad)
    request.assert_not_awaited()


async def test_an_upload_is_private_under_its_full_name():
    posted = {}

    class _Session:
        def post(self, url, data=None, headers=None):
            posted.update(url=url, fields={f[0]["name"]: f[2] for f in data._fields}, headers=headers)

            class _Exchange:
                async def __aenter__(self):
                    return SimpleNamespace(status=200, text=AsyncMock(return_value='{"id": "%s", "displayName": "notes.json", "size": 2}' % FILE))

                async def __aexit__(self, *exc):
                    return False

            return _Exchange()

    client = _client(base_url="https://platform.example")
    client._get_session = AsyncMock(return_value=_Session())
    response = await client.upload_file("notes.json", b"{}", content_type="application/json")
    assert response.success and response.data["id"] == FILE
    assert posted["url"] == "https://platform.example/api/content/upload"
    assert posted["fields"]["visibility"] == "private"
    assert posted["fields"]["displayName"] == "notes.json"
    assert posted["headers"]["Authorization"] == "Bearer rok_test"


class _Files:
    """A stand-in for the client's content calls, holding one agent's files."""

    def __init__(self, *names):
        self.files = [
            {"id": f"{i:08d}-0000-4000-8000-000000000000", "displayName": n, "createdAt": f"2026-09-25T00:00:0{i}Z", "text": '{"v": %d}' % i}
            for i, n in enumerate(names, 1)
        ]
        self.uploads = []
        self.deleted = []

    async def list_agent_content(self, agent_id, query=None, limit=100, page=1):
        items = [f for f in self.files if not query or query in f["displayName"]]
        return ApiResponse(success=True, data={"items": items})

    async def read_agent_content(self, agent_id, content_id, format="text"):
        found = next(f for f in self.files if f["id"] == content_id)
        return ApiResponse(success=True, data={"id": content_id, "text": found["text"]})

    async def upload_file(self, filename, data, content_type="application/octet-stream", visibility="private"):
        self.uploads.append((filename, visibility))
        new = {"id": "99999999-0000-4000-8000-000000000000", "displayName": filename, "createdAt": "2026-09-26T00:00:00Z", "text": data.decode()}
        self.files.append(new)
        return ApiResponse(success=True, data={"id": new["id"], "displayName": filename, "size": len(data)})

    async def delete_agent_content(self, agent_id, content_id):
        self.deleted.append(content_id)
        self.files = [f for f in self.files if f["id"] != content_id]
        return ApiResponse(success=True, data={})


def _json_skill(files):
    from webagents.agents.skills.robutler.storage.json.skill import RobutlerJSONSkill

    skill = RobutlerJSONSkill({"api_key": "rok_test", "agent_id": AGENT, "portal_url": "https://platform.example"})
    skill.client = files
    return skill


async def test_json_documents_are_stored_private():
    files = _Files()
    result = json.loads(await _json_skill(files).store_json_data("notes", {"a": 1}))
    assert result["success"] and result["filename"] == "notes.json"
    assert files.uploads == [("notes.json", "private")]


async def test_json_documents_read_back_newest_first():
    files = _Files("notes.json", "notes.json.bak", "notes.json")
    result = json.loads(await _json_skill(files).retrieve_json_data("notes"))
    assert result["success"] and result["data"] == {"v": 3}


async def test_a_missing_document_names_the_ones_that_exist():
    result = json.loads(await _json_skill(_Files("todo.json", "photo.png")).retrieve_json_data("notes"))
    assert result["success"] is False and result["available_json_files"] == ["todo.json"]


async def test_an_update_stores_first_then_removes_the_older_versions():
    files = _Files("notes.json", "other.json", "notes.json")
    result = json.loads(await _json_skill(files).update_json_data("notes", {"v": 9}))
    assert result["success"] and result["replaced"] == 2
    assert files.uploads == [("notes.json", "private")]
    assert [f["displayName"] for f in files.files] == ["other.json", "notes.json"]
    assert json.loads(await _json_skill(files).retrieve_json_data("notes"))["data"] == {"v": 9}


async def test_a_delete_removes_every_document_with_the_name():
    files = _Files("notes.json", "notes.json", "other.json")
    result = json.loads(await _json_skill(files).delete_json_file("notes"))
    assert result["success"] and result["deleted"] == 2
    assert [f["displayName"] for f in files.files] == ["other.json"]


async def test_without_a_key_the_tools_say_what_is_missing(monkeypatch):
    from webagents.agents.skills.robutler.storage.json.skill import RobutlerJSONSkill

    monkeypatch.setattr("webagents.server.core.registration.resolve_agent_token", lambda agent=None: None)
    monkeypatch.delenv("WEBAGENTS_API_KEY", raising=False)
    skill = RobutlerJSONSkill({"portal_url": "https://platform.example"})
    await skill.initialize(SimpleNamespace(name="keeper"))
    result = json.loads(await skill.store_json_data("notes", {}))
    assert result["success"] is False and "webagents publish" in result["error"]


def test_the_agent_id_is_the_claim_in_its_own_key():
    from webagents.agents.skills.robutler.storage.json.skill import _agent_id_in

    def token(claims):
        body = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
        return f"e30.{body}.sig"

    assert _agent_id_in(token({"agent_id": AGENT, "typ": "api_key"})) == AGENT
    assert _agent_id_in(token({"typ": "api_key"})) is None
    assert _agent_id_in("rok_not_a_jwt") is None
    assert _agent_id_in(None) is None


# ---------------------------------------------------------------------------
# One platform URL (S-252)
# ---------------------------------------------------------------------------


def test_with_no_url_the_client_asks_the_sdks_lookup(no_platform_named, monkeypatch):
    assert _client().base_url == DEFAULT_PLATFORM_URL == "https://robutler.ai"
    monkeypatch.setenv("ROBUTLER_INTERNAL_API_URL", "http://internal:3000")
    assert _client().base_url == "http://internal:3000"
    # The public variable first, as in TypeScript; the source client tried the
    # in-cluster one first.
    monkeypatch.setenv("ROBUTLER_API_URL", "https://public.example/")
    assert _client().base_url == "https://public.example"
    assert _client(base_url="https://named.example/").base_url == "https://named.example"


@pytest.mark.parametrize(
    "module, cls, attr, suffix",
    [
        ("namespace", "NamespaceSkill", "webagents_api_url", ""),
        ("publish", "PublishSkill", "webagents_api_url", ""),
    ],
)
def test_a_platform_skill_with_no_url_calls_the_platform(no_platform_named, monkeypatch, module, cls, attr, suffix):
    skill_class = getattr(importlib.import_module(f"webagents.agents.skills.robutler.{module}.skill"), cls)
    assert getattr(skill_class({}), attr) == DEFAULT_PLATFORM_URL + suffix
    monkeypatch.setenv("ROBUTLER_API_URL", "https://env.example")
    assert getattr(skill_class({}), attr) == "https://env.example" + suffix
    # The skill's own config outranks the variable, as in every platform skill.
    assert getattr(skill_class({"webagents_api_url": "https://mine.example"}), attr) == "https://mine.example" + suffix


async def test_message_history_with_no_url_calls_the_platform(no_platform_named):
    from webagents.agents.skills.robutler.message_history.skill import MessageHistorySkill

    skill = MessageHistorySkill({"api_key": "rok_test"})
    await skill.initialize(SimpleNamespace(name="history", id="history", api_key="rok_test"))
    assert skill.api_client.base_url == DEFAULT_PLATFORM_URL


def _docstrings(tree: ast.AST) -> set:
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                found.add(id(first.value))
    return found


def test_no_code_names_the_old_site_as_an_address():
    """S-252's shape, caught at the source: a string in code (comments and
    docstrings aside) anywhere in the SDK that names webagents.ai. The platform
    skills defaulted to it, and so did the UCP skill's merchant endpoint, buyer
    profile, token-check URL and handler spec."""
    offenders = []
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        docstrings = _docstrings(tree)
        offenders += [
            f"{path.relative_to(PYTHON_ROOT)}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and "webagents.ai" in node.value
        ]
    assert offenders == []
