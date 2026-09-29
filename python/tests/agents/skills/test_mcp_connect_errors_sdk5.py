"""
What the MCP client says when a remote server answers 401 or 403 (2026-09-29,
the skills and MCP e2e), against the shared fixture
`tests/fixtures/mcp_tool/connect_errors.json`, which the TypeScript suite
reads too (`tests/unit/skills/mcp-connect-errors-sdk5.test.ts`).

`webagents doctor` pointed at the portal's `/mcp` with no credential printed
`✗ mcp robutler: unhandled errors in a TaskGroup (1 sub-exception)`, the
`str()` of the anyio exception group the SDK's transport raised through, and
its fix line said to fix the server's entry. Pinned here:

  * the words are the fixture's, in both SDKs;
  * an exception group is unwrapped to its HTTP error, however deep;
  * a 401 and a 403 are said as the credential the server wants, with the
    `${secret:<SERVER>_TOKEN}` reference and the OAuth caveat; other errors
    keep their own first line;
  * a real server answering 401 gives the report row that sentence with
    `needs_credential`, over `http` and over `auto` (the SSE fallback's own
    failure is not the one said), and `doctor`'s fix line is the bearer
    recipe, never "Fix the server's entry".
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import httpx
import pytest

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.local.mcp.connect_errors import (
    CREDENTIAL_STATUSES,
    NEEDS_CREDENTIAL,
    credential_secret_name,
    describe_connect_error,
    http_status_of,
    needs_credential_sentence,
    root_cause,
)
from webagents.agents.skills.local.mcp.skill import LocalMcpSkill, owner_reference_sources
from webagents.cli.config_store import cli_command
from webagents.cli.doctor import MCP_CHECK_WORDS, mcp_check

try:
    ExceptionGroup  # noqa: B018 - built in from 3.11
except NameError:  # pragma: no cover - 3.10 runs anyio's backport
    from exceptiongroup import ExceptionGroup  # type: ignore[no-redef]

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
WORDS = json.loads((FIXTURES / "mcp_tool" / "connect_errors.json").read_text())
SECRETS = json.loads((FIXTURES / "cli" / "secrets.json").read_text())


def _status_error(status: int, reason: str = "Unauthorized") -> httpx.HTTPStatusError:
    """The error httpx raises from `raise_for_status()`: its message names the url and adds an MDN line."""
    request = httpx.Request("POST", "http://127.0.0.1:9/mcp")
    response = httpx.Response(status, request=request, text=reason)
    return httpx.HTTPStatusError(
        f"Client error '{status} {reason}' for url 'http://127.0.0.1:9/mcp'\nFor more information check: https://developer.mozilla.org/",
        request=request,
        response=response,
    )


def _task_group(error: BaseException) -> ExceptionGroup:
    """The shape anyio raises: a group inside a group, both saying "unhandled errors in a TaskGroup"."""
    return ExceptionGroup("unhandled errors in a TaskGroup", [ExceptionGroup("unhandled errors in a TaskGroup", [error])])


# -- the words -------------------------------------------------------------------------------------


def test_the_words_are_the_fixtures():
    assert NEEDS_CREDENTIAL == WORDS["needs_credential"]
    assert list(CREDENTIAL_STATUSES) == WORDS["credential_statuses"]
    assert MCP_CHECK_WORDS["fixCredential"] == SECRETS["doctor"]["words"]["fixCredential"]
    assert "—" not in NEEDS_CREDENTIAL and "—" not in MCP_CHECK_WORDS["fixCredential"]


@pytest.mark.parametrize("case", WORDS["cases"], ids=[f"{c['server']} {c['status']}" for c in WORDS["cases"]])
def test_a_401_and_a_403_are_said_as_the_credential_the_server_wants(case):
    assert credential_secret_name(case["server"]) == case["name"]
    assert needs_credential_sentence(case["server"], case["status"]) == case["says"]
    assert describe_connect_error(case["server"], _status_error(case["status"])) == (case["says"], True)
    assert describe_connect_error(case["server"], _task_group(_status_error(case["status"]))) == (case["says"], True)


def test_an_exception_group_is_unwrapped_to_its_http_error():
    error = _status_error(401)
    group = _task_group(error)
    assert "sub-exception" in str(group)  # what doctor printed
    assert root_cause(group) is error
    assert http_status_of(group) is None and http_status_of(error) == 401
    # A group that holds an HTTP error among others answers the HTTP error.
    mixed = ExceptionGroup("unhandled errors in a TaskGroup", [RuntimeError("closed"), error])
    assert root_cause(mixed) is error
    # A group with no HTTP error answers its first leaf, and a plain error itself.
    plain = RuntimeError("boom")
    assert root_cause(_task_group(plain)) is plain
    assert root_cause(plain) is plain
    assert describe_connect_error("x", _task_group(plain)) == ("boom", False)


@pytest.mark.parametrize("status", WORDS["not_a_credential"])
def test_other_http_errors_keep_their_own_first_line(status):
    said, needs = describe_connect_error("robutler", _status_error(status, "Nope"))
    assert needs is False
    assert said == f"Client error '{status} Nope' for url 'http://127.0.0.1:9/mcp'" or said.startswith(f"Client error '{status}") or said.startswith(f"Server error '{status}")
    assert "\n" not in said and "developer.mozilla.org" not in said


# -- a real server that answers 401 ----------------------------------------------------------------


class _Refuses(BaseHTTPRequestHandler):
    """Every request is refused with 401, as the portal's `/mcp` refuses a request with no credential."""

    def _refuse(self) -> None:
        body = json.dumps({"jsonrpc": "2.0", "error": {"code": -32000, "message": "Unauthorized"}, "id": None}).encode()
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("WWW-Authenticate", 'Bearer realm="portal"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = _refuse
    do_POST = _refuse
    do_DELETE = _refuse

    def log_message(self, *_args) -> None:  # quiet
        return


@pytest.fixture
def refusing_url(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    for var in ("WEBAGENTS_PROFILE", "WEBAGENTS_SECRETS_DIR"):
        monkeypatch.delenv(var, raising=False)
    server = HTTPServer(("127.0.0.1", 0), _Refuses)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/mcp"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("transport", ["http", "auto"])
def test_a_server_that_answers_401_is_reported_as_wanting_a_credential(refusing_url, transport):
    skill = LocalMcpSkill({"mcp": {"robutler": {"url": refusing_url, "transport": transport}}, "references": owner_reference_sources()})
    agent = BaseAgent(name="client", instructions="x", skills={"mcp": skill})

    async def report_after_initialize():
        await agent._ensure_skills_initialized()
        try:
            return skill.server_report()
        finally:
            await skill.cleanup()

    report = asyncio.run(report_after_initialize())
    expected = needs_credential_sentence("robutler", 401)
    assert expected == next(c["says"] for c in WORDS["cases"] if c["server"] == "robutler")
    row = report[0]
    assert row["connected"] is False
    assert row["error"] == expected
    assert row["needs_credential"] is True
    assert "TaskGroup" not in json.dumps(report)

    # `doctor`: the row's sentence in the detail, the bearer recipe as the fix.
    check = mcp_check(report)
    assert check.status == "fail"
    assert check.detail == f"robutler: {expected}"
    assert check.fix == MCP_CHECK_WORDS["fixCredential"].replace("{name}", "ROBUTLER_TOKEN").replace("{server}", "robutler").replace(
        "{hint}", cli_command("secrets set ROBUTLER_TOKEN")
    )
    assert MCP_CHECK_WORDS["fixEntry"] not in check.fix
