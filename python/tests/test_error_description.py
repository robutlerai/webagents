"""
What the CLI says when a request fails (2026-09-24, `webagents/utils/errors.py`
and `cli/client/daemon_client.py:daemon_error_detail`).

A mistyped OPENAI_BASE_URL reached the person as "Error: Connection error."
(the OpenAI client keeps the reason in `__cause__`), and through the daemon as
httpx's "Server error '500 Internal Server Error' for url ..." with a link to
MDN, while the daemon's answer, which said why, went unread.
"""

import httpx
import openai

from webagents.cli.client.daemon_client import daemon_error_detail
from webagents.utils.errors import describe_exception


def _connection_error(url: str) -> openai.APIConnectionError:
    error = openai.APIConnectionError(request=httpx.Request("POST", url))
    error.__cause__ = httpx.ConnectError("All connection attempts failed")
    return error


def test_a_connection_failure_names_the_server_and_the_reason():
    text = describe_exception(_connection_error("http://127.0.0.1:9/v1/chat/completions"))
    assert text == "Could not reach http://127.0.0.1:9: All connection attempts failed"


def test_only_the_origin_is_named():
    # A path or a query can carry a key on some gateways.
    text = describe_exception(_connection_error("https://gw.example/v1/sk-secret/chat?key=secret"))
    assert "https://gw.example" in text and "secret" not in text


def test_an_answer_from_the_server_is_left_as_it_is():
    request = httpx.Request("GET", "http://127.0.0.1:9/x")
    error = httpx.HTTPStatusError("Server error", request=request, response=httpx.Response(500, request=request))
    assert describe_exception(error) == "Server error"
    assert describe_exception(ValueError("plain")) == "plain"


def test_the_daemon_error_says_what_the_daemon_said():
    request = httpx.Request("POST", "http://127.0.0.1:8765/agents/a/chat/completions")
    plain = httpx.HTTPStatusError("500", request=request, response=httpx.Response(500, request=request, text="Connection error."))
    assert daemon_error_detail(plain) == "Connection error."
    as_json = httpx.HTTPStatusError(
        "404", request=request, response=httpx.Response(404, request=request, json={"detail": "Agent not found: a"})
    )
    assert daemon_error_detail(as_json) == "Agent not found: a"
    # No answer at all: where the daemon was supposed to be.
    down = httpx.ConnectError("All connection attempts failed", request=request)
    assert daemon_error_detail(down) == "Could not reach http://127.0.0.1:8765: All connection attempts failed"
