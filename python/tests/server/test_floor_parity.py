"""The two SDKs' credential floors are ONE decision written twice, so this test
reads the OTHER language's source and asserts the two copies still agree.

Why it has to read the file rather than compare two Python constants: the drift
this catches is somebody adding a billable path in TypeScript and not in Python
(or a header name in one and not the other). Nothing inside a single language
can see that. The TypeScript half asserts the same thing from the other side, in
``typescript/tests/unit/server/floor-parity.test.ts``, so either suite alone
catches the drift.

The pair matters because the SDKs serve the SAME endpoint for the same platform.
A caller probing both must not find ``chat/completions`` gated on one and open on
the other, and the observable refusal must be the same 401 with the same message
— the last time these disagreed, one answered 401 and the other a JSON-parser
500 for the same anonymous request.
"""

import re
from pathlib import Path

import pytest

from webagents.server.core.credential_floor import (
    BILLABLE_METHODS,
    BILLABLE_PATHS,
    BILLABLE_WS_PATHS,
    CREDENTIAL_HEADERS,
    PUBLIC_SUBPATHS,
    PUBLIC_WS_SUBPATHS,
    UNAUTHORIZED_MESSAGE,
    WS_CREDENTIAL_QUERY_PARAMS,
)

#: python/tests/server/ -> webagents/ -> typescript/src/server/
TS_FLOOR = (
    Path(__file__).resolve().parents[3]
    / "typescript"
    / "src"
    / "server"
    / "credential-floor.ts"
)


def _ts_literals(source: str, name: str) -> list:
    """Every quoted string literal in the right-hand side of ``export const NAME = ...``."""
    match = re.search(
        rf"export const {name}\s*=\s*(\[[\s\S]*?\]|[^;]+);",
        source,
    )
    if not match:
        raise AssertionError(f"{name} not found in {TS_FLOOR}")
    return [a or b for a, b in re.findall(r"'([^']*)'|\"([^\"]*)\"", match.group(1))]


@pytest.fixture(scope="module")
def ts_source() -> str:
    assert TS_FLOOR.exists(), (
        f"the TypeScript credential floor is not at {TS_FLOOR}. If it moved, fix this "
        "path — a parity test that silently stops comparing is worse than none."
    )
    return TS_FLOOR.read_text(encoding="utf-8")


class TestTheTwoSdkFloorsDoNotDriftApart:
    def test_the_typescript_module_is_where_this_test_thinks_it_is(self, ts_source):
        """Guard the guard."""
        assert "BILLABLE_PATHS" in ts_source

    def test_agrees_on_which_headers_can_carry_a_credential(self, ts_source):
        assert _ts_literals(ts_source, "CREDENTIAL_HEADERS") == list(CREDENTIAL_HEADERS)

    def test_agrees_on_the_billable_path_set(self, ts_source):
        assert _ts_literals(ts_source, "BILLABLE_PATHS") == list(BILLABLE_PATHS)

    def test_agrees_on_which_methods_the_floor_applies_to(self, ts_source):
        assert _ts_literals(ts_source, "BILLABLE_METHODS") == list(BILLABLE_METHODS)

    def test_agrees_on_the_billable_websocket_path_set(self, ts_source):
        assert _ts_literals(ts_source, "BILLABLE_WS_PATHS") == list(BILLABLE_WS_PATHS)

    def test_agrees_on_which_subpaths_are_declared_public(self, ts_source):
        """The allow-list is half of the classification, so it drifts the same
        way the billable set does — and drift here is worse, because a path
        declared public in one SDK and merely FORGOTTEN in the other looks
        identical from inside either language."""
        assert _ts_literals(ts_source, "PUBLIC_SUBPATHS") == list(PUBLIC_SUBPATHS)

    def test_agrees_on_which_websocket_subpaths_are_declared_public(self, ts_source):
        assert _ts_literals(ts_source, "PUBLIC_WS_SUBPATHS") == list(PUBLIC_WS_SUBPATHS)

    def test_no_subpath_is_declared_both_billable_and_public(self):
        """Classification means EXACTLY one of the two sets. A path in both
        would let the enumerating test pass while the floor gates a route the
        allow-list says is public -- or the reverse, which is how a route ends
        up looking classified and being open."""
        assert set(BILLABLE_PATHS) & set(PUBLIC_SUBPATHS) == set()
        assert set(BILLABLE_WS_PATHS) & set(PUBLIC_WS_SUBPATHS) == set()

    def test_agrees_on_the_websocket_credential_query_parameters(self, ts_source):
        assert _ts_literals(ts_source, "WS_CREDENTIAL_QUERY_PARAMS") == list(
            WS_CREDENTIAL_QUERY_PARAMS
        )

    def test_answers_with_the_same_refusal_message(self, ts_source):
        assert "".join(_ts_literals(ts_source, "UNAUTHORIZED_MESSAGE")) == UNAUTHORIZED_MESSAGE
