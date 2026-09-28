"""
`webagents budget <tokenId>` (webagents gap-closure plan 2.3, 2026-09-26):
the tree the platform answers is printed one line per token, as the fixture
`fixtures/payments/delegate_budget.json` pins it for both CLIs, with a totals
line; sign-in and answer failures are named.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from webagents.cli import budget_tree as module
from webagents.cli.budget_tree import budget_tree, budget_tree_totals, format_credits, render_budget_tree

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "payments" / "delegate_budget.json").read_text())
SAMPLE = FIXTURE["tree"]["sample"]


def test_rendering_matches_the_fixture_line_for_line():
    assert render_budget_tree(SAMPLE) == FIXTURE["tree"]["rendered"]
    assert budget_tree_totals(SAMPLE) == FIXTURE["tree"]["totals_line"]


def test_credits_are_formatted_without_trailing_zeros_or_float_noise():
    assert format_credits(0.1) == "0.1"
    assert format_credits(1) == "1"
    assert format_credits(0.1 + 0.2) == "0.3"
    assert format_credits(0.0123) == "0.0123"


def test_the_command_fetches_with_the_sign_in_and_renders(monkeypatch):
    seen = {}

    def fake_get(url, headers=None, timeout=None):
        seen["url"] = url
        seen["headers"] = headers
        return httpx.Response(200, json={"tree": SAMPLE}, request=httpx.Request("GET", url))

    monkeypatch.setattr(module, "cli_command", lambda rest: f"webagents {rest}")
    monkeypatch.setattr("webagents.cli.config_store.platform_url", lambda: "https://platform.test")
    monkeypatch.setattr("webagents.cli.credentials.get_token", lambda: "cli-token")
    monkeypatch.setattr(httpx, "get", fake_get)
    result = budget_tree(SAMPLE["id"])
    assert result.ok, result.message
    assert seen["url"] == "https://platform.test" + FIXTURE["tree"]["route"].replace("{id}", SAMPLE["id"])
    assert seen["headers"] == {"Authorization": "Bearer cli-token"}
    assert result.lines == FIXTURE["tree"]["rendered"]
    assert result.totals == FIXTURE["tree"]["totals_line"]


def test_failures_are_named(monkeypatch):
    monkeypatch.setattr("webagents.cli.config_store.platform_url", lambda: "https://platform.test")
    monkeypatch.setattr("webagents.cli.credentials.get_token", lambda: "cli-token")

    def answer(status):
        return lambda url, headers=None, timeout=None: httpx.Response(status, json={}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", answer(404))
    assert budget_tree(SAMPLE["id"]).code == "not_found"
    monkeypatch.setattr(httpx, "get", answer(401))
    assert budget_tree(SAMPLE["id"]).code == "expired"
    assert budget_tree("nope").code == "bad_token_id"
    monkeypatch.setattr("webagents.cli.credentials.get_token", lambda: None)
    assert budget_tree(SAMPLE["id"]).code == "not_signed_in"
