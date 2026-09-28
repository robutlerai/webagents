"""
Discovery results carry the TrustFlow score (plan item 2.5, 2026-09-26):
`format_agent` adds `trustflow` and `trustflow_for_query` from the platform's
row, as the shared cases pin them (`tests/fixtures/trust/trust_tool_definition.json`,
`discovery_agent_fields`; TypeScript
tests/unit/skills/discovery/format-agent-trustflow-w2trust.test.ts).
"""

import json
from pathlib import Path

import pytest

from webagents.agents.skills.robutler.discovery.skill import SEARCH_DEFINITION, format_agent

FIXTURE = json.loads((Path(__file__).resolve().parents[2] / "fixtures" / "trust" / "trust_tool_definition.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["discovery_agent_fields"]["cases"], ids=lambda c: c["name"])
def test_an_agent_result_carries_its_trustflow_score(case):
    assert format_agent(case["raw"]) == case["expect"]


def test_the_search_description_tells_the_model_what_the_score_is():
    assert "TrustFlow score (trustflow, 0 to 1, computed by the platform" in SEARCH_DEFINITION["function"]["description"]
