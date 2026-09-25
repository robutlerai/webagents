"""
What a failed turn says, in the chat and in `-p` (2026-09-25): the table the
TypeScript CLI runs too (`typescript/tests/unit/cli/failures.test.ts`), so a
refusal reads the same in both, whichever words each SDK raised it with.
"""

import json
from pathlib import Path

import pytest

from webagents.cli.repl.failures import present_failure

TABLE = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "failure_presentation.json").read_text()
)


@pytest.mark.parametrize("case", TABLE["cases"], ids=[case["name"] for case in TABLE["cases"]])
def test_the_table_both_sdks_run(case):
    proxy_url = TABLE["proxy_url"] if case["proxy"] else None
    assert present_failure(case["message"], proxy_url=proxy_url) == (case["headline"], case["hint"], case["code"])


def test_a_providers_error_gets_the_chats_own_advice():
    explained = present_failure("Error code: 401 - invalid api key", generic_hint=lambda text: "Check OPENAI_API_KEY.")
    assert explained == ("Error code: 401 - invalid api key", "Check OPENAI_API_KEY.", None)
