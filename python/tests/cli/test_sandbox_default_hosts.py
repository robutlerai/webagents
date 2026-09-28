"""
The agent-file line edit behind "always" in the chat's ask-on-first-use, and
the words it uses (the sandbox-default lane, 2026-09-27), against the shared
fixture `tests/fixtures/cli/sandbox_default_hosts.json` that
`typescript/tests/unit/cli/sandbox-default-hosts.test.ts` reads too. Every
edited file is read back through the loader, so the host really lands in
`sandbox.network.hosts`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from webagents.cli.loader.agent_md import parse_agent_file
from webagents.cli.sandbox_default_hosts import HOST_WORDS, add_network_host, host_answer

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "sandbox_default_hosts.json").read_text())


def test_the_words_and_the_answers():
    assert HOST_WORDS == FIXTURE["words"]
    for kind, answers in FIXTURE["answers"].items():
        for typed in answers:
            assert host_answer("xyz" if typed == "anything else" else typed) == kind, (kind, typed)
    assert host_answer(None) == "no"


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=[c["case"] for c in FIXTURE["cases"]])
def test_each_case(case, tmp_path):
    text, problem = add_network_host(case["before"], FIXTURE["host"])
    if "after" in case:
        assert problem is None
        assert text == case["after"]
        file = tmp_path / "AGENT.md"
        file.write_text(text)
        parsed = parse_agent_file(file)
        assert FIXTURE["host"] in parsed.metadata.sandbox.network.hosts
    else:
        assert text is None and problem == case["problem"]


def test_crlf_files_stay_crlf():
    text, problem = add_network_host("---\r\nname: a\r\n---\r\nBody.\r\n", "example.com")
    assert problem is None
    assert "\r\nsandbox:\r\n  network:\r\n    hosts:\r\n      - example.com\r\n---" in text
