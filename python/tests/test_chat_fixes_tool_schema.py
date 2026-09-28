"""
An array parameter declares its items (2026-09-27, the chat-fixes lane).

The `@tool` decorator mapped `List[str]` to `{"type": "array"}` and stopped
there. Nothing noticed while the Python proxy skill's tools never reached a
model; the first day they did, Google refused every turn of the built-in agent
with "parameters.properties[ignore].items: missing field" (`list_directory`
declares `ignore: Optional[List[str]]`). The TypeScript decorators declare
`items` for their arrays.
"""

from typing import Any, Dict, List, Optional

from webagents.agents.tools.decorators import tool


def _props(f):
    return f._webagents_tool_definition["function"]["parameters"]["properties"]


def test_a_list_of_strings_declares_string_items():
    @tool
    def f(ignore: Optional[List[str]] = None) -> str:
        """t"""
        return ""

    assert _props(f)["ignore"] == {"type": "array", "description": "Parameter ignore", "items": {"type": "string"}}


def test_the_item_type_follows_the_annotation():
    @tool
    def f(counts: List[int], weights: List[float], flags: List[bool], rows: List[Dict[str, Any]], anything: list) -> str:
        """t"""
        return ""

    props = _props(f)
    assert props["counts"]["items"] == {"type": "integer"}
    assert props["weights"]["items"] == {"type": "number"}
    assert props["flags"]["items"] == {"type": "boolean"}
    assert props["rows"]["items"] == {"type": "object"}
    # A bare `list` says nothing about its items: strings, the safe default.
    assert props["anything"] == {"type": "array", "description": "Parameter anything", "items": {"type": "string"}}


def test_the_built_in_agents_file_tools_are_acceptable_to_google():
    from webagents.agents.skills.local.filesystem.skill import FilesystemSkill

    for attr in dir(FilesystemSkill):
        definition = getattr(getattr(FilesystemSkill, attr), "_webagents_tool_definition", None)
        if not definition:
            continue
        for name, prop in definition["function"]["parameters"]["properties"].items():
            if prop["type"] == "array":
                assert "items" in prop, f"{definition['function']['name']}.{name} has no items"
