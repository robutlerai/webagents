"""
How the line-by-line chat draws a turn (2026-09-24, `cli/repl/render.py`).

Each case is something the old renderer got wrong, found by driving the real
chat through a pseudo-terminal against a stand-in model: tool results never
shown (the daemon's `{"type": "tool_result"}` events were not understood), tool
calls drawn above the text or at the end, a phantom second call from arguments
streamed under the call's index, the text either side of a call fused into one
paragraph, and a failed read painted green.
"""

from io import StringIO

from rich.console import Console

from webagents.cli.repl.render import (
    StreamError,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolResult,
    TurnRenderer,
    complete_blocks_end,
    events_from_chunk,
    tool_failed,
    tool_result_summary,
)


def _console():
    return Console(file=StringIO(), width=80, force_terminal=False, color_system=None, record=True)


def _turn(events):
    console = _console()
    renderer = TurnRenderer(console)
    for event in events:
        renderer.feed(event)
        renderer.flush()
    renderer.flush(final=True)
    return console.export_text(), renderer


def test_the_daemons_tool_events_are_understood():
    call = events_from_chunk({"type": "tool_call", "call_id": "c1", "name": "list_directory", "arguments": "{}"})
    result = events_from_chunk({"type": "tool_result", "id": "c1", "status": "success", "result": "ok"})
    assert isinstance(call[0], ToolCall) and call[0].id == "c1"
    # THE BUG: this shape was dropped, so no result was ever shown.
    assert isinstance(result[0], ToolResult) and result[0].result == "ok"


def test_arguments_streamed_by_index_belong_to_one_call():
    text, renderer = _turn([
        ToolCallDelta(0, "c1", "list_directory", ""),
        ToolCallDelta(0, None, None, '{"path":'),
        ToolCallDelta(0, None, None, ' "."}'),
        ToolResult("c1", "success", "Directory listing for /x:\n[DIR] a\nb"),
    ])
    tools = [s for s in renderer.segments if type(s).__name__ == "ToolSegment"]
    # THE BUG: the index-only chunks became a second, nameless call ("Ran ...").
    assert len(tools) == 1
    assert tools[0].arguments == '{"path": "."}'
    assert "Ran" not in text


def test_a_turn_is_drawn_in_the_order_it_happened():
    text, _ = _turn([
        TextDelta("Let me look first.\n"),
        ToolCallDelta(0, "c1", "list_directory", '{"path": "."}'),
        ToolCall("c1", "list_directory", '{"path": "."}'),
        ToolResult("c1", "success", "Directory listing for /x:\n[DIR] src\nREADME.md"),
        TextDelta("There are two entries.\n"),
    ])
    first = text.index("Let me look first.")
    tool = text.index("● list_directory(.)")
    summary = text.index("⎿  2 entries: src/, README.md")
    after = text.index("There are two entries.")
    # THE BUG: tools went above the text or to the end, and the two sentences
    # either side of the call were fused into one paragraph.
    assert first < tool < summary < after


def test_a_finished_block_is_printed_once_and_a_fence_is_never_split():
    text = "Intro paragraph.\n\n```python\nx = 1\n\ny = 2\n```\n\nstill writing"
    boundary = complete_blocks_end(text, 0)
    # Ends after the fenced block, not at the blank line INSIDE it.
    assert text[:boundary].endswith("```\n\n")
    printed, _ = _turn([TextDelta(text[:30]), TextDelta(text[30:])])
    assert printed.count("Intro paragraph.") == 1
    assert printed.count("y = 2") == 1


def test_a_failure_reported_as_text_is_shown_as_a_failure():
    # The local file tools RETURN "File not found: x" with a success status.
    assert tool_failed("success", "File not found: does-not-exist.txt")
    assert tool_result_summary("read_file", "File not found: does-not-exist.txt", "success") == (
        "File not found: does-not-exist.txt"
    )
    assert not tool_failed("success", "hello\nworld")
    assert tool_result_summary("read_file", "hello\nworld", "success") == "Read 2 lines"


def test_the_history_keeps_the_text_and_not_the_tool_calls():
    _, renderer = _turn([
        TextDelta("Before.\n"),
        ToolCallDelta(0, "c1", "list_directory", "{}"),
        ToolResult("c1", "success", "Directory listing for /x:"),
        TextDelta("After.\n"),
    ])
    # Separate paragraphs: one newline fused them when history was shown again.
    assert renderer.plain_text() == "Before.\n\nAfter."


def test_a_failed_turn_says_so_instead_of_drawing_nothing():
    # The shapes the servers stream when a run fails; both were dropped, so the
    # chat drew an empty reply (2026-09-24).
    assert events_from_chunk({"error": "Connection error."}) == [StreamError("Connection error.")]
    assert events_from_chunk({"error": {"message": "bad key", "type": "auth"}}) == [StreamError("bad key")]
    text, renderer = _turn([TextDelta("Partial answer.\n"), StreamError("Connection error.")])
    assert text.index("Partial answer.") < text.index("✗ Connection error.")
    assert renderer.failed
