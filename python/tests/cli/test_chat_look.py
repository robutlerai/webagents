"""
The Python chats' look (2026-09-24, `webagents/cli/ui/` and `cli/repl/render.py`).

Brought over from the TypeScript chat's redesign so the two SDKs look like
one product: the palette, the star and the shimmer, the terminal background
query, markdown drawn the chat's way, the welcome card, the input box's menu,
and the tool / thought / error / status lines both Python chats draw. Each
test is something a person would notice if it broke.
"""

from io import StringIO

from rich.console import Console
from rich.theme import Theme as RichTheme

from webagents.cli.repl.render import (
    ToolSegment,
    ThoughtSegment,
    compact_number,
    duration,
    error_hint,
    error_lines,
    stats_line,
    status_line,
    tool_lines,
    thought_lines,
)
from webagents.cli.ui.banner import WelcomeInfo, welcome_card, wordmark
from webagents.cli.ui.markdown import ChatMarkdown, preview_lines
from webagents.cli.ui.motion import STAR_FRAMES, shimmer, spark_at
from webagents.cli.ui.prompt_box import PromptBox, sent_message
from webagents.cli.ui.terminal import parse_background_reply
from webagents.cli.ui.theme import markdown_styles, mix, theme_for


def _console(colour=True, width=60):
    return Console(
        file=StringIO(), width=width, force_terminal=colour, color_system="truecolor" if colour else None, record=True
    )


def _render(markdown, colour=True, width=60):
    console = _console(colour, width)
    theme = theme_for(console, background="#101010")
    console.push_theme(RichTheme(markdown_styles(theme)))
    console.print(ChatMarkdown(markdown, theme))
    return console.export_text(clear=False), console.export_text(styles=True)


# -- the palette and the terminal ----------------------------------------------------


def test_the_bands_are_mixed_from_the_terminals_own_background():
    dark = theme_for(_console(), background="#000000")
    light = theme_for(_console(), background="#ffffff")
    assert not dark.light and light.light
    # A lift toward white on dark, toward black on light, the TypeScript values.
    assert dark.palette.surface == mix("#000000", "#ffffff", 0.12)
    assert light.palette.surface == mix("#ffffff", "#000000", 0.06)


def test_the_terminal_background_reply_is_read_and_a_silent_terminal_ends_the_wait():
    # The colour alone is not the end: the DA1 reply follows it, and must be read too.
    assert parse_background_reply(b"\x1b]11;rgb:1e1e/1e1e/1e1e\x1b\\") == ("#1e1e1e", False)
    assert parse_background_reply(b"\x1b]11;rgb:1e1e/1e1e/1e1e\x1b\\\x1b[?62;22c") == ("#1e1e1e", True)
    assert parse_background_reply(b"\x1b]11;rgb:ff/ff/ff\x07\x1b[?62;22c")[0] == "#ffffff"
    # DA1 alone: the terminal ignores OSC 11, so stop waiting, with nothing.
    assert parse_background_reply(b"\x1b[?1;2c") == (None, True)
    assert parse_background_reply(b"\x1b]11;rgb:1e") == (None, False)


def test_the_star_runs_forward_and_back_and_the_shimmer_leaves_the_text_alone():
    assert "".join(STAR_FRAMES) == "·✢✳✶✻✽✻✶✳✢"
    theme = theme_for(_console(), background="#101010", animate=True)
    assert shimmer(theme, "Writing…", theme.palette.agent, 0.7).plain == "Writing…"
    still = theme_for(_console(), background="#101010", animate=False)
    assert str(shimmer(still, "Writing…", still.palette.agent, 0.7).style) == still.palette.agent
    # The starfield twinkles the same way on every redraw of the same moment.
    assert [spark_at(theme, c, 3.3) for c in range(60)] == [spark_at(theme, c, 3.3) for c in range(60)]


# -- markdown ---------------------------------------------------------------------------


def test_headings_sit_on_the_left_and_lists_are_numbered_with_a_dot():
    text, _ = _render("# Big title\n\n## Section\n\n1. first\n2. second\n\n- item\n")
    lines = text.splitlines()
    assert lines[0].startswith("Big title")  # Rich centred H1
    assert "1. first" in text and "2. second" in text  # Rich wrote "1 first"
    assert "• item" in text


def test_code_is_a_shaded_band_with_its_language_and_nothing_on_its_lines_a_copy_picks_up():
    text, styled = _render("```python\ndef f():\n    return 1\n```\n")
    lines = text.splitlines()
    assert lines[0] == "▄" * 60 and lines[-1] == "▀" * 60
    assert lines[1].startswith(" def f():") and lines[1].rstrip().endswith("python")
    assert "│" not in text
    assert "\x1b[" in styled and "48;2;" in styled  # the band has a background


def test_without_colour_code_is_marked_by_rules():
    text, _ = _render("```python\nx = 1\n```\n", colour=False)
    lines = text.splitlines()
    assert lines[0].startswith("── python ──")
    assert lines[1] == " x = 1"
    assert set(lines[2]) == {"─"}


def test_tables_get_rounded_borders():
    text, _ = _render("| File | What |\n| --- | --- |\n| a.md | the agent |\n")
    assert "╭" in text and "╰" in text and "│ a.md │ the agent │" in text


def test_task_lists_become_boxes():
    text, _ = _render("- [ ] to do\n- [x] done\n")
    assert "☐ to do" in text and "✔ done" in text


def test_a_restored_reply_is_previewed_in_whole_blocks():
    # The preview was the first 8 rendered lines, which cut a table after its
    # first row and left it without a bottom edge.
    console = _console()
    theme = theme_for(console, background="#101010")
    console.push_theme(RichTheme(markdown_styles(theme)))
    reply = ("Let me look at the files first.\n\nI looked at the directory. Here is what is there:\n\n"
             "| File | What it is |\n| --- | --- |\n| AGENT.md | the agent definition |\n| notes.txt | scratch notes |\n\n"
             "Ask me to open any of them.\n")

    def plain(lines):
        return ["".join(segment.text for segment in line).rstrip() for line in lines]

    lines, more = preview_lines(console, reply, theme, 57, budget=8)
    assert more and plain(lines)[-1] == "I looked at the directory. Here is what is there:"
    lines, more = preview_lines(console, reply, theme, 57, budget=11)
    assert more and plain(lines)[-1].startswith("╰")  # the whole table, its bottom edge too
    lines, more = preview_lines(console, reply, theme, 57, budget=40)
    assert not more and plain(lines)[-1] == "Ask me to open any of them."
    # One block taller than the preview: its first lines, and a mark that there is more.
    tall = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(20)) + "\n```\n"
    lines, more = preview_lines(console, tall, theme, 57, budget=8)
    assert len(lines) == 8 and more


# -- the welcome card -------------------------------------------------------------------------


def test_the_card_lines_are_one_width_and_a_warning_wraps_rather_than_being_cut():
    theme = theme_for(_console(), background="#101010")
    info = WelcomeInfo(
        agent="helper",
        description="A tool agent",
        model="openai/gpt-4o-mini",
        tools=["read_file", "run_command"],
        folder="/a/very/long/path/that/goes/on/and/on/and/on/until/it/cannot/possibly/fit/in/the/card/work",
        warnings=["OPENAI_API_KEY is not set, and this agent's model needs it. Run `webagents secrets set OPENAI_API_KEY`."],
        version="0.3.6",
    )
    lines = welcome_card(theme, 100, info)
    box = lines[:-1]
    assert {line.cell_len for line in box} == {80}
    plain = [line.plain for line in lines]
    assert "webagents 0.3.6" in plain[0]
    assert any(line.rstrip(" │").endswith("/card/work") for line in plain)  # the folder keeps its end
    assert "webagents secrets set OPENAI_API_KEY" in " ".join(p.strip("│ ") for p in plain)


def test_the_wordmark_is_small_on_a_narrow_terminal():
    theme = theme_for(_console(), background="#101010")
    assert len(wordmark(theme, 120)) == 6
    assert len(wordmark(theme, 60)) == 3


# -- the input box --------------------------------------------------------------------------


def test_the_menu_narrows_as_you_type_and_follows_a_group_into_its_subcommands():
    box = PromptBox(
        theme_for(_console()),
        commands=[("/help", "Show help"), ("/agent list", "List agents"), ("/agent info", "Agent config"),
                  ("/history", "Show history")],
        footer=lambda: [],
    )
    assert [c[0] for c in box.menu_items("/")] == ["/help", "/agent list", "/agent info", "/history"]
    assert [c[0] for c in box.menu_items("/h")] == ["/help", "/history"]
    assert [c[0] for c in box.menu_items("/agent l")] == ["/agent list"]
    assert box.menu_items("hello") == [] and box.menu_items("/help\nmore") == []
    box.menu_dismissed = True
    assert box.menu_items("/") == []


async def test_history_suggests_nothing_while_the_menu_is_open():
    # The ghost of the last command sent sat beside a menu that would run another.
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.document import Document
    from prompt_toolkit.history import InMemoryHistory

    box = PromptBox(theme_for(_console()), commands=[("/help", "Show help"), ("/exit", "Leave")], footer=lambda: [])
    history = InMemoryHistory(["hello there", "/exit"])
    async for _ in history.load():  # the box's Buffer loads it this way, lazily
        pass
    buffer = Buffer(history=history)
    text = ""
    suggest = box.history_suggestions(lambda: text)
    for text, expected in (("/", None), ("/e", None), ("hel", "lo there")):
        suggestion = suggest.get_suggestion(buffer, Document(text))
        assert (suggestion.text if suggestion else None) == expected, text
    box.menu_dismissed = True  # esc closed the menu: history is back
    text = "/"
    assert suggest.get_suggestion(buffer, Document("/")).text == "exit"


def test_a_first_ctrl_c_or_esc_only_arms_the_second():
    box = PromptBox(theme_for(_console()), commands=[], footer=lambda: [])
    assert not box.exit_armed(1.0) and not box.esc_armed(1.0)  # never pressed is not "just pressed"
    box.exit_armed_at = 10.0
    assert box.exit_armed(11.0) and not box.exit_armed(13.0)


def test_a_sent_message_is_banded_only_where_a_band_can_be_drawn():
    plain = theme_for(_console(colour=False))
    assert [line.plain for line in sent_message(plain, 40, "hi\nthere")] == [" ❯ hi", "   there"]
    banded = theme_for(_console(), background="#101010")
    line = sent_message(banded, 40, "hi")[0]
    assert line.cell_len == 39 and "on " in str(line.spans[-1].style)


# -- the lines both chats draw ---------------------------------------------------------------


def test_a_tool_call_says_what_it_ran_what_it_produced_and_how_long_it_took():
    theme = theme_for(_console(), background="#101010")
    tool = ToolSegment(key="c1", id="c1", name="read_file", arguments='{"path": "a.txt"}', status="success",
                       result="one\ntwo", started=100.0, ended=101.5)
    lines = [line.plain for line in tool_lines(theme, tool, 102.0, 80)]
    assert lines == ["● read_file(a.txt)", "  ⎿  Read 2 lines  1.5s"]
    running = ToolSegment(key="c2", id="c2", name="list_directory", arguments="{}", started=100.0)
    assert tool_lines(theme, running, 100.5, 80)[0].plain.endswith(" list_directory")
    failed = ToolSegment(key="c3", id="c3", name="read_file", arguments="{}", status="success",
                         result="File not found: x", started=1.0, ended=1.1)
    summary = tool_lines(theme, failed, 2.0, 80)[1]
    assert summary.plain == "  ⎿  File not found: x"
    assert theme.palette.error in str(summary.spans[-1].style)


def test_an_error_says_what_failed_and_what_to_do():
    theme = theme_for(_console())
    lines = [line.plain for line in error_lines(theme, "Connection error.\nmore", "Check the network connection.", 80)]
    assert lines == ["✗ Connection error.", "  ⎿  Check the network connection."]
    assert error_hint("openai/gpt-4o-mini", "Error code: 401 - invalid api key") == (
        "The provider refused the key. Check OPENAI_API_KEY, then start the chat again."
    )
    assert error_hint("openai/gpt-4o-mini", "rate limit reached") == (
        "The provider is limiting requests. Wait a moment, then try again."
    )
    assert error_hint("openai/gpt-4o-mini", "something else entirely") is None


def test_the_status_and_stats_lines():
    theme = theme_for(_console(), background="#101010", animate=False)
    line = status_line(theme, "Writing", 100.0, 400, 103.2).plain
    assert line.endswith("Writing… (3s · ↓ 100 tokens · esc to interrupt)")
    assert line[0] in STAR_FRAMES
    assert stats_line(theme, 3.24, 1234).plain == "✻ Worked for 3.2s · 1,234 tokens"
    assert stats_line(theme, 75, 0).plain == "✻ Worked for 1m 15s"
    thought = ThoughtSegment(started=10.0, ended=12.1)
    assert thought_lines(theme, thought, 13.0)[0].plain == "∴ Thought for 2.1s"
    assert duration(9.94) == "9.9s" and compact_number(48_000) == "48k"
