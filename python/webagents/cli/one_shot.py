"""
`webagents -p "..."` (2026-09-24): one prompt, the answer, exit. The same
flags and output as the TypeScript CLI's `-p` (`typescript/src/cli/index.ts`
`chatAction`):

  * `--output-format text` (the default): the answer's text on stdout;
  * `json`: one document, `{"content", "content_items", "usage"}`;
  * `stream-json`: one JSON object per line as the turn happens (`delta`,
    `tool_call`, `tool_result`, then `done` or `error`), the TypeScript
    `streamJsonEvent` shapes.

The agent is built the way the chat builds it (`cli/agent_builder.py`), and
when there is nothing to run it on, the reason goes to stderr with exit 1
before anything is sent: the same sentence the chat shows.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

OUTPUT_FORMATS = ("text", "json", "stream-json")


def _fail(output_format: str, message: str, code: Optional[str], hint: Optional[str]) -> None:
    """A failed `-p` in the format asked for (2026-09-28, the e2e pass):
    `stream-json` an `error` line, `json` one `{"error": {...}}` document on
    stdout, `text` the headline and hint on stderr. The TypeScript `-p` answers
    the same (fixture `cli/final_sdk_low_items.json`, `prompt_failure`)."""
    body: Dict[str, Any] = {"message": message, **({"code": code} if code else {}), **({"hint": hint} if hint else {})}
    if output_format == "stream-json":
        _line({"type": "error", "error": body})
    elif output_format == "json":
        sys.stdout.write(json.dumps({"error": body}, indent=2, ensure_ascii=False) + "\n")
    else:
        sys.stderr.write(f"Error: {message}\n" if code != "no_model" else f"{message}\n")
        if hint:
            sys.stderr.write(f"{hint}\n")


def _line(event: Dict[str, Any]) -> None:
    # Compact, as `JSON.stringify` writes it: one line per event, the same bytes as the TypeScript CLI.
    sys.stdout.write(json.dumps(event, separators=(",", ":"), ensure_ascii=False) + "\n")
    sys.stdout.flush()


async def _run(agent_path: Optional[Path], prompt: str, model: Optional[str], output_format: str) -> int:
    from .agent_builder import build_agent
    from .credentials import get_token
    from .repl.render import Finish, Note, StreamError, TextDelta, ThinkingStart, ToolCall, ToolCallDelta, ToolResult, Usage, events_from_chunk

    # As in the chat: `discovery` searches as the person when the agent cannot.
    built = await build_agent(agent_path, working_dir=Path.cwd(), model=model, person_token=get_token)
    # An MCP server that did not load or connect, or a literal that looks
    # like a key, is said on stderr as the TypeScript `-p` says it (2026-09-26,
    # the e2e run: this was silent, the agent's log going to a file).
    from webagents.agents.skills.local.mcp.config import problem_lines

    mcp = built.agent.skills.get("mcp") if getattr(built.agent, "skills", None) else None
    if mcp is not None and hasattr(mcp, "server_report"):
        for line in problem_lines(mcp.server_report()):
            sys.stderr.write(line + "\n")
    if built.model_problem:
        # `--output-format json` and `stream-json` answer a failure in JSON
        # too (2026-09-28, the e2e pass: they printed "Error: ..." text).
        _fail(output_format, built.model_problem, "no_model", None)
        return 1

    answer = ""
    error: Optional[str] = None
    error_code: Optional[str] = None
    finish: Optional[Finish] = None
    thinking = False
    input_tokens = output_tokens = 0
    streamed_calls: Dict[int, Dict[str, str]] = {}
    stream = output_format == "stream-json"

    emitted_calls: set = set()

    def emit_call(call: Dict[str, str]) -> None:
        # One line per call: a call arrives both as streamed deltas and, whole,
        # as the agent's own event; the TypeScript CLI writes it once.
        if call.get("id") and call["id"] in emitted_calls:
            return
        emitted_calls.add(call.get("id") or f"#{len(emitted_calls)}")
        _line({"type": "tool_call", "tool_call": call})

    def flush_streamed_calls() -> None:
        # OpenAI-shaped deltas arrive in pieces; a call is whole once its result comes.
        for call in streamed_calls.values():
            emit_call(call)
        streamed_calls.clear()

    # The person at the terminal is the agent's owner (`access.caller`).
    from webagents.access import run_as_local_owner

    run_as_local_owner(built.agent)
    try:
        async for chunk in built.agent.run_streaming([{"role": "user", "content": prompt}]):
            if not isinstance(chunk, dict):
                continue
            for event in events_from_chunk(chunk):
                if isinstance(event, TextDelta):
                    answer += event.text
                    if stream:
                        _line({"type": "delta", "delta": event.text})
                elif isinstance(event, ToolCallDelta):
                    call = streamed_calls.setdefault(event.index, {"id": "", "name": "", "arguments": ""})
                    call["id"] = event.id or call["id"]
                    call["name"] = event.name or call["name"]
                    call["arguments"] += event.arguments or ""
                elif isinstance(event, ToolCall) and stream:
                    emit_call({"id": event.id, "name": event.name, "arguments": event.arguments})
                elif isinstance(event, ToolResult) and stream:
                    flush_streamed_calls()
                    _line(
                        {
                            "type": "tool_result",
                            "tool_result": {"call_id": event.id, "result": event.result, "is_error": event.status != "success"},
                        }
                    )
                elif isinstance(event, Usage):
                    input_tokens += event.prompt_tokens
                    output_tokens += event.completion_tokens
                elif isinstance(event, StreamError):
                    error = event.message
                elif isinstance(event, Note) and stream:
                    # The model failover's note (plan item 2.8), as the
                    # TypeScript stream-json writes it (2026-09-28).
                    _line({"type": "note", "note": event.text})
                elif isinstance(event, Finish):
                    finish = event
                elif isinstance(event, ThinkingStart):
                    thinking = True
    except Exception as failure:  # noqa: BLE001 - one line and exit 1, as the TypeScript CLI does
        from webagents.utils.errors import describe_exception

        error = describe_exception(failure)
        code = getattr(failure, "code", None)
        error_code = code if isinstance(code, str) and code else None

    if stream:
        flush_streamed_calls()
    # The agent's response as the TypeScript agent returns it (`RunResponse`).
    response: Dict[str, Any] = {"content": answer, "content_items": [{"type": "text", "text": answer}] if answer else []}
    if input_tokens or output_tokens:
        response["usage"] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
    if error is not None:
        # The headline and hint the chat shows (`repl/failures.py`), not the
        # platform's protocol text; the TypeScript `-p` does the same.
        from .model_access import platform_llm_url
        from .repl.failures import present_failure
        from .repl.render import error_hint

        on_robutler = getattr(built.access, "kind", None) == "proxy"
        explained = present_failure(
            error,
            proxy_url=platform_llm_url() if on_robutler else None,
            generic_hint=lambda text: error_hint(built.model_label, text),
            model=getattr(built.access, "model", None),
        )
        # Robutler's own refusals get their stable code; anything else keeps
        # the error's own, as the TypeScript `-p` does. The hint is `-p`'s
        # (`for_prompt`): no chat command.
        from .repl.failures import for_prompt

        _fail(output_format, explained.headline, explained.code or error_code, for_prompt(explained.hint))
        return 1
    # THE AGENT ENDED THE TURN (2026-09-28, `core/tool_budget.py`): its tool
    # rounds ran out, or it repeated one call, and its last, tool-less call
    # gave the answer printed here. The reason goes in the response's
    # `finish` (every format), a sentence on stderr, and the exit is 1, so a
    # script can tell it from a turn that finished. The TypeScript `-p`
    # answers the same.
    ended_by_agent = finish is not None and finish.reason in ("tool_round_limit", "tool_loop")
    if ended_by_agent:
        response["finish"] = {
            key: value
            for key, value in (("reason", finish.reason), ("rounds", finish.rounds), ("tool", finish.tool))
            if value is not None
        }
    if ended_by_agent and answer.strip() and output_format == "text":
        from webagents.agents.core.tool_budget import tool_loop_sentence, tool_round_limit_sentence

        said = tool_loop_sentence(finish.tool) if finish.reason == "tool_loop" else tool_round_limit_sentence(finish.rounds, answered=True)
        sys.stderr.write(f"Error: {said}\n")
    if not answer.strip() and not stream:
        # NOTHING SAID IS SAID, in `-p` too (the ptypass-fixes lane,
        # 2026-09-27; the chat's `present_empty_reply`, the TypeScript `-p`'s
        # twin): an empty reply printed an empty line and exited 0, so a
        # script could not tell it from an answer. The truthful line goes to
        # stderr; stdout keeps the answer channel's shape.
        from .repl.failures import present_empty_reply

        explained = present_empty_reply(
            finish.reason if finish else None,
            blocked=bool(finish and finish.blocked),
            retried=bool(finish and finish.retried),
            thinking=thinking,
            rounds=finish.rounds if finish else None,
            tool=finish.tool if finish else None,
        )
        sys.stderr.write(f"{'Error: ' if ended_by_agent else ''}{explained.headline}\n")
        if explained.hint:
            sys.stderr.write(f"{explained.hint}\n")
    if stream:
        _line({"type": "done", "response": response})
    elif output_format == "json":
        sys.stdout.write(json.dumps(response, indent=2, ensure_ascii=False) + "\n")
    else:
        sys.stdout.write(answer + ("\n" if not answer.endswith("\n") else ""))
    return 1 if ended_by_agent else 0


def run_prompt(agent_path: Optional[Path], prompt: str, model: Optional[str], output_format: str) -> int:
    import os

    if not os.environ.get("WEBAGENTS_DEBUG"):
        # THE AGENT'S LOG GOES TO A FILE, as the chat's does (2026-09-25). On
        # stderr, a failed run printed the agent's own ERROR line above the
        # `Error:` line that already says what failed; the TypeScript `-p`
        # prints only the latter. WEBAGENTS_DEBUG keeps the log on stderr.
        from webagents.utils.logging import setup_logging

        # The profile's own folder, as the chat's `repl_log_path` (the
        # ptypass-fixes lane, 2026-09-27): this went to `~/.webagents/logs`
        # whatever `--profile` said.
        from .config_store import global_dir

        log_dir = global_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(level="INFO", log_file=str(log_dir / "repl.log"), console_output=False)
    return asyncio.run(_run(agent_path, prompt, model, output_format))
