"""
`webagents acp` end to end (gap-closure plan item 1.6, 2026-09-26): the CLI
is spawned as an editor would spawn it, with HOME in a temporary folder and
an OpenAI-compatible model scripted on loopback (no network, a dummy key),
and each transcript in `tests/fixtures/acp/acp_transcripts.json` is driven
through its stdin and read back from its stdout. The TypeScript CLI runs the
same cases in `tests/unit/cli/acp-stdio.test.ts`. Every stdout line must be a
JSON-RPC message, and the startup line must be on stderr.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PROTOCOL = json.loads((FIXTURES / "acp" / "acp_protocol.json").read_text())
TRANSCRIPTS = json.loads((FIXTURES / "acp" / "acp_transcripts.json").read_text())
ECHO_SERVER = FIXTURES / "mcp_tool" / "echo_server.py"
LINE_TIMEOUT = 60.0


# -- the scripted model ------------------------------------------------------------------------


class StubModel:
    """`POST /v1/chat/completions` answered from a script, one entry per call
    (the last repeats), streamed as SSE when asked to; every request's
    messages are kept for `model_sees`."""

    def __init__(self) -> None:
        self.script: List[Dict[str, Any]] = []
        self.calls: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802 - http.server's name
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                with stub.lock:
                    index = len(stub.calls)
                    stub.calls.append(body)
                    entry = stub.script[min(index, len(stub.script) - 1)] if stub.script else {"text": ""}
                if entry.get("delay_ms"):
                    time.sleep(entry["delay_ms"] / 1000)
                try:
                    if body.get("stream"):
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.end_headers()
                        for chunk in stub.chunks(entry):
                            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                            self.wfile.flush()
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    else:
                        payload = json.dumps(stub.completion(entry)).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(payload)))
                        self.end_headers()
                        self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the agent hung up (a cancelled prompt)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def reset(self, script: List[Dict[str, Any]]) -> None:
        with self.lock:
            self.script = script
            self.calls = []

    @staticmethod
    def _message(entry: Dict[str, Any]) -> Dict[str, Any]:
        if "tool_call" in entry:
            call = entry["tool_call"]
            return {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}]}
        return {"role": "assistant", "content": entry.get("text", "")}

    def completion(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        message = self._message(entry)
        return {"id": "chatcmpl-stub", "object": "chat.completion", "created": 0, "model": "stub",
                "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if "tool_calls" in message else "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    def chunks(self, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        base = {"id": "chatcmpl-stub", "object": "chat.completion.chunk", "created": 0, "model": "stub"}
        if "tool_call" in entry:
            call = entry["tool_call"]
            delta = {"role": "assistant", "tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                                          "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}]}
            finish = "tool_calls"
        else:
            delta = {"role": "assistant", "content": entry.get("text", "")}
            finish = "stop"
        return [
            {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
            {**base, "choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
        ]


# -- the agent process -------------------------------------------------------------------------


class AgentProcess:
    """`webagents acp <folder>` with its stdout read line by line on a thread."""

    def __init__(self, folder: Path, env: Dict[str, str]) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-m", "webagents", "acp", str(folder)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
        )
        self.lines: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self.stderr = b""
        self.raw_stdout: List[bytes] = []
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()

    def _pump_stdout(self) -> None:
        assert self.process.stdout is not None
        for line in iter(self.process.stdout.readline, b""):
            self.raw_stdout.append(line)
            self.lines.put(line)
        self.lines.put(None)

    def _pump_stderr(self) -> None:
        assert self.process.stderr is not None
        self.stderr = self.process.stderr.read()

    def send_raw(self, text: str) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(text.encode() + b"\n")
        self.process.stdin.flush()

    def send(self, message: Dict[str, Any]) -> None:
        self.send_raw(json.dumps(message))

    def next_message(self) -> Dict[str, Any]:
        try:
            line = self.lines.get(timeout=LINE_TIMEOUT)
        except queue.Empty:
            raise AssertionError(f"no line from the agent within {LINE_TIMEOUT}s; stderr so far: {self.process.stderr}") from None
        if line is None:
            raise AssertionError(f"the agent exited (code {self.process.poll()}); stderr: {self.stderr.decode(errors='replace')}")
        return json.loads(line.decode())

    def close(self) -> int:
        assert self.process.stdin is not None
        self.process.stdin.close()
        code = self.process.wait(timeout=LINE_TIMEOUT)
        self.process.stdout.close()  # type: ignore[union-attr]
        return code


# -- matching --------------------------------------------------------------------------------


def substitute(value: Any, captures: Dict[str, Any]) -> Any:
    """Placeholders in a message to SEND, from what was captured."""
    if isinstance(value, str) and value.startswith("<") and value.endswith(">") and value[1:-1] in captures:
        return captures[value[1:-1]]
    if isinstance(value, str):
        return value.replace("<cwd>", str(captures["cwd"])) if "<cwd>" in value else value
    if isinstance(value, list):
        return [substitute(v, captures) for v in value]
    if isinstance(value, dict):
        return {k: substitute(v, captures) for k, v in value.items()}
    return value


def matches(expected: Any, actual: Any, captures: Dict[str, Any], path: str = "$") -> Optional[str]:
    """None when `actual` has the shape of `expected` (placeholders captured
    and held), else what differs."""
    if isinstance(expected, str) and expected.startswith("<") and expected.endswith(">"):
        name = expected[1:-1]
        if name == "any":
            return None
        if name == "version":
            return None if isinstance(actual, str) and actual else f"{path}: not a version: {actual!r}"
        if name == "capabilities":
            return None if actual == PROTOCOL["agent_capabilities"] else f"{path}: capabilities {actual!r}"
        if name == "auth_methods":
            return None if actual == PROTOCOL["auth_methods"] else f"{path}: authMethods {actual!r}"
        if name == "permission_options":
            return None if actual == PROTOCOL["permission"]["options"] else f"{path}: options {actual!r}"
        if name in captures:
            return None if captures[name] == actual else f"{path}: <{name}> was {captures[name]!r}, now {actual!r}"
        if name == "sessionId" and not (isinstance(actual, str) and actual.startswith(PROTOCOL["session_id_prefix"])):
            return f"{path}: not a session id: {actual!r}"
        if name == "toolCallId" and not (isinstance(actual, str) and actual):
            return f"{path}: not a tool call id: {actual!r}"
        if name == "requestId" and not isinstance(actual, int):
            return f"{path}: not a request id: {actual!r}"
        captures[name] = actual
        return None
    if isinstance(expected, dict) and set(expected) == {"<contains>"}:
        return None if isinstance(actual, str) and expected["<contains>"] in actual else f"{path}: {actual!r} lacks {expected['<contains>']!r}"
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return f"{path}: expected an object, got {actual!r}"
        if set(expected) != set(actual):
            return f"{path}: keys {sorted(actual)} != {sorted(expected)}"
        for key, value in expected.items():
            problem = matches(value, actual[key], captures, f"{path}.{key}")
            if problem:
                return problem
        return None
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return f"{path}: expected {len(expected)} items, got {actual!r}"
        for index, (e, a) in enumerate(zip(expected, actual)):
            problem = matches(e, a, captures, f"{path}[{index}]")
            if problem:
                return problem
        return None
    if isinstance(expected, str) and "<cwd>" in expected:
        expected = expected.replace("<cwd>", str(captures["cwd"]))
    return None if expected == actual else f"{path}: {actual!r} != {expected!r}"


def merge_chunks(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Consecutive `agent_message_chunk` updates as one, whatever the split."""
    merged: List[Dict[str, Any]] = []
    for line in lines:
        update = (line.get("params") or {}).get("update") if line.get("method") == "session/update" else None
        previous = (merged[-1].get("params") or {}).get("update") if merged and merged[-1].get("method") == "session/update" else None
        if update and previous and update.get("sessionUpdate") == "agent_message_chunk" == previous.get("sessionUpdate") \
                and update.get("content", {}).get("type") == "text" == previous.get("content", {}).get("type"):
            merged[-1] = json.loads(json.dumps(merged[-1]))
            merged[-1]["params"]["update"]["content"]["text"] += update["content"]["text"]
            continue
        merged.append(line)
    return merged


def run_steps(agent: AgentProcess, steps: List[Dict[str, Any]], captures: Dict[str, Any], stub: StubModel) -> None:
    for step in steps:
        if "send_raw" in step:
            agent.send_raw(step["send_raw"])
            request_id = None
        else:
            message = substitute(step["send"], captures)
            agent.send(message)
            request_id = message.get("id")
        if "then_send" in step:
            # A cancel must land while the model is being asked, not before the
            # request left, or it cancels nothing and the case races the stub.
            deadline = time.monotonic() + LINE_TIMEOUT
            while step.get("then_send_after_call") and time.monotonic() < deadline:
                with stub.lock:
                    if len(stub.calls) >= step["then_send_after_call"]:
                        break
                time.sleep(0.02)
            agent.send(substitute(step["then_send"], captures))
        expected = list(step["expect"])
        replies = [e for e in expected if "reply" in e]
        received: List[Dict[str, Any]] = []
        while True:
            line = agent.next_message()
            received.append(line)
            if "method" in line and "id" in line:
                # A request from the agent: answer it as the fixture says.
                assert replies, f"unexpected request from the agent: {line}"
                reply_spec = replies.pop(0)
                problem = matches({k: v for k, v in reply_spec.items() if k != "reply"}, line, captures)
                assert problem is None, problem
                agent.send({"jsonrpc": "2.0", "id": line["id"], "result": substitute(reply_spec["reply"], captures)})
                continue
            if "id" in line and ("result" in line or "error" in line) and (request_id is None or line["id"] == request_id):
                break
        received = merge_chunks(received)
        assert len(received) == len(expected), f"got {len(received)} lines, expected {len(expected)}:\n" + "\n".join(json.dumps(l) for l in received)
        for e, a in zip(expected, received):
            problem = matches({k: v for k, v in e.items() if k != "reply"}, a, captures)
            assert problem is None, f"{problem}\nline: {json.dumps(a)}"
        for check in step.get("model_sees", []):
            with stub.lock:
                call = stub.calls[check["call"] - 1]
            text = json.dumps(call.get("messages"))
            for needle in check["messages_contain"]:
                assert needle in text, f"the model's call {check['call']} did not see {needle!r}: {text}"


# -- the cases ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stub():
    model = StubModel()
    yield model
    model.close()


def make_env(home: Path, stub: StubModel) -> Dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("WEBAGENTS_PROFILE", "WEBAGENTS_TOKEN", "WEBAGENTS_DEBUG")}
    env.update(TRANSCRIPTS["env"])
    env.update({"HOME": str(home), "PYTHONUNBUFFERED": "1", TRANSCRIPTS["base_url_env"]: stub.base_url})
    return env


@pytest.mark.parametrize("case", TRANSCRIPTS["cases"], ids=lambda c: c["name"])
def test_transcript(case, tmp_path, stub):
    folder = tmp_path / "project"
    folder.mkdir()
    (folder / "AGENT.md").write_text(TRANSCRIPTS["agent_file"])
    home = tmp_path / "home"
    home.mkdir()
    env = make_env(home, stub)
    stub.reset(case["model"])
    captures: Dict[str, Any] = {"cwd": str(folder), "mcp_command": sys.executable, "mcp_args": [str(ECHO_SERVER)]}

    agent = AgentProcess(folder, env)
    try:
        run_steps(agent, case["steps"], captures, stub)
    finally:
        code = agent.close()
    assert code == 0, agent.stderr.decode(errors="replace")
    assert PROTOCOL["startup"] in agent.stderr.decode(errors="replace")
    for line in agent.raw_stdout:
        message = json.loads(line.decode())
        assert message.get("jsonrpc") == "2.0", f"not a protocol message on stdout: {line!r}"

    if case.get("restart"):
        second = AgentProcess(folder, env)
        try:
            run_steps(second, case["after_restart"], captures, stub)
        finally:
            assert second.close() == 0, second.stderr.decode(errors="replace")

    for name, content in (case.get("files_after") or {}).items():
        assert (folder / name).read_text() == content
    for name in case.get("files_absent") or []:
        assert not (folder / name).exists(), name
