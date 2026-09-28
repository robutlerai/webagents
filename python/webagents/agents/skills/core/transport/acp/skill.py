"""
ACP (Agent Client Protocol) transport: the agent a code editor spawns
(gap-closure plan item 1.6, 2026-09-26). https://agentclientprotocol.com/

`webagents acp [path]` builds the agent `serve` would build and hands it to
`ACPTransportSkill.serve_stdio`: JSON-RPC 2.0 over the process's stdin and
stdout, one message per line, stdout reserved for protocol messages (the CLI
points `sys.stdout` at stderr before the agent file is read, and this skill
writes the wire through the real stdout it is given). The TypeScript twin is
`src/skills/transport/acp/skill.ts`; both are pinned by
`tests/fixtures/acp/acp_protocol.json` and driven end to end, as spawned
processes, by `tests/fixtures/acp/acp_transcripts.json`.

WHAT IT ANSWERS: `initialize` (echoes protocol version 1, advertises only
what is served, and a `terminal` auth method that runs `webagents login`),
`authenticate`, `session/new` (the required `mcpServers` are attached to the
agent through the `mcp` client skill, in the session's `cwd`),
`session/prompt`, `session/cancel`, `$/cancel_request`, `session/load` (the
whole history replayed before the answer) and `session/list`. Sessions are
kept on disk under `sessions_dir` (the CLI passes the profile directory's
`acp/sessions`), so an editor that restarts the agent can load them again.
Requests and notifications are told apart by the PRESENCE of `id` (0 is an
id); anything else is `-32601`, the client's own `fs/*` and `terminal/*`
methods included: an agent calls those on its client, and serving them is
what S-269 was.

THE TURN, on the wire: the agent runs as the local owner (the person whose
editor this is), and its stream becomes `session/update` notifications: text
is `agent_message_chunk`, thinking is `agent_thought_chunk`, a tool call is
`tool_call` (with its `kind` from `protocol.tool_kind`), then
`tool_call_update` as it runs and finishes, and a todo tool's list is the
`plan`. A tool whose kind edits, deletes, moves or executes asks the client
first (`session/request_permission`, after its `tool_call`); a refusal is
what the model is told, through the loop's `tool_skipped` seam, so the turn
goes on. `session/cancel` cancels the prompt's task and the prompt answers
`stopReason: cancelled` after the last update it forwarded. A failed run is
the JSON-RPC error on `session/prompt`.

THE HOOK, not the event stream, asks for permission: the agent runs
`before_toolcall` BEFORE it yields the `tool_call` chunk, so the hook is the
first to see a call. It announces the call itself and the later chunk is
recognised by id, which keeps `tool_call` ahead of `session/request_permission`
as the spec wants.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import hook

from . import protocol as P
from .protocol import AcpError

#: A session id an editor may hand back: never joined into a path otherwise.
_SESSION_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

#: A tool result that starts like this failed (`base_agent._execute_single_tool`).
_ERROR_PREFIXES = ("Tool execution error:", "Error parsing tool arguments:")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class AcpSession:
    """One editor session: where it works, and the conversation so far."""

    session_id: str
    cwd: str
    agent: str
    created_at: str
    updated_at: str
    title: Optional[str] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    mcp_attached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "sessionId": self.session_id,
            "cwd": self.cwd,
            "agent": self.agent,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "messages": [{"role": m.get("role"), "content": m.get("content")} for m in self.messages],
        }
        if self.title:
            record["title"] = self.title
        return record

    @classmethod
    def from_dict(cls, record: Dict[str, Any]) -> "AcpSession":
        messages = [
            {"role": str(m.get("role") or "user"), "content": str(m.get("content") or "")}
            for m in (record.get("messages") or [])
            if isinstance(m, dict)
        ]
        return cls(
            session_id=str(record["sessionId"]),
            cwd=str(record.get("cwd") or ""),
            agent=str(record.get("agent") or ""),
            created_at=str(record.get("createdAt") or _now()),
            updated_at=str(record.get("updatedAt") or _now()),
            title=record.get("title") if isinstance(record.get("title"), str) else None,
            messages=messages,
        )

    def listing(self) -> Dict[str, Any]:
        entry: Dict[str, Any] = {"sessionId": self.session_id, "cwd": self.cwd, "updatedAt": self.updated_at}
        if self.title:
            entry["title"] = self.title
        return entry


@dataclass
class PromptRun:
    """A `session/prompt` in flight."""

    session: AcpSession
    request_id: Any
    task: Optional[asyncio.Task]
    cancelled: bool = False
    #: `$/cancel_request` answers `-32800`; `session/cancel` answers `stopReason: cancelled`.
    cancel_is_error: bool = False
    #: toolCallId -> tool name, for every call already announced.
    announced: Dict[str, str] = field(default_factory=dict)
    answer: str = ""
    error: Optional[str] = None
    #: The turn's finish when the agent's tool budget ended it (`webagents_finish`).
    finish: Optional[Dict[str, Any]] = None


class SessionStore:
    """Sessions as files, one per id, under `directory` (None keeps them in memory only)."""

    def __init__(self, directory: Optional[Path]):
        self.directory = directory

    def path_of(self, session_id: str) -> Optional[Path]:
        if self.directory is None or not _SESSION_ID.match(session_id or ""):
            return None
        return self.directory / f"{session_id}.json"

    def save(self, session: AcpSession) -> None:
        path = self.path_of(session.session_id)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(session.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def load(self, session_id: str) -> Optional[AcpSession]:
        path = self.path_of(session_id)
        if path is None or not path.is_file():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return AcpSession.from_dict(record) if isinstance(record, dict) and record.get("sessionId") == session_id else None

    def list_all(self) -> List[AcpSession]:
        if self.directory is None or not self.directory.is_dir():
            return []
        sessions: List[AcpSession] = []
        for path in self.directory.glob("*.json"):
            loaded = self.load(path.stem)
            if loaded is not None:
                sessions.append(loaded)
        return sessions


class AcpConnection:
    """One JSON-RPC connection: sends messages, and matches the client's
    answers to the requests this agent made (`session/request_permission`)."""

    def __init__(self, write_line: Callable[[str], None]):
        self._write_line = write_line
        self._next_id = 0
        self._pending: Dict[int, asyncio.Future] = {}

    def send(self, message: Dict[str, Any]) -> None:
        self._write_line(json.dumps(message, separators=(",", ":"), ensure_ascii=False))

    def respond(self, request_id: Any, result: Any) -> None:
        self.send({"jsonrpc": "2.0", "id": request_id, "result": result})

    def fail(self, request_id: Any, error: AcpError) -> None:
        self.send({"jsonrpc": "2.0", "id": request_id, "error": error.to_dict()})

    def notify(self, method: str, params: Dict[str, Any]) -> None:
        self.send({"jsonrpc": "2.0", "method": method, "params": params})

    async def request(self, method: str, params: Dict[str, Any]) -> Any:
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        self.send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        try:
            return await future
        finally:
            self._pending.pop(request_id, None)

    def resolve(self, message: Dict[str, Any]) -> None:
        """A response from the client to one of this agent's requests."""
        request_id = message.get("id")
        future = self._pending.get(request_id) if isinstance(request_id, int) else None
        if future is None or future.done():
            return
        if "error" in message and message["error"] is not None:
            error = message["error"] if isinstance(message["error"], dict) else {}
            future.set_exception(AcpError(int(error.get("code", P.INTERNAL_ERROR)), str(error.get("message", "error")), error.get("data")))
        else:
            future.set_result(message.get("result"))

    def close(self) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(AcpError(P.INTERNAL_ERROR, "The client closed the connection."))
        self._pending.clear()


async def stdin_lines(stream: Any) -> AsyncIterator[bytes]:
    """The lines of a blocking byte stream, read on a thread so the event loop
    keeps serving while the editor is quiet."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def pump() -> None:
        try:
            for line in iter(stream.readline, b""):
                loop.call_soon_threadsafe(queue.put_nowait, line)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=pump, name="acp-stdin", daemon=True).start()
    while True:
        line = await queue.get()
        if line is None:
            return
        yield line


class ACPTransportSkill(Skill):
    """The ACP agent over a line-delimited JSON-RPC connection (module docstring).

    Config: `sessions_dir`, where sessions are kept (None: `~/.webagents/acp/sessions`).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config, scope="all")
        config = config or {}
        sessions_dir = config.get("sessions_dir")
        self.settings: Dict[str, Any] = {
            "sessions_dir": str(sessions_dir) if isinstance(sessions_dir, (str, Path)) and str(sessions_dir) else None,
        }
        self._store: Optional[SessionStore] = None
        self._connection: Optional[AcpConnection] = None
        self._sessions: Dict[str, AcpSession] = {}
        self._runs: Dict[str, PromptRun] = {}
        self._client_capabilities: Dict[str, Any] = {}
        self._tasks: Set[asyncio.Task] = set()
        self._mcp_skills: Dict[str, Any] = {}

    async def initialize(self, agent: Any) -> None:
        await super().initialize(agent)

    # ------------------------------------------------------------------
    # Serving
    # ------------------------------------------------------------------

    @property
    def store(self) -> SessionStore:
        if self._store is None:
            configured = self.settings.get("sessions_dir")
            self._store = SessionStore(Path(configured) if configured else Path.home() / ".webagents" / "acp" / "sessions")
        return self._store

    async def serve_stdio(self, agent: Any, stdin: Any = None, stdout: Any = None) -> None:
        """Serve `agent` over `stdin` (bytes) and `stdout` (text, the REAL
        stdout `reserve_stdout()` returned) until the client closes stdin."""
        stdin = stdin if stdin is not None else sys.stdin.buffer
        stdout = stdout if stdout is not None else sys.stdout

        def write_line(text: str) -> None:
            stdout.write(text + "\n")
            stdout.flush()

        await self.serve(agent, stdin_lines(stdin), write_line)

    async def serve(self, agent: Any, lines: AsyncIterator[bytes], write_line: Callable[[str], None]) -> None:
        """Serve `agent` over any line source and sink (the stdio pair, or a test's)."""
        if not any(skill is self for skill in (agent.skills or {}).values()):
            agent.add_skill("acp", self)
        await agent._ensure_skills_initialized()
        self.agent = agent
        self._connection = AcpConnection(write_line)
        print(f"[webagents] {agent.name}: ACP over stdio", file=sys.stderr, flush=True)
        try:
            async for line in lines:
                self._on_line(line)
        finally:
            await self._shutdown()

    def _spawn(self, coroutine: Any) -> None:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _on_line(self, line: bytes) -> None:
        assert self._connection is not None
        text = line.decode("utf-8", "replace").strip()
        if not text:
            return
        try:
            message = json.loads(text)
        except ValueError:
            self._connection.fail(None, AcpError(P.PARSE_ERROR, "Parse error"))
            return
        if not isinstance(message, dict):
            self._connection.fail(None, AcpError(P.INVALID_REQUEST, "Invalid request"))
            return
        method = message.get("method")
        has_id = "id" in message
        params = message.get("params")
        params = params if isinstance(params, dict) else {}
        if isinstance(method, str):
            if has_id:
                self._spawn(self._answer(message["id"], method, params))
            else:
                self._spawn(self._notification(method, params))
        elif has_id and ("result" in message or "error" in message):
            self._connection.resolve(message)
        else:
            self._connection.fail(message.get("id") if has_id else None, AcpError(P.INVALID_REQUEST, "Invalid request"))

    async def _answer(self, request_id: Any, method: str, params: Dict[str, Any]) -> None:
        assert self._connection is not None
        try:
            result = await self._dispatch(request_id, method, params)
        except AcpError as error:
            self._connection.fail(request_id, error)
        except asyncio.CancelledError:
            self._connection.fail(request_id, AcpError(P.REQUEST_CANCELLED, "Request cancelled"))
        except Exception as error:  # noqa: BLE001 - every failure is an answer on the wire
            self._connection.fail(request_id, AcpError(P.INTERNAL_ERROR, str(error) or type(error).__name__))
        else:
            self._connection.respond(request_id, result)

    async def _dispatch(self, request_id: Any, method: str, params: Dict[str, Any]) -> Any:
        if method == "initialize":
            return self._initialize(params)
        if method == "authenticate":
            return self._authenticate(params)
        if method == "session/new":
            return await self._session_new(params)
        if method == "session/prompt":
            return await self._session_prompt(request_id, params)
        if method == "session/load":
            return await self._session_load(params)
        if method == "session/list":
            return self._session_list(params)
        raise AcpError(P.METHOD_NOT_FOUND, f"Method not found: {method}")

    async def _notification(self, method: str, params: Dict[str, Any]) -> None:
        if method == "session/cancel":
            self._cancel(params.get("sessionId"), is_error=False)
        elif method == "$/cancel_request":
            request_id = params.get("requestId")
            for session_id, run in list(self._runs.items()):
                if run.request_id == request_id:
                    self._cancel(session_id, is_error=True)

    async def _shutdown(self) -> None:
        for session_id in list(self._runs):
            self._cancel(session_id, is_error=False)
        for task in list(self._tasks):
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        for skill in self._mcp_skills.values():
            try:
                await skill.cleanup()
            except Exception:  # noqa: BLE001 - shutting down
                pass
        self._mcp_skills.clear()
        if self._connection is not None:
            self._connection.close()

    # ------------------------------------------------------------------
    # Agent methods
    # ------------------------------------------------------------------

    def _initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from webagents import __version__

        capabilities = params.get("clientCapabilities")
        self._client_capabilities = capabilities if isinstance(capabilities, dict) else {}
        return {
            "protocolVersion": P.PROTOCOL_VERSION,
            "agentCapabilities": P.AGENT_CAPABILITIES,
            "agentInfo": {"name": P.AGENT_INFO_NAME, "title": getattr(self.agent, "name", "") or "", "version": __version__},
            "authMethods": P.AUTH_METHODS,
        }

    def _authenticate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        method_id = params.get("methodId")
        if method_id not in {m["id"] for m in P.AUTH_METHODS}:
            raise AcpError(P.INVALID_PARAMS, f"Unknown auth method: {method_id!r}")
        return {}

    async def _session_new(self, params: Dict[str, Any]) -> Dict[str, Any]:
        cwd = self._cwd_of(params)
        servers = params.get("mcpServers")
        if not isinstance(servers, list):
            raise AcpError(P.INVALID_PARAMS, "mcpServers is required: an array, possibly empty")
        now = _now()
        session = AcpSession(
            session_id=f"{P.SESSION_ID_PREFIX}{uuid.uuid4().hex[:12]}",
            cwd=cwd,
            agent=getattr(self.agent, "name", "") or "",
            created_at=now,
            updated_at=now,
        )
        self._sessions[session.session_id] = session
        await self._attach_mcp(session, servers)
        self.store.save(session)
        return {"sessionId": session.session_id}

    async def _session_load(self, params: Dict[str, Any]) -> Dict[str, Any]:
        assert self._connection is not None
        session_id = params.get("sessionId")
        if not isinstance(session_id, str) or not _SESSION_ID.match(session_id):
            raise AcpError(P.INVALID_PARAMS, "sessionId must be a string")
        cwd = self._cwd_of(params)
        servers = params.get("mcpServers")
        if not isinstance(servers, list):
            raise AcpError(P.INVALID_PARAMS, "mcpServers is required: an array, possibly empty")
        session = self._sessions.get(session_id) or self.store.load(session_id)
        if session is None:
            raise AcpError(P.RESOURCE_NOT_FOUND, f"Session not found: {session_id}")
        session.cwd = cwd
        self._sessions[session_id] = session
        await self._attach_mcp(session, servers)
        for message in session.messages:
            text = message.get("content")
            if not isinstance(text, str) or not text:
                continue
            kind = "agent_message_chunk" if message.get("role") == "assistant" else "user_message_chunk"
            self._update(session, {"sessionUpdate": kind, "content": {"type": "text", "text": text}})
        return {}

    def _session_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        cwd = params.get("cwd") if isinstance(params.get("cwd"), str) else None
        by_id: Dict[str, AcpSession] = {s.session_id: s for s in self.store.list_all()}
        by_id.update(self._sessions)
        sessions = [s for s in by_id.values() if cwd is None or s.cwd == cwd]
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return {"sessions": [s.listing() for s in sessions]}

    async def _session_prompt(self, request_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        session_id = params.get("sessionId")
        session = self._sessions.get(session_id) if isinstance(session_id, str) else None
        if session is None:
            raise AcpError(P.RESOURCE_NOT_FOUND, f"Session not found: {session_id}")
        if session.session_id in self._runs:
            raise AcpError(P.INTERNAL_ERROR, "A prompt is already running for this session.")
        text = P.prompt_text(params.get("prompt"))
        run = PromptRun(session=session, request_id=request_id, task=asyncio.current_task())
        self._runs[session.session_id] = run
        session.messages.append({"role": "user", "content": text})
        if not session.title:
            session.title = text.strip().splitlines()[0][:60] if text.strip() else None
        try:
            try:
                await self._run_prompt(run)
            except asyncio.CancelledError:
                run.cancelled = True
            if run.cancelled:
                if run.cancel_is_error:
                    raise AcpError(P.REQUEST_CANCELLED, "Request cancelled")
                return {"stopReason": P.STOP_CANCELLED}
            if run.error is not None:
                raise AcpError(P.INTERNAL_ERROR, run.error)
            if run.finish is not None:
                # The agent's tool budget ended the turn (2026-09-28): ACP's
                # own reason, and the precise one beside it.
                return {"stopReason": P.STOP_MAX_TURN_REQUESTS, "_meta": {"webagents_finish": run.finish}}
            return {"stopReason": P.STOP_END_TURN}
        finally:
            self._runs.pop(session.session_id, None)
            if run.answer:
                session.messages.append({"role": "assistant", "content": run.answer})
            session.updated_at = _now()
            self.store.save(session)

    # ------------------------------------------------------------------
    # The turn
    # ------------------------------------------------------------------

    async def _run_prompt(self, run: PromptRun) -> None:
        from webagents.access import run_as_local_owner

        context = run_as_local_owner(self.agent)
        context.set("acp_session", run.session.session_id)
        messages = [{"role": m["role"], "content": m["content"]} for m in run.session.messages]
        thinking = False
        async for chunk in self.agent.run_streaming(messages):
            if run.cancelled:
                break
            if not isinstance(chunk, dict):
                continue
            finish = chunk.get("webagents_finish")
            if isinstance(finish, dict) and finish.get("reason") in ("tool_round_limit", "tool_loop"):
                run.finish = dict(finish)
            for event in _chunk_events(chunk):
                kind = event[0]
                if kind == "text":
                    if thinking:
                        self._update(run.session, {"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": event[1]}})
                    else:
                        run.answer += event[1]
                        self._update(run.session, {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": event[1]}})
                elif kind == "thinking_start":
                    thinking = True
                elif kind == "thinking_end":
                    thinking = False
                elif kind == "tool_call":
                    self._announce(run, event[1], event[2], P.parse_arguments(event[3]))
                elif kind == "tool_result":
                    self._finish_tool(run, event[1], event[2], event[3])
                elif kind == "error":
                    run.error = event[1]

    def _announce(self, run: PromptRun, call_id: str, name: str, raw_input: Dict[str, Any]) -> None:
        if call_id in run.announced:
            return
        run.announced[call_id] = name
        self._update(
            run.session,
            {
                "sessionUpdate": "tool_call",
                "toolCallId": call_id,
                "title": name,
                "kind": P.tool_kind(name),
                "status": "pending",
                "rawInput": raw_input,
            },
        )

    def _finish_tool(self, run: PromptRun, call_id: str, status: str, text: str) -> None:
        failed = status != "success" or text.startswith(_ERROR_PREFIXES)
        self._update(
            run.session,
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": call_id,
                "status": "failed" if failed else "completed",
                "content": [P.text_content(text)],
                "rawOutput": text,
            },
        )
        name = run.announced.get(call_id, "")
        if name.startswith("todo"):
            entries = P.plan_entries(self._todo_items())
            if entries is not None:
                self._update(run.session, {"sessionUpdate": "plan", "entries": entries})

    def _todo_items(self) -> Any:
        for skill in (getattr(self.agent, "skills", None) or {}).values():
            if type(skill).__name__ == "TodoSkill":
                return getattr(skill, "items", None)
        return None

    @hook("before_toolcall", priority=1)
    async def acp_before_toolcall(self, context: Any) -> Any:
        """Announce the call, ask the client when its kind needs permission,
        and tell the loop to skip a refused tool (module docstring)."""
        session_id = context.get("acp_session")
        run = self._runs.get(session_id) if session_id else None
        if run is None or self._connection is None:
            return context
        tool_call = context.get("tool_call") or {}
        function = tool_call.get("function") if isinstance(tool_call, dict) else None
        function = function if isinstance(function, dict) else {}
        name = str(function.get("name") or "")
        call_id = str(tool_call.get("id") or "") if isinstance(tool_call, dict) else ""
        call_id = call_id or f"call_{uuid.uuid4().hex[:8]}"
        raw_input = P.parse_arguments(function.get("arguments"))
        kind = P.tool_kind(name)
        self._announce(run, call_id, name, raw_input)
        if P.needs_permission(kind):
            outcome = await self._connection.request(
                "session/request_permission",
                {
                    "sessionId": run.session.session_id,
                    "toolCall": {"toolCallId": call_id, "title": name, "kind": kind, "status": "pending", "rawInput": raw_input},
                    "options": P.PERMISSION_OPTIONS,
                },
            )
            decision = _decision(outcome)
            if decision != "allow":
                context.set("tool_skipped", True)
                context.set("tool_result", (P.CANCELLED if decision == "cancelled" else P.REJECTED).format(tool=name))
                return context
        self._update(run.session, {"sessionUpdate": "tool_call_update", "toolCallId": call_id, "status": "in_progress"})
        return context

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update(self, session: AcpSession, update: Dict[str, Any]) -> None:
        if self._connection is not None:
            self._connection.notify("session/update", {"sessionId": session.session_id, "update": update})

    def _cancel(self, session_id: Any, *, is_error: bool) -> None:
        run = self._runs.get(session_id) if isinstance(session_id, str) else None
        if run is None or run.cancelled:
            return
        run.cancelled = True
        run.cancel_is_error = is_error
        if run.task is not None and not run.task.done():
            run.task.cancel()

    @staticmethod
    def _cwd_of(params: Dict[str, Any]) -> str:
        cwd = params.get("cwd")
        if not isinstance(cwd, str) or not os.path.isabs(cwd):
            raise AcpError(P.INVALID_PARAMS, "cwd must be an absolute path")
        return cwd

    async def _attach_mcp(self, session: AcpSession, entries: List[Any]) -> None:
        """The session's MCP servers, as an `mcp` client skill on the agent,
        started in the session's `cwd`. A server that fails is said on stderr
        and the session still opens: an editor's optional server must not
        make the agent unusable."""
        servers = P.mcp_servers_config(entries)
        if not servers or session.mcp_attached:
            return
        for server in servers.values():
            if "command" in server:
                server.setdefault("cwd", session.cwd)
        key = f"acp-mcp-{session.session_id}"
        try:
            from webagents.agents.skills.local.mcp.skill import LocalMcpSkill

            skill = LocalMcpSkill({"mcp": servers, "agent_name": getattr(self.agent, "name", ""), "agent_path": session.cwd})
            self.agent.add_skill(key, skill)
            await skill.initialize(self.agent)
        except Exception as error:  # noqa: BLE001 - said, and the session still opens
            print(f"[webagents] ACP session {session.session_id}: MCP servers not attached: {error}", file=sys.stderr, flush=True)
            return
        self._mcp_skills[key] = skill
        session.mcp_attached = True


def _decision(outcome: Any) -> str:
    """`allow`, `reject` or `cancelled` from a `session/request_permission` result."""
    selected = outcome.get("outcome") if isinstance(outcome, dict) else None
    if not isinstance(selected, dict):
        return "reject"
    if selected.get("outcome") == "cancelled":
        return "cancelled"
    option_id = selected.get("optionId")
    for option in P.PERMISSION_OPTIONS:
        if option["optionId"] == option_id:
            return "allow" if option["kind"].startswith("allow") else "reject"
    return "reject"


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _chunk_events(chunk: Dict[str, Any]) -> List[tuple]:
    """What one streamed chunk carries (the shapes `run_streaming` yields and
    the chat reads in `cli/repl/render.py`): text, thinking markers, a tool
    call, a tool result, or the run's error."""
    events: List[tuple] = []
    error = chunk.get("error")
    if error and not chunk.get("choices"):
        message = error.get("message") if isinstance(error, dict) else error
        return [("error", str(message or error))]
    kind = chunk.get("type")
    if kind == "tool_call":
        return [("tool_call", str(chunk.get("call_id") or chunk.get("id") or ""), str(chunk.get("name") or ""), chunk.get("arguments"))]
    if kind == "tool_result":
        return [("tool_result", str(chunk.get("id") or chunk.get("call_id") or ""), str(chunk.get("status") or "success"), _as_text(chunk.get("result")))]
    if chunk.get("object") == "metadata":
        payload = chunk.get("payload") or {}
        if kind == "tool_start":
            return [("tool_call", str(payload.get("id") or ""), str(payload.get("name") or ""), payload.get("arguments"))]
        if kind == "tool_result":
            return [("tool_result", str(payload.get("id") or ""), str(payload.get("status") or "success"), _as_text(payload.get("result")))]
        if kind == "thought_start":
            return [("thinking_start",)]
        if kind == "thought_end":
            return [("thinking_end",)]
        return events
    choices = chunk.get("choices") or []
    if choices:
        delta = (choices[0] or {}).get("delta") or {}
        content = delta.get("content")
        if content:
            events.append(("text", str(content)))
    return events
