"""
The schedule runner and the three deliverers (plan item 1.7, 2026-09-26): a
one-minute schedule runs a turn of the agent as its owner and delivers to a
file and to a local webhook server whose signature verifies, a restart neither
double-runs nor drops a missed slot, one run per schedule at a time, the
heartbeat sentinel suppresses delivery, and platform-chat delivery goes through
a mocked platform client. The words and vectors are the shared fixture's
(`tests/fixtures/daemon/cron.json`); the TypeScript suite runs the same in
`tests/unit/daemon/cron-runner-w1daemon.test.ts`.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from webagents.cli.daemon.deliver import (
    DeliveryContext,
    RunResult,
    chat_session_id,
    chat_turn,
    deliver,
    deliver_file,
    deliver_webhook,
    webhook_body,
)
from webagents.cli.daemon.schedule_runner import (
    HEARTBEAT_PROMPT,
    HEARTBEAT_SENTINEL,
    ScheduleRunner,
    is_quiet_heartbeat,
    iso_utc,
    next_cron_run,
    parse_iso,
    state_path,
)
from webagents.cli.loader.schedules import DeliverTarget, parse_cron_block
from webagents.crypto.http_signature import SigningKey
from webagents.crypto.identity import AgentSigningIdentity
from webagents.crypto.web_bot_auth_verify import DiscoveredKey, InboundRequest, KeySetOutcome, MemoryNonceStore, verify_web_bot_auth
from webagents.server.context.context_vars import get_context

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "daemon" / "cron.json").read_text())
T0 = parse_iso("2026-09-26T10:00:30Z")
MINUTE = 60_000
ISSUER = "https://agent.example/agents/reporter"


class _Keys:
    def __init__(self, key: SigningKey):
        self.key = key

    def held_ed25519_keys(self) -> List[SigningKey]:
        return [self.key]


def signing_identity() -> tuple[AgentSigningIdentity, SigningKey]:
    key = SigningKey.from_private_key(ed25519.Ed25519PrivateKey.generate())
    return AgentSigningIdentity(ISSUER, _Keys(key)), key


class FakeAgent:
    """A served agent for the runner: answers `reply`, records what it was asked and as whom."""

    def __init__(self, reply: Any = "Nothing happened.", identity: Any = None):
        self.name = "reporter"
        self.reply = reply
        self.signing_identity = identity
        self.calls: List[Dict[str, Any]] = []

    async def run(self, messages, **kwargs):
        context = get_context()
        auth = getattr(context, "auth", None)
        self.calls.append({"messages": messages, "scope": getattr(auth, "scope", None), "provider": getattr(auth, "provider", None)})
        reply = self.reply
        if callable(reply):
            reply = await reply()
        if isinstance(reply, BaseException):
            raise reply
        return {"choices": [{"message": {"role": "assistant", "content": reply}}]}


@dataclass
class Received:
    method: str
    path: str
    headers: Dict[str, str]
    body: bytes


@dataclass
class Hook:
    url: str
    authority: str
    received: List[Received] = field(default_factory=list)
    server: Any = None


@pytest.fixture
def webhook_server():
    """A local webhook: answers the queued statuses in order (200 once the queue is empty), keeps what it got."""
    servers: List[ThreadingHTTPServer] = []

    def start(statuses: Optional[List[int]] = None) -> Hook:
        queue = list(statuses or [])
        hook = Hook(url="", authority="")

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - http.server's name
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                hook.received.append(Received(self.command, self.path, {k.lower(): v for k, v in self.headers.items()}, body))
                self.send_response(queue.pop(0) if queue else 200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):  # quiet
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        hook.authority = f"127.0.0.1:{server.server_address[1]}"
        hook.url = f"http://{hook.authority}/hook"
        hook.server = server
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return hook

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


class NoSleep:
    def __init__(self):
        self.slept: List[float] = []

    async def __call__(self, seconds: float) -> None:
        self.slept.append(seconds)


def result(**overrides) -> RunResult:
    base = dict(agent="reporter", schedule="daily-report", kind="cron", prompt="Summarize yesterday's activity.", content="Nothing happened.", ran_at="2026-09-28T07:00:00Z")
    base.update(overrides)
    return RunResult(**base)


def agent_folder(tmp_path: Path, name: str = "agent") -> Path:
    folder = tmp_path / name
    folder.mkdir()
    (folder / "AGENT.md").write_text("---\nname: reporter\n---\nYou report.\n")
    return folder


def run(coro):
    return asyncio.run(coro)


# -- the words shared with TypeScript -------------------------------------------------------


def test_the_heartbeat_prompt_and_sentinel_are_the_fixture_s():
    assert HEARTBEAT_PROMPT == FIXTURE["heartbeat"]["prompt"]
    assert HEARTBEAT_SENTINEL == FIXTURE["heartbeat"]["sentinel"]


def test_reads_a_quiet_heartbeat_however_the_model_dresses_the_sentinel():
    for text in FIXTURE["heartbeat"]["quiet"]:
        assert is_quiet_heartbeat(text), repr(text)
    for text in FIXTURE["heartbeat"]["reports"]:
        assert not is_quiet_heartbeat(text), repr(text)


@pytest.mark.parametrize("expression, tz, after, expected", FIXTURE["next_run"])
def test_next_run_vectors(expression, tz, after, expected):
    assert iso_utc(next_cron_run(expression, tz, parse_iso(after))) == expected


def test_the_webhook_body_and_the_chat_turn_are_the_fixture_s():
    assert webhook_body(result()) == FIXTURE["webhook"]["body"]
    heartbeat = result(schedule="watch", kind="every", prompt=None, content="The queue is stuck.")
    assert webhook_body(heartbeat) == FIXTURE["webhook"]["heartbeat_body"]
    for key, uuid in FIXTURE["chat"]["sessions"].items():
        agent, schedule = key.split("/")
        assert chat_session_id(agent, schedule) == uuid
    assert chat_turn(result()) == FIXTURE["chat"]["turn"]
    assert chat_turn(heartbeat) == FIXTURE["chat"]["heartbeat_turn"]


# -- a one-minute schedule, run by the runner ------------------------------------------------


def test_runs_a_turn_as_the_owner_and_delivers_to_a_file_and_a_signed_webhook(tmp_path, webhook_server):
    folder = agent_folder(tmp_path)
    hook = webhook_server()
    identity, key = signing_identity()
    agent = FakeAgent("Nothing happened.", identity)
    clock = {"now": T0}
    lines: List[str] = []

    async def agent_for(name):
        return agent

    runner = ScheduleRunner(agent_for, clock=lambda: clock["now"], log=lines.append)
    schedules = parse_cron_block(
        [
            {"name": "minute-file", "schedule": "* * * * *", "prompt": "Report.", "deliver": {"file": "reports/minute.md"}},
            {"name": "minute-hook", "schedule": "* * * * *", "prompt": "Report.", "deliver": {"webhook": hook.url}},
        ]
    )
    runner.set_schedules("reporter", folder, schedules)

    # Scheduled for the next whole minute, persisted before anything ran.
    state = json.loads(state_path(folder, "reporter").read_text())
    assert list(state["schedules"]) == ["minute-file", "minute-hook"]
    assert state["schedules"]["minute-file"] == {"spec": "cron * * * * * UTC", "nextRun": "2026-09-26T10:01:00Z", "lastFire": None, "lastRun": None}
    assert list(state["schedules"]["minute-file"]) == list(FIXTURE["state_example"]["schedules"]["daily-report"])

    # Not due yet: nothing runs.
    run(runner.tick())
    assert agent.calls == []

    # Due: both run, once.
    clock["now"] = T0 + MINUTE
    run(runner.tick())
    assert len(agent.calls) == 2
    # The owner's own turn (the fixture's `turn.caller`), with the file's prompt.
    assert (agent.calls[0]["scope"], agent.calls[0]["provider"]) == (FIXTURE["turn"]["caller"]["scope"], FIXTURE["turn"]["caller"]["provider"])
    assert agent.calls[0]["messages"] == [{"role": "user", "content": "Report."}]

    # The file, with the fixture's entry.
    entry = FIXTURE["file_entry"].format(schedule="minute-file", ran_at="2026-09-26T10:01:30Z", content="Nothing happened.")
    assert (folder / "reports" / "minute.md").read_text() == entry

    # The webhook: the fixture's body, signed as the agent, and the signature verifies.
    assert len(hook.received) == 1
    got = hook.received[0]
    assert got.method == "POST"
    assert got.headers["content-type"] == FIXTURE["webhook"]["content_type"]
    assert json.loads(got.body) == {
        "agent": "reporter",
        "schedule": "minute-hook",
        "kind": "cron",
        "prompt": "Report.",
        "content": "Nothing happened.",
        "ran_at": "2026-09-26T10:01:30Z",
    }

    class KeySets:
        async def get(self, discovery):
            return KeySetOutcome(keys=(DiscoveredKey(thumbprint=key.thumbprint, x=key.public_jwk()["x"]),), ttl_s=300)

    outcome = run(
        verify_web_bot_auth(
            InboundRequest(method=got.method, target="/hook", headers=got.headers, body=got.body),
            authorities=[hook.authority],
            scheme="http",
            key_sets=KeySets(),
            nonces=MemoryNonceStore(),
        )
    )
    assert outcome.ok, outcome.refusal
    assert outcome.agent.principal == ISSUER

    # The records, in the state and in the log.
    listed = runner.list_schedules()
    assert [(s["name"], s["last_run"]["outcome"], s["last_run"]["detail"]) for s in listed] == [
        ("minute-file", "delivered", "file reports/minute.md"),
        ("minute-hook", "delivered", f"webhook {hook.url} (signed as {ISSUER})"),
    ]
    assert listed[0]["next_run"] == "2026-09-26T10:02:00Z"
    assert listed[0]["last_fire"] == "2026-09-26T10:01:30Z"
    assert "reporter/minute-file: delivered (file reports/minute.md)" in lines

    # The same minute again: nothing (the next fire is the next minute).
    run(runner.tick())
    assert len(agent.calls) == 2


def test_a_restart_does_not_double_run_and_one_missed_slot_catches_up_once(tmp_path):
    folder = agent_folder(tmp_path)
    agent = FakeAgent()
    schedules = parse_cron_block([{"name": "minute", "schedule": "* * * * *", "prompt": "Report.", "deliver": {"file": "out.md"}}])
    clock = {"now": T0}

    async def agent_for(name):
        return agent

    first = ScheduleRunner(agent_for, clock=lambda: clock["now"])
    first.set_schedules("reporter", folder, schedules)
    clock["now"] = T0 + MINUTE
    run(first.tick())
    assert len(agent.calls) == 1

    # Restarted ten seconds later: the slot already ran, nothing runs again.
    clock["now"] = T0 + MINUTE + 10_000
    second = ScheduleRunner(agent_for, clock=lambda: clock["now"])
    second.set_schedules("reporter", folder, schedules)
    assert second.list_schedules()[0]["next_run"] == "2026-09-26T10:02:00Z"
    assert second.list_schedules()[0]["last_run"] == {"at": "2026-09-26T10:01:30Z", "outcome": "delivered", "detail": "file out.md"}
    run(second.tick())
    assert len(agent.calls) == 1

    # Back after sleeping through five slots: one catch-up run, then the schedule resumes from now.
    clock["now"] = T0 + 6 * MINUTE + 10_000
    third = ScheduleRunner(agent_for, clock=lambda: clock["now"])
    third.set_schedules("reporter", folder, schedules)
    run(third.tick())
    assert len(agent.calls) == 2
    assert third.list_schedules()[0]["next_run"] == "2026-09-26T10:07:00Z"
    run(third.tick())
    assert len(agent.calls) == 2

    # A changed timing is rescheduled from now, not from the saved fire.
    changed = parse_cron_block([{"name": "minute", "schedule": "*/5 * * * *", "prompt": "Report.", "deliver": {"file": "out.md"}}])
    fourth = ScheduleRunner(agent_for, clock=lambda: clock["now"])
    fourth.set_schedules("reporter", folder, changed)
    assert fourth.list_schedules()[0]["next_run"] == "2026-09-26T10:10:00Z"
    # A disabled schedule keeps its record and has no next fire.
    off = parse_cron_block([{"name": "minute", "schedule": "* * * * *", "prompt": "Report.", "enabled": False, "deliver": {"file": "out.md"}}])
    fourth.set_schedules("reporter", folder, off)
    assert fourth.list_schedules()[0]["next_run"] is None
    run(fourth.tick())
    assert len(agent.calls) == 2


def test_runs_one_schedule_at_a_time_and_says_when_a_slot_is_skipped(tmp_path):
    folder = agent_folder(tmp_path)
    clock = {"now": T0}
    lines: List[str] = []

    async def scenario():
        release = asyncio.Event()

        async def blocked():
            await release.wait()
            return "Done."

        agent = FakeAgent(blocked)

        async def agent_for(name):
            return agent

        runner = ScheduleRunner(agent_for, clock=lambda: clock["now"], log=lines.append)
        runner.set_schedules("reporter", folder, parse_cron_block([{"name": "minute", "schedule": "* * * * *", "prompt": "Report.", "deliver": {"file": "out.md"}}]))
        clock["now"] = T0 + MINUTE
        ticking = asyncio.create_task(runner.tick())
        await asyncio.sleep(0.02)
        assert len(agent.calls) == 1
        assert runner.list_schedules()[0]["running"] is True
        # `webagents cron run` while it runs: refused with the fixture's words.
        assert await runner.run_now("reporter", "minute") == {"at": "2026-09-26T10:01:30Z", "outcome": "failed", "detail": FIXTURE["details"]["already_running"]}
        # The next slot comes due while it still runs: skipped, and said.
        clock["now"] = T0 + 2 * MINUTE
        await runner.tick()
        assert len(agent.calls) == 1
        release.set()
        await ticking
        assert runner.list_schedules()[0]["running"] is False
        assert runner.list_schedules()[0]["last_run"]["outcome"] == "delivered"

    run(scenario())
    assert "reporter/minute: due, but the previous run is still going" in lines


def test_records_a_failed_turn_and_an_unserved_agent_and_runs_now_on_request(tmp_path):
    folder = agent_folder(tmp_path)
    served: Dict[str, Any] = {"agent": FakeAgent(RuntimeError("no model key"))}

    async def agent_for(name):
        return served["agent"]

    runner = ScheduleRunner(agent_for, clock=lambda: T0)
    runner.set_schedules("reporter", folder, parse_cron_block([{"name": "daily", "schedule": "0 9 * * *", "prompt": "Report.", "deliver": {"file": "out.md"}}]))
    assert run(runner.run_now("reporter", "daily")) == {"at": "2026-09-26T10:00:30Z", "outcome": "failed", "detail": "turn failed: no model key"}
    served["agent"] = None
    assert run(runner.run_now("reporter", "daily")) == {"at": "2026-09-26T10:00:30Z", "outcome": "failed", "detail": FIXTURE["details"]["agent_missing"]}
    # Running now leaves the schedule's next fire alone.
    assert runner.list_schedules()[0]["next_run"] == "2026-09-27T09:00:00Z"
    with pytest.raises(KeyError):
        run(runner.run_now("reporter", "nope"))
    assert not (folder / "out.md").exists()


# -- the heartbeat -----------------------------------------------------------------------------


class MockChat:
    def __init__(self, where: Any = None, fail: Optional[Exception] = None):
        self.where = where or type("T", (), {"base": "https://robutler.test", "token": "t", "agent_id": "agent-1"})()
        self.fail = fail
        self.recorded: List[Any] = []
        self.asked: List[Any] = []

    async def target(self, agent_dir, agent_name):
        self.asked.append((Path(agent_dir), agent_name))
        return self.where

    async def record(self, target, turn, command):
        if self.fail is not None:
            raise self.fail
        self.recorded.append((target, turn))
        return "chat-1"


def test_the_heartbeat_runs_the_standing_instructions_and_delivers_only_a_report(tmp_path):
    folder = agent_folder(tmp_path)
    agent = FakeAgent(HEARTBEAT_SENTINEL)
    chat = MockChat()

    async def agent_for(name):
        return agent

    async def deliver_with_mock(target, result_, ctx):
        return await deliver(target, result_, DeliveryContext(agent_dir=ctx.agent_dir, agent=ctx.agent, chat=chat))

    runner = ScheduleRunner(agent_for, clock=lambda: T0, deliver_fn=deliver_with_mock)
    runner.set_schedules("reporter", folder, parse_cron_block([{"name": "watch", "every": "1h", "heartbeat": True, "deliver": {"chat": "owner"}}]))

    # Nothing to report: the sentinel, dressed or bare, and an empty reply.
    for quiet in FIXTURE["heartbeat"]["quiet"]:
        agent.reply = quiet
        assert run(runner.run_now("reporter", "watch")) == {"at": "2026-09-26T10:00:30Z", "outcome": "nothing", "detail": FIXTURE["details"]["quiet"]}
    assert chat.recorded == []
    assert agent.calls[0]["messages"] == [{"role": "user", "content": HEARTBEAT_PROMPT}]

    # A report: delivered, the report alone, into the schedule's own chat.
    agent.reply = "The queue is stuck."
    assert run(runner.run_now("reporter", "watch")) == {"at": "2026-09-26T10:00:30Z", "outcome": "delivered", "detail": "chat owner (chat-1)"}
    assert [turn for _, turn in chat.recorded] == [FIXTURE["chat"]["heartbeat_turn"]]


def test_an_empty_reply_to_a_prompt_schedule_is_nothing_too(tmp_path):
    folder = agent_folder(tmp_path)
    agent = FakeAgent("   ")

    async def agent_for(name):
        return agent

    runner = ScheduleRunner(agent_for, clock=lambda: T0)
    runner.set_schedules("reporter", folder, parse_cron_block([{"name": "daily", "schedule": "0 9 * * *", "prompt": "Report.", "deliver": {"file": "out.md"}}]))
    assert run(runner.run_now("reporter", "daily")) == {"at": "2026-09-26T10:00:30Z", "outcome": "nothing", "detail": FIXTURE["details"]["empty"]}
    assert not (folder / "out.md").exists()


# -- file delivery -----------------------------------------------------------------------------


def test_file_delivery_refuses_a_path_that_leaves_the_folder_through_a_link(tmp_path):
    folder = agent_folder(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, folder / "reports")
    ctx = DeliveryContext(agent_dir=folder)
    assert run(deliver_file(DeliverTarget(kind="file", path="reports/daily.md"), result(), ctx)) == (
        "failed",
        FIXTURE["details"]["file_outside"].format(path="reports/daily.md"),
    )
    assert not (outside / "daily.md").exists()
    # A link to a file outside, too.
    os.symlink(outside / "log.md", folder / "log.md")
    assert run(deliver_file(DeliverTarget(kind="file", path="log.md"), result(), ctx)) == ("failed", "file log.md: outside the agent's folder")
    # Inside: appended, twice is two entries.
    (folder / "inside").mkdir()
    assert run(deliver_file(DeliverTarget(kind="file", path="inside/daily.md"), result(), ctx)) == ("delivered", "file inside/daily.md")
    run(deliver_file(DeliverTarget(kind="file", path="inside/daily.md"), result(content="Again."), ctx))
    assert (folder / "inside" / "daily.md").read_text() == (
        "## daily-report, 2026-09-28T07:00:00Z\n\nNothing happened.\n\n## daily-report, 2026-09-28T07:00:00Z\n\nAgain.\n\n"
    )


# -- webhook delivery ---------------------------------------------------------------------------


def test_webhook_posts_unsigned_and_says_so_when_the_agent_holds_no_key(tmp_path, webhook_server):
    hook = webhook_server()
    ctx = DeliveryContext(agent_dir=agent_folder(tmp_path), agent=FakeAgent(), sleep=NoSleep())
    assert run(deliver_webhook(DeliverTarget(kind="webhook", url=hook.url, timeout=5, retries=0), result(), ctx)) == (
        "delivered",
        f"webhook {hook.url} (unsigned: {FIXTURE['details']['no_identity']})",
    )
    assert "signature" not in hook.received[0].headers
    assert hook.received[0].body.decode() == FIXTURE["webhook"]["body"]


def test_webhook_retries_a_5xx_with_the_fixture_backoff_signing_each_try_afresh(tmp_path, webhook_server):
    hook = webhook_server([500, 503])
    identity, _ = signing_identity()
    sleep = NoSleep()
    ctx = DeliveryContext(agent_dir=agent_folder(tmp_path), agent=FakeAgent(identity=identity), sleep=sleep)
    assert run(deliver_webhook(DeliverTarget(kind="webhook", url=hook.url, timeout=5, retries=3), result(), ctx)) == (
        "delivered",
        f"webhook {hook.url} (signed as {ISSUER})",
    )
    assert len(hook.received) == 3
    assert sleep.slept == FIXTURE["webhook"]["backoff_seconds"][:2]
    nonces = {re.search(r'nonce="([^"]+)"', r.headers["signature-input"]).group(1) for r in hook.received}
    assert len(nonces) == 3


def test_webhook_gives_up_after_the_retries_and_does_not_retry_a_final_answer(tmp_path, webhook_server):
    down = webhook_server([500, 500, 500])
    sleep = NoSleep()
    ctx = DeliveryContext(agent_dir=agent_folder(tmp_path), sleep=sleep)
    assert run(deliver_webhook(DeliverTarget(kind="webhook", url=down.url, timeout=5, retries=2), result(), ctx)) == (
        "failed",
        f"webhook {down.url}: answered 500 after 3 tries",
    )
    assert sleep.slept == [1, 2]

    refusing = webhook_server([403])
    assert run(deliver_webhook(DeliverTarget(kind="webhook", url=refusing.url, timeout=5, retries=3), result(), ctx)) == (
        "failed",
        f"webhook {refusing.url}: answered 403 after 1 try",
    )
    assert len(refusing.received) == 1

    closed = webhook_server()
    closed.server.shutdown()
    closed.server.server_close()
    assert run(deliver_webhook(DeliverTarget(kind="webhook", url=closed.url, timeout=5, retries=1), result(), ctx)) == (
        "failed",
        f"webhook {closed.url}: could not be reached after 2 tries",
    )


# -- the manager's turn ----------------------------------------------------------------------


def test_the_manager_runs_a_started_prompt_as_one_owner_turn_and_finishes(tmp_path):
    """`AgentManager.start(name, prompt)` was a task that slept under a TODO
    (`cli/daemon/manager.py`); it runs one real turn as the owner now, logs
    the reply and finishes, freeing the name."""
    from webagents.cli.daemon.manager import AgentManager
    from webagents.cli.daemon.registry import DaemonRegistry
    from webagents.cli.loader import AgentFile

    folder = agent_folder(tmp_path)
    registry = DaemonRegistry()
    registry.register(AgentFile(folder / "AGENT.md"))
    manager = AgentManager(registry)
    agent = FakeAgent("Done.")

    async def load(name):
        assert name == "reporter"
        return agent

    manager.get_or_load_agent = load  # type: ignore[method-assign]

    async def scenario():
        assert await manager.start("reporter", "Report.") is True
        assert manager.get_running_agents() == ["reporter"]
        await manager._running_agents["reporter"]

    run(scenario())
    assert agent.calls == [{"messages": [{"role": "user", "content": "Report."}], "scope": "owner", "provider": "local"}]
    assert registry.get("reporter").status == "stopped"
    assert any(line.endswith("Reply: Done.") for line in manager.get_logs("reporter"))
    assert manager.get_running_agents() == []


# -- chat delivery, against a mocked platform client ------------------------------------------


def test_chat_records_the_prompt_as_the_owner_and_the_reply_as_the_agent(tmp_path):
    folder = agent_folder(tmp_path)
    chat = MockChat()
    ctx = DeliveryContext(agent_dir=folder, chat=chat)
    assert run(deliver(DeliverTarget(kind="chat", to="owner"), result(), ctx)) == ("delivered", "chat owner (chat-1)")
    assert chat.asked == [(folder, "reporter")]
    assert [turn for _, turn in chat.recorded] == [FIXTURE["chat"]["turn"]]
    assert chat.recorded[0][0] is chat.where


def test_chat_fails_with_the_chat_s_own_sentences_when_signed_out_unpublished_or_refused(tmp_path):
    folder = agent_folder(tmp_path)
    login = FIXTURE["details"]["signed_out"].format(login="webagents login")
    publish = FIXTURE["details"]["not_published"].format(publish="webagents publish")
    assert run(deliver(DeliverTarget(kind="chat", to="owner"), result(), DeliveryContext(agent_dir=folder, chat=MockChat(where="signed_out")))) == (
        "failed",
        f"chat owner: {login}",
    )
    assert run(deliver(DeliverTarget(kind="chat", to="owner"), result(), DeliveryContext(agent_dir=folder, chat=MockChat(where="not_published")))) == (
        "failed",
        f"chat owner: {publish}",
    )
    refused = MockChat(fail=RuntimeError("Robutler answered 500"))
    assert run(deliver(DeliverTarget(kind="chat", to="owner"), result(), DeliveryContext(agent_dir=folder, chat=refused))) == (
        "failed",
        "chat owner: Robutler answered 500",
    )
