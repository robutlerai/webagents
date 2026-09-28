"""
Runs the `cron:` schedules of the agents a daemon serves (plan item 1.7,
2026-09-26): the same runner, minute for minute, as
`typescript/src/daemon/schedule-runner.ts`.

WHAT RAN BEFORE. `cli/daemon/cron.py` turned an agent's `cron:` string into a
job whose "execution" was `AgentManager.start`, a task that slept in a loop
under a TODO; the TypeScript daemon ignored the key. Nothing an agent file
scheduled ever ran, and the docs said it did (`docs/cli/daemon.md`).

WHAT RUNS NOW. Each schedule is a turn of the agent's normal runtime, as its
OWNER (`access.run_as_local_owner`, the caller `webagents -p` gives a turn),
with the schedule's `prompt` as the one user message; the reply goes where
the schedule says (`deliver.py`). Schedules come from agent files only: the
daemon's HTTP routes list them and can no longer add one (S-273), because an
added job with a prompt of the caller's choosing is a way to run the owner's
agent on the owner's key.

TIME. A cron schedule fires at the next whole minute that matches its
expression in its zone, strictly after the previous fire; day-of-month and
day-of-week are OR'd when both are restricted, as Vixie cron and croniter do.
A wall-clock minute that does not exist in the zone that day (a spring-forward
gap) is skipped; one that exists twice (a fall-back hour) fires at its first
occurrence. An `every` schedule fires one interval after the daemon first sees
it, then one interval after each fire. Both are computed here, not by
croniter, so the fixture's vectors (`tests/fixtures/daemon/cron.json`,
`next_run`) hold in both SDKs.

STATE, so a restart neither double-runs nor drops a schedule
(`.webagents/cron/<agent>.json` in the agent's folder). The next fire is
written BEFORE the turn runs, so a daemon that dies mid-run does not run the
same slot again when it comes back; a fire the daemon slept through is run
once, as soon as it is back, and the schedule then resumes from now rather
than replaying every missed slot. A schedule whose expression, interval or
zone changed is rescheduled from now. One run per schedule at a time: a slot
that comes due while the previous run is still going is skipped, and said.

HEARTBEAT. A `heartbeat: true` schedule has no prompt of its own: the turn is
`HEARTBEAT_PROMPT`, which asks the agent to follow its standing instructions
and answer `HEARTBEAT_SENTINEL` when there is nothing to say (OpenClaw's
heartbeat, Hermes' monitor mode). That reply, or an empty one, is the run's
`nothing` and is delivered nowhere; anything else is the report and goes where
the schedule says. The words are the fixture's (`heartbeat`), the same in
`schedule-runner.ts`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

from ..loader.schedules import CronSchedule, parse_cron_block
from .deliver import DeliveryContext, RunResult, deliver

logger = logging.getLogger("webagents.daemon.schedules")

#: Where an agent's schedule state lives, under the agent's folder.
STATE_DIR = Path(".webagents") / "cron"
#: The one user message of a heartbeat turn (the fixture's `heartbeat.prompt`).
HEARTBEAT_PROMPT = (
    "Heartbeat. Follow your standing instructions and check on whatever they ask you to keep an eye on. "
    "If there is nothing to report, reply with exactly HEARTBEAT_OK and nothing else. "
    "Otherwise reply with the report itself, written for the person who reads it."
)
#: The reply that means "nothing to report".
HEARTBEAT_SENTINEL = "HEARTBEAT_OK"
#: What a model wraps the sentinel in: whitespace, emphasis, quotes, end punctuation.
_SENTINEL_WRAPPING = re.compile(r"^[\s*_`\"'.!]+|[\s*_`\"'.!]+$")


def is_quiet_heartbeat(text: str) -> bool:
    """Whether a heartbeat reply says nothing: empty, or the sentinel however it is dressed."""
    bare = _SENTINEL_WRAPPING.sub("", text)
    return bare == "" or bare.upper() == HEARTBEAT_SENTINEL
#: How far ahead a cron expression is searched for its next slot (`0 0 29 2 *` needs four years).
SEARCH_DAYS = 366 * 5
MINUTE_MS = 60_000
#: How often the daemon looks for due schedules.
TICK_SECONDS = 5.0
#: (low, high) of each cron field: minute, hour, day of month, month, day of week.
_FIELD_RANGES = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 6))


def iso_utc(ms: int) -> str:
    """`2026-09-28T07:00:00Z`: the instant to the second, as both SDKs write it."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(text: Any) -> Optional[int]:
    """The milliseconds an ISO 8601 instant names, or None."""
    if not isinstance(text, str) or not text:
        return None
    try:
        when = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return int(when.timestamp() * 1000)


def _field_values(field: str, low: int, high: int) -> Set[int]:
    """The values one validated cron field allows."""
    values: Set[int] = set()
    for part in field.split(","):
        base, _, step_text = part.partition("/")
        step = int(step_text) if step_text else 1
        if base == "*":
            start, end = low, high
        elif "-" in base:
            a, b = base.split("-", 1)
            start, end = int(a), int(b)
        else:
            start = end = int(base)
        values.update(range(start, end + 1, step))
    return values


def local_to_utc_ms(year: int, month: int, day: int, hour: int, minute: int, tz: ZoneInfo) -> Optional[int]:
    """The instant a wall-clock minute names in `tz`: its first occurrence when
    it exists twice, None when it does not exist that day."""
    local = datetime(year, month, day, hour, minute, tzinfo=tz)
    utc = local.astimezone(timezone.utc)
    back = utc.astimezone(tz)
    if (back.year, back.month, back.day, back.hour, back.minute) != (year, month, day, hour, minute):
        return None
    return int(utc.timestamp() * 1000)


def next_cron_run(expression: str, tz_name: str, after_ms: int) -> Optional[int]:
    """The first whole minute strictly after `after_ms` that `expression`
    matches in `tz_name` (file comment, TIME), or None within `SEARCH_DAYS`."""
    fields = expression.split()
    minutes, hours, doms, months, dows = (
        _field_values(field, low, high) for field, (low, high) in zip(fields, _FIELD_RANGES)
    )
    dom_any, dow_any = fields[2] == "*", fields[4] == "*"

    def day_matches(dom: int, month: int, dow: int) -> bool:
        if month not in months:
            return False
        if dom_any and dow_any:
            return True
        if not dom_any and not dow_any:
            return dom in doms or dow in dows
        return dom in doms if not dom_any else dow in dows

    tz = ZoneInfo(tz_name)
    start_ms = (after_ms // MINUTE_MS) * MINUTE_MS + MINUTE_MS
    start = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).astimezone(tz)
    first_day = start.date()
    for offset in range(SEARCH_DAYS + 1):
        day = first_day + timedelta(days=offset)
        # Python counts Monday as 0; cron counts Sunday as 0.
        if not day_matches(day.day, day.month, (day.weekday() + 1) % 7):
            continue
        for hour in sorted(hours):
            if offset == 0 and hour < start.hour:
                continue
            for minute in sorted(minutes):
                if offset == 0 and hour == start.hour and minute < start.minute:
                    continue
                instant = local_to_utc_ms(day.year, day.month, day.day, hour, minute, tz)
                if instant is None or instant <= after_ms:
                    continue
                return instant
    return None


def next_run_for(schedule: CronSchedule, after_ms: int) -> Optional[int]:
    """When `schedule` fires next, strictly after `after_ms`."""
    if schedule.kind == "cron":
        return next_cron_run(schedule.expression or "", schedule.timezone, after_ms)
    return after_ms + int(schedule.every_seconds or 0) * 1000


def schedule_spec(schedule: CronSchedule) -> str:
    """What a schedule's timing is, so a changed one is rescheduled from now."""
    if schedule.kind == "cron":
        return f"cron {schedule.expression} {schedule.timezone}"
    return f"every {schedule.every_seconds} {schedule.timezone}"


def state_path(agent_dir: Path, agent_name: str) -> Path:
    return Path(agent_dir) / STATE_DIR / f"{agent_name}.json"


@dataclass
class ScheduleEntry:
    """One schedule the runner holds, with its persisted state."""

    agent_name: str
    agent_dir: Path
    schedule: CronSchedule
    #: When it fires next, ms; None while disabled or when no slot exists.
    next_run: Optional[int]
    #: When it last fired, ms.
    last_fire: Optional[int] = None
    #: `{"at", "outcome", "detail"}` of the last run, and `finish` when the
    #: agent's tool budget ended its turn (2026-09-28, `core/tool_budget.py`).
    last_run: Optional[Dict[str, Any]] = None
    running: bool = False
    #: The running turn's finish, for `last_run`.
    turn_finish: Optional[Dict[str, Any]] = None

    def describe(self) -> Dict[str, Any]:
        return {
            "agent": self.agent_name,
            **self.schedule.to_dict(),
            "next_run": iso_utc(self.next_run) if self.next_run is not None else None,
            "last_fire": iso_utc(self.last_fire) if self.last_fire is not None else None,
            "last_run": self.last_run,
            "running": self.running,
        }

    def state(self) -> Dict[str, Any]:
        return {
            "spec": schedule_spec(self.schedule),
            "nextRun": iso_utc(self.next_run) if self.next_run is not None else None,
            "lastFire": iso_utc(self.last_fire) if self.last_fire is not None else None,
            "lastRun": self.last_run,
        }


def reply_text(response: Any) -> str:
    """The reply's text from what `agent.run` returns: OpenAI-shaped, or `content`."""
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
        if isinstance(response.get("content"), str):
            return response["content"]
        return ""
    content = getattr(response, "content", None)
    return content if isinstance(content, str) else ""


def _describe(error: BaseException) -> str:
    return str(error) or type(error).__name__


class ScheduleRunner:
    """Holds the served agents' schedules and runs the due ones (file comment)."""

    def __init__(
        self,
        agent_for: Callable[[str], Awaitable[Any]],
        *,
        clock: Optional[Callable[[], int]] = None,
        deliver_fn: Optional[Callable[..., Awaitable[Tuple[str, str]]]] = None,
        interval_s: float = TICK_SECONDS,
        persist: bool = True,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        """`persist=False` reads state files and writes none: `webagents cron
        list` must not start the clock on an `every` schedule the daemon has
        not seen yet, or the daemon would run it early. `log` also receives
        the runner's lines (the TypeScript runner's `log` option), beside the
        module logger."""
        self._agent_for = agent_for
        self._clock = clock or (lambda: int(time.time() * 1000))
        self._deliver = deliver_fn or deliver
        self.interval_s = interval_s
        self._persist = persist
        self._log = log
        self._entries: Dict[Tuple[str, str], ScheduleEntry] = {}
        self._running = False

    def _say(self, level: int, line: str) -> None:
        """One of the runner's lines: to the module logger, and to `log` when given."""
        logger.log(level, "%s", line)
        if self._log is not None:
            self._log(line)

    # -- what is scheduled ----------------------------------------------------------------

    def set_schedules(self, agent_name: str, agent_dir: Path, schedules: List[CronSchedule]) -> None:
        """The schedules of `agent_name`, replacing what it had: state on disk is
        kept for a schedule whose timing is unchanged, so a restart resumes it."""
        saved = self._load_state(agent_dir, agent_name)
        now = self._clock()
        kept: Dict[Tuple[str, str], ScheduleEntry] = {}
        for schedule in schedules:
            key = (agent_name, schedule.name)
            before = self._entries.get(key)
            entry = ScheduleEntry(agent_name=agent_name, agent_dir=Path(agent_dir), schedule=schedule, next_run=None)
            if before is not None:
                entry.running = before.running
            record = saved.get(schedule.name) if isinstance(saved.get(schedule.name), dict) else None
            same_timing = record is not None and record.get("spec") == schedule_spec(schedule)
            if record is not None:
                entry.last_fire = parse_iso(record.get("lastFire"))
                entry.last_run = record.get("lastRun") if isinstance(record.get("lastRun"), dict) else None
            if not schedule.enabled:
                entry.next_run = None
            elif same_timing and parse_iso(record.get("nextRun")) is not None:
                # A fire the daemon slept through stays due, and runs once.
                entry.next_run = parse_iso(record.get("nextRun"))
            else:
                entry.next_run = next_run_for(schedule, now)
            kept[key] = entry
        for key in [k for k in self._entries if k[0] == agent_name]:
            del self._entries[key]
        self._entries.update(kept)
        self._save_state(agent_name, agent_dir)

    def remove_agent(self, agent_name: str) -> None:
        for key in [k for k in self._entries if k[0] == agent_name]:
            del self._entries[key]

    def sync_from_registry(self, registry: Any) -> None:
        """The schedules of every registered agent (`cli/daemon/registry.py`),
        as their files declare them; an agent whose file lost its block is dropped."""
        seen: Set[str] = set()
        for agent in registry.list_agents():
            block = getattr(agent, "cron", None)
            if not isinstance(block, list) or not block:
                continue
            try:
                schedules = parse_cron_block(block)
            except Exception as error:  # noqa: BLE001 - the loader already refused the file; say so and go on
                logger.warning("%s: %s", agent.source_path, error)
                continue
            seen.add(agent.name)
            self.set_schedules(agent.name, Path(agent.source_path).parent, schedules)
        for name in {k[0] for k in self._entries} - seen:
            self.remove_agent(name)

    def entries(self) -> List[ScheduleEntry]:
        """Every schedule, by agent name, each agent's in the order its file declares them."""
        # A stable sort on the agent alone: `set_schedules` re-inserts an
        # agent's entries in file order, and the dict keeps insertion order.
        return sorted(self._entries.values(), key=lambda entry: entry.agent_name)

    def list_schedules(self) -> List[Dict[str, Any]]:
        """Every schedule with its state, for `GET /cron` and `webagents cron list`."""
        return [entry.describe() for entry in self.entries()]

    def find(self, agent_name: str, schedule_name: str) -> Optional[ScheduleEntry]:
        return self._entries.get((agent_name, schedule_name))

    # -- running ----------------------------------------------------------------------------

    async def tick(self) -> None:
        """Run every schedule that is due, each at most once, concurrently, and wait for them."""
        now = self._clock()
        runs = []
        for entry in self.entries():
            if not entry.schedule.enabled or entry.next_run is None or entry.next_run > now:
                continue
            if entry.running:
                self._say(logging.WARNING, f"{entry.agent_name}/{entry.schedule.name}: due, but the previous run is still going")
                continue
            # The next fire, written before the turn: a daemon that dies
            # mid-run does not run this slot again.
            entry.last_fire = now
            entry.next_run = next_run_for(entry.schedule, now)
            self._save_state(entry.agent_name, entry.agent_dir)
            runs.append(self._execute(entry, now))
        if runs:
            await asyncio.gather(*runs)

    async def run_now(self, agent_name: str, schedule_name: str) -> Dict[str, Any]:
        """Run one schedule now (`webagents cron run`), leaving its next fire as it was."""
        entry = self.find(agent_name, schedule_name)
        if entry is None:
            raise KeyError(f"{agent_name}/{schedule_name}")
        if entry.running:
            return {"at": iso_utc(self._clock()), "outcome": "failed", "detail": "the previous run is still going"}
        await self._execute(entry, self._clock())
        return dict(entry.last_run or {})

    async def _execute(self, entry: ScheduleEntry, fired_at: int) -> None:
        entry.running = True
        entry.turn_finish = None
        try:
            outcome, detail = await self._turn_and_deliver(entry, fired_at)
        finally:
            entry.running = False
        entry.last_run = {"at": iso_utc(fired_at), "outcome": outcome, "detail": detail}
        if entry.turn_finish is not None:
            entry.last_run["finish"] = entry.turn_finish
        self._save_state(entry.agent_name, entry.agent_dir)
        self._say(logging.INFO, f"{entry.agent_name}/{entry.schedule.name}: {outcome} ({detail})")

    async def _turn_and_deliver(self, entry: ScheduleEntry, fired_at: int) -> Tuple[str, str]:
        agent = await self._agent_for(entry.agent_name)
        if agent is None:
            return "failed", "the agent is not served"
        heartbeat = entry.schedule.heartbeat
        prompt = HEARTBEAT_PROMPT if heartbeat else (entry.schedule.prompt or "")
        try:
            from webagents.access import run_as_local_owner

            # The owner's own turn, as `webagents -p` runs one.
            run_as_local_owner(agent)
            response = await agent.run([{"role": "user", "content": prompt}])
        except Exception as error:  # noqa: BLE001 - the run's failure is the run's record
            return "failed", f"turn failed: {_describe(error)}"
        # The agent's tool budget ended the turn (2026-09-28): the record says
        # so, and the answer its last, tool-less call gave is delivered.
        finish = response.get("webagents_finish") if isinstance(response, dict) else None
        if isinstance(finish, dict) and finish.get("reason") in ("tool_round_limit", "tool_loop"):
            entry.turn_finish = {key: finish[key] for key in ("reason", "rounds", "tool") if finish.get(key) is not None}
        content = reply_text(response)
        # Nothing to deliver: a quiet heartbeat, or a reply with no words in it.
        if heartbeat and is_quiet_heartbeat(content):
            return "nothing", "nothing to report"
        if not content.strip():
            return "nothing", "the reply was empty"
        result = RunResult(
            agent=entry.agent_name,
            schedule=entry.schedule.name,
            kind=entry.schedule.kind,
            # A heartbeat's prompt is the runner's, not the file's: the record carries none.
            prompt=None if heartbeat else prompt,
            content=content,
            ran_at=iso_utc(fired_at),
        )
        try:
            return await self._deliver(entry.schedule.deliver, result, DeliveryContext(agent_dir=entry.agent_dir, agent=agent))
        except Exception as error:  # noqa: BLE001 - a deliverer's failure is the run's record
            return "failed", f"delivery failed: {_describe(error)}"

    async def run(self) -> None:
        """The daemon's loop: a tick every `interval_s` until `stop()`."""
        self._running = True
        try:
            while self._running:
                try:
                    await self.tick()
                except Exception as error:  # noqa: BLE001 - one bad tick must not end the loop
                    self._say(logging.ERROR, f"schedule tick failed: {_describe(error)}")
                await asyncio.sleep(self.interval_s)
        except asyncio.CancelledError:
            self._running = False
            raise

    def stop(self) -> None:
        self._running = False

    # -- state ------------------------------------------------------------------------------

    def _load_state(self, agent_dir: Path, agent_name: str) -> Dict[str, Any]:
        try:
            data = json.loads(state_path(agent_dir, agent_name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        schedules = data.get("schedules") if isinstance(data, dict) else None
        return schedules if isinstance(schedules, dict) else {}

    def _save_state(self, agent_name: str, agent_dir: Path) -> None:
        if not self._persist:
            return
        entries = [entry for entry in self.entries() if entry.agent_name == agent_name]
        path = state_path(agent_dir, agent_name)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"schedules": {e.schedule.name: e.state() for e in entries}}, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as error:
            logger.warning("%s: could not write schedule state: %s", path, error)
