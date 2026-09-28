"""
The `cron:` block of an agent file (plan item 1.7, 2026-09-26): what the
daemon runs on a schedule, as the agent's owner, and where the result goes.

    cron:
      - name: daily-report
        schedule: "0 9 * * 1-5"      # or `every: 30m`
        timezone: Europe/Berlin      # an IANA name; UTC when absent
        prompt: Summarize yesterday's activity.
        deliver:
          file: reports/daily.md     # or `webhook: https://...`, or `chat: owner`
      - name: watch
        every: 1h
        heartbeat: true              # the standing instructions, delivered only when there is something to report
        deliver:
          chat: owner

THE STRING FORM IS REFUSED. `cron: "0 9 * * *"` parsed for a year and ran
nowhere: the TypeScript daemon never read the key, and the Python daemon's
job loop was a sleep under a TODO. It named no prompt and no delivery target,
so there was nothing it could have run and nowhere the result could have gone.
A file that still says it is told the shape that works.

ONE GRAMMAR IN BOTH SDKS. The five cron fields take `*`, `n`, `a-b`, `*/n`,
`a-b/n` and comma lists, the platform's function-cron grammar
(`typescript/src/skills/cron/skill.ts`): no names, no `@daily`, no sixth field,
and no step on a single number (`5/10`), which Vixie cron and croniter read
differently. So a schedule means the same thing to croniter here and to the
TypeScript stepper. `every` is a whole number of s, m, h or d, at least one
minute: a heartbeat that wakes the model every few seconds is a bill, not a
schedule. `timezone` is checked against the zone database, UTC when absent
(the platform's function-cron is UTC too). `deliver` names exactly one target;
`channel` is reserved for the channel relay and refused with its own sentence
until then, so a file written for it fails now rather than delivering nowhere.

Every sentence here is the shared fixture's (`tests/fixtures/daemon/cron.json`),
word for word with `typescript/src/agents/schedules.ts`; the access block's
style (`access: unknown key "x". It takes ...`), which both loaders already
speak.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from .schema import AgentFormatError

#: The keys a schedule takes, in the order the sentence lists them.
SCHEDULE_KEYS = ("deliver", "enabled", "every", "heartbeat", "name", "prompt", "schedule", "timezone")
#: The delivery targets, in the order the sentence lists them.
DELIVER_KINDS = ("chat", "file", "webhook")
#: The keys a webhook mapping takes.
WEBHOOK_KEYS = ("retries", "timeout", "url")
#: `deliver: channel: ...` is the channel relay's, not yet available.
RESERVED_DELIVER_KINDS = ("channel",)

DEFAULT_TIMEZONE = "UTC"
WEBHOOK_TIMEOUT_DEFAULT = 15
WEBHOOK_RETRIES_DEFAULT = 3
EVERY_MIN_SECONDS = 60

_SHAPE = "Each takes name, schedule or every, prompt or heartbeat, and deliver."
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_EVERY_RE = re.compile(r"^(\d+)([smhd])$")
_TZ_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_+-]*(/[A-Za-z0-9_+-]+)*$")
_EVERY_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
#: (low, high) of each cron field: minute, hour, day of month, month, day of week.
_FIELD_RANGES = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 6))


@dataclass
class DeliverTarget:
    """Where a schedule's result goes: one of `file`, `webhook` or `chat`."""

    kind: str
    #: `file`: a path relative to the agent's folder.
    path: Optional[str] = None
    #: `webhook`: the URL, the timeout in seconds, the retries after the first attempt.
    url: Optional[str] = None
    timeout: Optional[float] = None
    retries: Optional[int] = None
    #: `chat`: whose chat, `owner`.
    to: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        if self.kind == "file":
            return {"kind": "file", "path": self.path}
        if self.kind == "webhook":
            return {"kind": "webhook", "url": self.url, "timeout": self.timeout, "retries": self.retries}
        return {"kind": "chat", "to": self.to}


@dataclass
class CronSchedule:
    """One entry of the block, as the daemon runs it."""

    name: str
    #: `cron` (a 5-field expression) or `every` (a fixed interval).
    kind: str
    expression: Optional[str]
    every_seconds: Optional[int]
    timezone: str
    #: The turn's message; None for a heartbeat, which runs the standing instructions.
    prompt: Optional[str]
    heartbeat: bool
    enabled: bool
    deliver: DeliverTarget

    def to_dict(self) -> Dict[str, Any]:
        """The schedule as the shared fixture spells it (the TypeScript `describeSchedule`)."""
        return {
            "name": self.name,
            "kind": self.kind,
            "expression": self.expression,
            "every_seconds": self.every_seconds,
            "timezone": self.timezone,
            "prompt": self.prompt,
            "heartbeat": self.heartbeat,
            "enabled": self.enabled,
            "deliver": self.deliver.to_dict(),
        }


def _is_cron_field(field: str, low: int, high: int) -> bool:
    """One field of the grammar in the file comment (the TypeScript `isCronField`)."""
    if field == "*":
        return True
    for part in field.split(","):
        step_match = re.fullmatch(r"([\d*\-]+)/(\d+)", part)
        base = step_match.group(1) if step_match else part
        if step_match is not None:
            if int(step_match.group(2)) <= 0:
                return False
            # A step walks `*` or a range; `5/10` means different things to
            # different crons, so it is not a schedule here.
            if base != "*" and not re.fullmatch(r"\d+-\d+", base):
                return False
        if base == "*":
            continue
        range_match = re.fullmatch(r"(\d+)-(\d+)", base)
        if range_match:
            a, b = int(range_match.group(1)), int(range_match.group(2))
            if not (low <= a and b <= high and a <= b):
                return False
            continue
        if not re.fullmatch(r"\d+", base):
            return False
        if not (low <= int(base) <= high):
            return False
    return True


def cron_expression(value: Any) -> Optional[str]:
    """The expression with single spaces between its five fields, or None when
    `value` is not a schedule in the grammar above."""
    if not isinstance(value, str):
        return None
    fields = value.split()
    if len(fields) != 5:
        return None
    if not all(_is_cron_field(field, low, high) for field, (low, high) in zip(fields, _FIELD_RANGES)):
        return None
    return " ".join(fields)


def every_seconds(value: Any) -> Optional[int]:
    """The seconds `every` names (`15m`, `2h`, `1d`, `90s`), or None: not a
    duration, or under a minute."""
    if not isinstance(value, str):
        return None
    match = _EVERY_RE.match(value)
    if match is None:
        return None
    seconds = int(match.group(1)) * _EVERY_UNITS[match.group(2)]
    return seconds if seconds >= EVERY_MIN_SECONDS else None


def is_timezone(value: Any) -> bool:
    """Whether `value` names a zone in the zone database, in its own spelling."""
    if not isinstance(value, str) or _TZ_RE.match(value) is None:
        return False
    try:
        from zoneinfo import ZoneInfo

        ZoneInfo(value)
    except Exception:  # noqa: BLE001 - not found, not a zone, or no database: not a timezone here
        return False
    return True


def is_inside_folder(path: Any) -> bool:
    """Whether a `file` target stays inside the agent's folder, by its spelling:
    relative, and never a `..` segment. The deliverer checks the real path again
    when it writes, in case a link on the way out points elsewhere."""
    if not isinstance(path, str) or not path.strip():
        return False
    if path.startswith(("/", "\\", "~")) or re.match(r"^[A-Za-z]:", path):
        return False
    return all(segment != ".." for segment in re.split(r"[\\/]", path))


def _is_http_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parts = urlsplit(value)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and bool(parts.netloc)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _deliver_target(value: Any, where: str) -> DeliverTarget:
    if not isinstance(value, dict) or len(value) != 1:
        raise AgentFormatError(f"{where}: deliver must name one target: chat, file or webhook.")
    kind, config = next(iter(value.items()))
    if kind in RESERVED_DELIVER_KINDS:
        raise AgentFormatError(f"{where}: deliver: {kind} targets are not available yet.")
    if kind not in DELIVER_KINDS:
        raise AgentFormatError(f'{where}: deliver: unknown target "{kind}". It takes chat, file or webhook.')
    if kind == "file":
        if not is_inside_folder(config):
            raise AgentFormatError(f"{where}: deliver: file must be a relative path inside the agent's folder.")
        return DeliverTarget(kind="file", path=config)
    if kind == "chat":
        if config != "owner":
            raise AgentFormatError(f"{where}: deliver: chat must be owner.")
        return DeliverTarget(kind="chat", to="owner")
    # webhook: a URL, or a mapping with url, timeout and retries.
    shape = f"{where}: deliver: webhook must be an http(s) URL, or a mapping with url, timeout and retries."
    if isinstance(config, str):
        if not _is_http_url(config):
            raise AgentFormatError(shape)
        return DeliverTarget(kind="webhook", url=config, timeout=WEBHOOK_TIMEOUT_DEFAULT, retries=WEBHOOK_RETRIES_DEFAULT)
    if not isinstance(config, dict) or not _is_http_url(config.get("url")):
        raise AgentFormatError(shape)
    for key in config:
        if key not in WEBHOOK_KEYS:
            raise AgentFormatError(f'{where}: deliver: webhook: unknown key "{key}". It takes retries, timeout and url.')
    timeout = config.get("timeout", WEBHOOK_TIMEOUT_DEFAULT)
    if not _is_number(timeout) or not 1 <= timeout <= 300:
        raise AgentFormatError(f"{where}: deliver: webhook: timeout must be a number of seconds from 1 to 300.")
    retries = config.get("retries", WEBHOOK_RETRIES_DEFAULT)
    if not _is_number(retries) or not float(retries).is_integer() or not 0 <= retries <= 10:
        raise AgentFormatError(f"{where}: deliver: webhook: retries must be a whole number from 0 to 10.")
    return DeliverTarget(kind="webhook", url=config["url"], timeout=timeout, retries=int(retries))


def _schedule(index: int, entry: Any, seen: Dict[str, int]) -> CronSchedule:
    if not isinstance(entry, dict):
        raise AgentFormatError(f"cron: schedule {index} must be a mapping. {_SHAPE}")
    name = entry.get("name")
    if not isinstance(name, str) or _NAME_RE.match(name) is None:
        raise AgentFormatError(f"cron: schedule {index}: name must be a non-empty string of letters, digits, - and _.")
    where = f"cron: schedule {index} ({name})"
    if name in seen:
        raise AgentFormatError(f"{where}: name is already used by schedule {seen[name]}.")
    seen[name] = index
    for key in entry:
        if key not in SCHEDULE_KEYS:
            raise AgentFormatError(
                f'{where}: unknown key "{key}". It takes deliver, enabled, every, heartbeat, name, prompt, schedule and timezone.'
            )

    has_schedule, has_every = "schedule" in entry, "every" in entry
    if has_schedule == has_every:
        raise AgentFormatError(f"{where}: exactly one of schedule or every is required.")
    expression: Optional[str] = None
    seconds: Optional[int] = None
    if has_schedule:
        expression = cron_expression(entry["schedule"])
        if expression is None:
            raise AgentFormatError(
                f"{where}: schedule must be a 5-field cron expression (minute hour day-of-month month day-of-week)."
            )
    else:
        seconds = every_seconds(entry["every"])
        if seconds is None:
            raise AgentFormatError(f"{where}: every must be a duration like 15m, 2h or 1d, at least 1m.")

    timezone = entry.get("timezone", DEFAULT_TIMEZONE)
    if not is_timezone(timezone):
        raise AgentFormatError(f"{where}: timezone must be an IANA zone name like Europe/Berlin.")

    has_prompt, has_heartbeat = "prompt" in entry, "heartbeat" in entry
    if has_heartbeat and entry["heartbeat"] is not True:
        raise AgentFormatError(f"{where}: heartbeat must be true; leave it out for a prompt schedule.")
    if has_prompt == has_heartbeat:
        raise AgentFormatError(f"{where}: exactly one of prompt or heartbeat is required.")
    prompt: Optional[str] = None
    if has_prompt:
        prompt = entry["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise AgentFormatError(f"{where}: prompt must be a non-empty string.")

    enabled = entry.get("enabled", True)
    if not isinstance(enabled, bool):
        raise AgentFormatError(f"{where}: enabled must be true or false.")

    deliver = _deliver_target(entry.get("deliver"), where)
    return CronSchedule(
        name=name,
        kind="cron" if has_schedule else "every",
        expression=expression,
        every_seconds=seconds,
        timezone=timezone,
        prompt=prompt,
        heartbeat=has_heartbeat,
        enabled=enabled,
        deliver=deliver,
    )


def parse_cron_block(value: Any) -> List[CronSchedule]:
    """The schedules a `cron:` value declares, or an `AgentFormatError` whose
    sentence is written for the file's author (the shared fixture's)."""
    if not isinstance(value, list):
        raise AgentFormatError(f"cron: must be a list of schedules. {_SHAPE}")
    seen: Dict[str, int] = {}
    return [_schedule(index, entry, seen) for index, entry in enumerate(value, start=1)]
