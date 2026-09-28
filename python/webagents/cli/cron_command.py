"""
`webagents cron list` and `webagents cron run <agent> <name>` (plan item 1.7,
2026-09-26): the `cron:` schedules the agent files in a folder declare, as the
daemon runs them (`cli/daemon/schedule_runner.py`).

`list` reads the folder's agent files exactly as the daemon would
(`cli/daemon/registry.py`, `discover_agent_files` and `AgentFile`) and the
runner's state under each agent's `.webagents/cron/`, and writes nothing: the
runner is built with `persist=False`, because a listing that started the clock
on an `every` schedule the daemon has not seen would have the daemon run it
early. `run` builds the one agent the schedule names the way the chat and
`webagents -p` do (`cli/agent_builder.py`), runs the schedule now as the
owner, delivers as configured, and records the run in that state, where the
daemon's next listing shows it.

The lines are the TypeScript CLI's, word for word
(`typescript/src/cli/cron-action.ts`), held by `tests/fixtures/cli/cron.json`.
This lives beside `mcp_serve.py` and `serve.py` rather than in `commands/`:
`tests/cli/test_phase1_fixes.py` pins that package to its three modules and
`cli/commands/cron.py` in particular to stay deleted.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from .daemon.registry import discover_agent_files
from .daemon.schedule_runner import ScheduleRunner
from .loader import AgentFile, AgentFormatError
from .loader.schedules import CronSchedule, parse_cron_block

#: The table's columns (the fixture's `list.columns`).
COLUMNS = ("AGENT", "SCHEDULE", "WHEN", "NEXT", "LAST")
_NONE = "No schedules: no agent file in {folder} declares cron:."
#: The folder held a file the loader refused (2026-09-28, the e2e pass): "no
#: agent file declares cron:" was untrue when the refused one did. The
#: fixture's `list.none_refused`.
_NONE_REFUSED = "No schedules: no agent file in {folder} that loads declares cron:."
_NO_AGENT = "No agent named {agent} in {folder}."
_NO_SCHEDULE = "No schedule named {name} for agent {agent} in {folder}."
#: The process's exit code per outcome (the fixture's `run.exit`).
EXIT_CODES = {"delivered": 0, "nothing": 0, "failed": 1}


def human_duration(seconds: int) -> str:
    """`1d`, `2h`, `30m` or `90s`: a whole number of the largest unit that divides it."""
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds % size == 0:
            return f"{seconds // size}{unit}"
    return f"{seconds}s"


def _when_column(row: Dict[str, Any]) -> str:
    when = f"{row['expression']} {row['timezone']}" if row["kind"] == "cron" else f"every {human_duration(int(row['every_seconds']))}"
    return f"{when} (heartbeat)" if row.get("heartbeat") else when


def _next_column(row: Dict[str, Any]) -> str:
    if row.get("enabled") is False:
        return "off"
    return row["next_run"] if isinstance(row.get("next_run"), str) else "-"


def _last_column(row: Dict[str, Any]) -> str:
    if row.get("running"):
        return f"running since {row['last_fire']}"
    last = row.get("last_run")
    if not last:
        return "never"
    if last["outcome"] == "failed":
        return f"failed {last['at']}: {last['detail']}"
    return f"{last['outcome']} {last['at']}"


def render_schedule_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """The table for the runner's `list_schedules()` rows (the fixture's `list.cases`)."""
    table = [list(COLUMNS)] + [[str(r["agent"]), str(r["name"]), _when_column(r), _next_column(r), _last_column(r)] for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(COLUMNS) - 1)]
    return ["  ".join([row[i].ljust(widths[i]) for i in range(len(COLUMNS) - 1)] + [row[-1]]).rstrip() for row in table]


def run_line(record: Dict[str, Any]) -> str:
    """One run's line (the fixture's `run.line`)."""
    return f"{record['agent']}/{record['name']}: {record['outcome']} ({record['detail']})"


@dataclass
class FolderSchedules:
    """One agent file the daemon would serve, with the schedules its `cron:` declares."""

    agent_file: AgentFile
    schedules: List[CronSchedule]

    @property
    def name(self) -> str:
        return self.agent_file.metadata.name

    @property
    def folder(self) -> Path:
        return Path(self.agent_file.path).parent


#: `list`'s exit code when a file was refused (the fixture's `list.refused_exit`, 2026-09-26).
LIST_REFUSED_EXIT = 1


def folder_schedules(
    folder: Path,
    error: Callable[[str], None] = lambda line: print(line, file=sys.stderr),
    refused: Optional[Dict[str, int]] = None,
) -> List[FolderSchedules]:
    """The folder's agents as the daemon would serve them: a file that does
    not load (its `cron:` block included, which the loader checks) is said on
    stderr and skipped, as the daemon says it and does not serve it. `refused`
    counts those files (`{"count": n}`), for `list` to exit non-zero on."""
    out: List[FolderSchedules] = []
    for path in discover_agent_files(Path(folder)):
        try:
            agent_file = AgentFile(path)
        except AgentFormatError as refused_file:
            if refused is not None:
                refused["count"] = refused.get("count", 0) + 1
            # `cron`'s own words (the fixture's `list.refused_line`): it read
            # "[daemon] ... The agent is not served.", the daemon's words, in
            # a command that serves nothing. The message names the file.
            error(f"{refused_file} Its schedules do not run until the file loads.")
            continue
        block = agent_file.metadata.cron
        out.append(FolderSchedules(agent_file=agent_file, schedules=parse_cron_block(block) if block else []))
    return out


async def _no_agent(name: str) -> Any:
    return None


def list_schedules_command(
    folder: Path,
    json_out: bool = False,
    *,
    clock: Optional[Callable[[], int]] = None,
    log: Callable[[str], None] = print,
    error: Callable[[str], None] = lambda line: print(line, file=sys.stderr),
    emit: Optional[Callable[[Dict[str, Any]], None]] = None,
    agent: Optional[str] = None,
) -> int:
    """`webagents cron list`: the exit code. `agent` keeps only that agent's
    schedules (the chat's `/cron`, interactive-mode spec 3.7, 2026-09-26). A
    file the daemon would refuse (the string `cron:` form, S-270/D7) is said,
    the listing goes on, and the command exits 1 (2026-09-26, the e2e run):
    it exited 0 after "No schedules", as if the file were fine."""
    folder = Path(folder).resolve()
    refused: Dict[str, int] = {"count": 0}
    runner = ScheduleRunner(_no_agent, clock=clock, persist=False)
    for found in folder_schedules(folder, error, refused):
        if agent is not None and found.name != agent:
            continue
        runner.set_schedules(found.name, found.folder, found.schedules)
    rows = runner.list_schedules()
    code = LIST_REFUSED_EXIT if refused["count"] else 0
    if json_out:
        from .output import emit as emit_json

        (emit or emit_json)({"schedules": rows, "refused": refused["count"]})
        return code
    if not rows:
        log((_NONE_REFUSED if refused["count"] else _NONE).format(folder=folder))
        return code
    for line in render_schedule_table(rows):
        log(line)
    return code


def attach_daemon_identity(agent: Any, agent_file: AgentFile, error: Callable[[str], None] = lambda line: print(line, file=sys.stderr)) -> Any:
    """The signing identity the daemon gives this agent (2026-09-27,
    `daemon/identity.py`): the key in the agent folder's `.webagents/keys`,
    the issuer from `WEBAGENTS_PUBLIC_URL` or the configured daemon address,
    so a webhook run here is signed exactly as the daemon signs it. A key file
    that cannot be used is said and the run goes out unsigned; it is never
    replaced."""
    from .daemon.identity import daemon_agent_identity, daemon_public_url
    from .daemon_address import DEFAULT_HOST, DEFAULT_PORT, DaemonAddressError, resolve_daemon_address

    try:
        try:
            address = resolve_daemon_address()
            host, port = address.host, address.port
        except DaemonAddressError:
            host, port = DEFAULT_HOST, DEFAULT_PORT
        agent.signing_identity = daemon_agent_identity(agent.name, Path(agent_file.path).parent, daemon_public_url(host, port))
    except Exception as failure:  # noqa: BLE001 - said, never fatal: the run still goes out
        error(f"[daemon] {agent_file.path}: {failure} The run goes out without a signing identity.")
    return agent


async def _build_as_the_chat_does(agent_file: AgentFile, folder: Path) -> Any:
    from .agent_builder import build_agent
    from .credentials import get_token

    built = await build_agent(Path(agent_file.path), working_dir=folder, person_token=get_token)
    return attach_daemon_identity(built.agent, agent_file)


async def run_schedule(
    agent_name: str,
    schedule_name: str,
    folder: Path,
    json_out: bool = False,
    *,
    build: Optional[Callable[[AgentFile, Path], Awaitable[Any]]] = None,
    clock: Optional[Callable[[], int]] = None,
    log: Callable[[str], None] = print,
    error: Callable[[str], None] = lambda line: print(line, file=sys.stderr),
    emit: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> int:
    """`webagents cron run <agent> <name>` inside a running loop: the exit
    code. The chat's `/cron run` (interactive-mode spec 3.7, 2026-09-26) calls
    this from its own loop; the CLI wraps it in `asyncio.run`
    (`run_schedule_command`)."""
    folder = Path(folder).resolve()
    found = next((f for f in folder_schedules(folder, error) if f.name == agent_name), None)
    if found is None:
        error(_NO_AGENT.format(agent=agent_name, folder=folder))
        return 1
    if not any(s.name == schedule_name for s in found.schedules):
        error(_NO_SCHEDULE.format(name=schedule_name, agent=agent_name, folder=folder))
        return 1

    agent = await (build or _build_as_the_chat_does)(found.agent_file, folder)

    async def agent_for(name: str) -> Any:
        return agent

    runner = ScheduleRunner(agent_for, clock=clock)
    runner.set_schedules(agent_name, found.folder, found.schedules)
    record = await runner.run_now(agent_name, schedule_name)
    full = {"agent": agent_name, "name": schedule_name, **record}
    if json_out:
        from .output import emit as emit_json

        (emit or emit_json)(full)
    else:
        log(run_line(full))
    return EXIT_CODES.get(str(record.get("outcome")), 1)


def run_schedule_command(
    agent_name: str,
    schedule_name: str,
    folder: Path,
    json_out: bool = False,
    *,
    build: Optional[Callable[[AgentFile, Path], Awaitable[Any]]] = None,
    clock: Optional[Callable[[], int]] = None,
    log: Callable[[str], None] = print,
    error: Callable[[str], None] = lambda line: print(line, file=sys.stderr),
    emit: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> int:
    """`webagents cron run <agent> <name>`: the exit code (`run_schedule`, in its own loop)."""
    return asyncio.run(
        run_schedule(agent_name, schedule_name, folder, json_out, build=build, clock=clock, log=log, error=error, emit=emit)
    )
