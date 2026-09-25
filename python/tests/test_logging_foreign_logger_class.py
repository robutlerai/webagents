"""
Another package's logger class must not decide what webagents prints (2026-09-24).

The `robutler` package, a dependency, calls `logging.setLoggerClass` at import
with a class that gives every NEW logger its own stdout handler at INFO and
turns propagation off. The platform skills import it lazily, partway through a
command, so every webagents logger created after that printed INFO on stdout
whatever `setup_logging` had configured. `webagents run -p` showed the agent's
internals between the prompt and the answer, despite the CLI's quiet default.

Simulated with a stand-in class, so the test does not depend on which
`robutler` version is installed or leave its global state behind.
"""

import logging
import sys

import pytest

from webagents.utils import logging as wl


class _Foreign(logging.Logger):
    """What robutler's class does to every logger created under it."""

    def __init__(self, name):
        super().__init__(name)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        self.addHandler(handler)
        self.setLevel(logging.INFO)
        self.propagate = False


@pytest.fixture
def foreign_class():
    previous = logging.getLoggerClass()
    logging.setLoggerClass(_Foreign)
    try:
        yield
    finally:
        logging.setLoggerClass(previous)


def test_get_logger_takes_its_loggers_back(foreign_class):
    made = wl.get_logger("test_foreign_class.via_get_logger")
    assert logging.getLoggerClass() is wl.WebAgentsLogger
    assert made.handlers == []
    assert made.propagate is True


def test_webagents_loggers_made_under_the_foreign_class_are_cleaned(foreign_class):
    # Many modules call logging.getLogger("webagents.x") directly at import.
    stray = logging.getLogger("webagents.test_foreign_class.direct")
    assert stray.handlers, "the stand-in should behave like robutler's class"

    wl.get_logger("test_foreign_class.trigger")

    assert stray.handlers == []
    assert stray.propagate is True


def test_other_packages_loggers_are_left_alone(foreign_class):
    theirs = logging.getLogger("not_webagents.test_foreign_class")
    before = list(theirs.handlers)
    assert before

    wl.get_logger("test_foreign_class.trigger_again")

    assert theirs.handlers == before


def test_the_cli_formatter_can_leave_tracebacks_out():
    """A failed model call printed the core's whole traceback before the CLI's
    one-line error (2026-09-24). The message stays; the trace goes."""
    try:
        raise RuntimeError("Connection error.")
    except RuntimeError:
        record = logging.LogRecord("webagents.base_agent", logging.ERROR, __file__, 1,
                                   "Agent execution error", None, sys.exc_info())
    quiet = wl.WebAgentsFormatter(tracebacks=False).format(record)
    loud = wl.WebAgentsFormatter().format(record)
    assert "Agent execution error" in quiet and "Traceback" not in quiet
    assert "Traceback" in loud


def test_setup_logging_retunes_the_handlers_our_class_handed_out():
    """`openai._base_client` kept the INFO stdout handler it got at import time,
    so its retry lines landed in `run -p`'s answer after the CLI had asked for
    WARNING on stderr (2026-09-24)."""
    import io

    previous = logging.getLoggerClass()
    logging.setLoggerClass(wl.WebAgentsLogger)
    try:
        wl.setup_logging(level="INFO")
        third_party = logging.getLogger("thirdparty_test_retune.client")
        auto = [h for h in third_party.handlers if getattr(h, "_webagents_auto", False)]
        assert auto, "the class should have handed this logger a handler"

        sink = io.StringIO()
        wl.setup_logging(level="WARNING", stream=sink)
        third_party.info("Retrying request")
        third_party.warning("worth seeing")

        assert auto[0].level == logging.WARNING
        assert "Retrying request" not in sink.getvalue()
        assert "worth seeing" in sink.getvalue()
    finally:
        logging.setLoggerClass(previous)
        wl.setup_logging()
