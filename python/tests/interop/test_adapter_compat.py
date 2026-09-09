"""
Cross-Language Adapter Compatibility Tests (Python side)

Verifies that Python adapter message conversion matches the shared fixtures
that the TypeScript adapters also consume. Both must produce identical output
for the same inputs.

Run with:
    python -m pytest tests/interop/test_adapter_compat.py -v
"""

import json
import importlib.util
import os
import sys
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "test-fixtures" / "adapter-compat"

# Direct module loader — bypasses the full webagents package chain (which drags
# in optional deps like robutler).  We only need the uamp_adapter modules.
_ADAPTER_PATHS = {
    "anthropic": "webagents/agents/skills/core/llm/anthropic/uamp_adapter.py",
    "openai": "webagents/agents/skills/core/llm/openai/uamp_adapter.py",
    "google": "webagents/agents/skills/core/llm/google/uamp_adapter.py",
}
_ADAPTER_CLASSES = {
    "anthropic": "AnthropicUAMPAdapter",
    "openai": "OpenAIUAMPAdapter",
    "google": "GoogleUAMPAdapter",
}

_PY_ROOT = Path(__file__).resolve().parent.parent.parent


_adapters_loaded = False


def _ensure_importable():
    """
    Stub the webagents top-level __init__ so that ``from webagents.uamp import …``
    works without pulling in the full package chain (which requires ``robutler``).
    """
    global _adapters_loaded
    if _adapters_loaded:
        return
    _adapters_loaded = True

    if str(_PY_ROOT) not in sys.path:
        sys.path.insert(0, str(_PY_ROOT))

    import types
    # Provide a lightweight stub for the top-level webagents package so that
    # ``from webagents.uamp import …`` resolves to the real uamp sub-package
    # without executing the heavy webagents/__init__.py.
    if "webagents" not in sys.modules or hasattr(sys.modules["webagents"], "__COMPAT_STUB__"):
        stub = types.ModuleType("webagents")
        stub.__path__ = [str(_PY_ROOT / "webagents")]
        stub.__COMPAT_STUB__ = True  # type: ignore[attr-defined]
        sys.modules["webagents"] = stub


def _import_adapter(provider: str):
    """Import an adapter class directly."""
    _ensure_importable()

    mod_path = f"webagents.agents.skills.core.llm.{provider}.uamp_adapter"
    if mod_path in sys.modules:
        return getattr(sys.modules[mod_path], _ADAPTER_CLASSES[provider])

    spec = importlib.util.spec_from_file_location(
        mod_path, _PY_ROOT / _ADAPTER_PATHS[provider],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_path] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, _ADAPTER_CLASSES[provider])


def load_fixture(name: str):
    with open(FIXTURES_DIR / name) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class TestAnthropicCompat:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.fixture = load_fixture("anthropic.json")

    def _convert(self, messages):
        adapter = _import_adapter("anthropic")
        return adapter.convert_messages(messages)

    def _convert_tools(self, tools):
        adapter = _import_adapter("anthropic")
        return adapter.convert_tools(tools)

    @pytest.mark.parametrize(
        "tc",
        [pytest.param(tc, id=tc["name"]) for tc in load_fixture("anthropic.json")["tests"]],
    )
    def test_message_conversion(self, tc):
        result = self._convert(tc["input"]["messages"])

        if "system" in tc["expected"]:
            assert result.get("system") == tc["expected"]["system"]
        else:
            assert "system" not in result

        assert result["messages"] == tc["expected"]["messages"]

        # No content_items leaks
        if tc.get("expected_no_field"):
            for m in result["messages"]:
                assert tc["expected_no_field"] not in m

    @pytest.mark.parametrize(
        "tc",
        [pytest.param(tc, id=tc["name"]) for tc in load_fixture("anthropic.json").get("tool_tests", [])],
    )
    def test_tool_conversion(self, tc):
        result = self._convert_tools(tc["input"])
        assert result == tc["expected"]


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class TestOpenAICompat:
    def _convert(self, messages):
        adapter = _import_adapter("openai")
        return adapter.convert_messages(messages)

    def _convert_tools(self, tools):
        adapter = _import_adapter("openai")
        return adapter.convert_tools(tools)

    @pytest.mark.parametrize(
        "tc",
        [pytest.param(tc, id=tc["name"]) for tc in load_fixture("openai.json")["tests"]],
    )
    def test_message_conversion(self, tc):
        result = self._convert(tc["input"]["messages"])
        assert result == tc["expected"]

        # No content_items leaks
        if tc.get("expected_no_field"):
            for m in result:
                assert tc["expected_no_field"] not in m

    @pytest.mark.parametrize(
        "tc",
        [pytest.param(tc, id=tc["name"]) for tc in load_fixture("openai.json").get("tool_tests", [])],
    )
    def test_tool_conversion(self, tc):
        result = self._convert_tools(tc["input"])
        assert result == tc["expected"]


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------

class TestGoogleCompat:
    def _convert(self, messages):
        adapter = _import_adapter("google")
        return adapter.convert_messages(messages)

    def _convert_tools(self, tools):
        adapter = _import_adapter("google")
        return adapter.convert_tools(tools)

    @pytest.mark.parametrize(
        "tc",
        [pytest.param(tc, id=tc["name"]) for tc in load_fixture("google.json")["tests"]],
    )
    def test_message_conversion(self, tc):
        result = self._convert(tc["input"]["messages"])

        assert result["contents"] == tc["expected"]["contents"]

        if "system_parts" in tc["expected"]:
            assert result.get("system_parts") == tc["expected"]["system_parts"]
        else:
            assert "system_parts" not in result

    @pytest.mark.parametrize(
        "tc",
        [pytest.param(tc, id=tc["name"]) for tc in load_fixture("google.json").get("tool_tests", [])],
    )
    def test_tool_conversion(self, tc):
        result = self._convert_tools(tc["input"])
        assert result == tc["expected"]
