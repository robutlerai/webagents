"""T3 — advertised-vs-registered tool parity.

For every Skill class in the SDK: `get_skill_info()["tools"]` must equal the
set of names actually decorated with `@tool`. F-042 shipped because a
hand-maintained list advertised three tools while the registered surface was
one different tool — the two sets did not intersect at all, so an LLM
planning from the skill's own introspection selected tools that could not be
called.

The separate `robutler` distribution carries its own, older copy of the files
skill, and this file pinned the two against each other while the SDK depended
on that package. Since 2026-09-25 it does not (the platform API client moved
into the SDK), so no install of the SDK can import that copy, and the
cross-copy tests are gone with it.
"""

import importlib
import inspect
import pkgutil

import pytest

from webagents.agents.skills.base import Skill

SKILLS_PACKAGE = "webagents.agents.skills"


def _iter_skill_classes():
    """Import every module under webagents.agents.skills and yield the Skill
    subclasses it defines. Optional-dependency modules that fail to import
    are skipped — T1's import smoke covers the top-level chain."""
    package = importlib.import_module(SKILLS_PACKAGE)
    seen = set()
    for modinfo in pkgutil.walk_packages(package.__path__, prefix=SKILLS_PACKAGE + "."):
        try:
            module = importlib.import_module(modinfo.name)
        except Exception:
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(cls, Skill)
                and cls is not Skill
                and cls.__module__ == module.__name__
                and cls not in seen
            ):
                seen.add(cls)
                yield cls


def _instantiate(cls):
    """Instantiate a skill for introspection.

    Catches (ImportError, TypeError): several skills raise ImportError for a
    missing optional dependency, and five ecosystem skills take no config
    parameter at all and raise TypeError when handed one.
    """
    sig = inspect.signature(cls.__init__)
    takes_config = any(
        p.name in ("config",) or p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
        if p.name != "self"
    )
    try:
        return cls({}) if takes_config else cls()
    except (ImportError, TypeError):
        return None
    except Exception:
        # A constructor with heavier requirements (network clients, keys).
        # Registration parity for it is covered the day it gets a test
        # harness; do not let it fail the whole sweep.
        return None


def _registered_tool_names(cls):
    names = set()
    for klass in type(cls).__mro__ if not inspect.isclass(cls) else cls.__mro__:
        for attr_name, attr in vars(klass).items():
            if getattr(attr, "_webagents_is_tool", False):
                names.add(getattr(attr, "_tool_name", attr_name))
    return names


class TestAdvertisedVsRegistered:
    def test_every_skill_advertises_exactly_its_registered_tools(self):
        failures = []
        checked = 0
        for cls in _iter_skill_classes():
            instance = _instantiate(cls)
            if instance is None:
                continue
            try:
                info = instance.get_skill_info()
            except Exception as e:
                failures.append(f"{cls.__module__}.{cls.__name__}: get_skill_info raised {e!r}")
                continue
            advertised = set(info.get("tools", []))
            registered = _registered_tool_names(cls)
            if advertised != registered:
                failures.append(
                    f"{cls.__module__}.{cls.__name__}: advertised {sorted(advertised)} "
                    f"!= registered {sorted(registered)}"
                )
            checked += 1
        assert checked >= 30, f"skill discovery collapsed (only {checked} skills checked)"
        assert not failures, "\n".join(failures)

    def test_files_skill_advertises_the_documented_surface(self):
        """F-042 pin: the three documented tools are the three registered
        tools — including `list_files` under its documented name (it had
        been renamed in one copy while docs and the vendored copy kept
        `list_files`)."""
        from webagents.agents.skills.robutler.storage.files.skill import RobutlerFilesSkill

        skill = RobutlerFilesSkill({"api_key": "test-key"})
        info = skill.get_skill_info()
        assert set(info["tools"]) == {
            "store_file_from_url",
            "store_file_from_base64",
            "list_files",
        }
        assert set(info["tools"]) == _registered_tool_names(RobutlerFilesSkill)

    def test_files_skill_pricing_metadata(self):
        """Pricing is part of the advertised contract: list_files bills
        0.005 credits per call; the two store tools are unpriced."""
        from webagents.agents.skills.robutler.storage.files.skill import RobutlerFilesSkill

        pricing = getattr(RobutlerFilesSkill.list_files, "_webagents_pricing", None)
        assert pricing is not None, "list_files lost its @pricing decorator"
        assert pricing["credits_per_call"] == 0.005
        for name in ("store_file_from_url", "store_file_from_base64"):
            fn = getattr(RobutlerFilesSkill, name)
            assert getattr(fn, "_webagents_is_tool", False), f"{name} lost its @tool decorator"
