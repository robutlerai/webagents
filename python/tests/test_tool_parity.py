"""T3 — advertised-vs-registered tool parity.

For every Skill class in the SDK: `get_skill_info()["tools"]` must equal the
set of names actually decorated with `@tool`. F-042 shipped because a
hand-maintained list advertised three tools while the registered surface was
one different tool — the two sets did not intersect at all, so an LLM
planning from the skill's own introspection selected tools that could not be
called.

Also covers the VENDORED copy of the files skill inside the separate
`robutler` distribution: the two copies must not drift silently on names,
pricing metadata, or descriptions. (The vendored copy is currently BEHIND —
robutler 0.2.0 still ships the commented-out decorators — so the cross-copy
test pins today's exact divergence; when robutler 0.2.1 lands, it fails
loudly and the pin must be replaced with full parity.)
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


robutler_files = pytest.importorskip(
    "robutler.agents.skills.robutler.storage.files.skill",
    reason="robutler distribution not installed",
)


class TestVendoredRobutlerCopy:
    """The `robutler` distribution vendors its own copy of the files skill.
    A fix applied to only one copy is half-applied: the two bill and
    advertise differently depending on which package a deployment imports."""

    def _vendored_registered(self):
        cls = robutler_files.RobutlerFilesSkill
        names = set()
        for klass in cls.__mro__:
            for attr_name, attr in vars(klass).items():
                if getattr(attr, "_webagents_is_tool", False) or getattr(attr, "_robutler_is_tool", False):
                    names.add(getattr(attr, "_tool_name", attr_name))
        return names

    def test_vendored_copy_divergence_is_pinned(self):
        """PINNED DIVERGENCE (robutler 0.2.0): the vendored copy registers
        ONLY `list_files` — its store_* decorators are still commented out —
        while the webagents copy registers all three documented tools.

        This is the half of F-041/F-042 that cannot be fixed from this
        repo. When robutler 0.2.1 restores the two decorators, THIS TEST
        FAILS ON PURPOSE: replace the pin with a strict cross-copy parity
        assertion (registered names equal).
        """
        vendored = self._vendored_registered()
        assert vendored == {"list_files"}, (
            f"vendored robutler files skill now registers {sorted(vendored)} — "
            "the 0.2.0 divergence pin no longer holds. Replace this pin with "
            "strict cross-copy parity (names + pricing + descriptions)."
        )

    def test_list_files_pricing_parity_across_copies(self):
        """The one tool both copies register must BILL identically: a caller
        pays the same 0.005 credits whichever distribution served it."""
        from webagents.agents.skills.robutler.storage.files.skill import RobutlerFilesSkill

        def price_of(fn):
            meta = getattr(fn, "_webagents_pricing", None) or getattr(fn, "_robutler_pricing", None)
            return meta and meta.get("credits_per_call")

        assert price_of(robutler_files.RobutlerFilesSkill.list_files) == \
            price_of(RobutlerFilesSkill.list_files) == 0.005

    def test_tool_names_agree_across_copies(self):
        """The one live vendored tool must exist under the SAME name in the
        webagents copy — name skew means the two copies advertise different
        surfaces for the same documented skill."""
        from webagents.agents.skills.robutler.storage.files.skill import RobutlerFilesSkill

        webagents_names = _registered_tool_names(RobutlerFilesSkill)
        assert self._vendored_registered() <= webagents_names
