"""
Phase 5a: the context file's name, its reach, and who wins a merge (2026-09-23).

Each test here pins a defect that was live in `loader/context.py` or
`loader/hierarchy.py`, so they are written as "this exact thing must not happen
again" rather than as coverage of the happy path, which `test_hierarchy.py`
already has.

The four, in the order they bite a user:

1. The loader read `AGENTS.md`, which since became a cross-vendor standard for
   CODING agents. Any repo with one had it merged into the agent's system
   prompt.
2. The upward walk had no bound, so it climbed to `/`.
3. A dict-form entry (`- mcp: {...}`) crashed deduplication.
4. Context beat the agent: an ancestor directory could override the namespace
   and the skill configuration the agent declared for itself.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from webagents.cli.loader.context import (
    CONTEXT_FILENAME,
    ContextFile,
    ContextHierarchy,
)
from webagents.cli.loader.hierarchy import AgentLoader


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


class TestTheFilename:
    def test_the_cross_vendor_agents_md_is_not_read(self, tmp_path):
        # The whole point of the rename. A file written for Codex, Cursor or
        # Claude Code must not become an agent's system prompt.
        write(tmp_path / "AGENTS.md", "---\nnamespace: someone.else\n---\nRun pnpm test.")
        write(tmp_path / ".git" / "config", "")

        assert ContextHierarchy().resolve(tmp_path) == []

    def test_there_is_no_back_compat_fallback(self, tmp_path):
        # Not even "read AGENTS.md when WEBAGENTS.md is absent": a fallback
        # keeps exactly the behaviour the rename exists to stop.
        write(tmp_path / "AGENTS.md", "---\nnamespace: someone.else\n---\nForeign.")
        write(tmp_path / "AGENT.md", "---\nname: a\n---\nMine.")
        write(tmp_path / ".git" / "config", "")

        merged = AgentLoader().load(tmp_path / "AGENT.md")
        assert "Foreign" not in merged.instructions
        assert merged.metadata.namespace == "local"

    def test_webagents_md_is_read(self, tmp_path):
        write(tmp_path / CONTEXT_FILENAME, "---\nnamespace: ours\n---\nOurs.")
        write(tmp_path / ".git" / "config", "")

        contexts = ContextHierarchy().resolve(tmp_path)
        assert len(contexts) == 1
        assert contexts[0].metadata.namespace == "ours"


class TestTheWalkIsBounded:
    def test_stops_at_a_git_marker_and_does_not_read_above_it(self, tmp_path):
        # Before: the walk climbed to the filesystem root, so a file anywhere
        # above the checkout joined the system prompt.
        write(tmp_path / CONTEXT_FILENAME, "OUTSIDE THE PROJECT")
        repo = tmp_path / "repo"
        write(repo / ".git" / "config", "")
        write(repo / CONTEXT_FILENAME, "repo root")
        sub = repo / "pkg" / "deep"
        write(sub / CONTEXT_FILENAME, "leaf")

        found = [c.content for c in ContextHierarchy().resolve(sub)]

        assert found == ["repo root", "leaf"]
        assert "OUTSIDE THE PROJECT" not in found

    def test_a_git_worktree_file_counts_as_a_marker(self, tmp_path):
        # In a worktree `.git` is a FILE, which is why the check is `exists()`.
        write(tmp_path / CONTEXT_FILENAME, "OUTSIDE")
        repo = tmp_path / "wt"
        write(repo / ".git", "gitdir: /elsewhere/.git/worktrees/wt\n")
        write(repo / CONTEXT_FILENAME, "worktree root")

        assert [c.content for c in ContextHierarchy().resolve(repo)] == ["worktree root"]

    def test_dot_webagents_is_the_marker_for_a_non_git_project(self, tmp_path):
        write(tmp_path / CONTEXT_FILENAME, "OUTSIDE")
        proj = tmp_path / "proj"
        (proj / ".webagents").mkdir(parents=True)
        write(proj / CONTEXT_FILENAME, "project root")

        assert [c.content for c in ContextHierarchy().resolve(proj)] == ["project root"]

    def test_the_home_directory_is_never_read(self, tmp_path, monkeypatch):
        # A dotfiles repo puts a `.git` directly in $HOME, so the marker rule
        # alone would happily inherit from it and give every agent on the
        # machine the same silent preamble. $HOME is checked first, and is
        # exclusive.
        home = tmp_path / "home"
        write(home / ".git" / "config", "")
        write(home / CONTEXT_FILENAME, "HOME DOTFILES")
        work = home / "scratch" / "agent"
        work.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

        assert ContextHierarchy().resolve(work) == []

    def test_an_explicit_stop_at_still_wins(self, tmp_path):
        # A caller that names the top gets the top it named, marker or not.
        outer = tmp_path / "outer"
        write(outer / CONTEXT_FILENAME, "outer")
        inner = outer / "repo"
        write(inner / ".git" / "config", "")
        write(inner / CONTEXT_FILENAME, "inner")

        found = ContextHierarchy(stop_at=outer).resolve(inner)
        assert [c.content for c in found] == ["outer", "inner"]


class TestDictFormEntriesDoNotCrash:
    def test_a_dict_skill_survives_deduplication(self, tmp_path):
        # `dict.fromkeys` raised TypeError: unhashable type: 'dict'.
        write(
            tmp_path / CONTEXT_FILENAME,
            "---\nskills:\n  - memory\n  - mcp:\n      servers: [fs]\n---\nctx",
        )
        write(tmp_path / ".git" / "config", "")

        hierarchy = ContextHierarchy()
        merged = hierarchy.merge_contexts(hierarchy.resolve(tmp_path))

        assert merged["skills"] == ["memory", {"mcp": {"servers": ["fs"]}}]

    def test_mcp_servers_is_string_only_and_says_so(self, tmp_path):
        # Recording the real shape rather than the one you would expect: only
        # `skills` is `List[Union[str, Dict]]`. `mcp_servers` and `tools` are
        # `List[str]` in both schemas, so a mapping is REJECTED here, not
        # silently mangled. That is the safe direction, but it surfaces as a
        # raw pydantic ValidationError out of `ContextFile._parse` with no file
        # name attached, which is the schema-ergonomics work still open in
        # Phase 5a. `merge_entries` tolerates dicts in all three lists so this
        # stays a schema decision rather than a crash waiting in the merge.
        write(
            tmp_path / CONTEXT_FILENAME,
            "---\nmcp_servers:\n  - fs:\n      command: npx\n---\nctx",
        )
        write(tmp_path / ".git" / "config", "")

        with pytest.raises(ValidationError):
            ContextHierarchy().resolve(tmp_path)

    def test_the_same_dict_entry_twice_collapses_to_one(self, tmp_path):
        write(tmp_path / CONTEXT_FILENAME, "---\nskills:\n  - mcp: {servers: [a]}\n---\nroot")
        sub = tmp_path / "sub"
        write(sub / CONTEXT_FILENAME, "---\nskills:\n  - mcp: {servers: [b]}\n---\nleaf")
        write(tmp_path / ".git" / "config", "")

        hierarchy = ContextHierarchy()
        merged = hierarchy.merge_contexts(hierarchy.resolve(sub))

        # Deduplicated by NAME, and the nearer file's configuration is the one
        # kept, matching "later contexts override earlier ones".
        assert merged["skills"] == [{"mcp": {"servers": ["b"]}}]


class TestTheAgentWins:
    def test_an_explicit_local_namespace_is_not_overridden(self, tmp_path):
        # `namespace` defaults to "local", so the old check could not tell an
        # agent that SAYS local from one that says nothing, and shipped the
        # agent into the context's namespace against its own declaration.
        write(tmp_path / CONTEXT_FILENAME, "---\nnamespace: ai.myorg\n---\nctx")
        write(tmp_path / "AGENT.md", "---\nname: a\nnamespace: local\n---\nBody")
        write(tmp_path / ".git" / "config", "")

        merged = AgentLoader().load(tmp_path / "AGENT.md")
        assert merged.metadata.namespace == "local"

    def test_an_absent_namespace_still_inherits(self, tmp_path):
        # The other half of the same rule, and the behaviour people rely on.
        write(tmp_path / CONTEXT_FILENAME, "---\nnamespace: ai.myorg\n---\nctx")
        write(tmp_path / "AGENT.md", "---\nname: a\n---\nBody")
        write(tmp_path / ".git" / "config", "")

        merged = AgentLoader().load(tmp_path / "AGENT.md")
        assert merged.metadata.namespace == "ai.myorg"

    def test_a_silent_context_does_not_reset_its_parents_namespace(self, tmp_path):
        # Same defect one level up: every child context file "declared"
        # namespace, because the schema default made it truthy.
        write(tmp_path / CONTEXT_FILENAME, "---\nnamespace: ai.myorg\n---\nroot")
        sub = tmp_path / "sub"
        write(sub / CONTEXT_FILENAME, "---\nmodel: openai/gpt-4o\n---\nleaf")
        write(tmp_path / ".git" / "config", "")

        hierarchy = ContextHierarchy()
        merged = hierarchy.merge_contexts(hierarchy.resolve(sub))

        assert merged["namespace"] == "ai.myorg"

    def test_the_agents_own_skill_config_beats_the_contexts(self, tmp_path):
        # Previously the context entry was kept and the agent's DISCARDED, the
        # exact opposite of the docstring on `_merge_metadata`.
        write(
            tmp_path / CONTEXT_FILENAME,
            "---\nskills:\n  - mcp: {servers: [context-server]}\n  - memory\n---\nctx",
        )
        write(
            tmp_path / "AGENT.md",
            "---\nname: a\nskills:\n  - mcp: {servers: [agent-server]}\n  - cron\n---\nBody",
        )
        write(tmp_path / ".git" / "config", "")

        merged = AgentLoader().load(tmp_path / "AGENT.md")

        # Position is inherited (context order first, agent additions appended)
        # but the VALUE at the conflicting name is the agent's.
        assert merged.metadata.skills == [
            {"mcp": {"servers": ["agent-server"]}},
            "memory",
            "cron",
        ]

    def test_string_skills_still_deduplicate_without_reordering(self, tmp_path):
        write(tmp_path / CONTEXT_FILENAME, "---\nskills: [memory, mcp]\n---\nctx")
        write(tmp_path / "AGENT.md", "---\nname: a\nskills: [mcp, cron]\n---\nBody")
        write(tmp_path / ".git" / "config", "")

        merged = AgentLoader().load(tmp_path / "AGENT.md")
        assert merged.metadata.skills == ["memory", "mcp", "cron"]


class TestDeclares:
    def test_reports_what_the_file_actually_set(self, tmp_path):
        path = write(tmp_path / CONTEXT_FILENAME, "---\nnamespace: x\n---\nBody")
        ctx = ContextFile(path)

        assert ctx.declares("namespace")
        assert not ctx.declares("visibility")

    def test_a_file_with_no_frontmatter_declares_nothing(self, tmp_path):
        path = write(tmp_path / CONTEXT_FILENAME, "Just prose, no frontmatter.\n")
        ctx = ContextFile(path)

        assert not ctx.declares("namespace")
        assert ctx.content == "Just prose, no frontmatter."
