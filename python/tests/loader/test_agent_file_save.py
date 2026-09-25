"""
Phase 5a: writing an agent file must not destroy what a person put in it.

`AgentFile._save()` re-serialised the whole frontmatter with `yaml.dump`, so
the first programmatic write (`/skill add`, `update_metadata`) silently lost
every comment, deleted every key whose value equalled the schema default, and
reordered the rest. Only the author could restore the comments.

These pin the replacement: an in-place edit of the keys that changed.
"""

from pathlib import Path

from webagents.cli.loader import AgentFile

ANNOTATED = """---
# Production agent. Do not change the namespace without telling ops.
name: ops-bot
description: Handles the pager
namespace: local        # pinned on purpose
model: openai/gpt-4o-mini

# Skills the on-call rotation depends on
skills:
  - memory
  - mcp
intents:
  - triage alerts
---

# Ops Bot

Body text, with its own # hash, stays put.
"""


def agent(tmp_path: Path, text: str = ANNOTATED) -> AgentFile:
    path = tmp_path / "AGENT.md"
    path.write_text(text)
    return AgentFile(path)


class TestCommentsSurvive:
    def test_comments_outside_the_changed_key_are_kept(self, tmp_path):
        af = agent(tmp_path)
        af.add_skill("cron")

        text = af.path.read_text()
        assert "# Production agent. Do not change the namespace" in text
        assert "# pinned on purpose" in text
        assert "# Skills the on-call rotation depends on" in text

    def test_the_body_is_untouched(self, tmp_path):
        af = agent(tmp_path)
        af.add_skill("cron")

        assert "Body text, with its own # hash, stays put." in af.path.read_text()

    def test_untouched_keys_keep_their_order_and_formatting(self, tmp_path):
        af = agent(tmp_path)
        af.add_skill("cron")

        lines = af.path.read_text().splitlines()
        assert lines[2] == "name: ops-bot"
        assert lines[4] == "namespace: local        # pinned on purpose"
        # `intents` was never touched, so it keeps its own indentation.
        assert "  - triage alerts" in lines


class TestValuesEqualToDefaultsSurvive:
    def test_an_explicit_namespace_local_is_not_deleted(self, tmp_path):
        # This stopped being cosmetic when the loader learned to tell
        # "declared" from "defaulted": deleting the key turns a
        # pinned-to-local agent into one that inherits from context.
        af = agent(tmp_path)
        af.add_skill("cron")

        reread = AgentFile(af.path)
        assert "namespace" in reread.metadata.model_fields_set
        assert reread.metadata.namespace == "local"


class TestTheEditIsCorrect:
    def test_the_new_skill_is_actually_written(self, tmp_path):
        af = agent(tmp_path)
        af.add_skill("cron")

        assert AgentFile(af.path).metadata.skills == ["memory", "mcp", "cron"]

    def test_removing_a_skill_rewrites_only_that_key(self, tmp_path):
        af = agent(tmp_path)
        af.remove_skill("mcp")

        reread = AgentFile(af.path)
        assert reread.metadata.skills == ["memory"]
        assert reread.metadata.model == "openai/gpt-4o-mini"
        assert "# Production agent." in af.path.read_text()

    def test_sequence_indentation_matches_what_it_replaces(self, tmp_path):
        # `yaml.dump` writes `- item` flush left; this file writes `  - item`.
        # Two styles in one file reads as damage even though both parse.
        af = agent(tmp_path)
        af.add_skill("cron")

        assert "  - cron" in af.path.read_text()

    def test_a_key_absent_from_the_file_is_appended(self, tmp_path):
        af = agent(tmp_path, "---\nname: minimal\n---\n\nBody.\n")
        af.update_metadata(model="openai/gpt-4o")

        assert AgentFile(af.path).metadata.model == "openai/gpt-4o"
        assert af.path.read_text().splitlines()[1] == "name: minimal"

    def test_a_no_op_save_does_not_rewrite_the_file(self, tmp_path):
        af = agent(tmp_path)
        before = af.path.read_text()
        af.add_skill("memory")  # already present

        assert af.path.read_text() == before

    def test_a_file_with_no_frontmatter_gets_one(self, tmp_path):
        path = tmp_path / "AGENT-plain.md"
        path.write_text("Just instructions.\n")
        af = AgentFile(path)
        af.add_skill("memory")

        reread = AgentFile(path)
        assert reread.metadata.skills == ["memory"]
        assert "Just instructions." in path.read_text()

    def test_a_trailing_comment_before_the_next_key_is_not_swallowed(self, tmp_path):
        af = agent(
            tmp_path,
            "---\nname: a\nskills:\n  - memory\n\n# introduces the model\nmodel: x\n---\n\nBody.\n",
        )
        af.add_skill("cron")

        text = af.path.read_text()
        assert "# introduces the model" in text
        assert "model: x" in text
        assert AgentFile(af.path).metadata.skills == ["memory", "cron"]


class TestOnlyRealEditsAreWritten:
    def test_a_partially_specified_nested_key_is_not_expanded(self, tmp_path):
        # The obvious diff -- current dump vs the raw YAML -- is wrong, because
        # `model_dump` fills defaults, so `sandbox: {preset: strict}` never
        # equals its own two-field source and was rewritten on every save with
        # three fields the author never typed.
        af = agent(
            tmp_path,
            "---\nname: a\nskills:\n  - memory\nsandbox:\n  preset: strict\n---\n\nBody.\n",
        )
        af.add_skill("cron")

        text = af.path.read_text()
        assert "allowed_folders" not in text
        assert "allowed_commands" not in text
        assert text.count("preset: strict") == 1

    def test_no_key_is_added_that_was_not_there(self, tmp_path):
        af = agent(tmp_path, "---\nname: a\nskills: [memory]\n---\n\nBody.\n")
        af.add_skill("cron")

        assert set(AgentFile(af.path)._raw_yaml) == {"name", "skills"}

    def test_a_body_that_contains_a_horizontal_rule_survives(self, tmp_path):
        # The frontmatter regex is non-greedy, but a `---` in the body is the
        # classic way to get a rewrite to truncate a file.
        af = agent(tmp_path, "---\nname: a\nskills: [memory]\n---\n\nIntro\n\n---\n\nSection two.\n")
        af.add_skill("cron")

        text = af.path.read_text()
        assert "Section two." in text
        assert "Intro" in text

    def test_a_skills_line_in_the_body_is_not_mistaken_for_frontmatter(self, tmp_path):
        af = agent(
            tmp_path,
            "---\nname: a\nskills: [memory]\n---\n\nThe skills: line below is prose.\nskills: not frontmatter\n",
        )
        af.add_skill("cron")

        assert "skills: not frontmatter" in af.path.read_text()
        assert AgentFile(af.path).metadata.skills == ["memory", "cron"]

    def test_a_crlf_file_keeps_its_line_endings(self, tmp_path):
        # `read_text` applies universal newlines and `write_text` emits the
        # platform's, so a CRLF file came back entirely LF: every line changed,
        # which is exactly the whole-file diff this method exists to avoid.
        path = tmp_path / "AGENT-crlf.md"
        path.write_text(
            "---\r\nname: a\r\nskills:\r\n  - memory\r\nmodel: x\r\n---\r\n\r\nBody.\r\n",
            newline="",
        )
        AgentFile(path).add_skill("cron")

        raw = path.read_bytes()
        assert b"\r\n" in raw
        assert b"\n" not in raw.replace(b"\r\n", b"")
        assert AgentFile(path).metadata.skills == ["memory", "cron"]

    def test_an_lf_file_does_not_gain_carriage_returns(self, tmp_path):
        af = agent(tmp_path)
        af.add_skill("cron")

        assert b"\r" not in af.path.read_bytes()
