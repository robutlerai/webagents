# Documentation Improvements

## Consistency & Links
- Add anchors in `docs/skills/overview.md` for sections referenced from other pages (e.g., `#2-implement-core-functionality`, `#best-practices`, `#4-implement-lifecycle-hooks`, `#5-enable-smart-routing`) or update all incoming links to point to the correct headings.
- Ensure all platform skill links consistently use `skills/platform/...` relative paths from root-level docs.

## Under Construction Notices
- Replace illustrative code in ecosystem skill pages with short notices once implementations land (filesystem, database, crewai, n8n), and add minimal working snippets.
- Consider a single "Roadmap" page linking all under-construction skills for transparency.

## Examples
- Add curl examples for `@http` endpoints in `docs/sdk/agent/endpoints.md` that include typical headers and error cases.
- Provide a minimal end-to-end sample in `examples/` that matches Quickstart (agent + one platform skill + serve).

## Navigation
- Revisit nav grouping names for clarity: "Skills Repository" vs "Architecture". Consider a short landing page that explains both “using skills” and “building skills.”

## Style
- Keep admonitions to 2–3 per page; convert informational notes to inline sentences where possible.
- Maintain consistent heading levels and intro summaries at the top of each page.


