/**
 * Skill dependency utilities
 *
 * `Skill.dependencies` is a declarative contract: it declares which other
 * mounted skills must be present (by stable, platform-relative name)
 * before a given skill can be initialized. The framework does NOT
 * auto-materialise missing dependencies from a global registry — every
 * skill the agent uses must be passed explicitly to
 * `BaseAgent({ skills: [...] })`. The author owns the skill list; the
 * framework only enforces:
 *
 *   1. Validation — every declared dependency name must correspond to a
 *      skill present in the mounted list. Missing deps throw an
 *      actionable error naming the offending skill and the missing
 *      dependency. This catches misconfiguration at construction time
 *      rather than as a vague runtime failure later.
 *
 *   2. Topological sort — `initialize()` runs dep-first so memory loads
 *      before anything that consumes it, function-runtime loads before
 *      cron / custom-http / custom-tools / host-self-edit, etc. Cycles
 *      throw with a chain trace of the offending names.
 *
 * Skills with the same name are allowed (BaseAgent has always supported
 * mounting multiple instances of one skill class); for dependency
 * resolution the first instance wins as the canonical target for that
 * name. Duplicate-name instances are still emitted unchanged.
 */

import type { ISkill } from './types';

/**
 * Validate that every dependency declared by every skill in `skills`
 * resolves to another skill in the same list. Throws on the first
 * missing dependency with a message that names the offending skill,
 * the missing dependency, and the names that were available — pointed
 * at the agent author so they can fix their skill list.
 */
export function validateSkillDependencies(skills: ISkill[]): void {
  const present = new Set(skills.map((s) => s.name));
  for (const skill of skills) {
    for (const dep of skill.dependencies ?? []) {
      if (present.has(dep)) continue;
      const known = Array.from(present).sort().join(', ') || '(none)';
      throw new Error(
        `Skill "${skill.name}" declares dependency "${dep}" but no skill ` +
          `with that name was passed to BaseAgent. Mounted skills: ${known}. ` +
          `Add the missing skill to the \`skills\` array explicitly — ` +
          `dependencies are not auto-mounted.`,
      );
    }
  }
}

/**
 * Topologically sort an arbitrary list of skills by their
 * `Skill.dependencies` declarations. Skills missing from `byName` are
 * left in their original relative order behind their dependents (they
 * are unknown leafs from the perspective of this graph) — but in
 * practice this never happens because `BaseAgent` calls
 * `validateSkillDependencies` before sorting.
 *
 * Throws with a chain trace on cycles.
 *
 * Returned array initialises in dependency-first order — so memory
 * loads before its consumers, function-runtime loads before cron /
 * custom-http / custom-tools, etc.
 */
export function topoSortSkills(skills: ISkill[]): ISkill[] {
  // First instance wins as the dependency-resolution target for that
  // name. Other instances with the same name are still kept and
  // emitted — BaseAgent has always allowed authors to mount multiple
  // instances of the same skill class.
  const byName = new Map<string, ISkill>();
  for (const s of skills) {
    if (!byName.has(s.name)) byName.set(s.name, s);
  }

  const result: ISkill[] = [];
  const visited = new WeakSet<ISkill>();
  const visiting = new Set<string>();

  const visit = (skill: ISkill, path: string[]): void => {
    if (visited.has(skill)) return;
    if (visiting.has(skill.name)) {
      const cycle = [...path, skill.name].join(' -> ');
      throw new Error(`Skill dependency cycle detected: ${cycle}`);
    }
    visiting.add(skill.name);

    for (const dep of skill.dependencies ?? []) {
      const depSkill = byName.get(dep);
      if (depSkill) {
        visit(depSkill, [...path, skill.name]);
      }
    }

    visiting.delete(skill.name);
    visited.add(skill);
    result.push(skill);
  };

  for (const skill of skills) visit(skill, []);
  return result;
}
