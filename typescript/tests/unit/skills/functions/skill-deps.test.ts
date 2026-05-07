/**
 * Skill.dependencies — validation + topological ordering.
 *
 * The framework does NOT auto-materialise missing dependencies; the
 * agent author owns the skill list and the framework's job is to
 * validate it and order the explicit instances dep-first.
 */

import { describe, it, expect } from 'vitest';
import { Skill } from '../../../../src/core/skill.js';
import {
  validateSkillDependencies,
  topoSortSkills,
} from '../../../../src/core/skill-registry.js';

describe('skill dependencies', () => {
  class A extends Skill {
    readonly name = 'a';
    readonly dependencies = ['b'] as const;
  }
  class B extends Skill {
    readonly name = 'b';
    readonly dependencies = [] as const;
  }
  class C extends Skill {
    readonly name = 'c';
    readonly dependencies = ['a', 'b'] as const;
  }

  it('topo-sorts in dependency order', () => {
    const a = new A();
    const b = new B();
    const c = new C();
    const sorted = topoSortSkills([c, a, b]);
    const idx = (n: string) => sorted.findIndex((s) => s.name === n);
    expect(idx('b')).toBeLessThan(idx('a'));
    expect(idx('a')).toBeLessThan(idx('c'));
  });

  it('throws on cycle', () => {
    class X extends Skill {
      readonly name = 'x';
      readonly dependencies = ['y'] as const;
    }
    class Y extends Skill {
      readonly name = 'y';
      readonly dependencies = ['x'] as const;
    }
    expect(() => topoSortSkills([new X(), new Y()])).toThrow(/cycle/i);
  });

  it('passes validation when every declared dep is mounted', () => {
    expect(() => validateSkillDependencies([new A(), new B()])).not.toThrow();
    expect(() => validateSkillDependencies([new A(), new B(), new C()])).not.toThrow();
  });

  it('throws when a declared dep is not in the skill list', () => {
    // `A` declares dep on `b`, but only `A` is mounted — author bug.
    expect(() => validateSkillDependencies([new A()])).toThrow(
      /Skill "a" declares dependency "b"/,
    );
  });

  it('throws message names the offending skill, missing dep, and mounted skills', () => {
    try {
      validateSkillDependencies([new A()]);
      throw new Error('should have thrown');
    } catch (err) {
      const msg = (err as Error).message;
      expect(msg).toContain('"a"');
      expect(msg).toContain('"b"');
      expect(msg).toContain('Mounted skills: a');
      expect(msg).toContain('not auto-mounted');
    }
  });
});
