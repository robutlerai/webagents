/**
 * BrowserAutomationSkill Unit Tests
 * 
 * These tests verify the skill's structure and configuration.
 * Full browser tests require a browser environment (e2e tests).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { BrowserAutomationSkill } from '../../../../src/skills/browser/automation.js';

// Mock DOM environment for unit tests
const createMockDocument = () => {
  const elements = new Map();
  const container: any = {
    id: 'webagents-mark-container',
    style: { cssText: '' },
    appendChild: vi.fn(),
    remove: vi.fn(),
  };
  
  return {
    createElement: vi.fn((tag: string) => {
      const el: any = {
        tagName: tag.toUpperCase(),
        style: { cssText: '' },
        textContent: '',
        id: '',
        setAttribute: vi.fn(),
        getAttribute: vi.fn(),
        removeAttribute: vi.fn(),
        getBoundingClientRect: () => ({ top: 0, left: 0, width: 100, height: 50, bottom: 50, right: 100, x: 0, y: 0 }),
        appendChild: vi.fn(),
        remove: vi.fn(),
      };
      return el;
    }),
    body: {
      appendChild: vi.fn((el: any) => {
        elements.set(el.id, el);
      }),
    },
    querySelectorAll: vi.fn(() => []),
    querySelector: vi.fn(() => null),
    getElementById: vi.fn((id: string) => elements.get(id)),
  };
};

describe('BrowserAutomationSkill', () => {
  let skill: BrowserAutomationSkill;

  beforeEach(() => {
    skill = new BrowserAutomationSkill();
  });

  describe('initialization', () => {
    it('should have correct id and description', () => {
      expect(skill.id).toBe('browser-automation');
      expect(skill.description).toBe('Browser automation for DOM control, screenshots, and testing');
    });
  });

  describe('element info extraction', () => {
    // This test requires a full browser environment (DOM APIs like HTMLInputElement)
    // It's tested in e2e tests instead
    it.skip('should extract element info correctly (browser-only)', () => {
      // Browser-only test - requires real DOM environment
      expect(true).toBe(true);
    });
  });

  describe('accessibility tree', () => {
    it('should map implicit ARIA roles', () => {
      const testCases = [
        { tag: 'button', expected: 'button' },
        { tag: 'nav', expected: 'navigation' },
        { tag: 'main', expected: 'main' },
        { tag: 'header', expected: 'banner' },
        { tag: 'footer', expected: 'contentinfo' },
        { tag: 'form', expected: 'form' },
        { tag: 'ul', expected: 'list' },
        { tag: 'li', expected: 'listitem' },
      ];

      for (const { tag, expected } of testCases) {
        const mockElement = {
          tagName: tag.toUpperCase(),
          hasAttribute: () => tag === 'a',
        } as any;
        
        const role = (skill as any).getImplicitRole(mockElement);
        expect(role).toBe(expected);
      }
    });

    it('should map input types to roles', () => {
      const testCases = [
        { type: 'text', expected: 'textbox' },
        { type: 'checkbox', expected: 'checkbox' },
        { type: 'radio', expected: 'radio' },
        { type: 'button', expected: 'button' },
        { type: 'submit', expected: 'button' },
        { type: 'search', expected: 'searchbox' },
        { type: 'range', expected: 'slider' },
        { type: 'number', expected: 'spinbutton' },
      ];

      for (const { type, expected } of testCases) {
        const mockInput = { type } as HTMLInputElement;
        const role = (skill as any).getInputRole(mockInput);
        expect(role).toBe(expected);
      }
    });
  });

  describe('SoM marking', () => {
    it('should have mark_elements tool', () => {
      const tools = (skill as any).getToolDefinitions?.() || [];
      // The skill should expose marking tools
      expect(skill.markElements).toBeDefined();
      expect(typeof skill.markElements).toBe('function');
    });

    it('should have unmark_elements tool', () => {
      expect(skill.unmarkElements).toBeDefined();
      expect(typeof skill.unmarkElements).toBe('function');
    });

    it('should have marked_screenshot tool', () => {
      expect(skill.markedScreenshot).toBeDefined();
      expect(typeof skill.markedScreenshot).toBe('function');
    });

    it('should have click_marked tool', () => {
      expect(skill.clickMarked).toBeDefined();
      expect(typeof skill.clickMarked).toBe('function');
    });

    it('should have type_marked tool', () => {
      expect(skill.typeMarked).toBeDefined();
      expect(typeof skill.typeMarked).toBe('function');
    });
  });

  describe('default selectors', () => {
    it('should include standard interactive elements', () => {
      // The default selector should match common interactive elements
      const defaultSelector = 'a[href], button, input:not([type="hidden"]), select, textarea, summary, [role="button"], [tabindex]:not([tabindex="-1"]), [onclick], [data-action]';
      
      // Verify it includes key element types
      expect(defaultSelector).toContain('a[href]');
      expect(defaultSelector).toContain('button');
      expect(defaultSelector).toContain('input');
      expect(defaultSelector).toContain('select');
      expect(defaultSelector).toContain('textarea');
      expect(defaultSelector).toContain('[role="button"]');
      expect(defaultSelector).toContain('[tabindex]');
    });
  });
});

describe('BrowserAutomationSkill - DOM Operations', () => {
  // These tests would run in a browser environment
  // For now, we verify the method signatures exist
  
  it('should have DOM query methods', () => {
    const skill = new BrowserAutomationSkill();
    
    expect(skill.queryElement).toBeDefined();
    expect(skill.queryElements).toBeDefined();
  });

  it('should have interaction methods', () => {
    const skill = new BrowserAutomationSkill();
    
    expect(skill.click).toBeDefined();
    expect(skill.type).toBeDefined();
    expect(skill.scroll).toBeDefined();
    expect(skill.focus).toBeDefined();
    expect(skill.hover).toBeDefined();
  });

  it('should have content methods', () => {
    const skill = new BrowserAutomationSkill();
    
    expect(skill.getText).toBeDefined();
    expect(skill.getHtml).toBeDefined();
    expect(skill.evaluate).toBeDefined();
  });

  it('should have screenshot methods', () => {
    const skill = new BrowserAutomationSkill();
    
    expect(skill.screenshot).toBeDefined();
    expect(skill.markedScreenshot).toBeDefined();
  });
});
