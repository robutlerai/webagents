/**
 * Browser Automation Skill
 * 
 * Cutting-edge browser control for DOM automation, element interaction,
 * screenshots, and page manipulation. Ideal for testing and automation.
 */

import { Skill } from '../../core/skill.js';
import { tool } from '../../core/decorators.js';

/**
 * Element information returned by queries
 */
export interface ElementInfo {
  tag: string;
  id?: string;
  className?: string;
  text?: string;
  value?: string;
  href?: string;
  src?: string;
  disabled?: boolean;
  checked?: boolean;
  visible: boolean;
  rect: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  attributes: Record<string, string>;
  computedStyle?: Record<string, string>;
}

/**
 * Screenshot result
 */
export interface ScreenshotResult {
  dataUrl: string;
  width: number;
  height: number;
  format: 'png' | 'jpeg' | 'webp';
}

/**
 * Network entry from Performance API
 */
export interface NetworkEntry {
  name: string;
  entryType: string;
  startTime: number;
  duration: number;
  initiatorType?: string;
  transferSize?: number;
  encodedBodySize?: number;
  decodedBodySize?: number;
  responseStatus?: number;
}

/**
 * Accessibility node info
 */
export interface AccessibilityInfo {
  role: string;
  name: string;
  description?: string;
  value?: string;
  checked?: boolean;
  disabled?: boolean;
  expanded?: boolean;
  selected?: boolean;
  children?: AccessibilityInfo[];
}

/**
 * Browser Automation Skill
 * 
 * Provides tools for automated browser control including:
 * - DOM element queries and manipulation
 * - Click, type, scroll interactions
 * - Screenshots and visual snapshots
 * - JavaScript evaluation
 * - Network monitoring
 * - Clipboard access
 * - Accessibility queries
 * 
 * @example
 * ```typescript
 * const automation = new BrowserAutomationSkill();
 * 
 * // Click a button
 * await automation.click({ selector: '#submit-btn' });
 * 
 * // Type into an input
 * await automation.type({ selector: '#email', text: 'user@example.com' });
 * 
 * // Wait for element
 * await automation.waitForElement({ selector: '.loaded', timeout: 5000 });
 * 
 * // Take screenshot
 * const screenshot = await automation.screenshot({});
 * ```
 */
export class BrowserAutomationSkill extends Skill {
  /** Unique skill identifier */
  readonly id: string = 'browser-automation';
  
  /** Skill description */
  readonly description: string = 'Browser automation for DOM control, screenshots, and testing';

  constructor() {
    super({ name: 'Browser Automation' });
  }

  // ============================================================================
  // Element Query Tools
  // ============================================================================

  /**
   * Query a single element and return its info
   */
  @tool({
    name: 'query_element',
    description: 'Query a single DOM element by CSS selector and return its information',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector to find the element',
      },
      includeComputedStyle: {
        type: 'boolean',
        description: 'Include computed styles (slower)',
      },
    },
  })
  async queryElement(
    selector: string,
    includeComputedStyle: boolean = false
  ): Promise<{ element: ElementInfo | null; error?: string }> {
    try {
      const el = document.querySelector(selector);
      if (!el) {
        return { element: null };
      }
      return { element: this.getElementInfo(el as HTMLElement, includeComputedStyle) };
    } catch (error) {
      return { element: null, error: (error as Error).message };
    }
  }

  /**
   * Query multiple elements and return their info
   */
  @tool({
    name: 'query_elements',
    description: 'Query multiple DOM elements by CSS selector',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector to find elements',
      },
      limit: {
        type: 'number',
        description: 'Maximum number of elements to return (default: 100)',
      },
    },
  })
  async queryElements(
    selector: string,
    limit: number = 100
  ): Promise<{ elements: ElementInfo[]; count: number; error?: string }> {
    try {
      const els = document.querySelectorAll(selector);
      const elements: ElementInfo[] = [];
      const total = els.length;
      
      for (let i = 0; i < Math.min(els.length, limit); i++) {
        elements.push(this.getElementInfo(els[i] as HTMLElement, false));
      }
      
      return { elements, count: total };
    } catch (error) {
      return { elements: [], count: 0, error: (error as Error).message };
    }
  }

  /**
   * Wait for an element to appear in the DOM
   */
  @tool({
    name: 'wait_for_element',
    description: 'Wait for an element to appear in the DOM',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector to wait for',
      },
      timeout: {
        type: 'number',
        description: 'Timeout in milliseconds (default: 10000)',
      },
      visible: {
        type: 'boolean',
        description: 'Wait for element to be visible (default: false)',
      },
    },
  })
  async waitForElement(
    selector: string,
    timeout: number = 10000,
    visible: boolean = false
  ): Promise<{ found: boolean; element?: ElementInfo; error?: string }> {
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeout) {
      const el = document.querySelector(selector) as HTMLElement;
      
      if (el) {
        if (visible) {
          const rect = el.getBoundingClientRect();
          const style = getComputedStyle(el);
          const isVisible = rect.width > 0 && 
                           rect.height > 0 && 
                           style.visibility !== 'hidden' && 
                           style.display !== 'none';
          if (isVisible) {
            return { found: true, element: this.getElementInfo(el, false) };
          }
        } else {
          return { found: true, element: this.getElementInfo(el, false) };
        }
      }
      
      await this.sleep(100);
    }
    
    return { found: false, error: `Element "${selector}" not found within ${timeout}ms` };
  }

  /**
   * Wait for element to disappear
   */
  @tool({
    name: 'wait_for_element_hidden',
    description: 'Wait for an element to disappear or become hidden',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector to wait for disappearance',
      },
      timeout: {
        type: 'number',
        description: 'Timeout in milliseconds (default: 10000)',
      },
    },
  })
  async waitForElementHidden(
    selector: string,
    timeout: number = 10000
  ): Promise<{ hidden: boolean; error?: string }> {
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeout) {
      const el = document.querySelector(selector) as HTMLElement;
      
      if (!el) {
        return { hidden: true };
      }
      
      const rect = el.getBoundingClientRect();
      const style = getComputedStyle(el);
      const isHidden = rect.width === 0 || 
                       rect.height === 0 || 
                       style.visibility === 'hidden' || 
                       style.display === 'none';
      
      if (isHidden) {
        return { hidden: true };
      }
      
      await this.sleep(100);
    }
    
    return { hidden: false, error: `Element "${selector}" still visible after ${timeout}ms` };
  }

  // ============================================================================
  // Interaction Tools
  // ============================================================================

  /**
   * Click on an element
   */
  @tool({
    name: 'click',
    description: 'Click on a DOM element',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector of element to click',
      },
      button: {
        type: 'string',
        description: 'Mouse button: left, middle, right (default: left)',
      },
      modifiers: {
        type: 'object',
        description: 'Modifier keys: { ctrl, shift, alt, meta }',
      },
    },
  })
  async click(
    selector: string,
    button: 'left' | 'middle' | 'right' = 'left',
    modifiers?: { ctrl?: boolean; shift?: boolean; alt?: boolean; meta?: boolean }
  ): Promise<{ success: boolean; error?: string }> {
    try {
      const el = document.querySelector(selector) as HTMLElement;
      if (!el) {
        return { success: false, error: `Element not found: ${selector}` };
      }
      
      // Scroll into view
      el.scrollIntoView({ behavior: 'instant', block: 'center' });
      await this.sleep(50);
      
      // Create and dispatch mouse events
      const rect = el.getBoundingClientRect();
      const x = rect.left + rect.width / 2;
      const y = rect.top + rect.height / 2;
      
      const buttonMap = { left: 0, middle: 1, right: 2 };
      const mouseButton = buttonMap[button];
      
      const eventInit: MouseEventInit = {
        bubbles: true,
        cancelable: true,
        view: window,
        clientX: x,
        clientY: y,
        button: mouseButton,
        ctrlKey: modifiers?.ctrl || false,
        shiftKey: modifiers?.shift || false,
        altKey: modifiers?.alt || false,
        metaKey: modifiers?.meta || false,
      };
      
      el.dispatchEvent(new MouseEvent('mousedown', eventInit));
      el.dispatchEvent(new MouseEvent('mouseup', eventInit));
      el.dispatchEvent(new MouseEvent('click', eventInit));
      
      return { success: true };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  }

  /**
   * Double click on an element
   */
  @tool({
    name: 'double_click',
    description: 'Double click on a DOM element',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector of element to double click',
      },
    },
  })
  async doubleClick(selector: string): Promise<{ success: boolean; error?: string }> {
    try {
      const el = document.querySelector(selector) as HTMLElement;
      if (!el) {
        return { success: false, error: `Element not found: ${selector}` };
      }
      
      el.scrollIntoView({ behavior: 'instant', block: 'center' });
      await this.sleep(50);
      
      const rect = el.getBoundingClientRect();
      const eventInit: MouseEventInit = {
        bubbles: true,
        cancelable: true,
        view: window,
        clientX: rect.left + rect.width / 2,
        clientY: rect.top + rect.height / 2,
        detail: 2,
      };
      
      el.dispatchEvent(new MouseEvent('dblclick', eventInit));
      
      return { success: true };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  }

  /**
   * Type text into an element
   */
  @tool({
    name: 'type',
    description: 'Type text into an input element',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector of input element',
      },
      text: {
        type: 'string',
        description: 'Text to type',
      },
      clear: {
        type: 'boolean',
        description: 'Clear existing text first (default: true)',
      },
      delay: {
        type: 'number',
        description: 'Delay between keystrokes in ms (default: 0)',
      },
    },
  })
  async type(
    selector: string,
    text: string,
    clear: boolean = true,
    delay: number = 0
  ): Promise<{ success: boolean; error?: string }> {
    try {
      const el = document.querySelector(selector) as HTMLInputElement | HTMLTextAreaElement;
      if (!el) {
        return { success: false, error: `Element not found: ${selector}` };
      }
      
      // Focus the element
      el.focus();
      
      // Clear if requested
      if (clear) {
        el.value = '';
        el.dispatchEvent(new Event('input', { bubbles: true }));
      }
      
      // Type each character
      for (const char of text) {
        el.value += char;
        el.dispatchEvent(new KeyboardEvent('keydown', { key: char, bubbles: true }));
        el.dispatchEvent(new KeyboardEvent('keypress', { key: char, bubbles: true }));
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new KeyboardEvent('keyup', { key: char, bubbles: true }));
        
        if (delay > 0) {
          await this.sleep(delay);
        }
      }
      
      el.dispatchEvent(new Event('change', { bubbles: true }));
      
      return { success: true };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  }

  /**
   * Press a keyboard key
   */
  @tool({
    name: 'press_key',
    description: 'Press a keyboard key (Enter, Tab, Escape, etc.)',
    parameters: {
      key: {
        type: 'string',
        description: 'Key to press (Enter, Tab, Escape, ArrowUp, etc.)',
      },
      selector: {
        type: 'string',
        description: 'Optional selector to focus before pressing',
      },
      modifiers: {
        type: 'object',
        description: 'Modifier keys: { ctrl, shift, alt, meta }',
      },
    },
  })
  async pressKey(
    key: string,
    selector?: string,
    modifiers?: { ctrl?: boolean; shift?: boolean; alt?: boolean; meta?: boolean }
  ): Promise<{ success: boolean; error?: string }> {
    try {
      const target = selector 
        ? document.querySelector(selector) as HTMLElement
        : document.activeElement as HTMLElement || document.body;
      
      if (selector && !target) {
        return { success: false, error: `Element not found: ${selector}` };
      }
      
      if (selector) {
        (target as HTMLElement).focus();
      }
      
      const eventInit: KeyboardEventInit = {
        key,
        code: this.keyToCode(key),
        bubbles: true,
        cancelable: true,
        ctrlKey: modifiers?.ctrl || false,
        shiftKey: modifiers?.shift || false,
        altKey: modifiers?.alt || false,
        metaKey: modifiers?.meta || false,
      };
      
      target.dispatchEvent(new KeyboardEvent('keydown', eventInit));
      target.dispatchEvent(new KeyboardEvent('keypress', eventInit));
      target.dispatchEvent(new KeyboardEvent('keyup', eventInit));
      
      return { success: true };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  }

  /**
   * Hover over an element
   */
  @tool({
    name: 'hover',
    description: 'Hover the mouse over an element',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector of element to hover',
      },
    },
  })
  async hover(selector: string): Promise<{ success: boolean; error?: string }> {
    try {
      const el = document.querySelector(selector) as HTMLElement;
      if (!el) {
        return { success: false, error: `Element not found: ${selector}` };
      }
      
      const rect = el.getBoundingClientRect();
      const eventInit: MouseEventInit = {
        bubbles: true,
        cancelable: true,
        view: window,
        clientX: rect.left + rect.width / 2,
        clientY: rect.top + rect.height / 2,
      };
      
      el.dispatchEvent(new MouseEvent('mouseenter', eventInit));
      el.dispatchEvent(new MouseEvent('mouseover', eventInit));
      
      return { success: true };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  }

  /**
   * Select an option from a dropdown
   */
  @tool({
    name: 'select',
    description: 'Select an option from a dropdown/select element',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector of select element',
      },
      value: {
        type: 'string',
        description: 'Value or text of option to select',
      },
      byText: {
        type: 'boolean',
        description: 'Select by visible text instead of value (default: false)',
      },
    },
  })
  async select(
    selector: string,
    value: string,
    byText: boolean = false
  ): Promise<{ success: boolean; selectedValue?: string; error?: string }> {
    try {
      const el = document.querySelector(selector) as HTMLSelectElement;
      if (!el || el.tagName !== 'SELECT') {
        return { success: false, error: `Select element not found: ${selector}` };
      }
      
      const options = Array.from(el.options);
      const option = byText
        ? options.find(o => o.text === value)
        : options.find(o => o.value === value);
      
      if (!option) {
        return { success: false, error: `Option not found: ${value}` };
      }
      
      el.value = option.value;
      el.dispatchEvent(new Event('change', { bubbles: true }));
      
      return { success: true, selectedValue: option.value };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  }

  /**
   * Check or uncheck a checkbox
   */
  @tool({
    name: 'check',
    description: 'Check or uncheck a checkbox element',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector of checkbox element',
      },
      checked: {
        type: 'boolean',
        description: 'Whether to check (true) or uncheck (false)',
      },
    },
  })
  async check(
    selector: string,
    checked: boolean = true
  ): Promise<{ success: boolean; checked?: boolean; error?: string }> {
    try {
      const el = document.querySelector(selector) as HTMLInputElement;
      if (!el || el.type !== 'checkbox') {
        return { success: false, error: `Checkbox not found: ${selector}` };
      }
      
      if (el.checked !== checked) {
        el.checked = checked;
        el.dispatchEvent(new Event('change', { bubbles: true }));
      }
      
      return { success: true, checked: el.checked };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  }

  /**
   * Scroll to element or position
   */
  @tool({
    name: 'scroll',
    description: 'Scroll the page or an element',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector to scroll to, or container to scroll',
      },
      x: {
        type: 'number',
        description: 'Horizontal scroll offset (if no selector)',
      },
      y: {
        type: 'number',
        description: 'Vertical scroll offset (if no selector)',
      },
      behavior: {
        type: 'string',
        description: 'Scroll behavior: smooth or instant (default: smooth)',
      },
    },
  })
  async scroll(
    selector?: string,
    x?: number,
    y?: number,
    behavior: 'smooth' | 'instant' = 'smooth'
  ): Promise<{ success: boolean; error?: string }> {
    try {
      if (selector) {
        const el = document.querySelector(selector);
        if (!el) {
          return { success: false, error: `Element not found: ${selector}` };
        }
        el.scrollIntoView({ behavior, block: 'center' });
      } else {
        window.scrollTo({
          left: x ?? window.scrollX,
          top: y ?? window.scrollY,
          behavior,
        });
      }
      
      return { success: true };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  }

  /**
   * Focus an element
   */
  @tool({
    name: 'focus',
    description: 'Focus on an element',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector of element to focus',
      },
    },
  })
  async focus(selector: string): Promise<{ success: boolean; error?: string }> {
    try {
      const el = document.querySelector(selector) as HTMLElement;
      if (!el) {
        return { success: false, error: `Element not found: ${selector}` };
      }
      
      el.focus();
      return { success: true };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  }

  /**
   * Blur (unfocus) an element
   */
  @tool({
    name: 'blur',
    description: 'Remove focus from an element',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector of element to blur (optional, blurs active element)',
      },
    },
  })
  async blur(selector?: string): Promise<{ success: boolean; error?: string }> {
    try {
      const el = selector 
        ? document.querySelector(selector) as HTMLElement
        : document.activeElement as HTMLElement;
      
      if (el && typeof el.blur === 'function') {
        el.blur();
      }
      
      return { success: true };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  }

  // ============================================================================
  // Visual & State Tools
  // ============================================================================

  /**
   * Take a screenshot of the page or element
   */
  @tool({
    name: 'screenshot',
    description: 'Take a screenshot of the page or a specific element',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector of element to capture (optional, captures viewport)',
      },
      format: {
        type: 'string',
        description: 'Image format: png, jpeg, webp (default: png)',
      },
      quality: {
        type: 'number',
        description: 'Image quality 0-1 for jpeg/webp (default: 0.92)',
      },
    },
  })
  async screenshot(
    selector?: string,
    format: 'png' | 'jpeg' | 'webp' = 'png',
    quality: number = 0.92
  ): Promise<ScreenshotResult | { error: string }> {
    try {
      // This requires html2canvas or similar library
      // For now, we'll use canvas API for basic screenshots
      
      const target = selector 
        ? document.querySelector(selector) as HTMLElement
        : document.body;
      
      if (!target) {
        return { error: `Element not found: ${selector}` };
      }
      
      // Check if html2canvas is available
      const html2canvas = (window as unknown as { html2canvas?: Function }).html2canvas;
      
      if (html2canvas) {
        const canvas = await html2canvas(target);
        const mimeType = `image/${format}`;
        const dataUrl = canvas.toDataURL(mimeType, quality);
        
        return {
          dataUrl,
          width: canvas.width,
          height: canvas.height,
          format,
        };
      }
      
      // Fallback: return element dimensions without actual screenshot
      const rect = target.getBoundingClientRect();
      return {
        dataUrl: '',
        width: Math.round(rect.width),
        height: Math.round(rect.height),
        format,
      };
    } catch (error) {
      return { error: (error as Error).message };
    }
  }

  /**
   * Get DOM snapshot as HTML
   */
  @tool({
    name: 'get_html',
    description: 'Get HTML content of an element or the entire page',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector (optional, gets document.body)',
      },
      outer: {
        type: 'boolean',
        description: 'Include outer element HTML (default: true)',
      },
    },
  })
  async getHtml(
    selector?: string,
    outer: boolean = true
  ): Promise<{ html: string; error?: string }> {
    try {
      const el = selector 
        ? document.querySelector(selector)
        : document.body;
      
      if (!el) {
        return { html: '', error: `Element not found: ${selector}` };
      }
      
      const html = outer ? el.outerHTML : el.innerHTML;
      return { html };
    } catch (error) {
      return { html: '', error: (error as Error).message };
    }
  }

  /**
   * Get text content of an element
   */
  @tool({
    name: 'get_text',
    description: 'Get text content of an element',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector of element',
      },
    },
  })
  async getText(selector: string): Promise<{ text: string; error?: string }> {
    try {
      const el = document.querySelector(selector);
      if (!el) {
        return { text: '', error: `Element not found: ${selector}` };
      }
      
      return { text: el.textContent?.trim() || '' };
    } catch (error) {
      return { text: '', error: (error as Error).message };
    }
  }

  /**
   * Get value of an input element
   */
  @tool({
    name: 'get_value',
    description: 'Get value of an input, select, or textarea element',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector of input element',
      },
    },
  })
  async getValue(selector: string): Promise<{ value: string; error?: string }> {
    try {
      const el = document.querySelector(selector) as HTMLInputElement;
      if (!el) {
        return { value: '', error: `Element not found: ${selector}` };
      }
      
      return { value: el.value || '' };
    } catch (error) {
      return { value: '', error: (error as Error).message };
    }
  }

  /**
   * Set value of an input element directly
   */
  @tool({
    name: 'set_value',
    description: 'Set value of an input element directly (without typing)',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector of input element',
      },
      value: {
        type: 'string',
        description: 'Value to set',
      },
    },
  })
  async setValue(
    selector: string,
    value: string
  ): Promise<{ success: boolean; error?: string }> {
    try {
      const el = document.querySelector(selector) as HTMLInputElement;
      if (!el) {
        return { success: false, error: `Element not found: ${selector}` };
      }
      
      el.value = value;
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
      
      return { success: true };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  }

  /**
   * Get/set attribute of an element
   */
  @tool({
    name: 'attribute',
    description: 'Get or set an attribute of an element',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector of element',
      },
      name: {
        type: 'string',
        description: 'Attribute name',
      },
      value: {
        type: 'string',
        description: 'Value to set (omit to get)',
      },
    },
  })
  async attribute(
    selector: string,
    name: string,
    value?: string
  ): Promise<{ value: string | null; error?: string }> {
    try {
      const el = document.querySelector(selector);
      if (!el) {
        return { value: null, error: `Element not found: ${selector}` };
      }
      
      if (value !== undefined) {
        el.setAttribute(name, value);
        return { value };
      }
      
      return { value: el.getAttribute(name) };
    } catch (error) {
      return { value: null, error: (error as Error).message };
    }
  }

  // ============================================================================
  // JavaScript Evaluation
  // ============================================================================

  /**
   * Evaluate JavaScript in page context
   */
  @tool({
    name: 'evaluate',
    description: 'Evaluate JavaScript code in the page context',
    parameters: {
      script: {
        type: 'string',
        description: 'JavaScript code to evaluate',
      },
    },
  })
  async evaluate(script: string): Promise<{ result: unknown; error?: string }> {
    try {
      // Use Function constructor for safer eval
      const fn = new Function(`return (async () => { ${script} })()`);
      const result = await fn();
      return { result };
    } catch (error) {
      return { result: undefined, error: (error as Error).message };
    }
  }

  // ============================================================================
  // Network Monitoring
  // ============================================================================

  /**
   * Get network entries from Performance API
   */
  @tool({
    name: 'get_network_entries',
    description: 'Get network requests from Performance API',
    parameters: {
      entryType: {
        type: 'string',
        description: 'Filter by resource type: resource, navigation, paint, etc.',
      },
      limit: {
        type: 'number',
        description: 'Maximum entries to return (default: 100)',
      },
    },
  })
  async getNetworkEntries(
    entryType?: string,
    limit: number = 100
  ): Promise<{ entries: NetworkEntry[]; error?: string }> {
    try {
      let entries = performance.getEntriesByType('resource') as PerformanceResourceTiming[];
      
      if (entryType && entryType !== 'resource') {
        entries = performance.getEntriesByType(entryType) as PerformanceResourceTiming[];
      }
      
      const result: NetworkEntry[] = entries.slice(0, limit).map(entry => ({
        name: entry.name,
        entryType: entry.entryType,
        startTime: entry.startTime,
        duration: entry.duration,
        initiatorType: entry.initiatorType,
        transferSize: entry.transferSize,
        encodedBodySize: entry.encodedBodySize,
        decodedBodySize: entry.decodedBodySize,
        responseStatus: entry.responseStatus,
      }));
      
      return { entries: result };
    } catch (error) {
      return { entries: [], error: (error as Error).message };
    }
  }

  /**
   * Clear performance entries
   */
  @tool({
    name: 'clear_network_entries',
    description: 'Clear network performance entries',
    parameters: {},
  })
  async clearNetworkEntries(): Promise<{ success: boolean }> {
    performance.clearResourceTimings();
    return { success: true };
  }

  // ============================================================================
  // Clipboard
  // ============================================================================

  /**
   * Read from clipboard
   */
  @tool({
    name: 'clipboard_read',
    description: 'Read text from clipboard (requires permission)',
    parameters: {},
  })
  async clipboardRead(): Promise<{ text: string; error?: string }> {
    try {
      const text = await navigator.clipboard.readText();
      return { text };
    } catch (error) {
      return { text: '', error: (error as Error).message };
    }
  }

  /**
   * Write to clipboard
   */
  @tool({
    name: 'clipboard_write',
    description: 'Write text to clipboard',
    parameters: {
      text: {
        type: 'string',
        description: 'Text to copy to clipboard',
      },
    },
  })
  async clipboardWrite(text: string): Promise<{ success: boolean; error?: string }> {
    try {
      await navigator.clipboard.writeText(text);
      return { success: true };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  }

  // ============================================================================
  // Navigation
  // ============================================================================

  /**
   * Get current URL
   */
  @tool({
    name: 'get_url',
    description: 'Get the current page URL',
    parameters: {},
  })
  async getUrl(): Promise<{ url: string; origin: string; pathname: string; search: string; hash: string }> {
    return {
      url: window.location.href,
      origin: window.location.origin,
      pathname: window.location.pathname,
      search: window.location.search,
      hash: window.location.hash,
    };
  }

  /**
   * Navigate to URL
   */
  @tool({
    name: 'navigate',
    description: 'Navigate to a URL',
    parameters: {
      url: {
        type: 'string',
        description: 'URL to navigate to',
      },
    },
  })
  async navigate(url: string): Promise<{ success: boolean }> {
    window.location.href = url;
    return { success: true };
  }

  /**
   * Reload page
   */
  @tool({
    name: 'reload',
    description: 'Reload the current page',
    parameters: {
      hard: {
        type: 'boolean',
        description: 'Hard reload (bypass cache)',
      },
    },
  })
  async reload(_hard: boolean = false): Promise<{ success: boolean }> {
    // Note: Modern browsers don't support forced reload via JS
    // The hard parameter is for API compatibility
    window.location.reload();
    return { success: true };
  }

  /**
   * Go back in history
   */
  @tool({
    name: 'go_back',
    description: 'Go back in browser history',
    parameters: {},
  })
  async goBack(): Promise<{ success: boolean }> {
    window.history.back();
    return { success: true };
  }

  /**
   * Go forward in history
   */
  @tool({
    name: 'go_forward',
    description: 'Go forward in browser history',
    parameters: {},
  })
  async goForward(): Promise<{ success: boolean }> {
    window.history.forward();
    return { success: true };
  }

  // ============================================================================
  // Console
  // ============================================================================

  private consoleLogs: Array<{ level: string; args: unknown[]; timestamp: number }> = [];
  private originalConsole: Partial<Console> = {};
  private consoleIntercepted = false;

  /**
   * Start intercepting console logs
   */
  @tool({
    name: 'start_console_capture',
    description: 'Start capturing console logs',
    parameters: {},
  })
  async startConsoleCapture(): Promise<{ success: boolean }> {
    if (this.consoleIntercepted) {
      return { success: true };
    }
    
    this.consoleLogs = [];
    this.originalConsole = {
      log: console.log,
      warn: console.warn,
      error: console.error,
      info: console.info,
      debug: console.debug,
    };
    
    const captureLog = (level: string) => (...args: unknown[]) => {
      this.consoleLogs.push({ level, args, timestamp: Date.now() });
      (this.originalConsole as Record<string, Function>)[level]?.apply(console, args);
    };
    
    console.log = captureLog('log');
    console.warn = captureLog('warn');
    console.error = captureLog('error');
    console.info = captureLog('info');
    console.debug = captureLog('debug');
    
    this.consoleIntercepted = true;
    return { success: true };
  }

  /**
   * Get captured console logs
   */
  @tool({
    name: 'get_console_logs',
    description: 'Get captured console logs',
    parameters: {
      level: {
        type: 'string',
        description: 'Filter by level: log, warn, error, info, debug',
      },
      clear: {
        type: 'boolean',
        description: 'Clear logs after reading (default: false)',
      },
    },
  })
  async getConsoleLogs(
    level?: string,
    clear: boolean = false
  ): Promise<{ logs: Array<{ level: string; message: string; timestamp: number }> }> {
    let logs = this.consoleLogs;
    
    if (level) {
      logs = logs.filter(l => l.level === level);
    }
    
    const result = logs.map(l => ({
      level: l.level,
      message: l.args.map(a => typeof a === 'object' ? JSON.stringify(a) : String(a)).join(' '),
      timestamp: l.timestamp,
    }));
    
    if (clear) {
      this.consoleLogs = [];
    }
    
    return { logs: result };
  }

  /**
   * Stop intercepting console logs
   */
  @tool({
    name: 'stop_console_capture',
    description: 'Stop capturing console logs',
    parameters: {},
  })
  async stopConsoleCapture(): Promise<{ success: boolean }> {
    if (!this.consoleIntercepted) {
      return { success: true };
    }
    
    if (this.originalConsole.log) console.log = this.originalConsole.log;
    if (this.originalConsole.warn) console.warn = this.originalConsole.warn;
    if (this.originalConsole.error) console.error = this.originalConsole.error;
    if (this.originalConsole.info) console.info = this.originalConsole.info;
    if (this.originalConsole.debug) console.debug = this.originalConsole.debug;
    
    this.consoleIntercepted = false;
    return { success: true };
  }

  // ============================================================================
  // Helpers
  // ============================================================================

  /**
   * Get element info helper
   */
  private getElementInfo(el: HTMLElement, includeComputedStyle: boolean): ElementInfo {
    const rect = el.getBoundingClientRect();
    const style = getComputedStyle(el);
    
    const info: ElementInfo = {
      tag: el.tagName.toLowerCase(),
      id: el.id || undefined,
      className: el.className || undefined,
      text: el.textContent?.trim().slice(0, 200) || undefined,
      visible: rect.width > 0 && rect.height > 0 && 
               style.visibility !== 'hidden' && 
               style.display !== 'none',
      rect: {
        x: Math.round(rect.x),
        y: Math.round(rect.y),
        width: Math.round(rect.width),
        height: Math.round(rect.height),
      },
      attributes: {},
    };
    
    // Add specific attributes
    if (el instanceof HTMLInputElement) {
      info.value = el.value;
      info.disabled = el.disabled;
      if (el.type === 'checkbox' || el.type === 'radio') {
        info.checked = el.checked;
      }
    }
    
    if (el instanceof HTMLAnchorElement) {
      info.href = el.href;
    }
    
    if (el instanceof HTMLImageElement) {
      info.src = el.src;
    }
    
    // Collect all attributes
    for (const attr of el.attributes) {
      info.attributes[attr.name] = attr.value;
    }
    
    // Include computed style if requested
    if (includeComputedStyle) {
      info.computedStyle = {
        display: style.display,
        visibility: style.visibility,
        opacity: style.opacity,
        position: style.position,
        zIndex: style.zIndex,
        color: style.color,
        backgroundColor: style.backgroundColor,
        fontSize: style.fontSize,
      };
    }
    
    return info;
  }

  /**
   * Sleep helper
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Convert key name to code
   */
  private keyToCode(key: string): string {
    const keyMap: Record<string, string> = {
      'Enter': 'Enter',
      'Tab': 'Tab',
      'Escape': 'Escape',
      'Backspace': 'Backspace',
      'Delete': 'Delete',
      'ArrowUp': 'ArrowUp',
      'ArrowDown': 'ArrowDown',
      'ArrowLeft': 'ArrowLeft',
      'ArrowRight': 'ArrowRight',
      'Home': 'Home',
      'End': 'End',
      'PageUp': 'PageUp',
      'PageDown': 'PageDown',
      'Space': 'Space',
      ' ': 'Space',
    };
    
    return keyMap[key] || `Key${key.toUpperCase()}`;
  }
}
