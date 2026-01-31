"""
Browser Automation Skill

Provides tools for browser automation with Set-of-Mark (SoM) prompting support.
Works with Playwright for actual browser control.

Set-of-Mark (SoM) prompting overlays numbered labels and bounding boxes on
interactive elements, improving vision-language model understanding of web pages.
"""

import asyncio
import base64
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field

from ...base import Skill
from ....tools.decorators import tool

# Optional Playwright import
try:
    from playwright.async_api import async_playwright, Page, Browser, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Page = Any
    Browser = Any
    BrowserContext = Any


@dataclass
class ElementInfo:
    """Information about a DOM element."""
    tag: str
    label: Optional[str] = None
    text: Optional[str] = None
    id: Optional[str] = None
    class_name: Optional[str] = None
    href: Optional[str] = None
    value: Optional[str] = None
    disabled: bool = False
    visible: bool = True
    rect: Dict[str, float] = field(default_factory=dict)
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass  
class ScreenshotResult:
    """Result of a screenshot operation."""
    data: bytes
    width: int
    height: int
    format: str = 'png'
    
    @property
    def data_url(self) -> str:
        """Get base64 data URL."""
        b64 = base64.b64encode(self.data).decode('utf-8')
        return f"data:image/{self.format};base64,{b64}"


class BrowserAutomationSkill(Skill):
    """
    Browser Automation Skill with Set-of-Mark (SoM) Prompting.
    
    Provides cutting-edge browser control capabilities:
    - DOM element queries and manipulation
    - Click, type, scroll interactions
    - Screenshots with element marking (SoM prompting)
    - Accessibility tree access
    - Network monitoring
    
    Requires Playwright: pip install playwright && playwright install
    
    Example:
        ```python
        browser_skill = BrowserAutomationSkill()
        agent.add_skill(browser_skill)
        
        # Agent can then use browser tools
        # - browser_goto: Navigate to URL
        # - browser_click: Click elements
        # - browser_type: Type text
        # - browser_marked_screenshot: Screenshot with SoM labels
        ```
    """
    
    # JavaScript for marking elements (SoM prompting)
    MARK_ELEMENTS_JS = '''
    (options) => {
        const {
            selector = 'a[href], button, input:not([type="hidden"]), select, textarea, summary, [role="button"], [tabindex]:not([tabindex="-1"]), [onclick]',
            showBoundingBoxes = true,
            viewportOnly = false,
            markStyle = { backgroundColor: '#ff0000', color: '#ffffff', fontSize: '12px', padding: '2px 6px' },
            boundingBoxStyle = { outline: '2px solid #ff0000', backgroundColor: 'rgba(255, 0, 0, 0.1)' }
        } = options || {};
        
        // Remove existing marks
        document.querySelectorAll('[data-webagents-mark]').forEach(el => el.remove());
        document.querySelectorAll('[data-mark-label]').forEach(el => el.removeAttribute('data-mark-label'));
        
        // Create container
        const container = document.createElement('div');
        container.setAttribute('data-webagents-mark', 'container');
        container.style.cssText = 'position: absolute; top: 0; left: 0; width: 0; height: 0; overflow: visible; pointer-events: none; z-index: 999999;';
        document.body.appendChild(container);
        
        const elements = document.querySelectorAll(selector);
        const results = [];
        let index = 0;
        
        for (const el of elements) {
            const rect = el.getBoundingClientRect();
            const style = getComputedStyle(el);
            
            // Skip hidden elements
            if (rect.width === 0 || rect.height === 0 || style.visibility === 'hidden' || style.display === 'none') {
                continue;
            }
            
            // Skip elements outside viewport if viewportOnly
            if (viewportOnly) {
                if (rect.bottom < 0 || rect.top > window.innerHeight || rect.right < 0 || rect.left > window.innerWidth) {
                    continue;
                }
            }
            
            const label = String(index);
            
            // Create bounding box
            if (showBoundingBoxes) {
                const bbox = document.createElement('div');
                bbox.setAttribute('data-webagents-mark', 'bbox');
                bbox.style.cssText = `
                    position: fixed;
                    top: ${rect.top}px;
                    left: ${rect.left}px;
                    width: ${rect.width}px;
                    height: ${rect.height}px;
                    outline: ${boundingBoxStyle.outline};
                    background-color: ${boundingBoxStyle.backgroundColor};
                    pointer-events: none;
                    box-sizing: border-box;
                `;
                container.appendChild(bbox);
            }
            
            // Create label
            const mark = document.createElement('div');
            mark.setAttribute('data-webagents-mark', 'label');
            mark.textContent = label;
            mark.style.cssText = `
                position: fixed;
                top: ${Math.max(0, rect.top - 20)}px;
                left: ${rect.left}px;
                background-color: ${markStyle.backgroundColor};
                color: ${markStyle.color};
                font-size: ${markStyle.fontSize};
                font-weight: bold;
                padding: ${markStyle.padding};
                border-radius: 3px;
                font-family: monospace;
                pointer-events: none;
                z-index: 1000000;
            `;
            container.appendChild(mark);
            
            // Set data attribute
            el.setAttribute('data-mark-label', label);
            
            results.push({
                label,
                tag: el.tagName.toLowerCase(),
                text: el.textContent?.trim().slice(0, 50) || null,
                id: el.id || null,
                href: el.href || null,
                rect: {
                    x: Math.round(rect.x),
                    y: Math.round(rect.y),
                    width: Math.round(rect.width),
                    height: Math.round(rect.height)
                }
            });
            
            index++;
        }
        
        return results;
    }
    '''
    
    UNMARK_ELEMENTS_JS = '''
    () => {
        document.querySelectorAll('[data-webagents-mark]').forEach(el => el.remove());
        document.querySelectorAll('[data-mark-label]').forEach(el => el.removeAttribute('data-mark-label'));
        return true;
    }
    '''
    
    def __init__(
        self,
        headless: bool = True,
        browser_type: Literal['chromium', 'firefox', 'webkit'] = 'chromium',
        viewport: Optional[Dict[str, int]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.headless = headless
        self.browser_type = browser_type
        self.viewport = viewport or {'width': 1280, 'height': 720}
        
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
    
    async def initialize(self, agent) -> None:
        """Initialize the skill and register tools."""
        await super().initialize(agent)
        
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright is required for BrowserAutomationSkill. "
                "Install with: pip install playwright && playwright install"
            )
        
        # Register tools
        self.register_tool(self.browser_launch)
        self.register_tool(self.browser_close)
        self.register_tool(self.browser_goto)
        self.register_tool(self.browser_click)
        self.register_tool(self.browser_type)
        self.register_tool(self.browser_press_key)
        self.register_tool(self.browser_screenshot)
        self.register_tool(self.browser_mark_elements)
        self.register_tool(self.browser_unmark_elements)
        self.register_tool(self.browser_marked_screenshot)
        self.register_tool(self.browser_click_marked)
        self.register_tool(self.browser_type_marked)
        self.register_tool(self.browser_get_text)
        self.register_tool(self.browser_get_html)
        self.register_tool(self.browser_evaluate)
        self.register_tool(self.browser_wait_for)
        self.register_tool(self.browser_scroll)
        self.register_tool(self.browser_get_url)
    
    async def cleanup(self) -> None:
        """Cleanup browser resources."""
        await self._close_browser()
    
    async def _ensure_browser(self) -> Page:
        """Ensure browser is launched and return page."""
        if not self._page:
            await self._launch_browser()
        return self._page
    
    async def _launch_browser(self) -> None:
        """Launch browser instance."""
        self._playwright = await async_playwright().start()
        
        browser_launcher = getattr(self._playwright, self.browser_type)
        self._browser = await browser_launcher.launch(headless=self.headless)
        self._context = await self._browser.new_context(viewport=self.viewport)
        self._page = await self._context.new_page()
    
    async def _close_browser(self) -> None:
        """Close browser instance."""
        if self._page:
            await self._page.close()
            self._page = None
        if self._context:
            await self._context.close()
            self._context = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
    
    # ============================================================================
    # Browser Control Tools
    # ============================================================================
    
    @tool(description="Launch the browser")
    async def browser_launch(self, headless: Optional[bool] = None) -> Dict[str, Any]:
        """Launch a browser instance."""
        if headless is not None:
            self.headless = headless
        await self._launch_browser()
        return {'success': True, 'browser': self.browser_type, 'headless': self.headless}
    
    @tool(description="Close the browser")
    async def browser_close(self) -> Dict[str, Any]:
        """Close the browser instance."""
        await self._close_browser()
        return {'success': True}
    
    @tool(description="Navigate to a URL")
    async def browser_goto(self, url: str, wait_until: str = 'load') -> Dict[str, Any]:
        """
        Navigate to a URL.
        
        Args:
            url: URL to navigate to
            wait_until: Wait condition: load, domcontentloaded, networkidle
        """
        page = await self._ensure_browser()
        await page.goto(url, wait_until=wait_until)
        return {'success': True, 'url': page.url}
    
    @tool(description="Click on an element by CSS selector")
    async def browser_click(self, selector: str) -> Dict[str, Any]:
        """Click on an element."""
        page = await self._ensure_browser()
        try:
            await page.click(selector)
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @tool(description="Type text into an element")
    async def browser_type(
        self,
        selector: str,
        text: str,
        clear: bool = True
    ) -> Dict[str, Any]:
        """Type text into an input element."""
        page = await self._ensure_browser()
        try:
            if clear:
                await page.fill(selector, text)
            else:
                await page.type(selector, text)
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @tool(description="Press a keyboard key")
    async def browser_press_key(self, key: str, selector: Optional[str] = None) -> Dict[str, Any]:
        """Press a keyboard key (Enter, Tab, Escape, etc.)."""
        page = await self._ensure_browser()
        try:
            if selector:
                await page.press(selector, key)
            else:
                await page.keyboard.press(key)
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @tool(description="Take a screenshot of the page")
    async def browser_screenshot(
        self,
        selector: Optional[str] = None,
        full_page: bool = False
    ) -> Dict[str, Any]:
        """Take a screenshot."""
        page = await self._ensure_browser()
        try:
            if selector:
                element = await page.query_selector(selector)
                if element:
                    data = await element.screenshot(type='png')
                else:
                    return {'success': False, 'error': f'Element not found: {selector}'}
            else:
                data = await page.screenshot(type='png', full_page=full_page)
            
            b64 = base64.b64encode(data).decode('utf-8')
            return {
                'success': True,
                'data_url': f'data:image/png;base64,{b64}',
                'size': len(data)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # ============================================================================
    # Set-of-Mark (SoM) Tools
    # ============================================================================
    
    @tool(description="Mark interactive elements with numbered labels for vision AI (SoM prompting)")
    async def browser_mark_elements(
        self,
        selector: Optional[str] = None,
        viewport_only: bool = True
    ) -> Dict[str, Any]:
        """
        Mark interactive elements with numbered labels and bounding boxes.
        
        This implements Set-of-Mark (SoM) prompting to improve vision-language
        model understanding of web page elements.
        
        Args:
            selector: CSS selector for elements to mark (default: interactive elements)
            viewport_only: Only mark elements visible in viewport
        """
        page = await self._ensure_browser()
        try:
            options = {
                'viewportOnly': viewport_only
            }
            if selector:
                options['selector'] = selector
            
            results = await page.evaluate(self.MARK_ELEMENTS_JS, options)
            
            return {
                'success': True,
                'marked': len(results),
                'elements': results
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @tool(description="Remove element marks from the page")
    async def browser_unmark_elements(self) -> Dict[str, Any]:
        """Remove all element marks from the page."""
        page = await self._ensure_browser()
        try:
            await page.evaluate(self.UNMARK_ELEMENTS_JS)
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @tool(description="Mark elements and take screenshot for vision AI analysis")
    async def browser_marked_screenshot(
        self,
        selector: Optional[str] = None,
        viewport_only: bool = True
    ) -> Dict[str, Any]:
        """
        Mark interactive elements and take a screenshot for vision AI.
        
        This is the primary tool for SoM prompting - it:
        1. Marks all interactive elements with numbered labels
        2. Takes a screenshot
        3. Returns the screenshot and element list for LLM prompting
        
        Args:
            selector: CSS selector for elements to mark
            viewport_only: Only mark visible elements
            
        Returns:
            Screenshot data URL, element list, and suggested prompt
        """
        page = await self._ensure_browser()
        try:
            # Mark elements
            options = {'viewportOnly': viewport_only}
            if selector:
                options['selector'] = selector
            
            elements = await page.evaluate(self.MARK_ELEMENTS_JS, options)
            
            # Take screenshot
            screenshot_data = await page.screenshot(type='png')
            b64 = base64.b64encode(screenshot_data).decode('utf-8')
            data_url = f'data:image/png;base64,{b64}'
            
            # Generate prompt helper
            element_list = '\n'.join([
                f"  {e['label']}: {e['tag']}" + (f' "{e["text"]}"' if e.get('text') else '')
                for e in elements
            ])
            
            prompt = f"""The following screenshot shows a web page with interactive elements marked.
Each element has a red numbered label and bounding box.
Use the label numbers to refer to specific elements.

Marked elements:
{element_list}

When asked to interact with elements, respond with the label number."""
            
            return {
                'success': True,
                'screenshot': data_url,
                'elements': elements,
                'prompt': prompt
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @tool(description="Click a marked element by its label number")
    async def browser_click_marked(self, label: str) -> Dict[str, Any]:
        """Click a marked element by its label (e.g., '0', '5', '12')."""
        page = await self._ensure_browser()
        try:
            selector = f'[data-mark-label="{label}"]'
            await page.click(selector)
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @tool(description="Type into a marked element by its label number")
    async def browser_type_marked(
        self,
        label: str,
        text: str,
        clear: bool = True
    ) -> Dict[str, Any]:
        """Type text into a marked element by its label."""
        page = await self._ensure_browser()
        try:
            selector = f'[data-mark-label="{label}"]'
            if clear:
                await page.fill(selector, text)
            else:
                await page.type(selector, text)
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # ============================================================================
    # Content Tools
    # ============================================================================
    
    @tool(description="Get text content of an element")
    async def browser_get_text(self, selector: str) -> Dict[str, Any]:
        """Get text content of an element."""
        page = await self._ensure_browser()
        try:
            text = await page.text_content(selector)
            return {'text': text or '', 'success': True}
        except Exception as e:
            return {'text': '', 'success': False, 'error': str(e)}
    
    @tool(description="Get HTML content of an element or page")
    async def browser_get_html(
        self,
        selector: Optional[str] = None,
        outer: bool = True
    ) -> Dict[str, Any]:
        """Get HTML content."""
        page = await self._ensure_browser()
        try:
            if selector:
                if outer:
                    html = await page.eval_on_selector(selector, 'el => el.outerHTML')
                else:
                    html = await page.eval_on_selector(selector, 'el => el.innerHTML')
            else:
                html = await page.content()
            return {'html': html, 'success': True}
        except Exception as e:
            return {'html': '', 'success': False, 'error': str(e)}
    
    @tool(description="Evaluate JavaScript in page context")
    async def browser_evaluate(self, script: str) -> Dict[str, Any]:
        """Evaluate JavaScript and return result."""
        page = await self._ensure_browser()
        try:
            result = await page.evaluate(script)
            return {'result': result, 'success': True}
        except Exception as e:
            return {'result': None, 'success': False, 'error': str(e)}
    
    @tool(description="Wait for an element to appear")
    async def browser_wait_for(
        self,
        selector: str,
        timeout: int = 10000,
        state: str = 'visible'
    ) -> Dict[str, Any]:
        """
        Wait for an element to appear.
        
        Args:
            selector: CSS selector to wait for
            timeout: Timeout in milliseconds
            state: Wait state: visible, hidden, attached, detached
        """
        page = await self._ensure_browser()
        try:
            await page.wait_for_selector(selector, timeout=timeout, state=state)
            return {'found': True, 'success': True}
        except Exception as e:
            return {'found': False, 'success': False, 'error': str(e)}
    
    @tool(description="Scroll the page or to an element")
    async def browser_scroll(
        self,
        selector: Optional[str] = None,
        x: Optional[int] = None,
        y: Optional[int] = None
    ) -> Dict[str, Any]:
        """Scroll to element or position."""
        page = await self._ensure_browser()
        try:
            if selector:
                element = await page.query_selector(selector)
                if element:
                    await element.scroll_into_view_if_needed()
            elif x is not None or y is not None:
                await page.evaluate(f'window.scrollTo({x or 0}, {y or 0})')
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @tool(description="Get current page URL")
    async def browser_get_url(self) -> Dict[str, Any]:
        """Get the current page URL."""
        page = await self._ensure_browser()
        url = page.url
        return {
            'url': url,
            'success': True
        }
