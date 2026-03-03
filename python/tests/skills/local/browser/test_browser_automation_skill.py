"""
BrowserAutomationSkill Unit Tests

Note: Full browser automation tests require Playwright installed.
These tests verify the skill structure and configuration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from webagents.agents.skills.local.browser.automation import (
    BrowserAutomationSkill,
    ElementInfo,
    ScreenshotResult,
    PLAYWRIGHT_AVAILABLE,
)


class TestBrowserAutomationSkillInit:
    """Test BrowserAutomationSkill initialization."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        skill = BrowserAutomationSkill()
        
        assert skill.headless is True
        assert skill.browser_type == 'chromium'
        assert skill.viewport == {'width': 1280, 'height': 720}
    
    def test_custom_configuration(self):
        """Test custom configuration."""
        skill = BrowserAutomationSkill(
            headless=False,
            browser_type='firefox',
            viewport={'width': 1920, 'height': 1080},
        )
        
        assert skill.headless is False
        assert skill.browser_type == 'firefox'
        assert skill.viewport == {'width': 1920, 'height': 1080}


class TestMarkElementsJS:
    """Test the JavaScript code for marking elements."""
    
    def test_mark_elements_js_exists(self):
        """Test that mark elements JS is defined."""
        assert hasattr(BrowserAutomationSkill, 'MARK_ELEMENTS_JS')
        assert 'data-mark-label' in BrowserAutomationSkill.MARK_ELEMENTS_JS
    
    def test_unmark_elements_js_exists(self):
        """Test that unmark elements JS is defined."""
        assert hasattr(BrowserAutomationSkill, 'UNMARK_ELEMENTS_JS')
        assert 'data-webagents-mark' in BrowserAutomationSkill.UNMARK_ELEMENTS_JS
    
    def test_mark_elements_js_includes_interactive_elements(self):
        """Test that mark JS targets interactive elements."""
        js = BrowserAutomationSkill.MARK_ELEMENTS_JS
        
        assert 'a[href]' in js
        assert 'button' in js
        assert 'input' in js
        assert 'select' in js
        assert 'textarea' in js
        assert '[role="button"]' in js
        assert '[tabindex]' in js
    
    def test_mark_elements_js_creates_visual_markers(self):
        """Test that mark JS creates visual elements."""
        js = BrowserAutomationSkill.MARK_ELEMENTS_JS
        
        # Should create bounding box
        assert 'boundingBoxStyle' in js
        assert 'outline' in js
        
        # Should create label
        assert 'markStyle' in js
        assert 'backgroundColor' in js
        assert 'fontSize' in js


class TestDataClasses:
    """Test data classes."""
    
    def test_element_info(self):
        """Test ElementInfo dataclass."""
        info = ElementInfo(
            tag='button',
            label='0',
            text='Submit',
            id='submit-btn',
            class_name='btn primary',
        )
        
        assert info.tag == 'button'
        assert info.label == '0'
        assert info.text == 'Submit'
        assert info.id == 'submit-btn'
        assert info.class_name == 'btn primary'
        assert info.visible is True  # default
        assert info.disabled is False  # default
    
    def test_screenshot_result(self):
        """Test ScreenshotResult dataclass."""
        data = b'PNG binary data here'
        result = ScreenshotResult(
            data=data,
            width=1280,
            height=720,
            format='png',
        )
        
        assert result.data == data
        assert result.width == 1280
        assert result.height == 720
        assert result.format == 'png'
    
    def test_screenshot_data_url(self):
        """Test ScreenshotResult data_url property."""
        import base64
        
        data = b'test data'
        result = ScreenshotResult(data=data, width=100, height=100)
        
        expected_b64 = base64.b64encode(data).decode('utf-8')
        assert result.data_url == f'data:image/png;base64,{expected_b64}'


class TestBrowserMethods:
    """Test that browser methods exist."""
    
    @pytest.fixture
    def skill(self):
        """Create a skill instance."""
        return BrowserAutomationSkill()
    
    def test_has_navigation_methods(self, skill):
        """Test navigation methods exist."""
        assert hasattr(skill, 'browser_launch')
        assert hasattr(skill, 'browser_close')
        assert hasattr(skill, 'browser_goto')
    
    def test_has_interaction_methods(self, skill):
        """Test interaction methods exist."""
        assert hasattr(skill, 'browser_click')
        assert hasattr(skill, 'browser_type')
        assert hasattr(skill, 'browser_press_key')
        assert hasattr(skill, 'browser_scroll')
    
    def test_has_som_methods(self, skill):
        """Test Set-of-Mark (SoM) methods exist."""
        assert hasattr(skill, 'browser_mark_elements')
        assert hasattr(skill, 'browser_unmark_elements')
        assert hasattr(skill, 'browser_marked_screenshot')
        assert hasattr(skill, 'browser_click_marked')
        assert hasattr(skill, 'browser_type_marked')
    
    def test_has_content_methods(self, skill):
        """Test content methods exist."""
        assert hasattr(skill, 'browser_screenshot')
        assert hasattr(skill, 'browser_get_text')
        assert hasattr(skill, 'browser_get_html')
        assert hasattr(skill, 'browser_evaluate')
        assert hasattr(skill, 'browser_wait_for')
        assert hasattr(skill, 'browser_get_url')


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
class TestPlaywrightIntegration:
    """Tests requiring Playwright installation."""
    
    @pytest.mark.asyncio
    async def test_browser_lifecycle(self):
        """Test browser launch and close."""
        skill = BrowserAutomationSkill(headless=True)
        
        # Mock agent for initialization
        mock_agent = MagicMock()
        mock_agent.register_tool = MagicMock()
        
        await skill.initialize(mock_agent)
        
        # Launch browser
        result = await skill.browser_launch()
        assert result['success'] is True
        
        # Close browser
        result = await skill.browser_close()
        assert result['success'] is True
        
        await skill.cleanup()


class TestSkillConfiguration:
    """Test skill configuration options."""
    
    def test_supported_browser_types(self):
        """Test that common browser types are supported."""
        for browser in ['chromium', 'firefox', 'webkit']:
            skill = BrowserAutomationSkill(browser_type=browser)
            assert skill.browser_type == browser
    
    def test_headless_mode(self):
        """Test headless mode configuration."""
        headless_skill = BrowserAutomationSkill(headless=True)
        assert headless_skill.headless is True
        
        headed_skill = BrowserAutomationSkill(headless=False)
        assert headed_skill.headless is False
    
    def test_viewport_configuration(self):
        """Test viewport size configuration."""
        # Default viewport
        default_skill = BrowserAutomationSkill()
        assert default_skill.viewport['width'] == 1280
        assert default_skill.viewport['height'] == 720
        
        # Custom viewport
        custom_skill = BrowserAutomationSkill(
            viewport={'width': 1920, 'height': 1080}
        )
        assert custom_skill.viewport['width'] == 1920
        assert custom_skill.viewport['height'] == 1080


class TestPlaywrightAvailability:
    """Test Playwright availability detection."""
    
    def test_playwright_available_flag(self):
        """Test that PLAYWRIGHT_AVAILABLE is a boolean."""
        assert isinstance(PLAYWRIGHT_AVAILABLE, bool)
