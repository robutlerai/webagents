"""
Tests for XAISkill (xAI / Grok LLM integration).

Covers initialization, model catalog, usage record formatting,
cached-token tracking, and reasoning-model parameter injection.
"""

import pytest
from unittest.mock import Mock, patch

from webagents.agents.skills.core.llm.xai.skill import XAISkill, XAIModelConfig


@pytest.fixture
def xai_skill():
    """XAISkill with a test API key (no real network calls)."""
    with patch.dict('os.environ', {'XAI_API_KEY': 'xai-test-key'}):
        skill = XAISkill({'api_key': 'xai-test-key'})
    return skill


class TestXAISkillInitialization:

    def test_default_model(self):
        skill = XAISkill({'api_key': 'k'})
        assert skill.model == 'grok-3'

    def test_custom_model(self):
        skill = XAISkill({'api_key': 'k', 'model': 'grok-4-0709'})
        assert skill.model == 'grok-4-0709'

    def test_default_temperature(self):
        skill = XAISkill({'api_key': 'k'})
        assert skill.temperature == 0.7

    def test_base_url(self):
        skill = XAISkill({'api_key': 'k'})
        assert skill.base_url == 'https://api.x.ai/v1'

    def test_custom_base_url(self):
        skill = XAISkill({'api_key': 'k', 'base_url': 'https://custom.api/v1'})
        assert skill.base_url == 'https://custom.api/v1'

    def test_requires_openai_sdk(self):
        with patch('webagents.agents.skills.core.llm.xai.skill.OPENAI_AVAILABLE', False):
            with pytest.raises(ImportError, match="OpenAI SDK"):
                XAISkill({'api_key': 'k'})


class TestXAIDefaultModels:

    def test_catalog_contains_grok3(self):
        assert 'grok-3' in XAISkill.DEFAULT_MODELS

    def test_catalog_contains_grok3_mini(self):
        assert 'grok-3-mini' in XAISkill.DEFAULT_MODELS

    def test_catalog_contains_grok4(self):
        assert 'grok-4-0709' in XAISkill.DEFAULT_MODELS

    def test_catalog_contains_grok4_fast_reasoning(self):
        assert 'grok-4-fast-reasoning' in XAISkill.DEFAULT_MODELS

    def test_catalog_contains_grok4_fast_non_reasoning(self):
        assert 'grok-4-fast-non-reasoning' in XAISkill.DEFAULT_MODELS

    def test_catalog_contains_grok_code_fast(self):
        assert 'grok-code-fast-1' in XAISkill.DEFAULT_MODELS

    def test_grok3_is_not_reasoning(self):
        assert XAISkill.DEFAULT_MODELS['grok-3'].is_reasoning is False

    def test_grok3_mini_is_reasoning(self):
        assert XAISkill.DEFAULT_MODELS['grok-3-mini'].is_reasoning is True

    def test_grok4_supports_vision(self):
        assert XAISkill.DEFAULT_MODELS['grok-4-0709'].supports_vision is True

    def test_grok4_fast_non_reasoning_is_not_reasoning(self):
        assert XAISkill.DEFAULT_MODELS['grok-4-fast-non-reasoning'].is_reasoning is False


class TestXAIUsageRecordFormatting:

    def test_append_usage_record_xai_prefix(self, xai_skill):
        """Usage records must prefix model with 'xai/'."""
        mock_context = Mock()
        mock_context.usage = []
        xai_skill.agent = Mock()
        xai_skill.agent.context = mock_context

        xai_skill._append_usage_record(
            {'prompt_tokens': 100, 'completion_tokens': 50}, 'grok-3'
        )

        assert len(mock_context.usage) == 1
        record = mock_context.usage[0]
        assert record['model'] == 'xai/grok-3'
        assert record['type'] == 'llm'
        assert record['prompt_tokens'] == 100
        assert record['completion_tokens'] == 50

    def test_append_usage_record_creates_usage_list(self, xai_skill):
        """If context.usage doesn't exist, it is created."""
        mock_context = Mock(spec=[])
        xai_skill.agent = Mock()
        xai_skill.agent.context = mock_context

        xai_skill._append_usage_record(
            {'prompt_tokens': 10, 'completion_tokens': 5}, 'grok-3-mini'
        )

        assert hasattr(mock_context, 'usage')
        assert len(mock_context.usage) == 1
        assert mock_context.usage[0]['model'] == 'xai/grok-3-mini'

    def test_append_usage_record_noop_without_agent(self, xai_skill):
        """No-op when agent is None (pre-initialization)."""
        xai_skill.agent = None
        xai_skill._append_usage_record({'prompt_tokens': 10, 'completion_tokens': 5}, 'grok-3')

    def test_append_usage_record_noop_without_context(self, xai_skill):
        """No-op when agent has no context attribute."""
        xai_skill.agent = Mock(spec=[])
        xai_skill._append_usage_record({'prompt_tokens': 10, 'completion_tokens': 5}, 'grok-3')


class TestXAICachedTokenTracking:

    def test_cached_tokens_included(self, xai_skill):
        """cached_read_tokens is set when prompt_tokens_details.cached_tokens exists."""
        mock_context = Mock()
        mock_context.usage = []
        xai_skill.agent = Mock()
        xai_skill.agent.context = mock_context

        xai_skill._append_usage_record(
            {
                'prompt_tokens': 200,
                'completion_tokens': 50,
                'prompt_tokens_details': {'cached_tokens': 150},
            },
            'grok-3',
        )

        record = mock_context.usage[0]
        assert record['cached_read_tokens'] == 150

    def test_no_cached_tokens_when_absent(self, xai_skill):
        """cached_read_tokens is omitted when there are no cached tokens."""
        mock_context = Mock()
        mock_context.usage = []
        xai_skill.agent = Mock()
        xai_skill.agent.context = mock_context

        xai_skill._append_usage_record(
            {'prompt_tokens': 200, 'completion_tokens': 50}, 'grok-3'
        )

        record = mock_context.usage[0]
        assert 'cached_read_tokens' not in record

    def test_cached_tokens_zero_is_omitted(self, xai_skill):
        """cached_read_tokens is omitted when cached_tokens is 0."""
        mock_context = Mock()
        mock_context.usage = []
        xai_skill.agent = Mock()
        xai_skill.agent.context = mock_context

        xai_skill._append_usage_record(
            {
                'prompt_tokens': 200,
                'completion_tokens': 50,
                'prompt_tokens_details': {'cached_tokens': 0},
            },
            'grok-3',
        )

        record = mock_context.usage[0]
        assert 'cached_read_tokens' not in record


class TestXAIReasoningModels:

    def test_is_reasoning_grok3_mini(self, xai_skill):
        assert xai_skill._is_reasoning_model('grok-3-mini') is True

    def test_is_reasoning_grok4_fast_reasoning(self, xai_skill):
        assert xai_skill._is_reasoning_model('grok-4-fast-reasoning') is True

    def test_is_not_reasoning_grok3(self, xai_skill):
        assert xai_skill._is_reasoning_model('grok-3') is False

    def test_is_not_reasoning_grok4_fast_non_reasoning(self, xai_skill):
        assert xai_skill._is_reasoning_model('grok-4-fast-non-reasoning') is False

    def test_prepare_params_reasoning_effort(self, xai_skill):
        """Reasoning models get reasoning_effort when thinking is enabled."""
        xai_skill.thinking_config = {'enabled': True, 'effort': 'high'}
        params = xai_skill._prepare_params(
            messages=[{'role': 'user', 'content': 'hi'}],
            model='grok-3-mini',
            tools=None,
            stream=False,
            reasoning_effort='high',
        )
        assert params['reasoning_effort'] == 'high'

    def test_prepare_params_no_reasoning_effort_for_non_reasoning(self, xai_skill):
        """Non-reasoning models do NOT get reasoning_effort."""
        params = xai_skill._prepare_params(
            messages=[{'role': 'user', 'content': 'hi'}],
            model='grok-3',
            tools=None,
            stream=False,
        )
        assert 'reasoning_effort' not in params

    def test_prepare_params_temperature_for_non_reasoning(self, xai_skill):
        """Non-reasoning models get temperature; reasoning models do not."""
        params = xai_skill._prepare_params(
            messages=[{'role': 'user', 'content': 'hi'}],
            model='grok-3',
            tools=None,
            stream=False,
        )
        assert 'temperature' in params

    def test_prepare_params_no_temperature_for_reasoning(self, xai_skill):
        """Reasoning models should not include temperature."""
        params = xai_skill._prepare_params(
            messages=[{'role': 'user', 'content': 'hi'}],
            model='grok-3-mini',
            tools=None,
            stream=False,
        )
        assert 'temperature' not in params

    def test_prepare_params_stream_options(self, xai_skill):
        """Streaming requests include stream_options with include_usage."""
        params = xai_skill._prepare_params(
            messages=[{'role': 'user', 'content': 'hi'}],
            model='grok-3',
            tools=None,
            stream=True,
        )
        assert params['stream_options'] == {'include_usage': True}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
