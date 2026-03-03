"""
Tests for FireworksAISkill (Fireworks AI LLM integration).

Covers initialization, model catalog, usage record formatting,
and the accounts/fireworks/models/ prefix in _prepare_params.
"""

import pytest
from unittest.mock import Mock, patch

from webagents.agents.skills.core.llm.fireworks.skill import FireworksAISkill, FireworksModelConfig


@pytest.fixture
def fireworks_skill():
    """FireworksAISkill with a test API key (no real network calls)."""
    with patch.dict('os.environ', {'FIREWORKS_API_KEY': 'fw-test-key'}):
        skill = FireworksAISkill({'api_key': 'fw-test-key'})
    return skill


class TestFireworksSkillInitialization:

    def test_default_model(self):
        skill = FireworksAISkill({'api_key': 'k'})
        assert skill.model == 'deepseek-v3p2'

    def test_custom_model(self):
        skill = FireworksAISkill({'api_key': 'k', 'model': 'glm-5'})
        assert skill.model == 'glm-5'

    def test_default_temperature(self):
        skill = FireworksAISkill({'api_key': 'k'})
        assert skill.temperature == 0.7

    def test_base_url(self):
        assert FireworksAISkill.BASE_URL == 'https://api.fireworks.ai/inference/v1'

    def test_instance_base_url(self):
        skill = FireworksAISkill({'api_key': 'k'})
        assert skill.base_url == 'https://api.fireworks.ai/inference/v1'

    def test_custom_base_url(self):
        skill = FireworksAISkill({'api_key': 'k', 'base_url': 'https://custom/v1'})
        assert skill.base_url == 'https://custom/v1'

    def test_api_key_from_config(self):
        skill = FireworksAISkill({'api_key': 'my-key'})
        assert skill.api_key == 'my-key'

    def test_api_key_from_env(self):
        with patch.dict('os.environ', {'FIREWORKS_API_KEY': 'env-key'}):
            skill = FireworksAISkill({})
            assert skill.api_key == 'env-key'

    def test_requires_openai_sdk(self):
        with patch('webagents.agents.skills.core.llm.fireworks.skill.OPENAI_AVAILABLE', False):
            with pytest.raises(ImportError, match="OpenAI SDK"):
                FireworksAISkill({'api_key': 'k'})


class TestFireworksDefaultModels:

    def test_catalog_contains_deepseek_v3p2(self):
        assert 'deepseek-v3p2' in FireworksAISkill.DEFAULT_MODELS

    def test_catalog_contains_deepseek_v3p1(self):
        assert 'deepseek-v3p1' in FireworksAISkill.DEFAULT_MODELS

    def test_catalog_contains_glm5(self):
        assert 'glm-5' in FireworksAISkill.DEFAULT_MODELS

    def test_catalog_contains_kimi_k2p5(self):
        assert 'kimi-k2p5' in FireworksAISkill.DEFAULT_MODELS

    def test_catalog_contains_minimax_m2p5(self):
        assert 'minimax-m2p5' in FireworksAISkill.DEFAULT_MODELS

    def test_catalog_contains_llama(self):
        assert 'llama-v3p3-70b-instruct' in FireworksAISkill.DEFAULT_MODELS

    def test_catalog_contains_qwen3_8b(self):
        assert 'qwen3-8b' in FireworksAISkill.DEFAULT_MODELS

    def test_catalog_contains_cogito(self):
        assert 'cogito-671b-v2' in FireworksAISkill.DEFAULT_MODELS

    def test_kimi_k2p5_supports_vision(self):
        assert FireworksAISkill.DEFAULT_MODELS['kimi-k2p5'].supports_vision is True

    def test_deepseek_no_vision(self):
        assert FireworksAISkill.DEFAULT_MODELS['deepseek-v3p2'].supports_vision is False

    def test_model_config_type(self):
        for cfg in FireworksAISkill.DEFAULT_MODELS.values():
            assert isinstance(cfg, FireworksModelConfig)


class TestFireworksUsageRecordFormatting:

    def test_append_usage_record_fireworks_prefix(self, fireworks_skill):
        """Usage records must prefix model with 'fireworks/'."""
        mock_context = Mock()
        mock_context.usage = []
        fireworks_skill.agent = Mock()
        fireworks_skill.agent.context = mock_context

        fireworks_skill._append_usage_record(
            {'prompt_tokens': 300, 'completion_tokens': 100}, 'deepseek-v3p2'
        )

        assert len(mock_context.usage) == 1
        record = mock_context.usage[0]
        assert record['model'] == 'fireworks/deepseek-v3p2'
        assert record['type'] == 'llm'
        assert record['prompt_tokens'] == 300
        assert record['completion_tokens'] == 100

    def test_append_usage_record_creates_usage_list(self, fireworks_skill):
        """If context.usage doesn't exist, it is created."""
        mock_context = Mock(spec=[])
        fireworks_skill.agent = Mock()
        fireworks_skill.agent.context = mock_context

        fireworks_skill._append_usage_record(
            {'prompt_tokens': 10, 'completion_tokens': 5}, 'glm-5'
        )

        assert hasattr(mock_context, 'usage')
        assert len(mock_context.usage) == 1
        assert mock_context.usage[0]['model'] == 'fireworks/glm-5'

    def test_cached_tokens_included(self, fireworks_skill):
        """cached_read_tokens is set when prompt_tokens_details.cached_tokens exists."""
        mock_context = Mock()
        mock_context.usage = []
        fireworks_skill.agent = Mock()
        fireworks_skill.agent.context = mock_context

        fireworks_skill._append_usage_record(
            {
                'prompt_tokens': 500,
                'completion_tokens': 100,
                'prompt_tokens_details': {'cached_tokens': 400},
            },
            'deepseek-v3p2',
        )

        record = mock_context.usage[0]
        assert record['cached_read_tokens'] == 400

    def test_no_cached_tokens_when_absent(self, fireworks_skill):
        mock_context = Mock()
        mock_context.usage = []
        fireworks_skill.agent = Mock()
        fireworks_skill.agent.context = mock_context

        fireworks_skill._append_usage_record(
            {'prompt_tokens': 500, 'completion_tokens': 100}, 'deepseek-v3p2'
        )

        assert 'cached_read_tokens' not in mock_context.usage[0]

    def test_noop_without_agent(self, fireworks_skill):
        fireworks_skill.agent = None
        fireworks_skill._append_usage_record(
            {'prompt_tokens': 10, 'completion_tokens': 5}, 'deepseek-v3p2'
        )

    def test_noop_without_context(self, fireworks_skill):
        fireworks_skill.agent = Mock(spec=[])
        fireworks_skill._append_usage_record(
            {'prompt_tokens': 10, 'completion_tokens': 5}, 'deepseek-v3p2'
        )


class TestFireworksPrepareParams:

    def test_model_prefix(self, fireworks_skill):
        """_prepare_params prepends accounts/fireworks/models/ to the model name."""
        params = fireworks_skill._prepare_params(
            messages=[{'role': 'user', 'content': 'hi'}],
            model='deepseek-v3p2',
            tools=None,
            stream=False,
        )
        assert params['model'] == 'accounts/fireworks/models/deepseek-v3p2'

    def test_model_prefix_custom_model(self, fireworks_skill):
        params = fireworks_skill._prepare_params(
            messages=[{'role': 'user', 'content': 'hi'}],
            model='glm-5',
            tools=None,
            stream=False,
        )
        assert params['model'] == 'accounts/fireworks/models/glm-5'

    def test_temperature_default(self, fireworks_skill):
        params = fireworks_skill._prepare_params(
            messages=[{'role': 'user', 'content': 'hi'}],
            model='deepseek-v3p2',
            tools=None,
            stream=False,
        )
        assert params['temperature'] == 0.7

    def test_temperature_override(self, fireworks_skill):
        params = fireworks_skill._prepare_params(
            messages=[{'role': 'user', 'content': 'hi'}],
            model='deepseek-v3p2',
            tools=None,
            stream=False,
            temperature=0.2,
        )
        assert params['temperature'] == 0.2

    def test_stream_options(self, fireworks_skill):
        params = fireworks_skill._prepare_params(
            messages=[{'role': 'user', 'content': 'hi'}],
            model='deepseek-v3p2',
            tools=None,
            stream=True,
        )
        assert params['stream_options'] == {'include_usage': True}

    def test_no_stream_options_when_not_streaming(self, fireworks_skill):
        params = fireworks_skill._prepare_params(
            messages=[{'role': 'user', 'content': 'hi'}],
            model='deepseek-v3p2',
            tools=None,
            stream=False,
        )
        assert 'stream_options' not in params

    def test_tools_forwarded(self, fireworks_skill):
        tools = [{'type': 'function', 'function': {'name': 'foo', 'parameters': {}}}]
        params = fireworks_skill._prepare_params(
            messages=[{'role': 'user', 'content': 'hi'}],
            model='deepseek-v3p2',
            tools=tools,
            stream=False,
        )
        assert params['tools'] == tools

    def test_max_tokens(self, fireworks_skill):
        fireworks_skill.max_tokens = 2048
        params = fireworks_skill._prepare_params(
            messages=[{'role': 'user', 'content': 'hi'}],
            model='deepseek-v3p2',
            tools=None,
            stream=False,
        )
        assert params['max_tokens'] == 2048

    def test_passthrough_kwargs(self, fireworks_skill):
        params = fireworks_skill._prepare_params(
            messages=[{'role': 'user', 'content': 'hi'}],
            model='deepseek-v3p2',
            tools=None,
            stream=False,
            top_p=0.9,
            stop=['\n'],
        )
        assert params['top_p'] == 0.9
        assert params['stop'] == ['\n']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
