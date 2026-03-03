"""
xAI LLM UAMP Adapter.

Reuses the OpenAI adapter since xAI uses an OpenAI-compatible API.
Only overrides model capabilities for Grok models.
"""

from webagents.agents.skills.core.llm.openai.uamp_adapter import OpenAIUAMPAdapter

from webagents.uamp import (
    ModelCapabilities,
    ImageCapabilities,
    ToolCapabilities,
)


class XAIUAMPAdapter(OpenAIUAMPAdapter):
    """
    UAMP adapter for xAI. Inherits all conversion logic from OpenAI adapter
    since xAI uses an OpenAI-compatible API format.
    """

    MODEL_CAPABILITIES = {
        "grok-3": ModelCapabilities(
            model_id="grok-3",
            provider="xai",
            modalities=["text"],
            supports_streaming=True,
            supports_thinking=False,
            context_window=131072,
            max_output_tokens=131072,
            tools=ToolCapabilities(
                supports_tools=True,
                supports_parallel_tools=True,
            ),
        ),
        "grok-3-mini": ModelCapabilities(
            model_id="grok-3-mini",
            provider="xai",
            modalities=["text"],
            supports_streaming=True,
            supports_thinking=True,
            context_window=131072,
            max_output_tokens=131072,
            tools=ToolCapabilities(
                supports_tools=True,
            ),
        ),
        "grok-4-0709": ModelCapabilities(
            model_id="grok-4-0709",
            provider="xai",
            modalities=["text", "image"],
            supports_streaming=True,
            supports_thinking=True,
            context_window=256000,
            max_output_tokens=131072,
            image=ImageCapabilities(
                formats=["jpeg", "png", "gif", "webp"],
            ),
            tools=ToolCapabilities(
                supports_tools=True,
                supports_parallel_tools=True,
            ),
        ),
        "grok-4-fast-reasoning": ModelCapabilities(
            model_id="grok-4-fast-reasoning",
            provider="xai",
            modalities=["text", "image"],
            supports_streaming=True,
            supports_thinking=True,
            context_window=2000000,
            max_output_tokens=131072,
            image=ImageCapabilities(
                formats=["jpeg", "png", "gif", "webp"],
            ),
            tools=ToolCapabilities(
                supports_tools=True,
                supports_parallel_tools=True,
            ),
        ),
        "grok-4-fast-non-reasoning": ModelCapabilities(
            model_id="grok-4-fast-non-reasoning",
            provider="xai",
            modalities=["text", "image"],
            supports_streaming=True,
            supports_thinking=False,
            context_window=2000000,
            max_output_tokens=131072,
            image=ImageCapabilities(
                formats=["jpeg", "png", "gif", "webp"],
            ),
            tools=ToolCapabilities(
                supports_tools=True,
                supports_parallel_tools=True,
            ),
        ),
        "grok-code-fast-1": ModelCapabilities(
            model_id="grok-code-fast-1",
            provider="xai",
            modalities=["text"],
            supports_streaming=True,
            supports_thinking=False,
            context_window=256000,
            max_output_tokens=131072,
            tools=ToolCapabilities(
                supports_tools=True,
            ),
        ),
    }

    def __init__(self, model: str = "grok-3"):
        super().__init__(model=model)

    def get_capabilities(self, model=None):
        target_model = model or self.model
        for model_prefix, caps in self.MODEL_CAPABILITIES.items():
            if target_model.startswith(model_prefix):
                return caps
        return ModelCapabilities(
            model_id=target_model,
            provider="xai",
            modalities=["text"],
            supports_streaming=True,
        )
