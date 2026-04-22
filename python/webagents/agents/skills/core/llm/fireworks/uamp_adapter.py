"""
Fireworks AI UAMP Adapter.

Reuses the OpenAI adapter since Fireworks uses an OpenAI-compatible API.
Only overrides model capabilities for Fireworks models.
"""

from webagents.agents.skills.core.llm.openai.uamp_adapter import OpenAIUAMPAdapter

from webagents.uamp import (
    ModelCapabilities,
    ImageCapabilities,
    ToolCapabilities,
)


class FireworksUAMPAdapter(OpenAIUAMPAdapter):
    """
    UAMP adapter for Fireworks AI. Inherits all conversion logic from OpenAI adapter.
    """

    MODEL_CAPABILITIES = {
        "deepseek-v3p2": ModelCapabilities(
            model_id="deepseek-v3p2",
            provider="fireworks",
            modalities=["text"],
            supports_streaming=True,
            context_window=163840,
            max_output_tokens=131072,
            tools=ToolCapabilities(
                supports_tools=True,
                supports_parallel_tools=True,
            ),
        ),
        "glm-5": ModelCapabilities(
            model_id="glm-5",
            provider="fireworks",
            modalities=["text"],
            supports_streaming=True,
            context_window=202752,
            max_output_tokens=131072,
            tools=ToolCapabilities(
                supports_tools=True,
            ),
        ),
        "kimi-k2p6": ModelCapabilities(
            model_id="kimi-k2p6",
            provider="fireworks",
            modalities=["text", "image"],
            supports_streaming=True,
            context_window=262144,
            max_output_tokens=131072,
            image=ImageCapabilities(
                formats=["jpeg", "png", "gif", "webp"],
            ),
            tools=ToolCapabilities(
                supports_tools=True,
            ),
        ),
        "kimi-k2p5": ModelCapabilities(
            model_id="kimi-k2p5",
            provider="fireworks",
            modalities=["text", "image"],
            supports_streaming=True,
            context_window=262144,
            max_output_tokens=131072,
            image=ImageCapabilities(
                formats=["jpeg", "png", "gif", "webp"],
            ),
            tools=ToolCapabilities(
                supports_tools=True,
            ),
        ),
        "minimax-m2p5": ModelCapabilities(
            model_id="minimax-m2p5",
            provider="fireworks",
            modalities=["text"],
            supports_streaming=True,
            context_window=196608,
            max_output_tokens=131072,
            tools=ToolCapabilities(
                supports_tools=True,
            ),
        ),
        "llama-v3p3-70b-instruct": ModelCapabilities(
            model_id="llama-v3p3-70b-instruct",
            provider="fireworks",
            modalities=["text"],
            supports_streaming=True,
            context_window=131072,
            max_output_tokens=131072,
            tools=ToolCapabilities(
                supports_tools=True,
            ),
        ),
    }

    def __init__(self, model: str = "deepseek-v3p2"):
        super().__init__(model=model)

    def get_capabilities(self, model=None):
        target_model = model or self.model
        for model_prefix, caps in self.MODEL_CAPABILITIES.items():
            if target_model.startswith(model_prefix):
                return caps
        return ModelCapabilities(
            model_id=target_model,
            provider="fireworks",
            modalities=["text"],
            supports_streaming=True,
        )
