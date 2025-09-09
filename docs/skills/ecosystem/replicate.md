# Replicate Skill

!!! warning "Alpha Software Notice"

    This skill is in **alpha stage** and under active development. APIs, features, and functionality may change without notice. Use with caution in production environments and expect potential breaking changes in future releases.

Advanced machine learning model execution via Replicate's API. Run any public model, from text generation to image creation, with secure credential management and real-time monitoring.

## Features

- **Run ML models** - Execute text, image, audio, and video models
- **Model discovery** - Browse and get detailed information about available models
- **Real-time monitoring** - Track prediction progress and get results
- **Secure API token storage** via auth and KV skills
- **Cancel predictions** to save compute costs
- **Per-user isolation** with authentication

## Quick Setup

```python
from webagents.agents import BaseAgent
from webagents.agents.skills.ecosystem.replicate import ReplicateSkill

agent = BaseAgent(
    name="replicate-agent",
    model="openai/gpt-4o",
    skills={
        "replicate": ReplicateSkill()  # Auto-resolves: auth, kv
    }
)
```

## Core Tools

### `replicate_setup(api_token)`
Set up Replicate API credentials with automatic validation.

### `replicate_list_models(owner)`
List available models, optionally filtered by owner (e.g., "stability-ai").

### `replicate_get_model_info(model)`
Get detailed information about a specific model including parameters.

### `replicate_run_prediction(model, input_data)`
Run a prediction with a model and input parameters.

### `replicate_get_prediction(prediction_id)`
Get prediction status and results.

### `replicate_cancel_prediction(prediction_id)`
Cancel a running prediction to save costs.

## Usage Examples

### Text Generation

```python
messages = [
    {"role": "user", "content": "Generate a haiku about AI using LLaMA-2"}
]
response = await agent.run(messages=messages)
```

### Image Generation

```python
messages = [
    {"role": "user", "content": "Create an image of a dragon flying over mountains using Stable Diffusion"}
]
response = await agent.run(messages=messages)
```

### Model Discovery

```python
messages = [
    {"role": "user", "content": "Show me what video generation models are available from Stability AI"}
]
response = await agent.run(messages=messages)
```

### Audio Processing

```python
messages = [
    {"role": "user", "content": "Transcribe this audio file using Whisper: https://example.com/audio.mp3"}
]
response = await agent.run(messages=messages)
```

## Getting Your Replicate API Token

1. Sign up at [replicate.com](https://replicate.com)
2. Go to Account Settings → API tokens
3. Create token and copy it (starts with `r8_`)

## Model Categories

**Text Models**: LLaMA, GPT alternatives, chat models, code generation
**Image Models**: Stable Diffusion, DALL-E alternatives, image editing, upscaling
**Audio Models**: Whisper transcription, text-to-speech, music generation
**Video Models**: Text-to-video, video editing, style transfer
**Multimodal**: Vision-language models, image captioning

## Pricing

- **Free tier**: Limited compute credits per month
- **Pay-per-use**: Charged based on compute time and model complexity
- **Model-specific pricing**: Check individual model pages for costs

## Troubleshooting

**Authentication Issues** - Verify API token starts with `r8_` and is correctly copied
**Model Not Found** - Use model discovery tools to find correct model names
**Invalid Input** - Get model info to see required parameters and types
**Long Predictions** - Some models take minutes; use status monitoring
**Failed Predictions** - Check error messages and verify input data format

## Architecture

The skill integrates with Replicate's REST API v1, using secure credential storage via KV skill and per-user authentication via Auth skill. Supports both synchronous and asynchronous model execution patterns.

## Integration Tips

1. **Monitor costs** - Check model pricing before expensive predictions
2. **Use appropriate models** - Choose right model size for your task  
3. **Cancel unused predictions** - Save costs by canceling long runs
4. **Cache results** - Store outputs to avoid re-running identical predictions
5. **Test first** - Try models with simple inputs before complex tasks
