# Replicate Skill for WebAgents

Advanced machine learning model execution via Replicate's API. This skill allows you to securely manage Replicate API credentials and run any public model on the Replicate platform.

## Features

- 🔐 **Secure credential storage** via auth and KV skills
- 🤖 **Run ML models** with any input data type
- 📋 **List models** from specific owners or browse popular models
- 📊 **Monitor predictions** in real-time with status tracking
- 🛑 **Cancel predictions** to save compute costs
- 🔍 **Model discovery** with detailed parameter information
- 🔒 **Per-user isolation** with authentication
- 🛡️ **API token validation** during setup

## Prerequisites

1. **Replicate account** (free or paid)
2. **Replicate API token** (generated from Replicate account settings)
3. **WebAgents framework** with auth and kv skills enabled

## Quick Start

### 1. Set up Replicate API credentials

```python
# The agent will use the replicate_setup tool when asked
response = await agent.run([
    {"role": "user", "content": "Set up Replicate with token r8_your_token_here"}
])
```

### 2. Discover available models

```python
# Browse models from a specific owner
response = await agent.run([
    {"role": "user", "content": "List Stability AI models on Replicate"}
])
```

### 3. Get model information

```python
# Get detailed info about a model
response = await agent.run([
    {"role": "user", "content": "Tell me about the stability-ai/stable-diffusion model"}
])
```

### 4. Run a prediction

```python
# Run image generation
response = await agent.run([
    {"role": "user", "content": "Generate an image of a sunset over the ocean using Stable Diffusion"}
])
```

## Tools Reference

### `replicate_setup(api_token)`
Set up Replicate API credentials securely.

**Parameters:**
- `api_token` (str): Your Replicate API token from account settings

**Returns:**
- Success message with connection verification
- Error message if API token is invalid

**Example:**
```
✅ Replicate credentials saved successfully!
🔑 API token configured
📊 Connection to Replicate API verified
```

### `replicate_list_models(owner=None)`
List available models on Replicate.

**Parameters:**
- `owner` (str, optional): Filter models by owner (e.g., "stability-ai")

**Returns:**
- List of models with names, descriptions, and visibility
- Empty message if no models found

**Example:**
```
🤖 Available Models from stability-ai:

🌍 **stability-ai/stable-diffusion**
   📝 Generate images from text prompts

🌍 **stability-ai/stable-video-diffusion**
   📝 Generate videos from images

💡 Use replicate_run_prediction(model, input_data) to run a model
```

### `replicate_get_model_info(model)`
Get detailed information about a specific model.

**Parameters:**
- `model` (str): Model name in format "owner/model-name"

**Returns:**
- Detailed model information including parameters and schema
- Error message if model not found

**Example:**
```
🤖 Model Information: stability-ai/stable-diffusion
🌍 Visibility: public
📝 Description: Generate images from text prompts
🆔 Latest Version: v1.2.3
📅 Created: 2024-01-15T10:30:00Z

📋 Input Parameters:
  ⚠️ prompt (string): Text description of the image
  📝 width (integer): Width of the output image
  📝 height (integer): Height of the output image

💡 Use replicate_run_prediction() to run this model
```

### `replicate_run_prediction(model, input_data)`
Run a prediction with a Replicate model.

**Parameters:**
- `model` (str): Model name in format "owner/model-name"
- `input_data` (dict): Input parameters for the model

**Returns:**
- Success message with prediction ID and initial status
- Error message if model not found or input invalid

**Example:**
```
🚀 Prediction started successfully!
🆔 Prediction ID: pred_abc123def456
🤖 Model: stability-ai/stable-diffusion
📊 Status: starting
⏳ Use replicate_get_prediction() to check status and get results
```

### `replicate_get_prediction(prediction_id)`
Get prediction status and results.

**Parameters:**
- `prediction_id` (str): The prediction ID returned from replicate_run_prediction

**Returns:**
- Detailed prediction status report with results if completed
- Error message if prediction not found

**Example:**
```
📊 Prediction Status Report
🆔 Prediction ID: pred_abc123def456
🤖 Model: stability-ai/stable-diffusion
✅ Status: succeeded
🕐 Created: 2024-01-15T10:30:00Z
▶️ Started: 2024-01-15T10:30:05Z
🏁 Completed: 2024-01-15T10:32:15Z
📋 Output: https://replicate.delivery/pbxt/abc123.png
```

### `replicate_cancel_prediction(prediction_id)`
Cancel a running prediction.

**Parameters:**
- `prediction_id` (str): The prediction ID to cancel

**Returns:**
- Success message if canceled
- Error message if prediction cannot be canceled

**Example:**
```
✅ Prediction pred_abc123def456 canceled successfully
```

## Setup Guide

### Getting your Replicate API Token

1. Sign up for a [Replicate account](https://replicate.com)
2. Go to **Account Settings** → **API tokens**
3. Click **Create token**
4. Give it a name (e.g., "WebAgent Integration")
5. Copy the generated token (starts with `r8_`)

### Replicate Pricing

- **Free tier**: Limited compute credits per month
- **Pay-per-use**: Charged based on compute time and model complexity
- **Different models** have different pricing (check model pages)

## Usage Examples

### Simple Text Generation

```python
# Setup agent with Replicate skill
agent = BaseAgent(
    name="ai-assistant",
    instructions="You are an AI assistant that can run machine learning models.",
    model="openai/gpt-4o-mini",
    skills={
        "replicate": ReplicateSkill(),
        "auth": AuthSkill(),
        "kv": KVSkill()
    }
)

# Setup credentials (one-time)
response = await agent.run([
    {"role": "user", "content": "Set up Replicate with my API token r8_your_token_here"}
])

# Generate text
response = await agent.run([
    {"role": "user", "content": "Generate a haiku about AI using the LLaMA model"}
])
```

### Image Generation Workflow

```python
# Generate an image
response = await agent.run([
    {"role": "user", "content": "Create an image of a cyberpunk cityscape using Stable Diffusion"}
])

# Check generation status
response = await agent.run([
    {"role": "user", "content": "Check the status of my last image generation"}
])
```

### Audio Processing

```python
# Transcribe audio
response = await agent.run([
    {"role": "user", "content": "Transcribe this audio file: https://example.com/audio.mp3"}
])
```

### Model Discovery

```python
# Discover available models
response = await agent.run([
    {"role": "user", "content": "Show me what image generation models are available from Stability AI"}
])

# Get model details
response = await agent.run([
    {"role": "user", "content": "Tell me more about the stable-video-diffusion model"}
])
```

### Conversation Examples

#### Setting up Replicate

**User:** "I want to use Replicate models. My API token is r8_abc123def456"

**Assistant:** "I'll set up Replicate for you with that API token."

*[Agent uses replicate_setup tool]*

**Assistant:** "✅ Replicate credentials saved successfully! 🔑 API token configured 📊 Connection to Replicate API verified"

#### Generating an Image

**User:** "Generate an image of a dragon flying over mountains"

**Assistant:** "I'll create that image for you using Stable Diffusion."

*[Agent uses replicate_run_prediction tool with stability-ai/stable-diffusion]*

**Assistant:** "🚀 Prediction started successfully! 🆔 Prediction ID: pred_abc123def456 🤖 Model: stability-ai/stable-diffusion 📊 Status: starting. I'll check on the progress for you."

*[Agent uses replicate_get_prediction tool]*

**Assistant:** "✅ Your image is ready! Here's your dragon flying over mountains: https://replicate.delivery/pbxt/abc123.png"

#### Model Discovery

**User:** "What AI models are available for video generation?"

**Assistant:** "Let me check what video models are available on Replicate."

*[Agent uses replicate_list_models and replicate_get_model_info tools]*

**Assistant:** "I found several video generation models:

🌍 **stability-ai/stable-video-diffusion** - Generate videos from images
🌍 **anotherjesse/zeroscope-v2-xl** - Text-to-video generation

Would you like me to generate a video using one of these models?"

## Error Handling

The skill provides clear error messages for common scenarios:

- **❌ Authentication required** - User not authenticated
- **❌ API token is required** - Empty or missing API token
- **❌ Invalid API token** - API token rejected by Replicate
- **❌ Model not found** - Invalid model name or model doesn't exist
- **❌ Invalid input data** - Input doesn't match model requirements
- **❌ Prediction not found** - Invalid prediction ID
- **❌ Cannot cancel prediction** - Prediction already completed or canceled

## Security

- **API tokens** are stored securely using the KV skill with per-user namespacing
- **User isolation** ensures each user can only access their own credentials
- **Authentication required** for all operations
- **Automatic validation** of API tokens during setup
- **Memory fallback** available if KV skill unavailable

## Architecture

```
WebAgent
├── Auth Skill (user context)
├── KV Skill (secure storage)
└── Replicate Skill
    ├── Credential Management
    ├── API Communication
    ├── Model Discovery
    ├── Prediction Management
    └── Status Monitoring
```

## Dependencies

- `auth` skill - For user authentication and context
- `kv` skill - For secure credential storage
- `httpx` - For HTTP API communication
- `json` - For data serialization
- `datetime` - For timestamp handling

## Troubleshooting

### "Authentication required"
Ensure your agent has proper authentication configured and the user is logged in.

### "Invalid API token" 
- Verify the API token is correct and starts with `r8_`
- Check if the token has been revoked in your Replicate account
- Ensure you copied the full token without extra spaces

### "API token doesn't have required permissions"
- Check if your Replicate account has API access enabled
- Verify the token has necessary permissions
- Contact Replicate support if issues persist

### "Model not found"
- Ask the agent to list available models first
- Check if the model name format is correct (owner/model-name)
- Verify the model exists and is public (or you have access)

### "Invalid input data"
- Ask the agent to get model information to see required parameters
- Check the model's documentation on Replicate
- Ensure input data types match the model's schema

### "Prediction taking too long"
- Some models can take several minutes to complete
- Ask the agent to check prediction status periodically
- Consider canceling and trying a smaller/faster model

### "Prediction failed"
- Check the error message in the prediction status
- Verify your input data is valid
- Check if you have sufficient credits in your Replicate account

## Limitations

- Supports Replicate REST API v1 endpoints
- Limited to public models (unless you have private model access)
- Rate limited by Replicate's API limits
- Prediction results depend on model availability and performance
- Some models may have usage restrictions or require special permissions

## Model Categories

Replicate hosts various types of models:

### Text Models
- **Language models**: GPT, LLaMA, Claude alternatives
- **Chat models**: Conversational AI models
- **Code generation**: Programming assistance models

### Image Models
- **Text-to-image**: Stable Diffusion, DALL-E alternatives
- **Image-to-image**: Style transfer, upscaling, editing
- **Image analysis**: Object detection, classification

### Audio Models
- **Speech-to-text**: Whisper, transcription models
- **Text-to-speech**: Voice synthesis models
- **Audio generation**: Music, sound effects

### Video Models
- **Text-to-video**: Video generation from prompts
- **Video-to-video**: Style transfer, editing
- **Video analysis**: Object tracking, classification

### Multimodal Models
- **Vision-language**: CLIP, image captioning
- **Audio-visual**: Video understanding models

## Integration Tips

1. **Monitor costs** - Check model pricing before running expensive predictions
2. **Use appropriate models** - Choose the right model size for your task
3. **Cache results** - Store outputs to avoid re-running identical predictions
4. **Cancel unused predictions** - Save costs by canceling long-running predictions
5. **Test models first** - Try models with simple inputs before complex tasks

## Contributing

To extend this skill:

1. Add new tools following the existing pattern
2. Update tests in `tests/test_replicate_skill.py`
3. Update this documentation
4. Ensure proper error handling and user feedback
5. Test with various model types and scenarios

## License

Part of the WebAgents framework. See main project license.