# Google Veo 3.1 Video Generation

Generate high-quality 8-second videos using Google's Veo 3.1 models.

## Overview

The Veo skill provides access to Google's state-of-the-art video generation models through the Gemini API. Generate videos from text descriptions with options for speed/quality tradeoffs and audio control.

## Features

- **Two quality variants**: Fast and Standard models
- **Audio control**: Generate with or without audio (Standard model only)
- **Fixed 8-second duration** per API specification
- **HD quality output** in MP4 format
- **100% cashback** - Users receive full refund after generation
- **Portal integration** - Automatic upload to content management
- **Custom filenames** - Optional descriptive naming

## Installation

### Requirements

```bash
pip install google-genai httpx
```

### Environment Variables

```bash
export GOOGLE_GEMINI_API_KEY="your-gemini-api-key"
export ROBUTLER_INTERNAL_API_URL="https://robutler.ai"  # Optional
export ROBUTLER_PLATFORM_MARKUP="1.75"                   # Optional
export CASHBACK_MULTIPLIER="2"                           # Optional
```

Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Usage

### Basic Example

```python
from webagents import BaseAgent
from webagents.agents.skills.ecosystem.google.veo import VeoSkill

agent = BaseAgent(
    name="video-creator",
    instructions="You are a video generation assistant.",
    model="litellm/gpt-4o-mini",
    skills={
        "veo": VeoSkill()
    }
)

# Fast generation (with audio)
response = await agent.run("Generate a video of a sunset over the ocean")
```

### Advanced Example with Options

```python
# Standard quality with audio
result = await agent.skills["veo"].generate_video(
    prompt="A cinematic shot of a mountain landscape at golden hour, camera slowly panning right",
    model="standard",
    with_audio=True,
    filename="mountain_golden_hour"
)

# Standard quality without audio (cheaper)
result = await agent.skills["veo"].generate_video(
    prompt="Abstract geometric shapes morphing and rotating",
    model="standard",
    with_audio=False,
    filename="geometric_abstract"
)
```

## Model Variants

### Fast (Default)
- **Cost**: $4.20 per video (with 100% cashback = $0 net)
- **Speed**: Faster generation
- **Quality**: Good quality
- **Audio**: Always included
- **Best for**: Quick prototypes, testing, high volume

### Standard with Audio
- **Cost**: $11.20 per video (with 100% cashback = $0 net)
- **Speed**: Slower generation
- **Quality**: Best quality
- **Audio**: Included
- **Best for**: High-quality final videos with sound

### Standard Video-Only
- **Cost**: $5.60 per video (with 100% cashback = $0 net)
- **Speed**: Slower generation
- **Quality**: Best quality
- **Audio**: None
- **Best for**: Silent videos, music videos with separate audio

## API Reference

### `generate_video()`

Generate an 8-second video from a text prompt.

**Parameters:**

- `prompt` (str, required): Detailed description of the video scene, action, and visual style
- `model` (str, optional): `"fast"` (default) or `"standard"`
- `with_audio` (bool, optional): Include audio in video (default: `True`). Only applies to standard model; fast model always includes audio.
- `filename` (str, optional): Descriptive filename without extension (e.g., `"sunset_beach"`)

**Returns:**

Dictionary with the following keys:
- `video_url`: URL to the generated video
- `filename`: Final filename (sanitized)
- `format`: Always `"mp4"`
- `duration`: Always `8` seconds
- `model`: Model name used
- `status`: `"success"` or error status
- `markdown`: Pre-formatted markdown link for chat display

**Example Response:**

```python
{
    "video_url": "/api/content/public/user-id/sunset_beach.mp4",
    "filename": "sunset_beach.mp4",
    "format": "mp4",
    "duration": 8,
    "model": "veo-3.1-fast-generate-preview",
    "status": "success",
    "markdown": "[🎬 sunset_beach.mp4](/api/content/public/user-id/sunset_beach.mp4)"
}
```

## Pricing & Cashback

### Cost Structure

All costs calculated as: `base_cost × platform_markup × cashback_multiplier`

| Variant | Base Cost | Platform Cost | Cashback | Net Cost |
|---------|-----------|---------------|----------|----------|
| Fast (with audio) | $1.20 | $4.20 | -$4.20 | **$0.00** |
| Standard (with audio) | $3.20 | $11.20 | -$11.20 | **$0.00** |
| Standard (video-only) | $1.60 | $5.60 | -$5.60 | **$0.00** |

### How Cashback Works

1. User is charged upfront for generation
2. Video is generated and uploaded to portal
3. After successful completion, full amount is transferred back to user
4. Net cost to user: **$0** (fully subsidized)

The cashback is automatic and requires no action from the user.

## Best Practices

### Prompt Writing

Write detailed, specific prompts for best results:

```python
# ❌ Too vague
"A car driving"

# ✅ Detailed and specific
"A red sports car driving along a coastal highway at sunset, camera following from the side, waves crashing on rocks below, golden light reflecting off the car's surface"
```

**Include these elements:**
- Subject and action
- Camera angle/movement
- Lighting conditions
- Atmosphere and mood
- Visual details

### Choosing the Right Model

**Use Fast when:**
- Testing different prompts
- Need quick turnaround
- Volume generation
- Quality is acceptable

**Use Standard when:**
- Final production videos
- Maximum quality needed
- Specific cinematic requirements

**Use Video-Only when:**
- Creating music videos (separate audio track)
- Silent content (text overlays, tutorials)
- Want to save 50% on standard quality

## Limitations

- **Fixed duration**: All videos are exactly 8 seconds
- **No multi-shot**: Single continuous scene only
- **No editing**: Cannot combine or edit generated videos
- **Public URLs required**: Input references must be publicly accessible
- **Generation time**: Standard model may take several minutes

## Error Handling

```python
try:
    result = await agent.skills["veo"].generate_video(
        prompt="A serene forest scene",
        model="standard"
    )
    
    if result.get("status") == "success":
        video_url = result["video_url"]
        print(f"Video generated: {video_url}")
    else:
        error = result.get("error", "Unknown error")
        print(f"Generation failed: {error}")
        
except Exception as e:
    print(f"Error: {e}")
```

## Integration with Agents

The Veo skill integrates seamlessly with WebAgents:

```python
agent = BaseAgent(
    name="video-assistant",
    instructions="""
    You create videos based on user descriptions.
    Use the generate_video tool with detailed prompts.
    Include camera angles, lighting, and mood.
    Default to fast model unless user requests highest quality.
    Always provide descriptive filenames.
    """,
    model="litellm/gpt-4o-mini",
    skills={
        "veo": VeoSkill()
    }
)

# Agent automatically uses the skill
response = await agent.run("Create a video of a sunrise over mountains")
```

## Technical Details

### Architecture

1. **Generation**: Calls Google Gemini API `generate_videos()` method
2. **Polling**: Monitors operation status until completion
3. **Download**: Retrieves video file from Google
4. **Upload**: Uploads to portal content management
5. **Tracking**: Records cost in context for cashback
6. **Cashback**: Finalize hook transfers credits back to user

### File Handling

- Automatic filename sanitization (removes invalid characters)
- Falls back to hash-based names if not provided
- Always adds `.mp4` extension
- Uploads with public visibility to portal

### Cost Tracking

Costs are tracked in `context.veo_costs`:
```python
{
    "total_charged": 4.20,
    "operations": [
        {
            "type": "video_generation",
            "model": "fast",
            "cost": 4.20,
            "duration": 8
        }
    ]
}
```

## Troubleshooting

### "GOOGLE_GEMINI_API_KEY not configured"
- Ensure the environment variable is set correctly
- Verify API key is active in Google AI Studio

### "Video generation timed out"
- Standard model can take several minutes
- Check Google API status
- Retry with fast model

### "Upload failed"
- Check portal URL is accessible
- Verify bearer token/API key is valid
- Check file size limits

## Related Skills

- **Sonauto Generate**: Create music tracks to accompany videos
- **Sonauto Inpaint**: Edit existing music for custom soundtracks

## Support

For issues or questions:
- Check [WebAgents Documentation](https://docs.robutler.ai)
- Review [Google Gemini API Docs](https://ai.google.dev/gemini-api/docs/video)
- Open an issue on GitHub

