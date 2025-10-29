# Fal.ai Sonauto Music Generation

Generate original music tracks from text prompts using Sonauto v2.

## Overview

The Sonauto Generation skill creates original music tracks from text descriptions and genre tags using Fal.ai's Sonauto v2 text-to-music model. Perfect for creating background music, theme songs, or custom soundtracks.

## Features

- **Text-to-music generation** from lyrics or descriptions
- **Genre/style tags** for precise control
- **Configurable duration** (5-180 seconds, default 30s)
- **High-quality MP3 output**
- **100% cashback** - Users receive full refund after generation
- **Portal integration** - Automatic upload to content management
- **Custom filenames** - Optional descriptive naming

## Installation

### Requirements

```bash
pip install fal-client httpx
```

### Environment Variables

```bash
export FAL_KEY="your-fal-api-key"
export ROBUTLER_INTERNAL_API_URL="https://robutler.ai"  # Optional
export ROBUTLER_PLATFORM_MARKUP="1.75"                   # Optional
export CASHBACK_MULTIPLIER="2"                           # Optional
```

Get your FAL API key from [fal.ai](https://fal.ai/).

## Usage

### Basic Example

```python
from webagents import BaseAgent
from webagents.agents.skills.ecosystem.fal.sonauto import SonautoGenerateSkill

agent = BaseAgent(
    name="music-creator",
    instructions="You are a music generation assistant.",
    model="litellm/gpt-4o-mini",
    skills={
        "sonauto": SonautoGenerateSkill()
    }
)

# Generate music
response = await agent.run("Create an upbeat pop song about summer")
```

### Advanced Example with Tags

```python
# Electronic chill music
result = await agent.skills["sonauto"].generate_music(
    lyrics_prompt="Relaxing evening vibes, peaceful atmosphere",
    tags=["electronic", "chill", "ambient", "slow"],
    duration=60,
    filename="evening_chill"
)

# Energetic rock track
result = await agent.skills["sonauto"].generate_music(
    lyrics_prompt="Fast-paced adventure, high energy",
    tags=["rock", "energetic", "fast", "electric guitar"],
    duration=45,
    filename="adventure_rock"
)
```

## Available Tags

### Genres
- `pop`, `rock`, `jazz`, `classical`, `electronic`
- `hip-hop`, `country`, `indie`, `folk`, `blues`
- `metal`, `punk`, `reggae`, `r&b`, `soul`

### Moods
- `upbeat`, `chill`, `energetic`, `melancholic`
- `happy`, `dark`, `ambient`, `peaceful`
- `intense`, `calm`, `dramatic`, `playful`

### Tempos
- `fast`, `slow`, `moderate`, `upbeat`

### Styles/Instruments
- `acoustic`, `synth`, `orchestral`, `minimal`
- `lo-fi`, `electric guitar`, `piano`, `strings`
- `drums`, `bass`, `vocal`, `instrumental`

**Best Practice**: Combine 2-4 tags for optimal results.

## API Reference

### `generate_music()`

Generate an original music track from a text prompt.

**Parameters:**

- `lyrics_prompt` (str, required): Lyrics or lyrical theme/description
- `tags` (list[str], optional): List of genre/style tags (default: `[]`)
- `duration` (int, optional): Duration in seconds (5-180, default: `30`)
- `filename` (str, optional): Descriptive filename without extension (e.g., `"summer_vibes"`)

**Returns:**

Dictionary with the following keys:
- `audio_url`: URL to the generated music
- `filename`: Final filename (sanitized)
- `format`: Always `"mp3"`
- `duration`: Duration in seconds
- `seed`: Generation seed (for reproducibility)
- `status`: `"success"` or error status
- `markdown`: Pre-formatted markdown link for chat display

**Example Response:**

```python
{
    "audio_url": "/api/content/public/user-id/summer_vibes.mp3",
    "filename": "summer_vibes.mp3",
    "format": "mp3",
    "duration": 30,
    "seed": 42,
    "status": "success",
    "markdown": "[🎵 summer_vibes.mp3](/api/content/public/user-id/summer_vibes.mp3)"
}
```

## Pricing & Cashback

### Cost Structure

All tracks cost the same regardless of duration:

| Item | Base Cost | Platform Cost | Cashback | Net Cost |
|------|-----------|---------------|----------|----------|
| Any track | $0.075 | $0.2625 | -$0.2625 | **$0.00** |

Calculation: `$0.075 × 1.75 (markup) × 2 (cashback) = $0.2625`

### How Cashback Works

1. User is charged $0.2625 upfront
2. Music is generated and uploaded to portal
3. After successful completion, full $0.2625 is transferred back
4. Net cost to user: **$0** (fully subsidized)

The cashback is automatic and requires no action from the user.

## Best Practices

### Writing Effective Prompts

**Be specific about:**
- Musical mood/emotion
- Tempo and energy level
- Lyrical theme or story
- Intended use case

```python
# ❌ Too vague
"A happy song"

# ✅ Specific and descriptive
"Uplifting morning energy, positive vibes, piano and acoustic guitar, perfect for a motivational video"
```

### Choosing Tags

**Combine complementary tags:**
```python
# Good combinations
["pop", "upbeat", "electronic"]        # Upbeat dance pop
["jazz", "chill", "piano"]              # Smooth jazz
["rock", "energetic", "electric guitar"] # High-energy rock
["ambient", "slow", "minimal"]          # Ambient meditation music
```

### Duration Guidelines

- **5-15s**: Jingles, logos, short intros
- **15-30s**: Social media content, ads
- **30-60s**: Background music, loops
- **60-120s**: Full tracks, themes
- **120-180s**: Extended compositions

## Comparison with Sonauto Inpaint

| Feature | **Generate** (This Skill) | **Inpaint** |
|---------|---------------------------|-------------|
| Purpose | Create new tracks | Edit existing tracks |
| Input | Text prompt + tags | Existing audio + timestamps |
| Output | Original composition | Modified audio |
| Duration | 5-180 seconds | Max 60s per edit |
| Use Case | Background music, themes | Lyric replacement, remixes |
| Cost | $0.26 per track | $0.26 per edit |

**Use Generate for**: Original music creation  
**Use Inpaint for**: Editing/remixing existing songs

## Integration with Agents

```python
agent = BaseAgent(
    name="music-assistant",
    instructions="""
    You create music based on user descriptions.
    Ask about genre, mood, tempo, and intended use.
    Combine 2-4 tags for best results.
    Default duration is 30s unless specified.
    Always provide descriptive filenames based on the music style.
    """,
    model="litellm/gpt-4o-mini",
    skills={
        "sonauto": SonautoGenerateSkill()
    }
)

# Agent automatically uses the skill
response = await agent.run("Make me a calm lo-fi track for studying")
```

## Example Use Cases

### Background Music for Videos

```python
result = await generate_music(
    lyrics_prompt="Inspirational journey, building emotion",
    tags=["orchestral", "upbeat", "cinematic"],
    duration=60,
    filename="inspiring_journey"
)
```

### Podcast Intro Theme

```python
result = await generate_music(
    lyrics_prompt="Tech podcast intro, modern and professional",
    tags=["electronic", "moderate", "synth", "minimal"],
    duration=15,
    filename="tech_podcast_intro"
)
```

### Game Background Music

```python
result = await generate_music(
    lyrics_prompt="Adventure game soundtrack, mysterious forest",
    tags=["ambient", "mysterious", "orchestral", "slow"],
    duration=120,
    filename="forest_adventure_bg"
)
```

### Meditation Track

```python
result = await generate_music(
    lyrics_prompt="Deep relaxation, peaceful meditation",
    tags=["ambient", "minimal", "slow", "calm"],
    duration=180,
    filename="meditation_peace"
)
```

## Technical Details

### Architecture

1. **Submission**: Calls Fal.ai Sonauto v2 API
2. **Queue Processing**: Monitors job progress with logs
3. **Download**: Retrieves generated audio from Fal.ai
4. **Upload**: Uploads to portal content management
5. **Tracking**: Records cost in context for cashback
6. **Cashback**: Finalize hook transfers credits back to user

### File Handling

- Automatic filename sanitization
- Falls back to hash-based names if not provided
- Always adds `.mp3` extension
- Uploads with public visibility and AI-generated tags

### Cost Tracking

Costs are tracked in `context.sonauto_gen_costs`:
```python
{
    "total_charged": 0.2625,
    "operations": [
        {
            "type": "music_generation",
            "cost": 0.2625,
            "duration": 30
        }
    ]
}
```

## Error Handling

```python
try:
    result = await agent.skills["sonauto"].generate_music(
        lyrics_prompt="Upbeat summer song",
        tags=["pop", "upbeat"],
        duration=45
    )
    
    if result.get("status") == "success":
        audio_url = result["audio_url"]
        print(f"Music generated: {audio_url}")
    else:
        error = result.get("error", "Unknown error")
        print(f"Generation failed: {error}")
        
except Exception as e:
    print(f"Error: {e}")
```

## Limitations

- **Duration range**: 5-180 seconds only
- **No explicit lyrics**: Generates instrumental or generic vocals
- **Fixed cost**: Same price regardless of duration
- **Single track**: Cannot generate multi-track arrangements
- **No MIDI export**: MP3 audio only

## Troubleshooting

### "FAL_KEY not configured"
- Ensure the environment variable is set correctly
- Verify API key is active on fal.ai

### "Duration too short/long"
- Minimum duration: 5 seconds
- Maximum duration: 180 seconds (3 minutes)
- The tool will auto-clamp to valid range

### "Upload failed"
- Check portal URL is accessible
- Verify bearer token/API key is valid
- Check file size limits

## Advanced Features

### Using with Other Skills

Combine with Veo for complete audiovisual content:

```python
agent = BaseAgent(
    name="av-creator",
    skills={
        "veo": VeoSkill(),
        "sonauto": SonautoGenerateSkill()
    }
)

# Create video
video = await agent.skills["veo"].generate_video(
    prompt="Mountain landscape at sunrise"
)

# Create matching music
music = await agent.skills["sonauto"].generate_music(
    lyrics_prompt="Peaceful morning, inspiring nature",
    tags=["ambient", "peaceful", "orchestral"],
    duration=8  # Match video duration
)
```

### Reproducible Generation

Save the seed from the response to recreate similar tracks:

```python
result = await generate_music(
    lyrics_prompt="Energetic workout music",
    tags=["electronic", "fast", "energetic"]
)

seed = result["seed"]  # Save for future reference
```

## Related Skills

- **Sonauto Inpaint**: Edit existing music tracks
- **Veo**: Generate videos to accompany music

## Support

For issues or questions:
- Check [WebAgents Documentation](https://docs.robutler.ai)
- Review [Fal.ai Documentation](https://fal.ai/models/sonauto/v2/text-to-music)
- Open an issue on GitHub

