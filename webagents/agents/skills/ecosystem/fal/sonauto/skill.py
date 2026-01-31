"""
Sonauto Music Generation Skill
Generates music tracks from text prompts using Fal.ai's Sonauto v2 model.

Features:
- Text-to-music generation with genre tags
- Configurable duration (default 30s)
- 100% cashback after successful generation
- Automatic portal upload for seamless integration
"""
import os
import logging
import json
import time
import hashlib
import re
from typing import Optional, Dict, Any, List

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, prompt, hook
from webagents.agents.skills.robutler.payments.skill import pricing, PricingInfo
from webagents.server.context.context_vars import get_context

logger = logging.getLogger('webagents.skills.sonauto.generate')

# Sonauto pricing
SONAUTO_GENERATE_BASE_COST = 0.075  # $0.075 per generation


class SonautoGenerateSkill(Skill):
    """Skill for generating music using Sonauto V2 text-to-music model"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {})
        self.config = config or {}
        self.fal_api_key = os.getenv('FAL_KEY')

        if not self.fal_api_key:
            logger.warning("FAL_KEY not set - Sonauto music generation will not work")
    
    @hook("finalize_connection", priority=96)
    async def finalize_sonauto_gen_cashback(self, context) -> Any:
        """Send cashback to user after successful Sonauto generation operations"""
        # Only send cashback if payment was successful
        payment_context = getattr(context, 'payments', None)
        if payment_context:
            payment_successful = getattr(payment_context, 'payment_successful', False)
            if not payment_successful:
                logger.info("💸 Skipping Sonauto gen cashback - payment was not successful")
                return context
        else:
            logger.warning("💸 Skipping Sonauto gen cashback - no payment context available")
            return context
        
        # Get total Sonauto generation costs from context
        sonauto_gen_costs = getattr(context, 'sonauto_gen_costs', None)
        if not sonauto_gen_costs:
            return context
        
        total_charged = sonauto_gen_costs.get('total_charged', 0.0)
        if total_charged <= 0:
            return context
        
        # Get user ID from auth context or payment context
        actual_user_id = None
        
        # Try auth context first
        try:
            auth_ns = getattr(context, 'auth', None) or context.get('auth')
            if auth_ns:
                actual_user_id = getattr(auth_ns, 'user_id', None)
                if actual_user_id:
                    logger.debug(f"💸 Found user_id in auth context: {actual_user_id}")
        except Exception as e:
            logger.debug(f"💸 Error accessing auth context: {e}")
        
        # Fallback to payment context if auth context doesn't have user_id
        if not actual_user_id and payment_context:
            actual_user_id = getattr(payment_context, 'user_id', None)
            if actual_user_id:
                logger.debug(f"💸 Found user_id in payment context: {actual_user_id}")
        
        if not actual_user_id:
            logger.warning("💸 Cannot transfer Sonauto gen cashback - no caller user_id in auth or payment context")
            return context
        
        cashback_amount = total_charged
        logger.info(f"💸 Transferring Sonauto gen cashback of {cashback_amount:.4f} credits to user: {actual_user_id}")
        
        try:
            portal_url = os.getenv("ROBUTLER_INTERNAL_API_URL", "https://robutler.ai").rstrip('/')
            
            # Get agent API key - try skill config first (set by dynamic factory)
            api_key = self.config.get('robutler_api_key')
            if api_key:
                logger.debug(f"💸 Using API key from skill config for cashback")
            else:
                # Try context as fallback
                api_key = context.get("api_key")
                if api_key:
                    logger.debug(f"💸 Using API key from context for cashback")
                else:
                    # Last resort: environment variable
                    api_key = os.getenv('ROBUTLER_API_KEY')
                    if api_key:
                        logger.debug(f"💸 Using ROBUTLER_API_KEY environment variable for cashback")
            
            if not api_key:
                logger.error("💸 No agent API key available for Sonauto gen cashback transfer")
                return context
            
            logger.info(f"💸 Making Sonauto gen cashback transfer request to: {portal_url}/api/credits/transfer")
            
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{portal_url}/api/credits/transfer",
                    json={
                        "toUserId": actual_user_id,
                        "amount": str(cashback_amount),
                        "reason": f"Sonauto music generation cashback (${cashback_amount:.4f})",
                        "receipt": f"sonauto_gen_cashback_{actual_user_id}_{int(time.time())}"
                    },
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"💸✅ Sonauto gen cashback transfer successful: {cashback_amount:.4f} credits to {actual_user_id}")
                    logger.debug(f"💸 Transfer response: {result}")
                else:
                    error_text = response.text[:200] if hasattr(response, 'text') else str(response.content)[:200]
                    logger.error(f"💸❌ Sonauto gen cashback transfer failed: {response.status_code} - {error_text}")
        
        except Exception as e:
            logger.error(f"💸❌ Error during Sonauto gen cashback transfer: {e}", exc_info=True)
        
        return context
    
    @prompt(priority=10, scope="all")
    def instructions(self, context) -> str:
        return (
            "You can generate original music tracks from text prompts using Sonauto.\n\n"
            "HOW TO USE:\n"
            "   1. Get the user's music preferences (genre, mood, tempo, style)\n"
            "   2. Call generate_music with:\n"
            "      - lyrics_prompt: Lyrics or lyrical theme/description\n"
            "      - tags: List of genre/style tags (e.g., ['pop', 'upbeat', 'electronic'])\n"
            "      - duration: Length in seconds (default 30s, max 180s)\n"
            "      - filename: Optional descriptive name (e.g., 'summer_vibes')\n"
            "   3. The tool returns a markdown link to the generated music\n"
            "   4. Share the link with the user: [🎵 track_name.mp3](url)\n\n"
            "AVAILABLE TAGS (common genres/styles):\n"
            "   - Genres: pop, rock, jazz, classical, electronic, hip-hop, country, indie\n"
            "   - Moods: upbeat, chill, energetic, melancholic, happy, dark, ambient\n"
            "   - Tempos: fast, slow, moderate\n"
            "   - Styles: acoustic, synth, orchestral, minimal, lo-fi\n\n"
            "SPECIFICATIONS:\n"
            "   - Duration: 5-180 seconds (default 30s)\n"
            "   - Format: MP3\n"
            "   - Quality: High-quality AI-generated music\n\n"
            "COST & CASHBACK:\n"
            "   - Cost: $0.26 per track (any duration)\n"
            "   - User receives 100% cashback after successful generation\n"
            "   - Net cost to user: $0 (fully subsidized)\n\n"
            "BEST PRACTICES:\n"
            "   - Combine 2-4 tags for best results\n"
            "   - Be specific about mood and genre\n"
            "   - Use lyrics_prompt to guide the musical theme\n"
            "   - Provide descriptive filenames for easy identification\n\n"
            "COMPARISON WITH INPAINT:\n"
            "   - generate_music: Creates new tracks from scratch\n"
            "   - inpaint_music: Edits existing tracks (lyric replacement)\n"
            "   - Use generate_music for original compositions"
        )
    
    @tool(
        description=(
            "Generate an original music track from a text prompt using Sonauto. "
            "Provide lyrics or a lyrical theme, genre tags, and duration. "
            "Optional: specify a descriptive filename."
        ),
        scope="all"
    )
    @pricing(
        credits_per_call=SONAUTO_GENERATE_BASE_COST * float(os.getenv('ROBUTLER_PLATFORM_MARKUP', '1.75')) * float(os.getenv('CASHBACK_MULTIPLIER', '2')),
        reason="Sonauto music generation API call"
    )
    async def generate_music(
        self,
        lyrics_prompt: str,
        tags: Optional[List[str]] = None,
        duration: int = 30,
        filename: str = ""
    ) -> Dict[str, Any]:
        """
        Generate music using Sonauto V2 text-to-music model.
        
        Args:
            lyrics_prompt: Lyrics or lyrical theme/description
            tags: List of genre/style tags (e.g., ['pop', 'upbeat'])
            duration: Duration in seconds (default 30, max 180)
            filename: Optional descriptive filename (e.g., 'summer_vibes')
            
        Returns:
            Dict with 'audio_url', 'filename', 'status', and 'markdown' keys
        """
        if not self.fal_api_key:
            return {
                "error": "FAL_KEY not configured",
                "status": "configuration_error"
            }
        
        # Validate duration
        if duration < 5:
            duration = 5
            logger.warning(f"Duration too short, using minimum: 5s")
        elif duration > 180:
            duration = 180
            logger.warning(f"Duration too long, using maximum: 180s")
        
        # Default tags if none provided
        if not tags:
            tags = []
        
        logger.info(f"🎵 Sonauto music generation request")
        logger.info(f"   - Lyrics: '{lyrics_prompt[:100]}...'")
        logger.info(f"   - Tags: {tags}")
        logger.info(f"   - Duration: {duration}s")
        logger.info(f"   - FAL_KEY present: {bool(self.fal_api_key)}")
        
        # Build API request
        request_data = {
            "lyrics_prompt": lyrics_prompt,
            "tags": tags,
            "duration": duration
        }
        
        logger.debug(f"🎵 Full JSON request: {request_data}")
        
        try:
            # Use fal_client to make the request
            try:
                import fal_client  # type: ignore
            except ImportError:
                logger.error("fal_client not installed - run: pip install fal-client")
                return {
                    "error": "fal_client library not installed",
                    "status": "dependency_error"
                }
            
            logger.info("🎵 Submitting to Sonauto API...")
            
            # Ensure FAL_KEY is set in environment
            if self.fal_api_key:
                os.environ['FAL_KEY'] = self.fal_api_key
            
            # Submit request with progress tracking
            def on_queue_update(update):
                if hasattr(update, 'logs'):
                    for log in update.logs:
                        msg = log.get('message', '') if isinstance(log, dict) else str(log)
                        if msg:
                            logger.debug(f"🎵 Sonauto: {msg}")
            
            # Use async client for proper async/await
            try:
                # Try async version first
                result = await fal_client.subscribe_async(
                    "sonauto/v2/text-to-music",
                    arguments=request_data,
                    with_logs=True,
                    on_queue_update=on_queue_update
                )
            except AttributeError:
                # Fallback to running sync version in executor
                import asyncio
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: fal_client.subscribe(
                        "sonauto/v2/text-to-music",
                        arguments=request_data,
                        with_logs=True,
                        on_queue_update=on_queue_update
                    )
                )
            
            logger.info(f"✅ Sonauto generation complete")
            logger.debug(f"✅ Full result: {result}")
            
            # Extract audio URL(s) from result
            # Result format: { "audio": [{ "url": "...", "file_name": "...", "content_type": "...", "file_size": ... }], "seed": 123 }
            audio_list = result.get("audio", [])
            if not audio_list:
                return {
                    "error": "No audio returned from Sonauto API",
                    "status": "generation_error"
                }
            
            # Return first audio file
            audio_info = audio_list[0] if isinstance(audio_list, list) else audio_list
            
            # Use provided filename or generate hash-based fallback
            if not filename or not filename.strip():
                short_id = hashlib.md5(f"{lyrics_prompt}{time.time()}".encode()).hexdigest()[:8]
                final_filename = f"sonauto_{short_id}.mp3"
                logger.info(f"🎵 No filename provided, using: {final_filename}")
            else:
                # Clean up provided filename
                final_filename = filename.strip()
                # Ensure it has .mp3 extension
                if not final_filename.lower().endswith('.mp3'):
                    final_filename = f"{final_filename}.mp3"
                # Sanitize filename
                final_filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', final_filename)
                logger.info(f"🎵 Using provided filename: {final_filename}")
            
            # Handle both dict and File object responses
            if isinstance(audio_info, dict):
                fal_audio_url = audio_info.get("url")
                file_size = audio_info.get("file_size")
                content_type = audio_info.get("content_type") or "audio/mpeg"
            else:
                # File object
                fal_audio_url = getattr(audio_info, "url", None)
                file_size = getattr(audio_info, "file_size", None)
                content_type = getattr(audio_info, "content_type", "audio/mpeg")
            
            if not fal_audio_url:
                return {
                    "error": "No audio URL in Sonauto result",
                    "status": "generation_error"
                }
            
            # Download audio from Fal.ai and upload to portal
            try:
                logger.info(f"📥 Downloading generated audio from Fal.ai: {fal_audio_url[:80]}...")
                import httpx
                async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                    resp = await client.get(fal_audio_url)
                    resp.raise_for_status()
                    audio_bytes = resp.content
                    logger.info(f"✅ Downloaded {len(audio_bytes) / 1024 / 1024:.2f} MB")
                
                # Upload to portal content API
                portal_url = os.getenv("ROBUTLER_INTERNAL_API_URL", "https://robutler.ai").rstrip('/')
                upload_url = f"{portal_url}/api/content"
                
                context = get_context()
                agent_id = None
                agent_api_key = None
                
                # Try skill config first (set by dynamic factory)
                agent_api_key = self.config.get('robutler_api_key')
                if agent_api_key:
                    logger.info(f"🔑 Using API key from skill config")
                
                # Try context as fallback
                if not agent_api_key and context:
                    agent_id = context.get("agent_id") or context.get("current_agent_id")
                    if agent_id:
                        logger.info(f"🔑 Found agent_id in context: {agent_id}")
                    
                    # Get agent API key for content upload - try multiple possible keys
                    agent_api_key = (context.get("api_key") or 
                                    context.get("robutler_api_key") or 
                                    context.get("agent_api_key") or
                                    getattr(context, 'api_key', None))
                    
                    if agent_api_key:
                        logger.info(f"🔑 Found agent API key from context")
                    else:
                        logger.warning("⚠️ No agent API key found in context")
                elif not context:
                    logger.warning("⚠️ No context available for upload")
                
                # Get agent_id from context if not set
                if not agent_id and context:
                    agent_id = context.get("agent_id") or context.get("current_agent_id")
                
                if not agent_api_key:
                    logger.error("❌ No API key available for portal upload")
                    return {
                        "audio_url": fal_audio_url,
                        "filename": final_filename,
                        "format": "mp3",
                        "seed": result.get("seed"),
                        "status": "success",
                        "upload_failed": True,
                        "upload_error": "No API key available"
                    }
                
                files = {
                    'file': (final_filename, audio_bytes, content_type)
                }
                data = {
                    'visibility': 'public',
                    'description': f'AI-generated music: {lyrics_prompt[:100]}...' if lyrics_prompt else 'AI-generated music',
                    'tags': json.dumps(['ai-generated', 'sonauto', 'music-generation'] + (tags or [])),
                }
                if agent_id:
                    data['grantAgentAccess'] = json.dumps([agent_id])
                
                headers = {'User-Agent': 'Sonauto-Generate/1.0'}
                if agent_api_key:
                    headers['Authorization'] = f'Bearer {agent_api_key}'
                
                logger.info(f"⬆️  Uploading to portal: {upload_url}")
                async with httpx.AsyncClient(timeout=60.0) as client:
                    upload_resp = await client.post(upload_url, files=files, data=data, headers=headers)
                    if upload_resp.status_code not in (200, 201):
                        logger.error(f"❌ Upload failed: {upload_resp.status_code} {upload_resp.text[:200]}")
                        # Fallback to Fal URL if upload fails
                        return {
                            "audio_url": fal_audio_url,
                            "filename": final_filename,
                            "format": "mp3",
                            "seed": result.get("seed"),
                            "status": "success",
                            "upload_failed": True
                        }
                    
                    meta = upload_resp.json()
                    portal_audio_url = meta.get('url')
                    
                    # Use relative URL for browser compatibility (works with localhost and Tailscale)
                    if portal_audio_url and not portal_audio_url.startswith('http'):
                        # Already relative, use as-is
                        pass
                    elif portal_audio_url and portal_audio_url.startswith('http'):
                        # Convert absolute URL to relative
                        if '/api/content/public' in portal_audio_url:
                            portal_audio_url = '/api/content/public' + portal_audio_url.split('/api/content/public')[1]
                    
                    logger.info(f"✅ Upload complete: {portal_audio_url}")
                    
                    # Track cost in context for finalize hook
                    generate_cost = SONAUTO_GENERATE_BASE_COST * float(os.getenv('ROBUTLER_PLATFORM_MARKUP', '1.75')) * float(os.getenv('CASHBACK_MULTIPLIER', '2'))
                    context = get_context()
                    if context:
                        if not hasattr(context, 'sonauto_gen_costs'):
                            setattr(context, 'sonauto_gen_costs', {'total_charged': 0.0, 'operations': []})
                        
                        sonauto_gen_costs = getattr(context, 'sonauto_gen_costs')
                        sonauto_gen_costs['total_charged'] += generate_cost
                        sonauto_gen_costs['operations'].append({
                            'type': 'music_generation',
                            'cost': generate_cost,
                            'duration': duration
                        })
                        logger.info(f"💰 Tracked generation cost: ${generate_cost:.6f} (total so far: ${sonauto_gen_costs['total_charged']:.6f})")
                    
                    return {
                        "audio_url": portal_audio_url,
                        "filename": meta.get('fileName') or final_filename,
                        "format": "mp3",
                        "seed": result.get("seed"),
                        "duration": duration,
                        "file_size": file_size,
                        "content_type": content_type,
                        "status": "success",
                        "markdown": f"[🎵 {meta.get('fileName') or final_filename}]({portal_audio_url})"
                    }
            
            except Exception as upload_error:
                logger.error(f"❌ Failed to download/upload audio: {upload_error}")
                # Fallback to Fal URL
                return {
                    "audio_url": fal_audio_url,
                    "filename": final_filename,
                    "format": "mp3",
                    "seed": result.get("seed"),
                    "duration": duration,
                    "status": "success",
                    "upload_failed": True,
                    "upload_error": str(upload_error)
                }
        
        except Exception as e:
            logger.error(f"❌ Sonauto generation failed: {e}", exc_info=True)
            logger.error(f"❌ Error type: {type(e).__name__}")
            logger.error(f"❌ Error details: {str(e)}")
            
            return {
                "error": str(e),
                "status": "generation_error"
            }

