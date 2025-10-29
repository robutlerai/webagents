"""
Veo 3.1 Video Generation Skill
Generates 8-second videos using Google's Veo 3.1 models.

Features:
- Two model variants: Fast ($0.15/sec) and Standard ($0.40/sec)
- Fixed 8-second duration per Google API documentation
- 100% cashback after successful generation
- Automatic portal upload for seamless integration
"""
import os
import logging
import json
import time
import hashlib
import re
import asyncio
from typing import Optional, Dict, Any

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, prompt, hook
from webagents.agents.skills.robutler.payments.skill import pricing, PricingInfo
from webagents.server.context.context_vars import get_context

logger = logging.getLogger('webagents.skills.veo')

# Veo 3.1 pricing (per second of video)
VEO_FAST_COST_PER_SEC = 0.15  # $0.15/sec for fast variant
VEO_STANDARD_WITH_AUDIO_COST_PER_SEC = 0.40  # $0.40/sec for standard with audio
VEO_STANDARD_VIDEO_ONLY_COST_PER_SEC = 0.20  # $0.20/sec for standard video only
VEO_VIDEO_DURATION = 8  # Fixed 8 seconds per API docs


class VeoSkill(Skill):
    """Skill for generating videos using Google Veo 3.1 models"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {})
        self.config = config or {}
        self.gemini_api_key = os.getenv('GOOGLE_GEMINI_API_KEY')

        if not self.gemini_api_key:
            logger.warning("GOOGLE_GEMINI_API_KEY not set - Veo video generation will not work")
    
    @hook("finalize_connection", priority=96)
    async def finalize_veo_cashback(self, context) -> Any:
        """Send cashback to user after successful Veo operations"""
        # Only send cashback if payment was successful
        payment_context = getattr(context, 'payments', None)
        if payment_context:
            payment_successful = getattr(payment_context, 'payment_successful', False)
            if not payment_successful:
                logger.info("💸 Skipping Veo cashback - payment was not successful")
                return context
        else:
            logger.warning("💸 Skipping Veo cashback - no payment context available")
            return context
        
        # Get total Veo costs from context
        veo_costs = getattr(context, 'veo_costs', None)
        if not veo_costs:
            return context
        
        total_charged = veo_costs.get('total_charged', 0.0)
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
            logger.warning("💸 Cannot transfer Veo cashback - no caller user_id in auth or payment context")
            return context
        
        cashback_amount = total_charged
        logger.info(f"💸 Transferring Veo cashback of {cashback_amount:.4f} credits to user: {actual_user_id}")
        
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
                logger.error("💸 No agent API key available for Veo cashback transfer")
                return context
            
            logger.info(f"💸 Making Veo cashback transfer request to: {portal_url}/api/credits/transfer")
            
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{portal_url}/api/credits/transfer",
                    json={
                        "toUserId": actual_user_id,
                        "amount": str(cashback_amount),
                        "reason": f"Veo 3.1 video generation cashback (${cashback_amount:.4f})",
                        "receipt": f"veo_cashback_{actual_user_id}_{int(time.time())}"
                    },
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"💸✅ Veo cashback transfer successful: {cashback_amount:.4f} credits to {actual_user_id}")
                    logger.debug(f"💸 Transfer response: {result}")
                else:
                    error_text = response.text[:200] if hasattr(response, 'text') else str(response.content)[:200]
                    logger.error(f"💸❌ Veo cashback transfer failed: {response.status_code} - {error_text}")
        
        except Exception as e:
            logger.error(f"💸❌ Error during Veo cashback transfer: {e}", exc_info=True)
        
        return context
    
    @prompt(priority=10, scope="all")
    def instructions(self, context) -> str:
        return (
            "You can generate 8-second videos using Google Veo 3.1 models.\n\n"
            "AVAILABLE OPTIONS:\n"
            "   - Fast (default): $4.20 per video (faster generation, good quality, with audio)\n"
            "   - Standard with audio: $11.20 per video (best quality with audio)\n"
            "   - Standard video-only: $5.60 per video (best quality without audio)\n\n"
            "VIDEO SPECIFICATIONS:\n"
            "   - Duration: Always 8 seconds (fixed by API)\n"
            "   - Format: MP4\n"
            "   - Resolution: HD quality\n"
            "   - Audio: Optional (specify with_audio parameter)\n\n"
            "HOW TO USE:\n"
            "   1. Get a detailed description from the user about what they want to see\n"
            "   2. Call generate_video with:\n"
            "      - prompt: Detailed description of the video scene/action\n"
            "      - model: 'fast' (default) or 'standard'\n"
            "      - with_audio: True (default) or False\n"
            "      - filename: Optional descriptive name (e.g., 'sunset_beach')\n"
            "   3. The tool returns a markdown link to the generated video\n"
            "   4. Share the link with the user: [🎬 video_name.mp4](url)\n\n"
            "COST & CASHBACK:\n"
            "   - You are charged upfront for generation\n"
            "   - User receives 100% cashback after successful generation\n"
            "   - Net cost to user: $0 (fully subsidized)\n\n"
            "BEST PRACTICES:\n"
            "   - Use detailed, specific prompts for best results\n"
            "   - Describe motion, camera angles, lighting, atmosphere\n"
            "   - Start with 'fast' model unless user requests highest quality\n"
            "   - Include audio by default unless user specifically wants silent video\n"
            "   - Provide descriptive filenames for easy identification"
        )
    
    @tool(
        description=(
            "Generate an 8-second video using Google Veo 3.1. "
            "Provide a detailed prompt describing the scene, action, and visual style. "
            "Optional: specify 'fast' (default) or 'standard' model variant, "
            "with_audio (True/False), and a descriptive filename."
        ),
        scope="all"
    )
    @pricing()  # Dynamic pricing based on model variant and audio option
    async def generate_video(
        self,
        prompt: str,
        model: str = "fast",
        with_audio: bool = True,
        filename: str = ""
    ) -> Dict[str, Any]:
        """
        Generate a video using Google Veo 3.1 models.
        
        Args:
            prompt: Detailed description of the video to generate
            model: 'fast' (default, $0.15/sec) or 'standard' ($0.20-0.40/sec)
            with_audio: Include audio in video (default True). False for video-only (standard model only)
            filename: Optional descriptive filename (e.g., 'sunset_beach')
            
        Returns:
            Dict with 'video_url', 'filename', 'status', and 'markdown' keys
        """
        if not self.gemini_api_key:
            return ({
                "error": "GOOGLE_GEMINI_API_KEY not configured",
                "status": "configuration_error"
            }, PricingInfo(credits=0, reason="Configuration error"))
        
        # Validate model variant
        model = model.lower().strip()
        if model not in ['fast', 'standard']:
            return ({
                "error": f"Invalid model variant '{model}'. Must be 'fast' or 'standard'.",
                "status": "invalid_input"
            }, PricingInfo(credits=0, reason="Invalid model variant"))
        
        # Validate audio option
        if model == 'fast' and not with_audio:
            logger.warning("Fast model always includes audio, ignoring with_audio=False")
            with_audio = True
        
        # Calculate cost based on model and audio option
        if model == 'fast':
            cost_per_sec = VEO_FAST_COST_PER_SEC
            model_description = "fast (with audio)"
        else:  # standard
            if with_audio:
                cost_per_sec = VEO_STANDARD_WITH_AUDIO_COST_PER_SEC
                model_description = "standard (with audio)"
            else:
                cost_per_sec = VEO_STANDARD_VIDEO_ONLY_COST_PER_SEC
                model_description = "standard (video only)"
        
        base_cost = cost_per_sec * VEO_VIDEO_DURATION
        platform_markup = float(os.getenv('ROBUTLER_PLATFORM_MARKUP', '1.75'))
        cashback_multiplier = float(os.getenv('CASHBACK_MULTIPLIER', '2'))
        total_cost = base_cost * platform_markup * cashback_multiplier
        
        logger.info(f"🎬 Veo video generation request")
        logger.info(f"   - Model: veo-3.1-{model}-generate-preview")
        logger.info(f"   - Variant: {model_description}")
        logger.info(f"   - Duration: {VEO_VIDEO_DURATION}s")
        logger.info(f"   - With audio: {with_audio}")
        logger.info(f"   - Prompt: '{prompt[:100]}...'")
        logger.info(f"   - Cost: ${base_cost:.2f} base → ${total_cost:.2f} after markup")
        
        try:
            # Import google.genai
            try:
                from google import genai
                from google.genai import types
            except ImportError:
                logger.error("google-genai not installed - run: pip install google-genai")
                return ({
                    "error": "google-genai library not installed",
                    "status": "dependency_error"
                }, PricingInfo(credits=0, reason="Missing dependency"))
            
            # Initialize Gemini client
            client = genai.Client(api_key=self.gemini_api_key)
            model_name = f"veo-3.1-{model}-generate-preview"
            
            logger.info(f"🎬 Starting video generation with {model_name}...")
            
            # Generate video (async operation)
            generate_video_response = client.models.generate_videos(
                model=model_name,
                prompt=prompt
            )
            
            # Get operation name for polling
            operation_name = generate_video_response.operation.name
            logger.info(f"🎬 Video generation started, operation: {operation_name}")
            
            # Poll for completion
            max_wait_time = 600  # 10 minutes max
            poll_interval = 5  # Check every 5 seconds
            elapsed_time = 0
            
            while elapsed_time < max_wait_time:
                operation = client.operations.get(name=operation_name)
                
                if operation.done:
                    logger.info(f"✅ Video generation complete after {elapsed_time}s")
                    break
                
                logger.debug(f"🎬 Video generation in progress... ({elapsed_time}s elapsed)")
                await asyncio.sleep(poll_interval)
                elapsed_time += poll_interval
            
            if not operation.done:
                logger.error(f"❌ Video generation timed out after {max_wait_time}s")
                return ({
                    "error": f"Video generation timed out after {max_wait_time} seconds",
                    "status": "timeout_error"
                }, PricingInfo(
                    credits=total_cost,
                    reason=f"Veo {model_description} video generation (timeout)",
                    metadata={"model": f"veo-3.1-{model}-generate-preview", "duration": VEO_VIDEO_DURATION, "with_audio": with_audio}
                ))
            
            # Check for errors in operation
            if hasattr(operation, 'error') and operation.error:
                error_msg = str(operation.error)
                logger.error(f"❌ Video generation failed: {error_msg}")
                return ({
                    "error": f"Video generation failed: {error_msg}",
                    "status": "generation_error"
                }, PricingInfo(
                    credits=total_cost,
                    reason=f"Veo {model_description} video generation (failed)",
                    metadata={"model": f"veo-3.1-{model}-generate-preview", "duration": VEO_VIDEO_DURATION, "with_audio": with_audio}
                ))
            
            # Extract video file name from result
            if not hasattr(operation, 'result') or not operation.result:
                logger.error("❌ No result in operation response")
                return ({
                    "error": "No video generated",
                    "status": "generation_error"
                }, PricingInfo(
                    credits=total_cost,
                    reason=f"Veo {model_description} video generation (no result)",
                    metadata={"model": f"veo-3.1-{model}-generate-preview", "duration": VEO_VIDEO_DURATION, "with_audio": with_audio}
                ))
            
            result = operation.result
            
            # Get file name from result
            # Result structure: result.generated_samples[0].video.uri
            if not hasattr(result, 'generated_samples') or not result.generated_samples:
                logger.error("❌ No generated_samples in result")
                return ({
                    "error": "No video samples generated",
                    "status": "generation_error"
                }, PricingInfo(
                    credits=total_cost,
                    reason=f"Veo {model_description} video generation (no samples)",
                    metadata={"model": f"veo-3.1-{model}-generate-preview", "duration": VEO_VIDEO_DURATION, "with_audio": with_audio}
                ))
            
            sample = result.generated_samples[0]
            if not hasattr(sample, 'video') or not sample.video:
                logger.error("❌ No video in sample")
                return ({
                    "error": "No video in generated sample",
                    "status": "generation_error"
                }, PricingInfo(
                    credits=total_cost,
                    reason=f"Veo {model_description} video generation (no video)",
                    metadata={"model": f"veo-3.1-{model}-generate-preview", "duration": VEO_VIDEO_DURATION, "with_audio": with_audio}
                ))
            
            video_uri = sample.video.uri
            logger.info(f"✅ Video generated: {video_uri}")
            
            # Download video from Google
            logger.info(f"📥 Downloading video from Google...")
            video_bytes = client.files.download(file_name=video_uri)
            logger.info(f"✅ Downloaded {len(video_bytes) / 1024 / 1024:.2f} MB")
            
            # Generate filename
            if not filename or not filename.strip():
                # Generate hash-based fallback filename
                short_id = hashlib.md5(f"{prompt}{time.time()}".encode()).hexdigest()[:8]
                final_filename = f"veo_{model}_{short_id}.mp4"
                logger.info(f"🎬 No filename provided, using: {final_filename}")
            else:
                # Clean up provided filename
                final_filename = filename.strip()
                # Ensure it has .mp4 extension
                if not final_filename.lower().endswith('.mp4'):
                    final_filename = f"{final_filename}.mp4"
                # Sanitize filename
                final_filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', final_filename)
                logger.info(f"🎬 Using provided filename: {final_filename}")
            
            # Upload to portal
            try:
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
                    
                    # Get agent API key for content upload
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
                    return ({
                        "video_url": video_uri,
                        "filename": final_filename,
                        "format": "mp4",
                        "status": "success",
                        "upload_failed": True,
                        "upload_error": "No API key available"
                    }, PricingInfo(
                        credits=total_cost,
                        reason=f"Veo {model_description} video generation ({VEO_VIDEO_DURATION}s)",
                        metadata={"model": model_name, "duration": VEO_VIDEO_DURATION, "with_audio": with_audio}
                    ))
                
                files = {
                    'file': (final_filename, video_bytes, 'video/mp4')
                }
                data = {
                    'visibility': 'public',
                    'description': f'AI-generated video: {prompt[:100]}...' if prompt else 'AI-generated video',
                    'tags': json.dumps(['ai-generated', 'veo', 'video-generation', model]),
                }
                if agent_id:
                    data['grantAgentAccess'] = json.dumps([agent_id])
                
                headers = {'User-Agent': 'Veo-Skill/1.0'}
                if agent_api_key:
                    headers['Authorization'] = f'Bearer {agent_api_key}'
                
                logger.info(f"⬆️  Uploading to portal: {upload_url}")
                import httpx
                async with httpx.AsyncClient(timeout=120.0) as client:
                    upload_resp = await client.post(upload_url, files=files, data=data, headers=headers)
                    if upload_resp.status_code not in (200, 201):
                        logger.error(f"❌ Upload failed: {upload_resp.status_code} {upload_resp.text[:200]}")
                        # Fallback to Google URL if upload fails
                        return ({
                            "video_url": video_uri,
                            "filename": final_filename,
                            "format": "mp4",
                            "status": "success",
                            "upload_failed": True
                        }, PricingInfo(
                            credits=total_cost,
                            reason=f"Veo {model} video generation ({VEO_VIDEO_DURATION}s)",
                            metadata={"model": model_name, "duration": VEO_VIDEO_DURATION}
                        ))
                    
                    meta = upload_resp.json()
                    portal_video_url = meta.get('url')
                    
                    # Use relative URL for browser compatibility
                    if portal_video_url and not portal_video_url.startswith('http'):
                        # Already relative, use as-is
                        pass
                    elif portal_video_url and portal_video_url.startswith('http'):
                        # Convert absolute URL to relative
                        if '/api/content/public' in portal_video_url:
                            portal_video_url = '/api/content/public' + portal_video_url.split('/api/content/public')[1]
                    
                    logger.info(f"✅ Upload complete: {portal_video_url}")
                    
                    # Track cost in context for finalize hook
                    context = get_context()
                    if context:
                        if not hasattr(context, 'veo_costs'):
                            setattr(context, 'veo_costs', {'total_charged': 0.0, 'operations': []})
                        
                        veo_costs = getattr(context, 'veo_costs')
                        veo_costs['total_charged'] += total_cost
                        veo_costs['operations'].append({
                            'type': 'video_generation',
                            'model': model,
                            'cost': total_cost,
                            'duration': VEO_VIDEO_DURATION
                        })
                        logger.info(f"💰 Tracked video generation cost: ${total_cost:.2f} (total so far: ${veo_costs['total_charged']:.2f})")
                    
                    return ({
                        "video_url": portal_video_url,
                        "filename": meta.get('fileName') or final_filename,
                        "format": "mp4",
                        "duration": VEO_VIDEO_DURATION,
                        "model": model_name,
                        "status": "success",
                        "markdown": f"[🎬 {meta.get('fileName') or final_filename}]({portal_video_url})"
                    }, PricingInfo(
                        credits=total_cost,
                        reason=f"Veo {model_description} video generation ({VEO_VIDEO_DURATION}s)",
                        metadata={"model": model_name, "duration": VEO_VIDEO_DURATION, "with_audio": with_audio}
                    ))
            
            except Exception as upload_error:
                logger.error(f"❌ Failed to upload video: {upload_error}")
                # Fallback to Google URL
                return ({
                    "video_url": video_uri,
                    "filename": final_filename,
                    "format": "mp4",
                    "duration": VEO_VIDEO_DURATION,
                    "model": model_name,
                    "status": "success",
                    "upload_failed": True,
                    "upload_error": str(upload_error)
                }, PricingInfo(
                    credits=total_cost,
                    reason=f"Veo {model_description} video generation ({VEO_VIDEO_DURATION}s)",
                    metadata={"model": model_name, "duration": VEO_VIDEO_DURATION, "with_audio": with_audio}
                ))
        
        except Exception as e:
            logger.error(f"❌ Veo video generation failed: {e}", exc_info=True)
            return ({
                "error": str(e),
                "status": "generation_error"
            }, PricingInfo(
                credits=total_cost,
                reason=f"Veo {model_description} video generation (error)",
                metadata={"model": f"veo-3.1-{model}-generate-preview", "duration": VEO_VIDEO_DURATION, "with_audio": with_audio, "error": str(e)}
            ))

