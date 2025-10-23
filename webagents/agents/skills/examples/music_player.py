"""Example Music Player Skill demonstrating widget functionality"""

import html
import json
from typing import Optional
from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import widget
from webagents.server.context.context_vars import Context


class MusicPlayerSkill(Skill):
    """Example skill that demonstrates interactive widget rendering
    
    This skill shows how to create an interactive music player widget
    that can communicate with the chat interface via postMessage.
    """
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__(config or {})
        self.name = "MusicPlayer"
    
    async def initialize(self, agent):
        """Initialize the music player skill"""
        await super().initialize(agent)
        # Logger is now available after super().initialize()
        if hasattr(self, 'logger'):
            self.logger.info("🎵 MusicPlayerSkill initialized")
    
    @widget(
        name="play_music",
        description="Display an interactive music player for a given track. Use this to play music for the user.",
        scope="all"
    )
    async def play_music(
        self,
        song_url: str,
        title: str,
        artist: str = "Unknown Artist",
        album: str = "Unknown Album",
        context: Context = None
    ) -> str:
        """Create an interactive music player widget
        
        Args:
            song_url: URL of the audio file to play
            title: Title of the song
            artist: Artist name (optional)
            album: Album name (optional)
            context: Request context (auto-injected)
        
        Returns:
            HTML widget wrapped in <widget> tags
        """
        # Get the frontend origin for absolute URLs (needed for blob iframe context)
        # Note: We need the CHAT UI origin (where /api/proxy-media lives), not the agents API origin
        request_origin = 'http://localhost:3001'  # Default for development (chat UI port)
        
        if context and hasattr(context, 'request') and context.request:
            # Try to get the Referer or Origin header which points to the frontend
            request_headers = getattr(context.request, 'headers', {})
            if hasattr(request_headers, 'get'):
                # Check Referer header first (shows where the request came from)
                referer = request_headers.get('referer') or request_headers.get('origin')
                if hasattr(self, 'logger'):
                    self.logger.debug(f"🌐 Request headers - referer: {referer}, origin: {request_headers.get('origin')}")
                if referer:
                    try:
                        from urllib.parse import urlparse
                        parsed_referer = urlparse(referer)
                        request_origin = f"{parsed_referer.scheme}://{parsed_referer.netloc}"
                        if hasattr(self, 'logger'):
                            self.logger.debug(f"🌐 Extracted origin from referer: {request_origin}")
                    except Exception as e:
                        if hasattr(self, 'logger'):
                            self.logger.warning(f"⚠️ Failed to parse referer: {e}")
                        pass  # Use default if parsing fails
        
        # Proxy external audio URLs to bypass CORS restrictions
        # Only proxy truly external URLs (not localhost or relative paths)
        from urllib.parse import quote, urlparse
        proxied_song_url = song_url
        if song_url and (song_url.startswith('http://') or song_url.startswith('https://')):
            # Check if it's not a same-origin URL (localhost or relative)
            parsed = urlparse(song_url)
            is_localhost = parsed.hostname in ('localhost', '127.0.0.1')
            
            if not is_localhost:
                # Use absolute URL for blob iframe context (relative URLs don't work in blob:)
                proxied_song_url = f"{request_origin}/api/proxy-media?url={quote(song_url)}"
                if hasattr(self, 'logger'):
                    self.logger.debug(f"🎵 Proxying external audio: {song_url} -> {proxied_song_url}")
        
        # Create a modern, compact music player with custom controls
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: transparent;
        }}
        .progress-bar {{
            transition: width 0.1s linear;
        }}
        .play-button {{
            transition: all 0.2s ease;
        }}
        .play-button:hover {{
            transform: scale(1.05);
        }}
        .volume-slider {{
            -webkit-appearance: none;
            appearance: none;
            background: transparent;
            cursor: pointer;
        }}
        .volume-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 12px;
            height: 12px;
            background: white;
            border-radius: 50%;
            cursor: pointer;
        }}
        .volume-slider::-moz-range-thumb {{
            width: 12px;
            height: 12px;
            background: white;
            border-radius: 50%;
            border: none;
            cursor: pointer;
        }}
    </style>
</head>
<body>
             <!-- Hidden audio element -->
             <audio id="audioPlayer" preload="metadata">
                 <source src="{proxied_song_url}" type="audio/mpeg">
             </audio>
    
    <!-- Compact Music Player -->
    <div class="w-full bg-gradient-to-r from-purple-900 via-purple-800 to-indigo-900 rounded-xl shadow-2xl overflow-hidden">
        <div class="p-4">
            <!-- Top Row: Song Info + Controls -->
            <div class="flex items-center gap-4 mb-3">
                <!-- Album Art Icon -->
                <div class="flex-shrink-0 w-14 h-14 bg-gradient-to-br from-purple-400 to-pink-500 rounded-lg flex items-center justify-center shadow-lg">
                    <svg class="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M18 3a1 1 0 00-1.196-.98l-10 2A1 1 0 006 5v9.114A4.369 4.369 0 005 14c-1.657 0-3 .895-3 2s1.343 2 3 2 3-.895 3-2V7.82l8-1.6v5.894A4.37 4.37 0 0015 12c-1.657 0-3 .895-3 2s1.343 2 3 2 3-.895 3-2V3z"/>
                    </svg>
                </div>
                
                 <!-- Song Info -->
                 <div class="flex-1 min-w-0">
                     <h3 class="text-white font-bold text-base truncate">{title}</h3>
                     <p class="text-purple-200 text-sm truncate">{artist}</p>
                 </div>
                
                <!-- Play/Pause Button -->
                <button id="playPauseBtn" class="play-button flex-shrink-0 w-12 h-12 bg-white rounded-full flex items-center justify-center shadow-lg hover:shadow-xl">
                    <svg id="playIcon" class="w-6 h-6 text-purple-900" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z"/>
                    </svg>
                    <svg id="pauseIcon" class="w-6 h-6 text-purple-900 hidden" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M5.75 3a.75.75 0 00-.75.75v12.5c0 .414.336.75.75.75h1.5a.75.75 0 00.75-.75V3.75A.75.75 0 007.25 3h-1.5zM12.75 3a.75.75 0 00-.75.75v12.5c0 .414.336.75.75.75h1.5a.75.75 0 00.75-.75V3.75a.75.75 0 00-.75-.75h-1.5z"/>
                    </svg>
                </button>
            </div>
            
            <!-- Progress Bar -->
            <div class="mb-2">
                <div class="flex items-center gap-2 text-xs text-purple-200 mb-1">
                    <span id="currentTime">0:00</span>
                    <div class="flex-1 h-1.5 bg-purple-950 rounded-full overflow-hidden cursor-pointer" id="progressContainer">
                        <div id="progressBar" class="progress-bar h-full bg-gradient-to-r from-pink-400 to-purple-400 rounded-full" style="width: 0%"></div>
                    </div>
                    <span id="duration">0:00</span>
                </div>
            </div>
            
            <!-- Bottom Row: Volume + Actions -->
            <div class="flex items-center gap-2">
                <!-- Volume Control -->
                <div class="flex items-center gap-2 flex-1 min-w-0 max-w-[140px]">
                    <svg class="w-4 h-4 text-purple-300 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M10.5 3.75a.75.75 0 00-1.264-.546L5.203 7H3.167a.75.75 0 00-.7.48A6.985 6.985 0 002 10c0 .887.165 1.737.468 2.52a.75.75 0 00.7.48h2.036l4.033 3.796a.75.75 0 001.264-.546V3.75zM16.45 5.05a.75.75 0 00-1.06 1.06 5.5 5.5 0 010 7.78.75.75 0 001.06 1.06 7 7 0 000-9.9z"/>
                    </svg>
                    <input type="range" id="volumeSlider" class="volume-slider flex-1 h-1 bg-purple-700 rounded-lg min-w-0" min="0" max="100" value="70">
                </div>
                
                <!-- Action Buttons (responsive width) -->
                 <button onclick="sendChatMessage('Play next song')" class="px-3 py-1.5 bg-purple-700 hover:bg-purple-600 text-white text-sm font-medium rounded-lg transition-colors whitespace-nowrap">
                     Next
                 </button>
                 <button onclick="sendChatMessage('More by {artist}')" class="px-3 py-1.5 bg-purple-700 hover:bg-purple-600 text-white text-sm font-medium rounded-lg transition-colors whitespace-nowrap">
                     More
                 </button>
            </div>
        </div>
    </div>
    
    <script>
        // Audio element and UI controls
        const audio = document.getElementById('audioPlayer');
        const playPauseBtn = document.getElementById('playPauseBtn');
        const playIcon = document.getElementById('playIcon');
        const pauseIcon = document.getElementById('pauseIcon');
        const progressBar = document.getElementById('progressBar');
        const progressContainer = document.getElementById('progressContainer');
        const currentTimeEl = document.getElementById('currentTime');
        const durationEl = document.getElementById('duration');
        const volumeSlider = document.getElementById('volumeSlider');
        
        // Format time as MM:SS
        function formatTime(seconds) {{
            if (isNaN(seconds)) return '0:00';
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${{mins}}:${{secs.toString().padStart(2, '0')}}`;
        }}
        
        // Play/Pause toggle
        playPauseBtn.addEventListener('click', () => {{
            if (audio.paused) {{
                audio.play();
                playIcon.classList.add('hidden');
                pauseIcon.classList.remove('hidden');
                console.log('▶️ Playing');
            }} else {{
                audio.pause();
                playIcon.classList.remove('hidden');
                pauseIcon.classList.add('hidden');
                console.log('⏸️ Paused');
            }}
        }});
        
        // Update progress bar
        audio.addEventListener('timeupdate', () => {{
            if (audio.duration) {{
                const progress = (audio.currentTime / audio.duration) * 100;
                progressBar.style.width = progress + '%';
                currentTimeEl.textContent = formatTime(audio.currentTime);
            }}
        }});
        
        // Update duration when metadata loads
        audio.addEventListener('loadedmetadata', () => {{
            durationEl.textContent = formatTime(audio.duration);
            console.log('✅ Duration loaded:', audio.duration, 'seconds');
        }});
        
        // Seek on progress bar click
        progressContainer.addEventListener('click', (e) => {{
            const rect = progressContainer.getBoundingClientRect();
            const percent = (e.clientX - rect.left) / rect.width;
            audio.currentTime = percent * audio.duration;
            console.log('⏩ Seeked to:', formatTime(audio.currentTime));
        }});
        
        // Volume control (only if slider exists - hidden on mobile)
        if (volumeSlider) {{
            volumeSlider.addEventListener('input', (e) => {{
                audio.volume = e.target.value / 100;
            }});
        }}
        
        // Set initial volume
        audio.volume = 0.7;
        
        // Auto-reset play button on end
        audio.addEventListener('ended', () => {{
            playIcon.classList.remove('hidden');
            pauseIcon.classList.add('hidden');
            progressBar.style.width = '0%';
            console.log('⏹️ Playback ended');
        }});
        
        // Send messages to chat
        function sendChatMessage(text) {{
            window.parent.postMessage({{
                type: 'widget_message',
                content: text
            }}, '*');
            console.log('💬 Sent message:', text);
        }}
        
        // Send height to parent
        function sendHeight() {{
            const height = document.body.scrollHeight;
            window.parent.postMessage({{
                type: 'widget_resize',
                height: height
            }}, '*');
        }}
        
        window.addEventListener('load', sendHeight);
        setTimeout(sendHeight, 100);
        
        // Error handling
        audio.addEventListener('error', (e) => {{
            console.error('❌ Audio error:', audio.error?.code, audio.error?.message);
            alert('Failed to load audio. Please try a different track.');
        }});
        
        console.log('🎵 Custom music player initialized');
        console.log('🎵 Audio src:', audio.querySelector('source')?.src);
    </script>
</body>
</html>"""
        
        # Prepare data attribute with widget metadata
        widget_data = {
            'song_url': song_url,
            'title': title,
            'artist': artist,
            'album': album
        }
        
        # Escape data for safe HTML attribute embedding
        escaped_data = html.escape(json.dumps(widget_data), quote=True)
        
        # Return widget wrapped in <widget> tags with kind="webagents"
        return f'<widget kind="webagents" id="music_player" data="{escaped_data}">{html_content}</widget>'

