"""Text-to-Speech module"""

from .base import TTSProvider
from .local_tts import LocalTTS
from .factory import TTSFactory

# Optional: ElevenLabs (only if installed)
try:
    from .elevenlabs_tts import ElevenLabsTTS
    __all__ = ["TTSProvider", "ElevenLabsTTS", "LocalTTS", "TTSFactory"]
except ImportError:
    __all__ = ["TTSProvider", "LocalTTS", "TTSFactory"]
