"""TTS provider factory"""

import logging
from typing import Optional

from .base import TTSProvider
from .local_tts import LocalTTS

# Optional: ElevenLabs (only if installed)
try:
    from .elevenlabs_tts import ElevenLabsTTS
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False

logger = logging.getLogger(__name__)


class TTSFactory:
    """Factory for creating TTS provider instances"""

    @staticmethod
    def create(config, api_key: Optional[str] = None) -> TTSProvider:
        """
        Create a TTS provider based on configuration.

        Args:
            config: Configuration object with tts settings
            api_key: Optional API key (for ElevenLabs)

        Returns:
            TTSProvider instance

        Raises:
            ValueError: If provider is not supported
        """
        provider = config.tts.provider.lower()

        if provider == "elevenlabs":
            if not ELEVENLABS_AVAILABLE:
                raise ValueError(
                    "ElevenLabs provider requested but not installed. "
                    "Install with: pip install elevenlabs"
                )
            return ElevenLabsTTS(api_key=api_key, config=config)
        elif provider == "local" or provider == "xtts":
            return LocalTTS(config=config)
        else:
            raise ValueError(
                f"Unsupported TTS provider: {provider}. "
                f"Supported providers: elevenlabs, local"
            )

    @staticmethod
    def list_providers():
        """List all available TTS providers with voice cloning support"""
        return {
            "local": {
                "name": "Coqui TTS (XTTS-v2)",
                "description": "Free local neural TTS with voice cloning",
                "requires_api_key": False,
                "supports_cloning": True,
                "quality": "High",
                "speed": "0.5-2s/sentence with GPU, 2-5s/sentence CPU-only",
                "optimizations": "GPU (MPS/CUDA) + Streaming",
                "cost": "Free",
                "install": "pip install TTS torch"
            },
            "elevenlabs": {
                "name": "ElevenLabs",
                "description": "Cloud-based TTS with professional voice cloning",
                "requires_api_key": True,
                "supports_cloning": True,
                "quality": "Very High",
                "speed": "0.5-1s/sentence",
                "optimizations": "N/A (cloud service)",
                "cost": "Paid API ($5-11/month)",
                "install": "pip install elevenlabs"
            }
        }
