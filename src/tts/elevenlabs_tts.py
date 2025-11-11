"""ElevenLabs TTS integration"""

import os
import logging
from pathlib import Path
from typing import Optional

from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings

from .base import TTSProvider

logger = logging.getLogger(__name__)


class ElevenLabsTTS(TTSProvider):
    """Text-to-Speech using ElevenLabs API"""

    def __init__(self, api_key: Optional[str] = None, config=None):
        """
        Initialize ElevenLabs TTS.

        Args:
            api_key: ElevenLabs API key (or set ELEVENLABS_API_KEY env var)
            config: Optional configuration object
        """
        if api_key is None:
            api_key = os.getenv("ELEVENLABS_API_KEY")

        if not api_key:
            raise ValueError(
                "ELEVENLABS_API_KEY not found in environment variables or provided"
            )

        self.client = ElevenLabs(api_key=api_key)
        self.config = config

        logger.info("Initialized ElevenLabs TTS")

    def generate_speech(
        self,
        text: str,
        voice_id: str,
        save_path: Optional[str] = None,
        **kwargs
    ) -> bytes:
        """
        Generate speech from text using specified voice.

        Args:
            text: Text to convert to speech
            voice_id: ElevenLabs voice ID
            save_path: Optional path to save audio file
            **kwargs: Additional parameters (model, stability, similarity_boost)

        Returns:
            Audio bytes
        """
        # Extract kwargs with defaults
        model = kwargs.get('model', 'eleven_multilingual_v2')
        stability = kwargs.get('stability', 0.5)
        similarity_boost = kwargs.get('similarity_boost', 0.75)

        try:
            logger.debug(f"Generating speech for text: {text[:50]}...")

            # Generate audio
            audio = self.client.generate(
                text=text,
                voice=Voice(
                    voice_id=voice_id,
                    settings=VoiceSettings(
                        stability=stability,
                        similarity_boost=similarity_boost,
                    )
                ),
                model=model,
            )

            # Convert generator to bytes
            audio_bytes = b"".join(audio)

            # Save if path provided
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(audio_bytes)
                logger.info(f"Audio saved to: {save_path}")

            return audio_bytes

        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            raise

    def play_audio(self, audio_bytes: bytes):
        """
        Play audio bytes.

        Args:
            audio_bytes: Audio data to play
        """
        try:
            import simpleaudio as sa
            from io import BytesIO
            from pydub import AudioSegment

            # Convert to playable format
            audio = AudioSegment.from_mp3(BytesIO(audio_bytes))
            playback = sa.play_buffer(
                audio.raw_data,
                num_channels=audio.channels,
                bytes_per_sample=audio.sample_width,
                sample_rate=audio.frame_rate
            )

            logger.debug("Playing audio...")
            playback.wait_for_done()

        except ImportError as e:
            logger.error(f"Audio playback dependencies not installed: {e}")
            logger.error("Install with: pip install simpleaudio pydub")
            raise
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            raise

    def list_voices(self):
        """List all available voices"""
        try:
            voices = self.client.voices.get_all()
            return voices.voices
        except Exception as e:
            logger.error(f"Error listing voices: {e}")
            raise

    def clone_voice(
        self,
        name: str,
        audio_files: list[str],
        description: Optional[str] = None,
    ) -> str:
        """
        Clone a voice from audio files.

        Args:
            name: Name for the cloned voice
            audio_files: List of paths to audio files
            description: Optional description

        Returns:
            Voice ID of cloned voice
        """
        try:
            logger.info(f"Cloning voice '{name}' from {len(audio_files)} file(s)...")

            # Open audio files
            files = []
            for path in audio_files:
                with open(path, "rb") as f:
                    files.append(f.read())

            # Clone voice
            voice = self.client.clone(
                name=name,
                description=description or f"Cloned voice: {name}",
                files=files,
            )

            voice_id = voice.voice_id
            logger.info(f"Voice cloned successfully! Voice ID: {voice_id}")

            return voice_id

        except Exception as e:
            logger.error(f"Error cloning voice: {e}")
            raise

    def get_voice_info(self, voice_id: str):
        """Get information about a voice"""
        try:
            voice = self.client.voices.get(voice_id)
            return voice
        except Exception as e:
            logger.error(f"Error getting voice info: {e}")
            raise
