"""Base TTS provider interface"""

from abc import ABC, abstractmethod
from typing import Optional


class TTSProvider(ABC):
    """Abstract base class for Text-to-Speech providers"""

    @abstractmethod
    def generate_speech(
        self,
        text: str,
        voice_id: str,
        save_path: Optional[str] = None,
        **kwargs
    ) -> bytes:
        """
        Generate speech from text.

        Args:
            text: Text to convert to speech
            voice_id: Voice identifier (provider-specific)
            save_path: Optional path to save audio file
            **kwargs: Provider-specific parameters

        Returns:
            Audio bytes
        """
        pass

    @abstractmethod
    def play_audio(self, audio_bytes: bytes):
        """
        Play audio bytes through speakers.

        Args:
            audio_bytes: Audio data to play
        """
        pass

    @abstractmethod
    def list_voices(self):
        """List all available voices for this provider"""
        pass

    def clone_voice(
        self,
        name: str,
        audio_files: list[str],
        description: Optional[str] = None,
    ) -> str:
        """
        Clone a voice from audio files (optional, not all providers support this).

        Args:
            name: Name for the cloned voice
            audio_files: List of paths to audio files
            description: Optional description

        Returns:
            Voice ID of cloned voice

        Raises:
            NotImplementedError: If provider doesn't support voice cloning
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support voice cloning"
        )

    def get_voice_info(self, voice_id: str):
        """
        Get information about a voice (optional).

        Args:
            voice_id: Voice identifier

        Returns:
            Voice information

        Raises:
            NotImplementedError: If provider doesn't support voice info
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support voice info retrieval"
        )
