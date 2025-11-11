"""Local TTS with voice cloning using Coqui TTS (XTTS-v2)"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Optional, Generator
import json
import re
from concurrent.futures import ThreadPoolExecutor
import threading

from .base import TTSProvider

logger = logging.getLogger(__name__)


class LocalTTS(TTSProvider):
    """
    Local Text-to-Speech with voice cloning using Coqui TTS (XTTS-v2).

    XTTS-v2 is a state-of-the-art open-source TTS model that supports:
    - Voice cloning from short audio samples (6+ seconds)
    - Multiple languages
    - High quality synthesis
    - Runs entirely locally (no API calls)

    Installation:
        pip install TTS

    Repository: https://github.com/coqui-ai/TTS
    """

    def __init__(self, config=None):
        """
        Initialize Local TTS with Coqui XTTS-v2.

        Args:
            config: Optional configuration object
        """
        self.config = config
        self.model = None
        self.voices_dir = Path.home() / ".local" / "share" / "castor" / "voices"
        self.voices_dir.mkdir(parents=True, exist_ok=True)

        # Voice metadata file
        self.voices_file = self.voices_dir / "voices.json"
        self.voices_metadata = self._load_voices_metadata()

        # Lazy load the model (only when needed)
        logger.info("Initialized Local TTS (Coqui XTTS-v2)")

    def _load_model(self):
        """Lazy load the TTS model with GPU acceleration if available"""
        if self.model is not None:
            return

        try:
            from TTS.api import TTS as CoquiTTS
            import torch

            # Detect best available device
            if torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon GPU
                logger.info("Using MPS (Apple Silicon GPU) acceleration")
            elif torch.cuda.is_available():
                device = "cuda"  # NVIDIA GPU
                logger.info("Using CUDA GPU acceleration")
            else:
                device = "cpu"
                logger.info("Using CPU (GPU not available)")

            logger.info("Loading XTTS-v2 model (this may take a minute on first run)...")

            # Load model with GPU acceleration
            self.model = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            self.device = device

            logger.info(f"XTTS-v2 model loaded successfully on {device}")
        except ImportError:
            raise ImportError(
                "Coqui TTS not installed. Install with: pip install TTS"
            )
        except Exception as e:
            logger.error(f"Error loading TTS model: {e}")
            raise

    def _load_voices_metadata(self):
        """Load voices metadata from disk"""
        if self.voices_file.exists():
            with open(self.voices_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_voices_metadata(self):
        """Save voices metadata to disk"""
        with open(self.voices_file, 'w') as f:
            json.dump(self.voices_metadata, f, indent=2)

    def generate_speech(
        self,
        text: str,
        voice_id: str,
        save_path: Optional[str] = None,
        **kwargs
    ) -> bytes:
        """
        Generate speech from text using cloned voice.

        Args:
            text: Text to convert to speech
            voice_id: Voice identifier (reference audio file path or voice name)
            save_path: Optional path to save audio file
            **kwargs: Additional parameters (language, speaker_wav for inline cloning)

        Returns:
            Audio bytes (WAV format)
        """
        try:
            self._load_model()

            logger.debug(f"Generating speech for text: {text[:50]}...")

            # Determine speaker audio file
            if voice_id in self.voices_metadata:
                # Use registered voice
                speaker_wav = self.voices_metadata[voice_id]["audio_path"]
                language = kwargs.get('language', self.voices_metadata[voice_id].get('language', 'en'))
            else:
                # Try to use voice_id as direct path to audio file
                speaker_wav = voice_id
                language = kwargs.get('language', 'en')

                if not Path(speaker_wav).exists():
                    raise ValueError(
                        f"Voice '{voice_id}' not found. "
                        f"Either register it with clone_voice() or provide path to audio file."
                    )

            # Create temporary file if no save path provided
            if save_path is None:
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                output_path = temp_file.name
                temp_file.close()
            else:
                output_path = save_path
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Generate speech with voice cloning
            self.model.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
                file_path=output_path
            )

            # Read generated audio
            with open(output_path, "rb") as f:
                audio_bytes = f.read()

            # Clean up temporary file if we created one
            if save_path is None:
                try:
                    os.unlink(output_path)
                except:
                    pass
            else:
                logger.info(f"Audio saved to: {output_path}")

            return audio_bytes

        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            raise

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences for streaming"""
        # Simple sentence splitting (could be enhanced with nltk)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def generate_speech_streaming(
        self,
        text: str,
        voice_id: str,
        **kwargs
    ) -> Generator[bytes, None, None]:
        """
        Generate speech in chunks (sentence by sentence) for streaming.

        Args:
            text: Text to convert to speech
            voice_id: Voice identifier
            **kwargs: Additional parameters

        Yields:
            Audio bytes for each sentence
        """
        sentences = self._split_into_sentences(text)

        for sentence in sentences:
            if not sentence:
                continue

            try:
                audio_bytes = self.generate_speech(
                    text=sentence,
                    voice_id=voice_id,
                    save_path=None,
                    **kwargs
                )
                yield audio_bytes
            except Exception as e:
                logger.error(f"Error generating sentence '{sentence[:50]}...': {e}")
                continue

    def play_audio_async(self, audio_bytes: bytes):
        """
        Play audio asynchronously (non-blocking).

        Args:
            audio_bytes: Audio data to play
        """
        def _play():
            try:
                self.play_audio(audio_bytes)
            except Exception as e:
                logger.error(f"Error in async playback: {e}")

        thread = threading.Thread(target=_play, daemon=True)
        thread.start()
        return thread

    def play_audio(self, audio_bytes: bytes):
        """
        Play audio bytes.

        Args:
            audio_bytes: Audio data to play (WAV format)
        """
        try:
            import simpleaudio as sa
            from io import BytesIO
            from pydub import AudioSegment

            # Convert to playable format
            audio = AudioSegment.from_wav(BytesIO(audio_bytes))
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
        """List all registered voices"""
        voices = []
        for voice_id, metadata in self.voices_metadata.items():
            voices.append({
                "voice_id": voice_id,
                "name": metadata.get("name", voice_id),
                "description": metadata.get("description", "Custom voice"),
                "language": metadata.get("language", "en"),
                "audio_path": metadata["audio_path"]
            })
        return voices

    def clone_voice(
        self,
        name: str,
        audio_files: list[str],
        description: Optional[str] = None,
    ) -> str:
        """
        Register a voice for cloning from audio file(s).

        For XTTS-v2, only the first audio file is used. It should be:
        - At least 6 seconds long
        - Clear speech without background noise
        - Single speaker

        Args:
            name: Name for the voice
            audio_files: List of paths to audio files (first one will be used)
            description: Optional description

        Returns:
            Voice ID (same as name)
        """
        try:
            if not audio_files:
                raise ValueError("At least one audio file is required")

            reference_audio = audio_files[0]

            if not Path(reference_audio).exists():
                raise FileNotFoundError(f"Audio file not found: {reference_audio}")

            logger.info(f"Registering voice '{name}' from {reference_audio}...")

            # Copy reference audio to voices directory
            voice_id = name.lower().replace(" ", "_")

            # Keep original file extension - XTTS supports multiple formats
            ref_audio_path = Path(reference_audio)
            audio_dest = self.voices_dir / f"{voice_id}{ref_audio_path.suffix}"

            # Just copy the audio file - XTTS handles multiple formats natively
            import shutil
            shutil.copy(reference_audio, audio_dest)

            logger.info(f"Audio file copied (format: {ref_audio_path.suffix})")

            # Store metadata
            self.voices_metadata[voice_id] = {
                "name": name,
                "description": description or f"Cloned voice: {name}",
                "audio_path": str(audio_dest),
                "language": "en",  # Default, can be changed
                "original_file": reference_audio
            }
            self._save_voices_metadata()

            logger.info(f"Voice '{name}' registered successfully! Voice ID: {voice_id}")
            logger.info(f"Reference audio stored at: {audio_dest}")

            return voice_id

        except Exception as e:
            logger.error(f"Error registering voice: {e}")
            raise

    def get_voice_info(self, voice_id: str):
        """Get information about a registered voice"""
        if voice_id not in self.voices_metadata:
            raise ValueError(f"Voice '{voice_id}' not found")

        return self.voices_metadata[voice_id]

    def delete_voice(self, voice_id: str):
        """Delete a registered voice"""
        if voice_id not in self.voices_metadata:
            raise ValueError(f"Voice '{voice_id}' not found")

        # Delete audio file
        audio_path = Path(self.voices_metadata[voice_id]["audio_path"])
        if audio_path.exists():
            audio_path.unlink()

        # Remove from metadata
        del self.voices_metadata[voice_id]
        self._save_voices_metadata()

        logger.info(f"Voice '{voice_id}' deleted")
