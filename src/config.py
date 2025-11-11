"""Configuration management for Anima"""

import os
from typing import Optional, List
from pathlib import Path
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class PersonaConfig(BaseModel):
    """Persona configuration"""
    name: str
    corpus_path: str
    collection_name: str
    description: Optional[str] = None
    # Optional per-persona overrides
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    similarity_threshold: Optional[float] = None
    # TTS configuration
    voice_id: Optional[str] = None  # Voice identifier for TTS
    voice_enabled: bool = False  # Enable TTS for this persona


class ModelSpecificConfig(BaseModel):
    """Model-specific configuration"""
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 1.0
    max_iterations: int = 20


class ModelConfig(BaseModel):
    """Model configuration"""
    primary: str
    fallback: str
    openai: ModelSpecificConfig
    claude: ModelSpecificConfig
    deepseek: ModelSpecificConfig
    hermes: ModelSpecificConfig


class AgentConfig(BaseModel):
    """Agent configuration"""
    max_tool_calls_per_iteration: int = 3
    system_prompt_dir: str = "src/agent/prompts/"
    force_tool_use: bool = True  # Require model to use tools


class VectorDBConfig(BaseModel):
    """Vector database configuration"""
    provider: str = "qdrant"
    host: str = "localhost"
    port: int = 6333


class EmbeddingConfig(BaseModel):
    """Embedding configuration"""
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 100


class CorpusConfig(BaseModel):
    """Corpus processing configuration"""
    chunk_size: int = 800
    chunk_overlap: int = 100
    min_chunk_length: int = 100
    file_types: List[str] = Field(default_factory=lambda: [".txt", ".md", ".email", ".json"])


class IncrementalModeConfig(BaseModel):
    """Incremental reasoning configuration for OOD queries"""
    enabled: bool = True
    ood_check_model: str = "gpt-4o-mini"
    max_corpus_concepts: int = 5


class RetrievalConfig(BaseModel):
    """Retrieval configuration"""
    default_k: int = 5
    max_k: int = 20
    similarity_threshold: float = 0.7
    style_pack_enabled: bool = False
    style_pack_size: int = 10
    incremental_mode: IncrementalModeConfig = Field(default_factory=IncrementalModeConfig)


class StyleConfig(BaseModel):
    """Style verification configuration"""
    verification_enabled: bool = False
    similarity_threshold: float = 0.75
    verification_method: str = "embedding"


class CostTrackingConfig(BaseModel):
    """Cost tracking configuration"""
    enabled: bool = True
    log_path: str = "logs/costs.json"
    budget_alert_threshold: float = 10.0


class TTSConfig(BaseModel):
    """Text-to-Speech configuration"""
    enabled: bool = False
    provider: str = "local"  # "local", "kokoro", or "elevenlabs"
    use_streaming: bool = True  # Stream audio sentence-by-sentence
    use_gpu: bool = True  # Use GPU acceleration if available
    api_key_env: str = "ELEVENLABS_API_KEY"
    model: str = "eleven_multilingual_v2"
    voice_stability: float = 0.5
    voice_similarity_boost: float = 0.75
    auto_play: bool = True
    save_audio: bool = False
    audio_output_dir: str = "outputs/audio"


class Config(BaseModel):
    """Main configuration"""
    personas: dict[str, PersonaConfig]
    default_persona: str
    model: ModelConfig
    agent: AgentConfig
    vector_db: VectorDBConfig
    embedding: EmbeddingConfig
    corpus: CorpusConfig
    retrieval: RetrievalConfig
    style: StyleConfig
    cost_tracking: CostTrackingConfig
    tts: TTSConfig = Field(default_factory=TTSConfig)

    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "Config":
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)

    def get_api_key(self, model_type: str) -> Optional[str]:
        """Get API key for specific model type"""
        model_config = getattr(self.model, model_type, None)
        if model_config and model_config.api_key_env:
            return os.getenv(model_config.api_key_env)
        return None

    def get_persona(self, persona_id: Optional[str] = None) -> PersonaConfig:
        """
        Get persona configuration by ID.

        Args:
            persona_id: Persona identifier. If None, returns default persona.

        Returns:
            PersonaConfig for the requested persona

        Raises:
            ValueError: If persona_id not found
        """
        if persona_id is None:
            persona_id = self.default_persona

        if persona_id not in self.personas:
            available = ", ".join(self.personas.keys())
            raise ValueError(
                f"Persona '{persona_id}' not found. Available personas: {available}"
            )

        return self.personas[persona_id]


# Global config instance
_config: Optional[Config] = None


def get_config(config_path: str = "config.yaml") -> Config:
    """Get or create global configuration instance"""
    global _config
    if _config is None:
        _config = Config.from_yaml(config_path)
    return _config


def reload_config(config_path: str = "config.yaml") -> Config:
    """Reload configuration from file"""
    global _config
    _config = Config.from_yaml(config_path)
    return _config
