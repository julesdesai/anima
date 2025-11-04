"""Configuration management for Castor"""

import os
from typing import Optional, List
from pathlib import Path
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class UserConfig(BaseModel):
    """User configuration"""
    name: str
    corpus_path: str


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


class VectorDBConfig(BaseModel):
    """Vector database configuration"""
    provider: str = "qdrant"
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "user_corpus"


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


class RetrievalConfig(BaseModel):
    """Retrieval configuration"""
    default_k: int = 5
    max_k: int = 20
    similarity_threshold: float = 0.7
    style_pack_enabled: bool = False
    style_pack_size: int = 10


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


class Config(BaseModel):
    """Main configuration"""
    user: UserConfig
    model: ModelConfig
    agent: AgentConfig
    vector_db: VectorDBConfig
    embedding: EmbeddingConfig
    corpus: CorpusConfig
    retrieval: RetrievalConfig
    style: StyleConfig
    cost_tracking: CostTrackingConfig

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
