"""Agent factory for creating model-specific agents"""

import logging
from typing import Optional

from .base import BaseAgent
from .claude_agent import ClaudeAgent
from .deepseek_agent import DeepSeekAgent
from .hermes_agent import HermesAgent
from .openai_agent import OpenAIAgent
from ..config import get_config, Config

logger = logging.getLogger(__name__)


class AgentFactory:
    """Factory for creating appropriate agent based on model selection"""

    @staticmethod
    def create(
        model_name: str,
        persona_id: str,
        config: Optional[Config] = None,
    ) -> BaseAgent:
        """
        Create an agent instance for the specified model.

        Args:
            model_name: Model identifier (e.g., "claude", "deepseek", "hermes")
            persona_id: Persona identifier (e.g., "jules", "heidegger")
            config: Optional configuration object

        Returns:
            Agent instance

        Raises:
            ValueError: If model_name is not supported
        """
        if config is None:
            config = get_config()

        model_name_lower = model_name.lower()
        persona = config.get_persona(persona_id)

        # OpenAI (gpt-4, gpt-3.5, etc)
        if "gpt" in model_name_lower or "openai" in model_name_lower:
            logger.info(f"Creating OpenAIAgent for persona: {persona.name}")
            return OpenAIAgent(
                persona_id=persona_id,
                config=config,
                model=model_name if model_name.startswith("gpt") else config.model.openai.model,
            )

        # Claude
        elif "claude" in model_name_lower:
            logger.info(f"Creating ClaudeAgent for persona: {persona.name}")
            return ClaudeAgent(
                persona_id=persona_id,
                config=config,
                model=model_name if model_name.startswith("claude") else config.model.primary,
            )

        # DeepSeek
        elif "deepseek" in model_name_lower:
            logger.info(f"Creating DeepSeekAgent for persona: {persona.name}")
            return DeepSeekAgent(
                persona_id=persona_id,
                config=config,
                model=model_name if model_name.startswith("deepseek") else config.model.deepseek.model,
            )

        # Hermes
        elif "hermes" in model_name_lower:
            logger.info(f"Creating HermesAgent for persona: {persona.name}")
            return HermesAgent(
                persona_id=persona_id,
                config=config,
                model=model_name if "/" in model_name else config.model.hermes.model,
            )

        else:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Supported: openai (gpt-4, gpt-3.5), claude, deepseek, hermes"
            )

    @staticmethod
    def create_primary(persona_id: str, config: Optional[Config] = None) -> BaseAgent:
        """
        Create agent using the primary model from config.

        Args:
            persona_id: Persona identifier (e.g., "jules", "heidegger")
            config: Optional configuration object

        Returns:
            Agent instance for primary model
        """
        if config is None:
            config = get_config()

        logger.info(f"Creating primary agent: {config.model.primary}")
        return AgentFactory.create(config.model.primary, persona_id, config)

    @staticmethod
    def create_fallback(persona_id: str, config: Optional[Config] = None) -> BaseAgent:
        """
        Create agent using the fallback model from config.

        Args:
            persona_id: Persona identifier (e.g., "jules", "heidegger")
            config: Optional configuration object

        Returns:
            Agent instance for fallback model
        """
        if config is None:
            config = get_config()

        logger.info(f"Creating fallback agent: {config.model.fallback}")
        return AgentFactory.create(config.model.fallback, persona_id, config)
