"""Hermes agent implementation for self-hosted deployment"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from openai import OpenAI
from .base import BaseAgent

logger = logging.getLogger(__name__)


class HermesAgent(BaseAgent):
    """Agent using Hermes 70B (self-hosted via vLLM)"""

    def __init__(
        self,
        persona_id: str,
        config=None,
        base_url: Optional[str] = None,
        model: str = "NousResearch/Hermes-2-Pro-Llama-3-70B",
    ):
        """
        Initialize Hermes agent.

        Args:
            persona_id: Persona identifier (e.g., "jules", "heidegger")
            config: Optional configuration object
            base_url: vLLM endpoint URL (default from config)
            model: Hermes model identifier
        """
        super().__init__(persona_id, config)

        if base_url is None:
            base_url = self.config.model.hermes.base_url

        self.client = OpenAI(
            api_key="not-needed",  # Local inference doesn't need API key
            base_url=base_url,
        )
        self.model = model
        self.max_iterations = self.config.model.hermes.max_iterations

        logger.info(f"Initialized HermesAgent with model: {model} at {base_url}")

    def _get_model_specific_prompt(self) -> Optional[str]:
        """Get Hermes-specific prompt additions"""
        prompt_path = Path(self.config.agent.system_prompt_dir) / "hermes.txt"
        with open(prompt_path, "r") as f:
            return f.read()

    def _call_model(self, system: str, messages: List[Dict]) -> Any:
        """Call Hermes via vLLM API"""
        # Add system message to messages
        full_messages = [{"role": "system", "content": system}] + messages

        tools = [self.search_tool.get_tool_definition_openai()]

        # Add incremental reasoning tool if enabled
        if self.config.retrieval.incremental_mode.enabled:
            tools.append(self.reasoning_tool.get_tool_definition_openai())

        # Determine tool_choice based on config and iteration state
        tool_choice = "required" if self._should_force_tool_use() else "auto"

        return self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=self.config.model.hermes.temperature,
            max_tokens=self.config.model.hermes.max_tokens,
        )

    def _is_complete(self, response: Any) -> bool:
        """Check if Hermes has finished"""
        finish_reason = response.choices[0].finish_reason
        return finish_reason in ["stop", "end_turn"]

    def _parse_tool_use(self, response: Any) -> List[Dict]:
        """Extract tool calls from Hermes response"""
        message = response.choices[0].message
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return []

        tools = []
        for tool_call in message.tool_calls:
            try:
                # Hermes sometimes outputs malformed JSON
                arguments = json.loads(tool_call.function.arguments)
                tools.append(
                    {
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "input": arguments,
                    }
                )
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool arguments: {e}")
                continue

        return tools

    def _extract_text(self, response: Any) -> str:
        """Extract text from Hermes response"""
        return response.choices[0].message.content or ""

    def _update_messages(
        self, messages: List[Dict], response: Any, tool_results: List[Any]
    ) -> List[Dict]:
        """Update messages with Hermes's response and tool results"""
        message = response.choices[0].message

        # Only add tool calls if they exist
        if hasattr(message, "tool_calls") and message.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": message.tool_calls,
                }
            )

            # Add tool results
            for tool_call, result in zip(message.tool_calls, tool_results):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    }
                )
        else:
            # No tool calls, just add the message
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                }
            )

        return messages
