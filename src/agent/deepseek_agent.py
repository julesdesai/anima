"""DeepSeek agent implementation"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from openai import OpenAI
from .base import BaseAgent

logger = logging.getLogger(__name__)


class DeepSeekAgent(BaseAgent):
    """Agent using DeepSeek R1 or V3"""

    def __init__(
        self,
        persona_id: str,
        config=None,
        model: str = "deepseek-reasoner",
    ):
        """
        Initialize DeepSeek agent.

        Args:
            persona_id: Persona identifier (e.g., "jules", "heidegger")
            config: Optional configuration object
            model: DeepSeek model identifier (deepseek-reasoner or deepseek-chat)
        """
        super().__init__(persona_id, config)

        # Get API key
        api_key = self.config.get_api_key("deepseek")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not found in environment variables"
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.config.model.deepseek.base_url,
        )
        self.model = model
        self.max_iterations = self.config.model.deepseek.max_iterations

        logger.info(f"Initialized DeepSeekAgent with model: {model}")

    def _get_model_specific_prompt(self) -> Optional[str]:
        """Get DeepSeek-specific prompt additions"""
        prompt_path = Path(self.config.agent.system_prompt_dir) / "deepseek.txt"
        with open(prompt_path, "r") as f:
            return f.read()

    def _call_model(self, system: str, messages: List[Dict]) -> Any:
        """Call DeepSeek API"""
        # Add system message to messages (OpenAI-style)
        full_messages = [{"role": "system", "content": system}] + messages

        return self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            tools=[self.search_tool.get_tool_definition_openai()],
            temperature=self.config.model.deepseek.temperature,
        )

    def _is_complete(self, response: Any) -> bool:
        """Check if DeepSeek has finished"""
        finish_reason = response.choices[0].finish_reason
        return finish_reason in ["stop", "end_turn"]

    def _parse_tool_use(self, response: Any) -> List[Dict]:
        """Extract tool calls from DeepSeek response"""
        message = response.choices[0].message
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return []

        tools = []
        for tool_call in message.tool_calls:
            tools.append(
                {
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "input": json.loads(tool_call.function.arguments),
                }
            )
        return tools

    def _extract_text(self, response: Any) -> str:
        """Extract text from DeepSeek response"""
        return response.choices[0].message.content or ""

    def _update_messages(
        self, messages: List[Dict], response: Any, tool_results: List[Any]
    ) -> List[Dict]:
        """Update messages with DeepSeek's response and tool results"""
        message = response.choices[0].message

        # Add assistant message with tool calls
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": message.tool_calls,
            }
        )

        # Add tool results
        if message.tool_calls:
            for tool_call, result in zip(message.tool_calls, tool_results):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    }
                )

        return messages
