"""Claude agent implementation"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from anthropic import Anthropic
from .base import BaseAgent

logger = logging.getLogger(__name__)


class ClaudeAgent(BaseAgent):
    """Agent using Claude Sonnet 4.5"""

    def __init__(
        self,
        persona_id: str,
        config=None,
        model: str = "claude-sonnet-4.5-20250929",
    ):
        """
        Initialize Claude agent.

        Args:
            persona_id: Persona identifier (e.g., "jules", "heidegger")
            config: Optional configuration object
            model: Claude model identifier
        """
        super().__init__(persona_id, config)

        # Get API key
        api_key = self.config.get_api_key("claude")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment variables"
            )

        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_iterations = self.config.model.claude.max_iterations

        logger.info(f"Initialized ClaudeAgent with model: {model}")

    def _call_model(self, system: str, messages: List[Dict]) -> Any:
        """Call Claude API"""
        return self.client.messages.create(
            model=self.model,
            system=system,
            messages=messages,
            tools=[self.search_tool.get_tool_definition_claude()],
            max_tokens=self.config.model.claude.max_tokens,
            temperature=self.config.model.claude.temperature,
        )

    def _is_complete(self, response: Any) -> bool:
        """Check if Claude has finished"""
        return response.stop_reason == "end_turn"

    def _parse_tool_use(self, response: Any) -> List[Dict]:
        """Extract tool calls from Claude response"""
        tools = []
        for block in response.content:
            if block.type == "tool_use":
                tools.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
        return tools

    def _extract_text(self, response: Any) -> str:
        """Extract text from Claude response"""
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""

    def _update_messages(
        self, messages: List[Dict], response: Any, tool_results: List[Any]
    ) -> List[Dict]:
        """Update messages with Claude's response and tool results"""
        # Add assistant message
        messages.append({"role": "assistant", "content": response.content})

        # Add tool results
        tool_uses = self._parse_tool_use(response)
        tool_result_content = []

        for tool_use, result in zip(tool_uses, tool_results):
            tool_result_content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use["id"],
                    "content": str(result),
                }
            )

        if tool_result_content:
            messages.append({"role": "user", "content": tool_result_content})

        return messages
