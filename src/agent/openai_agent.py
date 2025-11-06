"""OpenAI agent implementation (GPT-4, GPT-3.5)"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Generator
from pathlib import Path

from openai import OpenAI
from .base import BaseAgent

logger = logging.getLogger(__name__)


class OpenAIAgent(BaseAgent):
    """Agent using OpenAI GPT models"""

    def __init__(
        self,
        persona_id: str,
        config=None,
        model: str = "gpt-4o",
    ):
        """
        Initialize OpenAI agent.

        Args:
            persona_id: Persona identifier (e.g., "jules", "heidegger")
            config: Optional configuration object
            model: OpenAI model identifier (gpt-4, gpt-4-turbo-preview, gpt-3.5-turbo)
        """
        super().__init__(persona_id, config)

        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables"
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_iterations = 20

        logger.info(f"Initialized OpenAIAgent with model: {model}")

    def _call_model(self, system: str, messages: List[Dict]) -> Any:
        """Call OpenAI API"""
        # Add system message to messages (OpenAI-style)
        full_messages = [{"role": "system", "content": system}] + messages

        return self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            tools=[self.search_tool.get_tool_definition_openai()],
            temperature=1.0,
        )

    def _is_complete(self, response: Any) -> bool:
        """Check if OpenAI has finished"""
        finish_reason = response.choices[0].finish_reason
        return finish_reason in ["stop", "end_turn"]

    def _parse_tool_use(self, response: Any) -> List[Dict]:
        """Extract tool calls from OpenAI response"""
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
        """Extract text from OpenAI response"""
        return response.choices[0].message.content or ""

    def _update_messages(
        self, messages: List[Dict], response: Any, tool_results: List[Any]
    ) -> List[Dict]:
        """Update messages with OpenAI's response and tool results"""
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

    def respond_stream(self, query: str, conversation_history: Optional[List[Dict]] = None) -> Generator[str, None, Dict]:
        """
        Stream response from the agent.

        Args:
            query: User query
            conversation_history: Optional conversation history

        Yields:
            Text chunks as they arrive

        Returns:
            Final result dict with metadata
        """
        system_prompt = self._build_system_prompt()

        # Start with conversation history if provided
        if conversation_history:
            messages = conversation_history.copy()
            messages.append({"role": "user", "content": query})
        else:
            messages = [{"role": "user", "content": query}]

        tool_calls_log = []

        logger.info(f"Starting streaming agent loop for query: {query[:100]}...")

        for iteration in range(self.max_iterations):
            logger.debug(f"Iteration {iteration + 1}/{self.max_iterations}")

            try:
                # Add system message
                full_messages = [{"role": "system", "content": system_prompt}] + messages

                # Call with streaming
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=full_messages,
                    tools=[self.search_tool.get_tool_definition_openai()],
                    temperature=1.0,
                    stream=True,
                )

                # Collect response
                collected_content = ""
                collected_tool_calls = []

                for chunk in stream:
                    delta = chunk.choices[0].delta

                    # Handle content
                    if delta.content:
                        collected_content += delta.content
                        yield delta.content

                    # Handle tool calls
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        for tool_call_delta in delta.tool_calls:
                            # Initialize tool call if needed
                            if tool_call_delta.index >= len(collected_tool_calls):
                                collected_tool_calls.append({
                                    "id": tool_call_delta.id or "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""}
                                })

                            # Update tool call
                            tc = collected_tool_calls[tool_call_delta.index]
                            if tool_call_delta.id:
                                tc["id"] = tool_call_delta.id
                            if tool_call_delta.function:
                                if tool_call_delta.function.name:
                                    tc["function"]["name"] = tool_call_delta.function.name
                                if tool_call_delta.function.arguments:
                                    tc["function"]["arguments"] += tool_call_delta.function.arguments

                    # Check if done
                    if chunk.choices[0].finish_reason in ["stop", "end_turn"]:
                        logger.info(f"Agent completed in {iteration + 1} iterations with {len(tool_calls_log)} tool calls")
                        return {
                            "response": collected_content,
                            "tool_calls": tool_calls_log,
                            "iterations": iteration + 1,
                            "model": self.__class__.__name__,
                        }

                # Handle tool calls if present
                if collected_tool_calls:
                    # Add assistant message with tool calls
                    messages.append({
                        "role": "assistant",
                        "content": collected_content,
                        "tool_calls": collected_tool_calls,
                    })

                    # Execute tools
                    tool_results = []
                    for tool_call in collected_tool_calls:
                        tool_use = {
                            "id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "input": json.loads(tool_call["function"]["arguments"]),
                        }
                        result = self._execute_tool(tool_use)
                        tool_results.append(result)
                        tool_calls_log.append({
                            "tool": tool_use["name"],
                            "input": tool_use["input"],
                            "result_count": len(result) if isinstance(result, list) else 1,
                        })

                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": str(result),
                        })

                    # Continue to next iteration for final response
                    continue

            except Exception as e:
                logger.error(f"Error in streaming iteration {iteration + 1}: {e}")
                raise

        # Max iterations reached
        logger.warning(f"Max iterations ({self.max_iterations}) reached")
        return {
            "response": collected_content if collected_content else "Max iterations reached",
            "tool_calls": tool_calls_log,
            "iterations": self.max_iterations,
            "model": self.__class__.__name__,
        }
