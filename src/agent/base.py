"""Base agent class for multi-model support"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path

from .tools import CorpusSearchTool
from ..config import get_config

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all model agents"""

    def __init__(self, persona_id: str, config=None):
        """
        Initialize base agent.

        Args:
            persona_id: Persona identifier (e.g., "jules", "heidegger")
            config: Optional configuration object
        """
        if config is None:
            config = get_config()

        self.config = config
        self.persona_id = persona_id
        self.persona = config.get_persona(persona_id)
        self.user_name = self.persona.name  # For backwards compatibility
        self.max_iterations = 20
        self.search_tool = CorpusSearchTool(self.persona.collection_name, config)

    @abstractmethod
    def _call_model(self, system: str, messages: List[Dict]) -> Any:
        """
        Call the underlying model API.

        Args:
            system: System prompt
            messages: Conversation messages

        Returns:
            Model response object
        """
        pass

    @abstractmethod
    def _parse_tool_use(self, response: Any) -> List[Dict]:
        """
        Extract tool calls from model response.

        Args:
            response: Model response object

        Returns:
            List of tool calls with format: [{"id": str, "name": str, "input": dict}]
        """
        pass

    @abstractmethod
    def _is_complete(self, response: Any) -> bool:
        """
        Check if model has finished responding.

        Args:
            response: Model response object

        Returns:
            True if model is done, False if it wants to use tools
        """
        pass

    @abstractmethod
    def _extract_text(self, response: Any) -> str:
        """
        Extract text content from model response.

        Args:
            response: Model response object

        Returns:
            Text content
        """
        pass

    @abstractmethod
    def _update_messages(
        self, messages: List[Dict], response: Any, tool_results: List[Any]
    ) -> List[Dict]:
        """
        Update message list with assistant response and tool results.

        Args:
            messages: Current message list
            response: Model response
            tool_results: Results from tool execution

        Returns:
            Updated message list
        """
        pass

    def _build_system_prompt(self) -> str:
        """
        Build system prompt from template.

        Returns:
            System prompt string
        """
        # Load base prompt
        prompt_dir = Path(self.config.agent.system_prompt_dir)
        base_prompt_path = prompt_dir / "base.txt"

        with open(base_prompt_path, "r") as f:
            base_prompt = f.read()

        # Format with user name
        prompt = base_prompt.format(user_name=self.user_name)

        # Add model-specific additions
        model_specific = self._get_model_specific_prompt()
        if model_specific:
            prompt += "\n\n" + model_specific.format(user_name=self.user_name)

        # Add style pack for grounding (diverse writing samples)
        if self.config.retrieval.style_pack_enabled:
            style_pack = self.search_tool.get_style_pack()
            if style_pack:
                prompt += "\n\n" + "="*70
                prompt += f"\n\nSTYLE GROUNDING - {self.user_name}'s Writing Examples:\n"
                prompt += "The following are representative samples of how {user_name} writes. Use these to match their style, tone, and communication patterns.\n\n"

                for i, sample in enumerate(style_pack, 1):
                    source = sample["metadata"].get("source", "unknown")
                    prompt += f"\n--- Example {i} (from {source}) ---\n"
                    # Truncate if too long
                    text = sample["text"]
                    if len(text) > 500:
                        text = text[:500] + "..."
                    prompt += text + "\n"

                prompt += "\n" + "="*70

        return prompt

    def _get_model_specific_prompt(self) -> Optional[str]:
        """
        Get model-specific prompt additions.
        Override in subclasses.

        Returns:
            Model-specific prompt or None
        """
        return None

    def _execute_tool(self, tool_use: Dict) -> Any:
        """
        Execute a tool call.

        Args:
            tool_use: Tool call dict with name and input

        Returns:
            Tool execution result
        """
        if tool_use["name"] == "search_corpus":
            try:
                result = self.search_tool.search(**tool_use["input"])
                logger.debug(f"Tool search returned {len(result)} results")
                return result
            except Exception as e:
                logger.error(f"Error executing search_corpus: {e}")
                return {"error": str(e)}
        else:
            logger.error(f"Unknown tool: {tool_use['name']}")
            return {"error": f"Unknown tool: {tool_use['name']}"}

    def respond(self, query: str, conversation_history: Optional[List[Dict]] = None) -> Dict:
        """
        Main agent loop - model self-orchestrates retrieval.

        Args:
            query: User query
            conversation_history: Optional list of previous messages [{"role": "user/assistant", "content": "..."}]

        Returns:
            Dict with response, tool_calls, iterations, and model name
        """
        system_prompt = self._build_system_prompt()

        # Start with conversation history if provided
        if conversation_history:
            messages = conversation_history.copy()
            messages.append({"role": "user", "content": query})
        else:
            messages = [{"role": "user", "content": query}]

        tool_calls_log = []

        logger.info(f"Starting agent loop for query: {query[:100]}...")

        for iteration in range(self.max_iterations):
            logger.debug(f"Iteration {iteration + 1}/{self.max_iterations}")

            try:
                # Call model
                response = self._call_model(system_prompt, messages)

                # Check if model is done
                if self._is_complete(response):
                    final_response = self._extract_text(response)
                    logger.info(
                        f"Agent completed in {iteration + 1} iterations with {len(tool_calls_log)} tool calls"
                    )
                    return {
                        "response": final_response,
                        "tool_calls": tool_calls_log,
                        "iterations": iteration + 1,
                        "model": self.__class__.__name__,
                    }

                # Execute tool calls
                tool_uses = self._parse_tool_use(response)
                if tool_uses:
                    tool_results = []
                    for tool_use in tool_uses:
                        result = self._execute_tool(tool_use)
                        tool_results.append(result)
                        tool_calls_log.append(
                            {
                                "tool": tool_use["name"],
                                "input": tool_use["input"],
                                "result_count": len(result)
                                if isinstance(result, list)
                                else 1,
                            }
                        )

                    # Update conversation
                    messages = self._update_messages(messages, response, tool_results)
                else:
                    # No tools but not complete - add response and continue
                    text = self._extract_text(response)
                    if text:
                        logger.info(f"Agent completed with text response")
                        return {
                            "response": text,
                            "tool_calls": tool_calls_log,
                            "iterations": iteration + 1,
                            "model": self.__class__.__name__,
                        }

            except Exception as e:
                logger.error(f"Error in iteration {iteration + 1}: {e}")
                return {
                    "response": f"Error: {str(e)}",
                    "tool_calls": tool_calls_log,
                    "iterations": iteration + 1,
                    "model": self.__class__.__name__,
                    "error": str(e),
                }

        logger.warning(f"Max iterations ({self.max_iterations}) reached")
        return {
            "response": "Max iterations reached without completion",
            "tool_calls": tool_calls_log,
            "iterations": self.max_iterations,
            "model": self.__class__.__name__,
        }
