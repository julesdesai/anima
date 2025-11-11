#!/usr/bin/env python3
"""Debate system for having two animas engage in dialogue"""

import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console, Group
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich import box

from src.agent.factory import AgentFactory
from src.config import get_config

console = Console()
logger = logging.getLogger(__name__)


class DebateMode(ABC):
    """Base class for debate orchestration modes"""

    def __init__(self, agent1, agent2, persona1, persona2, config):
        self.agent1 = agent1
        self.agent2 = agent2
        self.persona1 = persona1
        self.persona2 = persona2
        self.config = config
        self.conversation_history = []

    @abstractmethod
    def run_debate(self, topic: str, turns: int) -> List[Dict[str, Any]]:
        """Run the debate and return transcript"""
        pass

    def _add_to_history(self, speaker: str, content: str, persona_name: str):
        """Add a turn to conversation history"""
        self.conversation_history.append({
            "speaker": speaker,
            "persona_name": persona_name,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def _get_agent_and_persona(self, speaker: str):
        """Get agent and persona for speaker"""
        if speaker == "persona1":
            return self.agent1, self.persona1
        else:
            return self.agent2, self.persona2

    def _display_turn(self, persona_name: str, content: str, turn_num: int = None):
        """Display a debate turn"""
        title = f"[bold cyan]{persona_name} Anima"
        if turn_num is not None:
            title += f" (Turn {turn_num})"

        md = Markdown(content)
        console.print()
        console.print(Panel(
            md,
            title=title,
            border_style="cyan",
            box=box.ROUNDED,
            padding=(1, 2)
        ))


class SequentialDebateMode(DebateMode):
    """Simple turn-based debate: A speaks, B responds, A responds, etc."""

    def run_debate(self, topic: str, turns: int) -> List[Dict[str, Any]]:
        console.print(f"\n[bold green]Starting Sequential Debate[/bold green]")
        console.print(f"[dim]Topic: {topic}[/dim]")
        console.print(f"[dim]Format: Alternating turns between personas[/dim]\n")

        # First turn - persona1 addresses the topic
        current_speaker = "persona1"

        for turn in range(turns):
            agent, persona = self._get_agent_and_persona(current_speaker)

            # Build prompt based on turn
            if turn == 0:
                # Opening statement
                prompt = f"The topic is: {topic}\n\nGive your opening statement on this topic."
            else:
                # Response to previous speaker
                other_persona = self.persona2.name if current_speaker == "persona1" else self.persona1.name
                prompt = f"Respond to {other_persona}'s previous point."

            # Get response
            console.print(f"[dim]⚙ {persona.name} is thinking...[/dim]")

            # Convert conversation history to agent format
            agent_history = []
            for entry in self.conversation_history:
                role = "assistant" if entry["speaker"] == current_speaker else "user"
                agent_history.append({
                    "role": role,
                    "content": entry["content"]
                })

            result = agent.respond(prompt, conversation_history=agent_history)
            response = result["response"]

            # Display and record
            self._display_turn(persona.name, response, turn + 1)
            self._add_to_history(current_speaker, response, persona.name)

            # Switch speaker
            current_speaker = "persona2" if current_speaker == "persona1" else "persona1"

        return self.conversation_history


class ModeratedDebateMode(DebateMode):
    """Structured debate with moderator questions and independent responses"""

    def run_debate(self, topic: str, turns: int) -> List[Dict[str, Any]]:
        console.print(f"\n[bold green]Starting Moderated Debate[/bold green]")
        console.print(f"[dim]Topic: {topic}[/dim]")
        console.print(f"[dim]Format: Moderator poses questions, both respond[/dim]\n")

        # Opening question
        questions = [
            f"Let's begin our discussion on: {topic}. What is your position?",
            f"How do you respond to each other's opening positions?",
            f"Can you address the key disagreement between your views?",
            f"What would you say is the strongest aspect of your opponent's argument?",
            f"How would you defend your position against these critiques?",
            f"What are the implications of your respective positions?",
        ]

        for turn in range(min(turns, len(questions))):
            question = questions[turn]

            # Display moderator question
            console.print()
            console.print(Panel(
                Text(question, style="yellow"),
                title="[bold yellow]Moderator",
                border_style="yellow",
                box=box.ROUNDED
            ))

            # Get responses from both personas
            for speaker, agent, persona in [
                ("persona1", self.agent1, self.persona1),
                ("persona2", self.agent2, self.persona2)
            ]:
                console.print(f"[dim]⚙ {persona.name} is thinking...[/dim]")

                # Build context: moderator question + other persona's previous responses
                agent_history = []
                for entry in self.conversation_history:
                    if entry.get("is_moderator"):
                        # Don't include moderator questions in history
                        continue
                    role = "assistant" if entry["speaker"] == speaker else "user"
                    agent_history.append({
                        "role": role,
                        "content": f"{entry['persona_name']}: {entry['content']}"
                    })

                result = agent.respond(question, conversation_history=agent_history)
                response = result["response"]

                # Display and record
                self._display_turn(persona.name, response, turn + 1)
                self._add_to_history(speaker, response, persona.name)

        return self.conversation_history


class FreeFormDebateMode(DebateMode):
    """Natural dialogue where personas decide when to interject"""

    def run_debate(self, topic: str, turns: int) -> List[Dict[str, Any]]:
        console.print(f"\n[bold green]Starting Free-Form Debate[/bold green]")
        console.print(f"[dim]Topic: {topic}[/dim]")
        console.print(f"[dim]Format: Personas naturally engage and interject[/dim]\n")

        current_speaker = "persona1"

        for turn in range(turns):
            agent, persona = self._get_agent_and_persona(current_speaker)
            other_persona = self.persona2.name if current_speaker == "persona1" else self.persona1.name

            # Build prompt that encourages natural dialogue
            if turn == 0:
                prompt = f"You are in a philosophical dialogue with {other_persona} about: {topic}\n\nBegin the discussion."
            else:
                prompt = f"You are in dialogue with {other_persona}. Respond to their point, raise objections, ask questions, or develop the discussion as you see fit."

            console.print(f"[dim]⚙ {persona.name} is thinking...[/dim]")

            # Convert history
            agent_history = []
            for entry in self.conversation_history:
                role = "assistant" if entry["speaker"] == current_speaker else "user"
                agent_history.append({
                    "role": role,
                    "content": f"{entry['persona_name']}: {entry['content']}"
                })

            result = agent.respond(prompt, conversation_history=agent_history)
            response = result["response"]

            # Display and record
            self._display_turn(persona.name, response, turn + 1)
            self._add_to_history(current_speaker, response, persona.name)

            # Decide who speaks next (alternate, but could be more sophisticated)
            current_speaker = "persona2" if current_speaker == "persona1" else "persona1"

        return self.conversation_history


def main():
    parser = argparse.ArgumentParser(description="Have two animas debate a topic")
    parser.add_argument(
        "--persona1",
        "-p1",
        type=str,
        required=True,
        help="First persona (e.g., 'heidegger')"
    )
    parser.add_argument(
        "--persona2",
        "-p2",
        type=str,
        required=True,
        help="Second persona (e.g., 'wittgenstein')"
    )
    parser.add_argument(
        "--topic",
        "-t",
        type=str,
        required=True,
        help="Debate topic"
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["sequential", "moderated", "freeform"],
        default="sequential",
        help="Debate mode: sequential (alternating turns), moderated (structured with questions), freeform (natural dialogue)"
    )
    parser.add_argument(
        "--turns",
        "-n",
        type=int,
        default=6,
        help="Number of turns/exchanges"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use for both personas"
    )
    parser.add_argument(
        "--save",
        "-s",
        type=str,
        default=None,
        help="Path to save debate transcript"
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Configure logging
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s: %(message)s",
        )

    # Load config
    config = get_config()

    # Create agents
    try:
        if args.model:
            agent1 = AgentFactory.create(args.model, args.persona1, config)
            agent2 = AgentFactory.create(args.model, args.persona2, config)
        else:
            agent1 = AgentFactory.create_primary(args.persona1, config)
            agent2 = AgentFactory.create_primary(args.persona2, config)

        persona1 = config.get_persona(args.persona1)
        persona2 = config.get_persona(args.persona2)

    except ConnectionError as e:
        console.print(f"\n[bold red]Connection Error:[/]")
        console.print(f"[red]{str(e)}[/]\n")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error initializing agents:[/] {e}\n")
        if args.debug:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/]")
        sys.exit(1)

    # Display header
    header = Table.grid(padding=1)
    header.add_column(style="cyan", justify="right")
    header.add_column(style="white")

    header.add_row("System:", "Debate")
    header.add_row("Mode:", args.mode.title())
    header.add_row("Persona 1:", f"{persona1.name} ({args.persona1})")
    header.add_row("Persona 2:", f"{persona2.name} ({args.persona2})")
    header.add_row("Model:", agent1.__class__.__name__)
    header.add_row("Turns:", str(args.turns))

    console.print(Panel(header, title="[bold cyan]Debate Configuration", border_style="cyan", box=box.DOUBLE))

    # Create debate mode
    mode_classes = {
        "sequential": SequentialDebateMode,
        "moderated": ModeratedDebateMode,
        "freeform": FreeFormDebateMode
    }

    debate_mode = mode_classes[args.mode](agent1, agent2, persona1, persona2, config)

    # Run debate
    try:
        transcript = debate_mode.run_debate(args.topic, args.turns)

        # Display summary
        console.print()
        console.print(Panel(
            f"[green]Debate completed with {len(transcript)} exchanges[/green]",
            border_style="green"
        ))

        # Save transcript if requested
        if args.save:
            save_path = Path(args.save)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            debate_data = {
                "timestamp": datetime.now().isoformat(),
                "topic": args.topic,
                "mode": args.mode,
                "persona1": {
                    "id": args.persona1,
                    "name": persona1.name
                },
                "persona2": {
                    "id": args.persona2,
                    "name": persona2.name
                },
                "model": agent1.__class__.__name__,
                "turns": args.turns,
                "transcript": transcript
            }

            with open(save_path, "w") as f:
                json.dump(debate_data, f, indent=2)

            console.print(f"[green]Transcript saved to:[/] [cyan]{save_path}[/]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Debate interrupted[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Error during debate:[/] {e}\n")
        if args.debug:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
