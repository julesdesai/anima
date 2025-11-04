#!/usr/bin/env python3
"""Interactive chat interface for testing the agent"""

import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from rich import box

from src.agent.factory import AgentFactory
from src.config import get_config

# Create rich console
console = Console()

# Logger will be configured after parsing args
logger = logging.getLogger(__name__)


def main():
    """Run interactive chat"""
    parser = argparse.ArgumentParser(description="Chat with user-aligned assistant")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Model to use (claude, deepseek, hermes) - default from config",
    )
    parser.add_argument(
        "--user",
        "-u",
        type=str,
        default=None,
        help="User name - default from config",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Single query (non-interactive mode)",
    )
    parser.add_argument(
        "--save-history",
        "-s",
        action="store_true",
        help="Save conversation history to file on exit",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug mode (show detailed logging and stats)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming responses (use for non-OpenAI models)",
    )

    args = parser.parse_args()

    # Configure logging based on debug flag
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        # Only show warnings and errors in normal mode
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s: %(message)s",
        )

    # Load config
    config = get_config()

    # Get user name
    user_name = args.user or config.user.name

    # Create agent
    if args.model:
        agent = AgentFactory.create(args.model, user_name, config)
    else:
        agent = AgentFactory.create_primary(user_name, config)

    if args.debug:
        logger.info(f"Using model: {agent.__class__.__name__}")
        logger.info(f"Modeling user: {user_name}")

    # Single query mode
    if args.query:
        console.print(Panel(args.query, title="[bold cyan]Query", border_style="cyan"))

        # Check if streaming is supported and enabled
        use_streaming = not args.no_stream and hasattr(agent, 'respond_stream')

        if use_streaming:
            # Streaming mode
            collected_response = ""
            result = None

            with Live(console=console, refresh_per_second=10) as live:
                live.update(Panel(
                    Text("Searching and thinking...", style="dim"),
                    title="[bold green]Response",
                    border_style="green",
                    box=box.ROUNDED
                ))

                stream_gen = agent.respond_stream(args.query)

                try:
                    while True:
                        chunk = next(stream_gen)
                        if isinstance(chunk, str):
                            collected_response += chunk
                            md = Markdown(collected_response)
                            live.update(Panel(
                                md,
                                title="[bold green]Response",
                                border_style="green",
                                box=box.ROUNDED
                            ))
                except StopIteration as e:
                    result = e.value if hasattr(e, 'value') else {
                        "response": collected_response,
                        "tool_calls": [],
                        "iterations": 1,
                        "model": agent.__class__.__name__,
                    }
        else:
            # Non-streaming mode
            with console.status("[bold green]Thinking...", spinner="dots"):
                result = agent.respond(args.query)

            md = Markdown(result['response'])
            console.print(Panel(md, title="[bold green]Response", border_style="green", box=box.ROUNDED))

        # Display stats in debug mode only
        if args.debug:
            stats = Text()
            stats.append(f"Tool calls: {len(result['tool_calls'])} ", style="dim")
            stats.append(f"• Iterations: {result['iterations']}", style="dim")
            console.print(stats)
        return

    # Interactive mode
    console.print()

    # Header
    header = Table.grid(padding=1)
    header.add_column(style="cyan", justify="right")
    header.add_column(style="white")

    header.add_row("System:", "Castor - User-Aligned Assistant")
    header.add_row("Model:", agent.__class__.__name__)
    header.add_row("User:", user_name)

    # Show mode info
    modes = []
    if args.debug:
        modes.append("[yellow]Debug[/]")
    if not args.no_stream and hasattr(agent, 'respond_stream'):
        modes.append("[green]Streaming[/]")
    if modes:
        header.add_row("Mode:", " • ".join(modes))

    console.print(Panel(header, title="[bold cyan]Welcome", border_style="cyan", box=box.DOUBLE))

    # Commands help
    commands_table = Table(show_header=False, box=None, padding=(0, 2))
    commands_table.add_column(style="yellow")
    commands_table.add_column(style="dim")
    commands_table.add_row("exit/quit", "End the session")
    commands_table.add_row("clear", "Reset conversation history")
    commands_table.add_row("history", "Show conversation history")

    console.print(Panel(commands_table, title="[bold yellow]Commands", border_style="yellow"))
    console.print()

    # Track conversation history
    conversation_history = []

    while True:
        try:
            # Get user input
            query = Prompt.ask("\n[bold cyan]You[/]")

            if not query or not query.strip():
                continue

            query = query.strip()

            if query.lower() in ["exit", "quit"]:
                # Save conversation history if requested
                if args.save_history and conversation_history:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"conversation_{timestamp}.json"
                    filepath = Path("logs") / filename

                    # Create logs directory if it doesn't exist
                    filepath.parent.mkdir(parents=True, exist_ok=True)

                    # Save conversation
                    with open(filepath, "w") as f:
                        json.dump({
                            "timestamp": timestamp,
                            "user": user_name,
                            "model": agent.__class__.__name__,
                            "conversation": conversation_history
                        }, f, indent=2)

                    console.print(f"\n[green]Conversation saved to[/] [cyan]{filepath}[/]")

                console.print("\n[yellow]Goodbye![/]\n")
                break

            if query.lower() == "clear":
                conversation_history = []
                console.print("\n[green]Conversation history cleared[/]\n")
                continue

            if query.lower() == "history":
                if not conversation_history:
                    console.print("\n[yellow]No conversation history[/]\n")
                else:
                    turns = len(conversation_history) // 2
                    history_table = Table(title=f"Conversation History ({turns} turn{'s' if turns != 1 else ''})",
                                        box=box.ROUNDED, border_style="blue")
                    history_table.add_column("Role", style="cyan", width=12)
                    history_table.add_column("Message", style="white")

                    for msg in conversation_history:
                        role = "You" if msg["role"] == "user" else "Assistant"
                        content_preview = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
                        history_table.add_row(role, content_preview)

                    console.print()
                    console.print(history_table)
                    console.print()
                continue

            # Get response with conversation history
            console.print()

            # Check if streaming is supported and enabled
            use_streaming = not args.no_stream and hasattr(agent, 'respond_stream')

            if use_streaming:
                # Streaming mode
                collected_response = ""
                result = None

                with Live(console=console, refresh_per_second=10) as live:
                    # Start with thinking status
                    live.update(Panel(
                        Text("Searching and thinking...", style="dim"),
                        title="[bold green]Assistant",
                        border_style="green",
                        box=box.ROUNDED,
                        padding=(1, 2)
                    ))

                    # Stream the response
                    stream_gen = agent.respond_stream(query, conversation_history=conversation_history)

                    try:
                        while True:
                            chunk = next(stream_gen)
                            if isinstance(chunk, str):
                                collected_response += chunk
                                # Update display with markdown
                                md = Markdown(collected_response)
                                live.update(Panel(
                                    md,
                                    title="[bold green]Assistant",
                                    border_style="green",
                                    box=box.ROUNDED,
                                    padding=(1, 2)
                                ))
                    except StopIteration as e:
                        # Generator returned final result
                        result = e.value if hasattr(e, 'value') else {
                            "response": collected_response,
                            "tool_calls": [],
                            "iterations": 1,
                            "model": agent.__class__.__name__,
                        }
            else:
                # Non-streaming mode
                with console.status("[bold green]Thinking...", spinner="dots"):
                    result = agent.respond(query, conversation_history=conversation_history)

                # Display response with markdown rendering
                md = Markdown(result['response'])
                console.print(Panel(
                    md,
                    title="[bold green]Assistant",
                    border_style="green",
                    box=box.ROUNDED,
                    padding=(1, 2)
                ))

            # Update conversation history
            conversation_history.append({"role": "user", "content": query})
            conversation_history.append({"role": "assistant", "content": result['response']})

            # Show stats in debug mode only
            if args.debug:
                turns = len(conversation_history) // 2
                stats = Text()
                stats.append(f"Tool calls: {len(result['tool_calls'])} ", style="dim cyan")
                stats.append("• ", style="dim")
                stats.append(f"Iterations: {result['iterations']} ", style="dim cyan")
                stats.append("• ", style="dim")
                stats.append(f"History: {turns} turn{'s' if turns != 1 else ''}", style="dim cyan")
                console.print(stats)

        except KeyboardInterrupt:
            # Save conversation history if requested
            if args.save_history and conversation_history:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"conversation_{timestamp}.json"
                filepath = Path("logs") / filename

                # Create logs directory if it doesn't exist
                filepath.parent.mkdir(parents=True, exist_ok=True)

                # Save conversation
                with open(filepath, "w") as f:
                    json.dump({
                        "timestamp": timestamp,
                        "user": user_name,
                        "model": agent.__class__.__name__,
                        "conversation": conversation_history
                    }, f, indent=2)

                console.print(f"\n[green]Conversation saved to[/] [cyan]{filepath}[/]")

            console.print("\n[yellow]Goodbye![/]\n")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            console.print(f"\n[red]Error:[/] {e}\n")


if __name__ == "__main__":
    main()
