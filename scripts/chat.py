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

from rich.console import Console, Group
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from rich import box
import time

from src.agent.factory import AgentFactory
from src.config import get_config
from src.tts import TTSFactory

# Create rich console
console = Console()

# Logger will be configured after parsing args
logger = logging.getLogger(__name__)


def handle_tts(tts_client, text: str, persona, config):
    """Generate and optionally play audio for response"""
    if not tts_client or not persona.voice_id:
        return

    try:
        # Check if streaming is available and enabled
        use_streaming = (
            config.tts.use_streaming and
            hasattr(tts_client, 'generate_speech_streaming') and
            hasattr(tts_client, 'play_audio_async')
        )

        if use_streaming and config.tts.auto_play:
            # Stream audio: generate and play in chunks
            console.print("[dim]🎤 Streaming audio...[/dim]")

            playback_threads = []
            for audio_chunk in tts_client.generate_speech_streaming(
                text=text,
                voice_id=persona.voice_id
            ):
                # Play each chunk as it's generated (non-blocking)
                thread = tts_client.play_audio_async(audio_chunk)
                playback_threads.append(thread)

            # Wait for all playback to complete
            for thread in playback_threads:
                thread.join()

            console.print("[dim]✓ Audio playback complete[/dim]")
        else:
            # Non-streaming: generate all audio first
            console.print("[dim]🎤 Generating audio...[/dim]")

            # Determine save path if saving is enabled
            save_path = None
            if config.tts.save_audio:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = Path(config.tts.audio_output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / f"{persona.collection_name}_{timestamp}.mp3"

            # Generate audio
            audio_bytes = tts_client.generate_speech(
                text=text,
                voice_id=persona.voice_id,
                model=config.tts.model,
                stability=config.tts.voice_stability,
                similarity_boost=config.tts.voice_similarity_boost,
                save_path=str(save_path) if save_path else None
            )

            # Play audio if enabled
            if config.tts.auto_play:
                console.print("[dim]🔊 Playing audio...[/dim]")
                tts_client.play_audio(audio_bytes)
                console.print("[dim]✓ Audio playback complete[/dim]")

    except ImportError as e:
        console.print(f"[yellow]⚠ Audio playback libraries not installed: {e}[/yellow]")
        console.print("[dim]Install with: pip install simpleaudio pydub[/dim]")
    except Exception as e:
        console.print(f"[yellow]⚠ TTS error: {e}[/yellow]")
        logger.debug(f"TTS error details: {e}", exc_info=True)


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
        "--persona",
        "-p",
        type=str,
        default=None,
        help="Persona to animate (e.g., 'jules', 'heidegger') - default from config",
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

    # Get persona ID
    persona_id = args.persona or config.default_persona

    # Create agent
    try:
        if args.model:
            agent = AgentFactory.create(args.model, persona_id, config)
        else:
            agent = AgentFactory.create_primary(persona_id, config)

        persona = config.get_persona(persona_id)
    except ConnectionError as e:
        console.print(f"\n[bold red]Connection Error:[/]")
        console.print(f"[red]{str(e)}[/]\n")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error initializing agent:[/] {e}\n")
        if args.debug:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/]")
        sys.exit(1)

    if args.debug:
        logger.info(f"Using model: {agent.__class__.__name__}")
        logger.info(f"Animating persona: {persona.name} ({persona_id})")

    # Initialize TTS if enabled for this persona
    tts_client = None
    if config.tts.enabled and persona.voice_enabled and persona.voice_id:
        try:
            tts_client = TTSFactory.create(config)
            if args.debug:
                logger.info(f"TTS enabled ({config.tts.provider}) for {persona.name} with voice ID: {persona.voice_id}")
        except ValueError as e:
            console.print(f"[yellow]Warning: TTS initialization failed: {e}[/yellow]")
            console.print("[dim]Audio features will be disabled[/dim]")
            tts_client = None
        except Exception as e:
            console.print(f"[yellow]Warning: TTS initialization error: {e}[/yellow]")
            console.print("[dim]Audio features will be disabled[/dim]")
            if args.debug:
                import traceback
                console.print(f"[dim]{traceback.format_exc()}[/]")
            tts_client = None

    # Single query mode
    if args.query:
        console.print(Panel(args.query, title="[bold cyan]Query", border_style="cyan"))

        # Check if streaming is supported and enabled
        use_streaming = not args.no_stream and hasattr(agent, 'respond_stream')

        if use_streaming:
            # Streaming mode
            collected_response = ""
            result = None
            status_messages = []

            with Live(console=console, refresh_per_second=20) as live:
                spinner = Spinner("dots", text="Thinking...", style="yellow")
                live.update(Panel(
                    spinner,
                    title=f"[bold green]{persona.name} Anima",
                    border_style="green",
                    box=box.ROUNDED
                ))

                stream_gen = agent.respond_stream(args.query)

                try:
                    while True:
                        chunk = next(stream_gen)

                        # Handle dict-based yields
                        if isinstance(chunk, dict):
                            if chunk.get("type") == "text":
                                collected_response += chunk["content"]
                                # Build display
                                display_content = []
                                if status_messages:
                                    status_text = "\n".join([f"⚙️  {msg}" for msg in status_messages])
                                    display_content.append(Text(status_text, style="dim yellow"))
                                    display_content.append(Text())
                                if collected_response:
                                    display_content.append(Markdown(collected_response))

                                live.update(Panel(
                                    Group(*display_content) if display_content else Text("Thinking...", style="dim"),
                                    title=f"[bold green]{persona.name} Anima",
                                    border_style="green",
                                    box=box.ROUNDED
                                ))
                            elif chunk.get("type") == "status":
                                status_messages.append(chunk["message"])
                                if len(status_messages) > 5:
                                    status_messages.pop(0)

                                # Update with current status (with spinner)
                                display_content = []

                                # Add spinner with latest status
                                spinner = Spinner("dots", text=status_messages[-1], style="yellow")
                                display_content.append(spinner)

                                # Show previous status messages
                                if len(status_messages) > 1:
                                    prev_status = "\n".join([f"  {msg}" for msg in status_messages[:-1]])
                                    display_content.append(Text(f"\n{prev_status}", style="dim"))

                                if collected_response:
                                    display_content.append(Text("\n"))
                                    display_content.append(Markdown(collected_response))

                                live.update(Panel(
                                    Group(*display_content),
                                    title=f"[bold green]{persona.name} Anima",
                                    border_style="green",
                                    box=box.ROUNDED
                                ))
                        # Backward compatibility
                        elif isinstance(chunk, str):
                            collected_response += chunk
                            md = Markdown(collected_response)
                            live.update(Panel(
                                md,
                                title=f"[bold green]{persona.name} Anima",
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

        # Generate and play audio if TTS is enabled
        handle_tts(tts_client, result['response'], persona, config)

        return

    # Interactive mode
    console.print()

    # Header
    header = Table.grid(padding=1)
    header.add_column(style="cyan", justify="right")
    header.add_column(style="white")

    header.add_row("System:", "Anima")
    header.add_row("Model:", agent.__class__.__name__)
    header.add_row("Persona:", f"{persona.name} ({persona_id})")

    # Show mode info
    modes = []
    if args.debug:
        modes.append("[yellow]Debug[/]")
    if not args.no_stream and hasattr(agent, 'respond_stream'):
        modes.append("[green]Streaming[/]")
    if tts_client:
        modes.append("[magenta]TTS[/]")
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
                            "persona_id": persona_id,
                            "persona_name": persona.name,
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
                        role = "You" if msg["role"] == "user" else f"{persona.name} Anima"
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
                status_messages = []  # Track status messages
                start_time = time.time()

                with Live(console=console, refresh_per_second=20) as live:
                    # Start with thinking status
                    spinner = Spinner("dots", text="Thinking...", style="yellow")
                    live.update(Panel(
                        spinner,
                        title=f"[bold green]{persona.name} Anima",
                        border_style="green",
                        box=box.ROUNDED,
                        padding=(1, 2)
                    ))

                    # Stream the response
                    stream_gen = agent.respond_stream(query, conversation_history=conversation_history)

                    try:
                        while True:
                            chunk = next(stream_gen)

                            # Handle dict-based yields (new format)
                            if isinstance(chunk, dict):
                                if chunk.get("type") == "text":
                                    collected_response += chunk["content"]
                                    # Build display with status + content
                                    display_content = []
                                    if status_messages:
                                        status_text = "\n".join([f"⚙️  {msg}" for msg in status_messages])
                                        display_content.append(Text(status_text, style="dim yellow"))
                                        display_content.append(Text())  # Blank line
                                    if collected_response:
                                        display_content.append(Markdown(collected_response))

                                    live.update(Panel(
                                        Group(*display_content) if display_content else Text("Thinking...", style="dim"),
                                        title=f"[bold green]{persona.name} Anima",
                                        border_style="green",
                                        box=box.ROUNDED,
                                        padding=(1, 2)
                                    ))
                                elif chunk.get("type") == "status":
                                    # Update status messages (keep last 5 for more context)
                                    status_messages.append(chunk["message"])
                                    if len(status_messages) > 5:
                                        status_messages.pop(0)

                                    # Update display with current status (with spinner)
                                    display_content = []

                                    # Add spinner with latest status
                                    spinner = Spinner("dots", text=status_messages[-1], style="yellow")
                                    display_content.append(spinner)

                                    # Show previous status messages
                                    if len(status_messages) > 1:
                                        prev_status = "\n".join([f"  {msg}" for msg in status_messages[:-1]])
                                        display_content.append(Text(f"\n{prev_status}", style="dim"))

                                    if collected_response:
                                        display_content.append(Text("\n"))  # Blank line
                                        display_content.append(Markdown(collected_response))

                                    live.update(Panel(
                                        Group(*display_content),
                                        title=f"[bold green]{persona.name} Anima",
                                        border_style="green",
                                        box=box.ROUNDED,
                                        padding=(1, 2)
                                    ))
                            # Handle legacy string yields (backward compatibility)
                            elif isinstance(chunk, str):
                                collected_response += chunk
                                md = Markdown(collected_response)
                                live.update(Panel(
                                    md,
                                    title=f"[bold green]{persona.name} Anima",
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
                    title=f"[bold green]{persona.name} Anima",
                    border_style="green",
                    box=box.ROUNDED,
                    padding=(1, 2)
                ))

            # Update conversation history
            conversation_history.append({"role": "user", "content": query})
            conversation_history.append({"role": "assistant", "content": result['response']})

            # Generate and play audio if TTS is enabled
            handle_tts(tts_client, result['response'], persona, config)

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
                        "persona_id": persona_id,
                        "persona_name": persona.name,
                        "model": agent.__class__.__name__,
                        "conversation": conversation_history
                    }, f, indent=2)

                console.print(f"\n[green]Conversation saved to[/] [cyan]{filepath}[/]")

            console.print("\n[yellow]Goodbye![/]\n")
            break
        except ConnectionError as e:
            # Connection errors get special treatment with helpful instructions
            console.print(f"\n[bold red]Connection Error:[/]", style="bold")
            console.print(f"[red]{str(e)}[/]\n")
        except Exception as e:
            # Log the full error for debugging
            logger.error(f"Error: {e}", exc_info=True)

            # Display user-friendly error message
            console.print(f"\n[bold red]Error:[/] {e}\n")

            if args.debug:
                # Show full traceback in debug mode
                import traceback
                console.print("[dim]Full traceback:[/]")
                console.print(f"[dim]{traceback.format_exc()}[/]")


if __name__ == "__main__":
    main()
