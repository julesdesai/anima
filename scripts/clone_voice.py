#!/usr/bin/env python3
"""
Voice cloning setup script for creating TTS voices.

Note: Voice cloning is only supported with ElevenLabs provider.
For local TTS (piper), use pre-trained models instead.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.tts import TTSFactory
from src.config import get_config, reload_config
import yaml

console = Console()
logger = logging.getLogger(__name__)


def update_persona_voice_id(persona_id: str, voice_id: str):
    """Update the persona's voice_id in config.yaml"""
    config_path = Path("config.yaml")

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    if persona_id not in config_data['personas']:
        raise ValueError(f"Persona '{persona_id}' not found in config.yaml")

    # Update voice_id
    config_data['personas'][persona_id]['voice_id'] = voice_id

    # Write back to file
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]Updated {persona_id} voice_id in config.yaml[/green]")


def list_existing_voices(tts, provider: str):
    """List all existing voices"""
    console.print("\n[bold cyan]Existing Voices:[/bold cyan]")

    try:
        voices = tts.list_voices()

        if not voices:
            console.print("[yellow]No voices found[/yellow]")
            return

        table = Table(box=box.ROUNDED)
        table.add_column("Name", style="cyan")
        table.add_column("Voice ID", style="green")
        table.add_column("Info", style="yellow")

        for voice in voices:
            # Handle different provider formats
            if isinstance(voice, dict):
                # Local provider returns dicts
                name = voice.get('name', voice.get('voice_id'))
                voice_id = voice['voice_id']
                info = voice.get('description', voice.get('language', 'N/A'))
            else:
                # ElevenLabs returns objects
                name = voice.name
                voice_id = voice.voice_id
                info = getattr(voice, 'category', 'N/A')

            table.add_row(name, voice_id, str(info))

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error listing voices: {e}[/red]")


def test_voice(tts, voice_id: str, persona_name: str):
    """Test the cloned voice with sample text"""
    console.print("\n[bold cyan]Testing voice...[/bold cyan]")

    test_text = f"Hello, I am {persona_name}. This is a test of my cloned voice."

    try:
        audio_bytes = tts.generate_speech(
            text=test_text,
            voice_id=voice_id
        )

        console.print("[green]✓ Voice generation successful[/green]")
        console.print(f"[dim]Generated {len(audio_bytes)} bytes of audio[/dim]")

        # Try to play it
        try:
            console.print("\n[yellow]Playing audio...[/yellow]")
            tts.play_audio(audio_bytes)
            console.print("[green]✓ Playback complete[/green]")
        except ImportError:
            console.print("[yellow]⚠ Audio playback libraries not installed[/yellow]")
            console.print("[dim]Install with: pip install simpleaudio pydub[/dim]")
        except Exception as e:
            console.print(f"[yellow]⚠ Playback error: {e}[/yellow]")

    except Exception as e:
        console.print(f"[red]Error testing voice: {e}[/red]")


def main():
    parser = argparse.ArgumentParser(
        description="Clone a voice for a persona using TTS provider"
    )
    parser.add_argument(
        "--persona",
        "-p",
        type=str,
        required=True,
        help="Persona ID (e.g., 'sasha')"
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        help="Name for the cloned voice (defaults to persona name)"
    )
    parser.add_argument(
        "--audio-files",
        "-a",
        type=str,
        nargs="+",
        help="Paths to audio files for cloning (at least 1 required). If not provided, looks in data/corpus/{persona}/audio/"
    )
    parser.add_argument(
        "--auto-discover",
        action="store_true",
        help="Automatically discover audio files in data/corpus/{persona}/audio/"
    )
    parser.add_argument(
        "--description",
        "-d",
        type=str,
        help="Description of the voice"
    )
    parser.add_argument(
        "--list-voices",
        "-l",
        action="store_true",
        help="List all existing voices and exit"
    )
    parser.add_argument(
        "--use-existing",
        "-e",
        type=str,
        help="Use an existing voice ID instead of cloning"
    )
    parser.add_argument(
        "--test-only",
        "-t",
        action="store_true",
        help="Test the persona's existing voice without cloning"
    )
    parser.add_argument(
        "--debug",
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
            level=logging.INFO,
            format="%(levelname)s: %(message)s",
        )

    # Load config
    config = get_config()

    # Initialize TTS
    try:
        tts = TTSFactory.create(config)
        provider = config.tts.provider
    except ValueError as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        if "ELEVENLABS_API_KEY" in str(e):
            console.print("[yellow]Set ELEVENLABS_API_KEY environment variable[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        console.print(f"[yellow]Check TTS provider configuration in config.yaml[/yellow]")
        if args.debug:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/]")
        sys.exit(1)

    # Display header
    console.print(Panel(
        f"[bold cyan]Voice Cloning Setup[/bold cyan]\n"
        f"[dim]Provider: {provider}[/dim]",
        border_style="cyan",
        box=box.DOUBLE
    ))

    # List voices and exit
    if args.list_voices:
        list_existing_voices(tts, provider)
        sys.exit(0)

    # Get persona
    try:
        persona = config.get_persona(args.persona)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    # Test existing voice
    if args.test_only:
        if not persona.voice_id:
            console.print(f"[red]Persona '{args.persona}' has no voice_id configured[/red]")
            sys.exit(1)
        test_voice(tts, persona.voice_id, persona.name)
        sys.exit(0)

    # Use existing voice
    if args.use_existing:
        console.print(f"\n[cyan]Using existing voice ID:[/cyan] {args.use_existing}")
        update_persona_voice_id(args.persona, args.use_existing)

        # Test it
        test_voice(tts, args.use_existing, persona.name)
        sys.exit(0)

    # Discover or validate audio files
    audio_paths = []

    if args.auto_discover or not args.audio_files:
        # Look for audio files in persona's audio directory
        audio_dir = Path(persona.corpus_path) / "audio"

        if audio_dir.exists():
            # Find all audio files
            audio_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac']
            for ext in audio_extensions:
                audio_paths.extend([str(f) for f in audio_dir.glob(f"*{ext}")])

            if audio_paths:
                console.print(f"\n[cyan]Discovered {len(audio_paths)} audio file(s) in:[/cyan] {audio_dir}")
                for path in audio_paths:
                    console.print(f"  • {Path(path).name}")
            else:
                console.print(f"[yellow]No audio files found in:[/yellow] {audio_dir}")
        else:
            console.print(f"[yellow]Audio directory not found:[/yellow] {audio_dir}")

        if not audio_paths and not args.audio_files:
            console.print("\n[red]No audio files found or provided[/red]")
            console.print(f"[dim]Create directory and add audio samples:[/dim]")
            console.print(f"[dim]  mkdir -p {audio_dir}[/dim]")
            console.print(f"[dim]  cp your_audio.mp3 {audio_dir}/[/dim]")
            console.print(f"[dim]Then run with --auto-discover or provide --audio-files[/dim]")
            sys.exit(1)

    # Add explicitly provided audio files
    if args.audio_files:
        for path_str in args.audio_files:
            path = Path(path_str)
            if not path.exists():
                console.print(f"[red]Audio file not found:[/red] {path}")
                sys.exit(1)
            audio_paths.append(str(path))

    if not audio_paths:
        console.print("[red]Error: No audio files to process[/red]")
        sys.exit(1)

    voice_name = args.name or f"{persona.name} Voice"
    voice_description = args.description or f"Cloned voice for {persona.name} persona"

    # Display cloning info
    info = Table.grid(padding=1)
    info.add_column(style="cyan", justify="right")
    info.add_column(style="white")

    info.add_row("Persona:", f"{persona.name} ({args.persona})")
    info.add_row("Voice Name:", voice_name)
    info.add_row("Audio Files:", str(len(audio_paths)))
    for i, path in enumerate(audio_paths, 1):
        info.add_row("", f"  {i}. {Path(path).name}")

    console.print(Panel(info, title="[bold]Cloning Configuration", border_style="cyan"))

    # Clone voice
    try:
        console.print("\n[yellow]Cloning voice...[/yellow]")
        console.print("[dim]This may take a minute...[/dim]")

        voice_id = tts.clone_voice(
            name=voice_name,
            audio_files=audio_paths,
            description=voice_description
        )

        console.print(f"\n[bold green]✓ Voice cloned successfully![/bold green]")
        console.print(f"[cyan]Voice ID:[/cyan] {voice_id}")

        # Update config
        update_persona_voice_id(args.persona, voice_id)

        # Test the voice
        test_voice(tts, voice_id, persona.name)

        console.print("\n[bold green]Setup complete![/bold green]")
        console.print(f"[dim]You can now use TTS with the '{args.persona}' persona[/dim]")

    except Exception as e:
        console.print(f"\n[bold red]Error cloning voice:[/bold red] {e}")
        if args.debug:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
