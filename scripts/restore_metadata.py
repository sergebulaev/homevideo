#!/usr/bin/env python3
"""
Restore stream metadata from a source file to a target file.

Useful when conversion stripped metadata (language tags, titles) but preserved streams.
Can probe remote files via SSH and apply metadata to local files.

Usage:
    python restore_metadata.py SOURCE TARGET [--output OUTPUT]

Examples:
    # From remote file
    python restore_metadata.py "ssh://samui001/srv/torrents/movie.mkv" local.mp4

    # From local file
    python restore_metadata.py original.mkv converted.mp4 -o fixed.mp4
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None


def probe_file(filepath: str) -> dict:
    """Probe a file for stream metadata. Supports ssh:// URLs."""

    if filepath.startswith("ssh://"):
        # Parse ssh://host/path format
        parts = filepath[6:].split("/", 1)
        host = parts[0]
        remote_path = "/" + parts[1] if len(parts) > 1 else "/"

        if console:
            console.print(f"[cyan]Probing remote:[/cyan] {host}:{remote_path}")

        # Stream file through SSH to ffprobe
        cmd = f'ssh {host} \'cat "{remote_path}"\' | ffprobe -v error -show_entries stream=index,codec_type,codec_name,channels -show_entries stream_tags=language,title -of json -'

        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                print(f"Error probing remote file: {result.stderr}")
                return {}
            return json.loads(result.stdout)
        except Exception as e:
            print(f"Error: {e}")
            return {}
    else:
        # Local file
        if console:
            console.print(f"[cyan]Probing local:[/cyan] {filepath}")

        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=index,codec_type,codec_name,channels",
            "-show_entries", "stream_tags=language,title",
            "-of", "json",
            filepath
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                print(f"Error probing file: {result.stderr}")
                return {}
            return json.loads(result.stdout)
        except Exception as e:
            print(f"Error: {e}")
            return {}


def display_streams(probe_data: dict, title: str):
    """Display stream information."""
    streams = probe_data.get("streams", [])

    if not streams:
        print("No streams found")
        return

    if RICH_AVAILABLE:
        table = Table(title=title)
        table.add_column("#", style="cyan")
        table.add_column("Type", style="white")
        table.add_column("Codec", style="green")
        table.add_column("Ch", style="yellow")
        table.add_column("Language", style="blue")
        table.add_column("Title", style="dim")

        for s in streams:
            tags = s.get("tags", {})
            table.add_row(
                str(s.get("index", "?")),
                s.get("codec_type", "?"),
                s.get("codec_name", "?"),
                str(s.get("channels", "")) if s.get("codec_type") == "audio" else "",
                tags.get("language", "und"),
                (tags.get("title", "")[:40] + "...") if len(tags.get("title", "")) > 40 else tags.get("title", "")
            )

        console.print(table)
    else:
        print(f"\n{title}")
        print("-" * 60)
        for s in streams:
            tags = s.get("tags", {})
            print(f"  {s.get('index', '?')}: {s.get('codec_type', '?')} "
                  f"({s.get('codec_name', '?')}) "
                  f"lang={tags.get('language', 'und')} "
                  f"title={tags.get('title', '')[:30]}")


def restore_metadata(source_probe: dict, target_file: str, output_file: str) -> bool:
    """Remux target file with metadata from source probe data."""

    source_streams = source_probe.get("streams", [])

    # Build ffmpeg command
    cmd = ["ffmpeg", "-y", "-i", target_file]

    # Map all streams
    cmd.extend(["-map", "0"])

    # Copy all codecs (no re-encoding)
    cmd.extend(["-c", "copy"])

    # Apply metadata from source streams
    audio_idx = 0
    video_idx = 0
    subtitle_idx = 0

    for s in source_streams:
        tags = s.get("tags", {})
        codec_type = s.get("codec_type", "")

        if codec_type == "video":
            stream_spec = f"s:v:{video_idx}"
            video_idx += 1
        elif codec_type == "audio":
            stream_spec = f"s:a:{audio_idx}"
            audio_idx += 1
        elif codec_type == "subtitle":
            stream_spec = f"s:s:{subtitle_idx}"
            subtitle_idx += 1
        else:
            continue

        # Set language
        if tags.get("language"):
            cmd.extend([f"-metadata:{stream_spec}", f"language={tags['language']}"])

        # Set title
        if tags.get("title"):
            cmd.extend([f"-metadata:{stream_spec}", f"title={tags['title']}"])

    # Set container title from filename
    output_title = Path(output_file).stem
    cmd.extend(["-metadata", f"title={output_title}"])

    # Output file
    cmd.append(output_file)

    if console:
        console.print(f"\n[bold]Running ffmpeg...[/bold]")
        console.print(f"[dim]{' '.join(cmd[:10])}... {cmd[-1]}[/dim]\n")
    else:
        print(f"\nRunning ffmpeg...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            if console:
                console.print(f"[green]Success![/green] Output: {output_file}")

                # Show output file size
                output_size = os.path.getsize(output_file)
                input_size = os.path.getsize(target_file)
                console.print(f"[dim]Size: {output_size / 1024 / 1024:.1f} MB[/dim]")
            else:
                print(f"Success! Output: {output_file}")
            return True
        else:
            print(f"FFmpeg error: {result.stderr}")
            return False

    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Restore stream metadata from source to target file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "ssh://server/path/original.mkv" converted.mp4
  %(prog)s original.mkv converted.mp4 -o fixed.mp4
  %(prog)s --probe-only "ssh://server/path/movie.mkv"
        """
    )
    parser.add_argument("source", help="Source file with metadata (supports ssh://host/path)")
    parser.add_argument("target", nargs="?", help="Target file to fix")
    parser.add_argument("-o", "--output", help="Output file (default: target with .fixed suffix)")
    parser.add_argument("--probe-only", action="store_true", help="Only probe source, don't remux")

    args = parser.parse_args()

    # Probe source
    source_probe = probe_file(args.source)

    if not source_probe.get("streams"):
        print("Error: Could not probe source file")
        sys.exit(1)

    display_streams(source_probe, "Source Streams")

    if args.probe_only or not args.target:
        return

    # Probe target
    target_probe = probe_file(args.target)

    if not target_probe.get("streams"):
        print("Error: Could not probe target file")
        sys.exit(1)

    display_streams(target_probe, "Target Streams (before)")

    # Verify stream count matches
    source_audio = len([s for s in source_probe["streams"] if s.get("codec_type") == "audio"])
    target_audio = len([s for s in target_probe["streams"] if s.get("codec_type") == "audio"])

    if source_audio != target_audio:
        if console:
            console.print(f"\n[yellow]Warning:[/yellow] Stream count mismatch! "
                         f"Source has {source_audio} audio, target has {target_audio}")
        else:
            print(f"\nWarning: Stream count mismatch! Source: {source_audio}, Target: {target_audio}")

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        target_path = Path(args.target)
        output_file = str(target_path.parent / f"{target_path.stem}.fixed{target_path.suffix}")

    # Restore metadata
    if console:
        console.print(f"\n[bold]Restoring metadata...[/bold]")

    success = restore_metadata(source_probe, args.target, output_file)

    if success:
        # Verify output
        if console:
            console.print()
        output_probe = probe_file(output_file)
        display_streams(output_probe, "Output Streams (after)")

        if console:
            console.print(Panel(
                f"[green]Metadata restored successfully![/green]\n\n"
                f"Output: {output_file}\n\n"
                f"[dim]You can now replace the original with:[/dim]\n"
                f"mv \"{output_file}\" \"{args.target}\"",
                title="Done",
                border_style="green"
            ))
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
