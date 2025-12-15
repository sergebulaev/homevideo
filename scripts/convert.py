#!/usr/bin/env python3
"""
Video converter for Samsung TV compatibility.
Converts videos to H.264/AAC format that plays on all Samsung TVs.

Usage:
    python convert.py <input_file> [output_file] [--preset PRESET]

Presets:
    samsung-safe    H.264 + AAC, maximum compatibility (default)
    samsung-4k      HEVC + AAC, for newer 4K TVs
    remux           Copy streams, just change container
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import ffmpeg
except ImportError:
    print("Error: ffmpeg-python not installed.")
    print("Run: source venv/bin/activate && pip install ffmpeg-python")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        TaskProgressColumn,
    )
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: rich not installed. Install with: pip install rich")

console = Console() if RICH_AVAILABLE else None


def format_size(bytes_size: int) -> str:
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f} PB"


def format_duration(seconds: float) -> str:
    """Format seconds to HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_bitrate(bps: float) -> str:
    """Format bitrate to human readable."""
    if bps >= 1_000_000:
        return f"{bps / 1_000_000:.2f} Mbps"
    elif bps >= 1_000:
        return f"{bps / 1_000:.1f} Kbps"
    return f"{bps:.0f} bps"


def get_video_info(input_path: str) -> dict:
    """Probe video file and return detailed codec information."""
    try:
        probe = ffmpeg.probe(input_path)
        video_stream = next(
            (s for s in probe['streams'] if s['codec_type'] == 'video'), None
        )
        audio_stream = next(
            (s for s in probe['streams'] if s['codec_type'] == 'audio'), None
        )
        format_info = probe.get('format', {})

        duration = float(format_info.get('duration', 0))
        file_size = int(format_info.get('size', 0))

        return {
            'video_codec': video_stream.get('codec_name') if video_stream else None,
            'video_codec_long': video_stream.get('codec_long_name') if video_stream else None,
            'width': video_stream.get('width') if video_stream else None,
            'height': video_stream.get('height') if video_stream else None,
            'fps': eval(video_stream.get('r_frame_rate', '0/1')) if video_stream else 0,
            'video_bitrate': int(video_stream.get('bit_rate', 0)) if video_stream and video_stream.get('bit_rate') else None,
            'audio_codec': audio_stream.get('codec_name') if audio_stream else None,
            'audio_codec_long': audio_stream.get('codec_long_name') if audio_stream else None,
            'audio_channels': audio_stream.get('channels') if audio_stream else None,
            'audio_sample_rate': audio_stream.get('sample_rate') if audio_stream else None,
            'audio_bitrate': int(audio_stream.get('bit_rate', 0)) if audio_stream and audio_stream.get('bit_rate') else None,
            'duration': duration,
            'duration_str': format_duration(duration),
            'file_size': file_size,
            'file_size_str': format_size(file_size),
            'total_bitrate': int(format_info.get('bit_rate', 0)) if format_info.get('bit_rate') else None,
            'format': format_info.get('format_long_name', 'Unknown'),
            'total_frames': int(float(duration) * eval(video_stream.get('r_frame_rate', '0/1'))) if video_stream and duration else 0,
        }
    except ffmpeg.Error as e:
        if console:
            console.print(f"[red]Error probing file:[/red] {e.stderr.decode() if e.stderr else e}")
        else:
            print(f"Error probing file: {e.stderr.decode() if e.stderr else e}")
        return {}
    except Exception as e:
        if console:
            console.print(f"[red]Error:[/red] {e}")
        else:
            print(f"Error: {e}")
        return {}


def display_file_info(info: dict, title: str = "File Information"):
    """Display file information in a nice table."""
    if not RICH_AVAILABLE or not info:
        return

    table = Table(title=title, show_header=False, border_style="blue")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    # Video info
    table.add_row("Video Codec", f"{info.get('video_codec', 'N/A')} ({info.get('video_codec_long', '')})")
    table.add_row("Resolution", f"{info.get('width', '?')}x{info.get('height', '?')}")
    table.add_row("Frame Rate", f"{info.get('fps', 0):.2f} fps")
    if info.get('video_bitrate'):
        table.add_row("Video Bitrate", format_bitrate(info['video_bitrate']))

    table.add_row("", "")  # Separator

    # Audio info
    table.add_row("Audio Codec", f"{info.get('audio_codec', 'N/A')} ({info.get('audio_codec_long', '')})")
    table.add_row("Channels", str(info.get('audio_channels', 'N/A')))
    table.add_row("Sample Rate", f"{info.get('audio_sample_rate', 'N/A')} Hz")
    if info.get('audio_bitrate'):
        table.add_row("Audio Bitrate", format_bitrate(info['audio_bitrate']))

    table.add_row("", "")  # Separator

    # File info
    table.add_row("Duration", info.get('duration_str', 'N/A'))
    table.add_row("File Size", info.get('file_size_str', 'N/A'))
    if info.get('total_bitrate'):
        table.add_row("Total Bitrate", format_bitrate(info['total_bitrate']))
    table.add_row("Container", info.get('format', 'N/A'))

    console.print(table)


def run_ffmpeg_with_progress(
    input_path: str,
    output_path: str,
    ffmpeg_args: list,
    preset_name: str,
    total_duration: float,
    total_frames: int
):
    """Run FFmpeg with real-time progress display."""

    # Build FFmpeg command manually for progress parsing
    cmd = ['ffmpeg', '-y', '-i', input_path, '-progress', 'pipe:1', '-nostats']

    # Add output arguments (now a list)
    cmd.extend(ffmpeg_args)

    cmd.append(output_path)

    if not RICH_AVAILABLE:
        # Fallback to simple progress
        print(f"Converting with {preset_name}...")
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )
        process.wait()
        if process.returncode == 0:
            print("Done!")
        else:
            print(f"Error: {process.stderr.read()}")
        return process.returncode == 0

    # Rich progress display
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TextColumn("[cyan]ETA:[/cyan]"),
        TimeRemainingColumn(),
        TextColumn("[cyan]Elapsed:[/cyan]"),
        TimeElapsedColumn(),
        console=console,
        expand=False,
    ) as progress:

        task = progress.add_task(f"[green]{preset_name}", total=total_duration)

        # Stats display
        stats_table = Table.grid(padding=(0, 2))
        stats_table.add_column(style="cyan", justify="right")
        stats_table.add_column(style="white")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )

        current_time = 0
        current_frame = 0
        current_fps = 0
        current_bitrate = "N/A"
        current_size = 0

        # Read progress from stdout
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break

            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)

                if key == 'out_time_ms' and value.isdigit():
                    current_time = int(value) / 1_000_000  # Convert microseconds to seconds
                    progress.update(task, completed=current_time)

                elif key == 'frame' and value.isdigit():
                    current_frame = int(value)

                elif key == 'fps' and value:
                    try:
                        current_fps = float(value)
                    except ValueError:
                        pass

                elif key == 'bitrate' and value != 'N/A':
                    current_bitrate = value

                elif key == 'total_size' and value.isdigit():
                    current_size = int(value)

                elif key == 'progress' and value == 'end':
                    progress.update(task, completed=total_duration)

        process.wait()

        elapsed = time.time() - start_time

        if process.returncode == 0:
            # Show completion stats
            console.print()

            completion_table = Table(title="Conversion Complete", border_style="green")
            completion_table.add_column("Metric", style="cyan")
            completion_table.add_column("Value", style="green")

            completion_table.add_row("Total Time", format_duration(elapsed))
            completion_table.add_row("Avg Speed", f"{total_duration / elapsed:.2f}x realtime")
            completion_table.add_row("Frames Processed", f"{current_frame:,}")
            if current_fps > 0:
                completion_table.add_row("Avg FPS", f"{current_fps:.1f}")

            # Output file size
            if os.path.exists(output_path):
                output_size = os.path.getsize(output_path)
                input_size = os.path.getsize(input_path)
                compression = (1 - output_size / input_size) * 100 if input_size > 0 else 0

                completion_table.add_row("Output Size", format_size(output_size))
                completion_table.add_row("Size Change", f"{compression:+.1f}%")

            completion_table.add_row("Output File", output_path)

            console.print(completion_table)
            return True
        else:
            stderr = process.stderr.read()
            console.print(f"[red]FFmpeg Error:[/red]\n{stderr}")
            return False


def convert_samsung_safe(input_path: str, output_path: str, info: dict) -> bool:
    """Convert to H.264 + AAC for maximum Samsung TV compatibility."""
    # Build args list - map all streams, convert audio to AAC
    ffmpeg_args = [
        '-map', '0:v:0',           # Map first video stream
        '-map', '0:a',             # Map ALL audio streams
        '-map_metadata:g', '-1',   # Strip global metadata, keep stream metadata
        '-c:v', 'libx264',         # H.264 video codec
        '-preset', 'medium',
        '-crf', '20',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',             # Convert all audio to AAC
        '-b:a', '192k',
        '-ac', '2',                # Stereo for compatibility
        '-movflags', 'faststart',
        '-metadata', f'title={Path(output_path).stem}',
    ]

    return run_ffmpeg_with_progress(
        input_path, output_path, ffmpeg_args,
        "Samsung Safe (H.264 + AAC Stereo)",
        info.get('duration', 0),
        info.get('total_frames', 0)
    )


def convert_samsung_surround(input_path: str, output_path: str, info: dict) -> bool:
    """Convert keeping surround sound (5.1 AAC) for Samsung TVs with good audio."""
    ffmpeg_args = [
        '-map', '0:v:0',           # Map first video stream
        '-map', '0:a',             # Map ALL audio streams
        '-map_metadata:g', '-1',   # Strip global metadata, keep stream metadata
        '-c:v', 'libx264',         # H.264 video codec
        '-preset', 'medium',
        '-crf', '20',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',             # Convert all audio to AAC
        '-b:a', '384k',            # Higher bitrate for surround
        '-movflags', 'faststart',
        '-metadata', f'title={Path(output_path).stem}',
    ]

    return run_ffmpeg_with_progress(
        input_path, output_path, ffmpeg_args,
        "Samsung Surround (H.264 + AAC 5.1)",
        info.get('duration', 0),
        info.get('total_frames', 0)
    )


def convert_samsung_4k(input_path: str, output_path: str, info: dict) -> bool:
    """Convert to HEVC + AAC for newer Samsung 4K TVs."""
    ffmpeg_args = [
        '-map', '0:v:0',           # Map first video stream
        '-map', '0:a',             # Map ALL audio streams
        '-map_metadata:g', '-1',   # Strip global metadata, keep stream metadata
        '-c:v', 'libx265',         # HEVC video codec
        '-preset', 'medium',
        '-crf', '22',
        '-pix_fmt', 'yuv420p',
        '-tag:v', 'hvc1',          # Apple/Samsung compatible tag
        '-c:a', 'aac',             # Convert all audio to AAC
        '-b:a', '384k',            # Higher bitrate for surround
        '-movflags', 'faststart',
        '-metadata', f'title={Path(output_path).stem}',
    ]

    return run_ffmpeg_with_progress(
        input_path, output_path, ffmpeg_args,
        "Samsung 4K (HEVC + AAC)",
        info.get('duration', 0),
        info.get('total_frames', 0)
    )


def remux_to_mp4(input_path: str, output_path: str, info: dict) -> bool:
    """Copy streams to MP4 container without re-encoding."""
    ffmpeg_args = [
        '-map', '0:v:0',           # Map first video stream
        '-map', '0:a',             # Map ALL audio streams
        '-map_metadata:g', '-1',   # Strip global metadata, keep stream metadata
        '-c:v', 'copy',            # Copy video
        '-c:a', 'copy',            # Copy audio
        '-movflags', 'faststart',
        '-metadata', f'title={Path(output_path).stem}',
    ]

    return run_ffmpeg_with_progress(
        input_path, output_path, ffmpeg_args,
        "Remux to MP4 (no re-encode)",
        info.get('duration', 0),
        info.get('total_frames', 0)
    )


def main():
    parser = argparse.ArgumentParser(
        description='Convert videos for Samsung TV compatibility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s movie.avi                         Convert to H.264+AAC stereo (default)
  %(prog)s movie.avi -o output.mp4           Specify output file
  %(prog)s movie.mkv -p surround             Keep 5.1 surround sound
  %(prog)s movie.mkv -p samsung-4k           Convert for 4K TVs (HEVC)
  %(prog)s movie.mkv -p remux                Just change container (fast)
  %(prog)s movie.avi --info                  Show file info only

Presets:
  samsung-safe    H.264 + AAC stereo, max compatibility (default)
  surround        H.264 + AAC 5.1, keeps all audio tracks
  samsung-4k      HEVC + AAC, for newer 4K TVs
  remux           Copy streams, just change container
        """
    )
    parser.add_argument('input', help='Input video file')
    parser.add_argument('-o', '--output', help='Output video file')
    parser.add_argument(
        '-p', '--preset',
        choices=['samsung-safe', 'surround', 'samsung-4k', 'remux'],
        default='samsung-safe',
        help='Conversion preset (default: samsung-safe)'
    )
    parser.add_argument(
        '-i', '--info',
        action='store_true',
        help='Show video info without converting'
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        if console:
            console.print(f"[red]Error:[/red] Input file not found: {args.input}")
        else:
            print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Get file info
    if console:
        console.print(f"\n[bold]Analyzing:[/bold] {args.input}\n")
    else:
        print(f"\nAnalyzing: {args.input}\n")

    info = get_video_info(args.input)

    if not info:
        sys.exit(1)

    # Show info
    if RICH_AVAILABLE:
        display_file_info(info, "Input File")
    else:
        print(f"Video: {info.get('video_codec')} {info.get('width')}x{info.get('height')}")
        print(f"Audio: {info.get('audio_codec')} ({info.get('audio_channels')} ch)")
        print(f"Duration: {info.get('duration_str')}")
        print(f"Size: {info.get('file_size_str')}")

    # Exit if only info requested
    if args.info:
        return

    # Generate output filename if not provided
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input)
        output_path = str(input_path.parent / f"{input_path.stem}.samsung.mp4")

    if console:
        console.print(f"\n[bold]Output:[/bold] {output_path}")
        console.print(f"[bold]Preset:[/bold] {args.preset}\n")
    else:
        print(f"\nOutput: {output_path}")
        print(f"Preset: {args.preset}\n")

    # Convert based on preset
    success = False
    if args.preset == 'samsung-safe':
        success = convert_samsung_safe(args.input, output_path, info)
    elif args.preset == 'surround':
        success = convert_samsung_surround(args.input, output_path, info)
    elif args.preset == 'samsung-4k':
        success = convert_samsung_4k(args.input, output_path, info)
    elif args.preset == 'remux':
        success = remux_to_mp4(args.input, output_path, info)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
