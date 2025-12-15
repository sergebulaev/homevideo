#!/usr/bin/env python3
"""
Media Sync - Production-grade media synchronization and conversion tool.

Syncs video files from a remote server, converts them to Samsung TV compatible
format, and maintains a database of processed files.

Features:
- SSH-based remote file discovery
- Rsync for efficient file transfer
- FFmpeg conversion with all audio tracks preserved
- JSON database for tracking processed files
- Rotating log files
- Automatic cleanup after successful conversion

Usage:
    media_sync sync                    # Sync and convert new files
    media_sync status                  # Show processing status
    media_sync list-remote             # List files on remote server
    media_sync convert <file>          # Convert a single local file
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from loguru import logger

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
        DownloadColumn,
        TransferSpeedColumn,
    )
    from rich.table import Table
    from rich.live import Live
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None


def format_size(bytes_size: int) -> str:
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format seconds to HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "remote_host": "samui001",
    "remote_path": "/srv/torrents",
    "local_sync_dir": "/home/sbulaev/p/homevideo/incoming",
    "output_dir": "/media/shared/Movies",
    "database_file": "/home/sbulaev/p/homevideo/data/processed_files.json",
    "log_dir": "/home/sbulaev/p/homevideo/logs",
    "video_extensions": [".mkv", ".avi", ".mp4", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".ts", ".mpg", ".mpeg"],
    "min_file_size_mb": 100,  # Ignore files smaller than this (likely not movies)
    "ffmpeg_preset": "medium",
    "ffmpeg_crf": 20,
    "audio_bitrate": "384k",
    "keep_surround": True,
}

# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(log_dir: str, verbose: bool = False):
    """Configure loguru logging with file rotation."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console handler
    console_level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=console_level,
        colorize=True,
    )

    # File handler with rotation
    logger.add(
        log_path / "media_sync_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        rotation="00:00",  # New file each day
        retention="30 days",
        compression="gz",
    )

    # Error-only log file
    logger.add(
        log_path / "errors.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="ERROR",
        rotation="10 MB",
        retention="90 days",
    )

# ============================================================================
# Database Operations
# ============================================================================

class ProcessedFilesDB:
    """JSON-based database for tracking processed files."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        """Load database from file."""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r') as f:
                    self.data = json.load(f)
            except json.JSONDecodeError:
                logger.warning("Corrupted database, starting fresh")
                self.data = {"processed": {}, "failed": {}, "in_progress": {}}
        else:
            self.data = {"processed": {}, "failed": {}, "in_progress": {}}

    def _save(self):
        """Save database to file."""
        with open(self.db_path, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)

    def is_processed(self, remote_path: str) -> bool:
        """Check if file has been processed."""
        return remote_path in self.data["processed"]

    def is_failed(self, remote_path: str) -> bool:
        """Check if file has previously failed."""
        return remote_path in self.data["failed"]

    def mark_in_progress(self, remote_path: str, local_path: str):
        """Mark file as currently being processed."""
        self.data["in_progress"][remote_path] = {
            "local_path": local_path,
            "started_at": datetime.now().isoformat(),
        }
        self._save()

    def mark_processed(self, remote_path: str, output_path: str, stats: dict):
        """Mark file as successfully processed."""
        self.data["processed"][remote_path] = {
            "output_path": output_path,
            "processed_at": datetime.now().isoformat(),
            "stats": stats,
        }
        # Remove from in_progress
        self.data["in_progress"].pop(remote_path, None)
        self._save()

    def mark_failed(self, remote_path: str, error: str):
        """Mark file as failed."""
        self.data["failed"][remote_path] = {
            "error": error,
            "failed_at": datetime.now().isoformat(),
            "attempts": self.data["failed"].get(remote_path, {}).get("attempts", 0) + 1,
        }
        # Remove from in_progress
        self.data["in_progress"].pop(remote_path, None)
        self._save()

    def get_stats(self) -> dict:
        """Get processing statistics."""
        return {
            "processed": len(self.data["processed"]),
            "failed": len(self.data["failed"]),
            "in_progress": len(self.data["in_progress"]),
        }

    def clear_in_progress(self):
        """Clear in-progress entries (for recovery after crash)."""
        self.data["in_progress"] = {}
        self._save()

# ============================================================================
# Remote Operations
# ============================================================================

def get_remote_files(host: str, path: str, extensions: list, min_size_mb: int) -> list:
    """Get list of video files from remote server via SSH."""
    logger.info(f"Scanning remote: {host}:{path}")

    # Build find command for video files
    ext_pattern = " -o ".join([f'-name "*{ext}"' for ext in extensions])
    min_size_bytes = min_size_mb * 1024 * 1024

    cmd = [
        "ssh", host,
        f'find "{path}" -type f \\( {ext_pattern} \\) -size +{min_size_bytes}c 2>/dev/null'
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.error(f"SSH command failed: {result.stderr}")
            return []

        files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        logger.info(f"Found {len(files)} video files on remote")
        return files

    except subprocess.TimeoutExpired:
        logger.error("SSH command timed out")
        return []
    except Exception as e:
        logger.error(f"Error scanning remote: {e}")
        return []

def get_remote_file_info(host: str, filepath: str) -> dict:
    """Get file info from remote server."""
    cmd = [
        "ssh", host,
        f'stat --printf="%s\\n%Y" "{filepath}" 2>/dev/null'
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            return {
                "size": int(lines[0]),
                "mtime": int(lines[1]),
            }
    except Exception as e:
        logger.debug(f"Could not get file info: {e}")

    return {}

def sync_file(host: str, remote_path: str, local_dir: str) -> Optional[str]:
    """Sync a single file from remote server using rsync with progress display."""
    local_dir_path = Path(local_dir)
    local_dir_path.mkdir(parents=True, exist_ok=True)

    filename = Path(remote_path).name
    local_path = local_dir_path / filename

    # Get remote file size first
    file_info = get_remote_file_info(host, remote_path)
    total_size = file_info.get("size", 0)

    logger.info(f"Syncing: {filename} ({format_size(total_size)})")

    cmd = [
        "rsync", "-avz", "--progress", "--partial",
        f"{host}:{remote_path}",
        str(local_path)
    ]

    if RICH_AVAILABLE and total_size > 0:
        # Use rich progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Downloading", total=total_size)

            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                )

                start_time = time.time()

                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break

                    # Parse rsync progress output
                    # Format: "  1,234,567   5%  123.45kB/s    0:01:23"
                    match = re.search(r'^\s*([\d,]+)\s+(\d+)%', line)
                    if match:
                        transferred = int(match.group(1).replace(',', ''))
                        progress.update(task, completed=transferred)

                process.wait()
                elapsed = time.time() - start_time

                if process.returncode == 0:
                    progress.update(task, completed=total_size)
                    logger.success(f"Downloaded in {format_duration(elapsed)}")
                    return str(local_path)
                else:
                    stderr = process.stderr.read()
                    logger.error(f"Rsync failed: {stderr}")
                    return None

            except Exception as e:
                logger.error(f"Sync error: {e}")
                return None
    else:
        # Fallback without rich
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            last_percent = 0
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break

                match = re.search(r'(\d+)%', line)
                if match:
                    percent = int(match.group(1))
                    if percent >= last_percent + 10:
                        logger.info(f"Download progress: {percent}%")
                        last_percent = percent

            process.wait()

            if process.returncode == 0:
                logger.success(f"Synced: {filename}")
                return str(local_path)
            else:
                stderr = process.stderr.read()
                logger.error(f"Rsync failed: {stderr}")
                return None

        except Exception as e:
            logger.error(f"Sync error: {e}")
            return None

# ============================================================================
# Video Conversion
# ============================================================================

def get_video_info(filepath: str) -> dict:
    """Get video file information using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,size,bit_rate",
        "-show_entries", "stream=index,codec_type,codec_name,width,height,channels",
        "-show_entries", "stream_tags=language,title",
        "-of", "json",
        filepath
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            data = json.loads(result.stdout)

            streams = data.get("streams", [])
            format_info = data.get("format", {})

            video_streams = [s for s in streams if s.get("codec_type") == "video"]
            audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

            return {
                "duration": float(format_info.get("duration", 0)),
                "size": int(format_info.get("size", 0)),
                "video_codec": video_streams[0].get("codec_name") if video_streams else None,
                "width": video_streams[0].get("width") if video_streams else None,
                "height": video_streams[0].get("height") if video_streams else None,
                "audio_tracks": len(audio_streams),
                "audio_codecs": [s.get("codec_name") for s in audio_streams],
                "audio_languages": [s.get("tags", {}).get("language", "und") for s in audio_streams],
            }
    except Exception as e:
        logger.error(f"Error probing video: {e}")

    return {}

def convert_video(
    input_path: str,
    output_path: str,
    preset: str = "medium",
    crf: int = 20,
    audio_bitrate: str = "384k",
    keep_surround: bool = True,
) -> tuple[bool, dict]:
    """
    Convert video to Samsung TV compatible format.

    Returns tuple of (success, stats).
    """
    logger.info(f"Converting: {Path(input_path).name}")

    # Get input info
    info = get_video_info(input_path)
    if not info:
        return False, {"error": "Could not probe input file"}

    logger.debug(f"Input: {info.get('video_codec')} {info.get('width')}x{info.get('height')}, "
                 f"{info.get('audio_tracks')} audio tracks")

    # Determine if we need to re-encode video
    video_codec = info.get("video_codec", "")
    needs_video_reencode = video_codec not in ["h264", "avc"]

    # Build FFmpeg command
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-progress", "pipe:1",
        "-nostats",
    ]

    # Map streams
    cmd.extend(["-map", "0:v:0"])  # First video stream
    cmd.extend(["-map", "0:a"])    # All audio streams

    # Strip global metadata but preserve stream metadata (language tags, etc.)
    output_title = Path(output_path).stem
    cmd.extend(["-map_metadata:g", "-1"])  # Strip only global metadata
    cmd.extend(["-metadata", f"title={output_title}"])

    # Video codec settings
    if needs_video_reencode:
        cmd.extend([
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
        ])
        logger.info(f"Re-encoding video: {video_codec} -> H.264")
    else:
        cmd.extend(["-c:v", "copy"])
        logger.info("Copying video stream (already H.264)")

    # Audio codec settings - convert all tracks to AAC
    audio_codecs = info.get("audio_codecs", [])
    needs_audio_reencode = any(c not in ["aac"] for c in audio_codecs)

    if needs_audio_reencode:
        cmd.extend([
            "-c:a", "aac",
            "-b:a", audio_bitrate,
        ])
        if not keep_surround:
            cmd.extend(["-ac", "2"])  # Downmix to stereo
        logger.info(f"Converting audio: {audio_codecs} -> AAC")
    else:
        cmd.extend(["-c:a", "copy"])
        logger.info("Copying audio streams (already AAC)")

    # Output settings
    cmd.extend(["-movflags", "faststart"])
    cmd.append(output_path)

    # Execute conversion
    start_time = time.time()
    duration = info.get("duration", 0)
    input_size = info.get("size", 0)

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        if RICH_AVAILABLE and duration > 0:
            # Rich progress bar for conversion
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TextColumn("[green]{task.fields[speed]}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Converting",
                    total=int(duration),
                    speed="0.0x"
                )

                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break

                    if "out_time_ms=" in line:
                        try:
                            ms = int(line.split("=")[1])
                            current_time = ms / 1_000_000
                            elapsed_so_far = time.time() - start_time
                            speed = f"{current_time / elapsed_so_far:.1f}x" if elapsed_so_far > 0 else "0.0x"
                            progress.update(task, completed=int(current_time), speed=speed)
                        except:
                            pass

                process.wait()

        else:
            # Fallback without rich
            last_progress = 0
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break

                if "out_time_ms=" in line:
                    try:
                        ms = int(line.split("=")[1])
                        current_time = ms / 1_000_000
                        if duration > 0:
                            pct = int(current_time / duration * 100)
                            if pct >= last_progress + 10:
                                logger.info(f"Progress: {pct}%")
                                last_progress = pct
                    except:
                        pass

            process.wait()

        elapsed = time.time() - start_time

        if process.returncode == 0:
            output_size = os.path.getsize(output_path)

            stats = {
                "duration": elapsed,
                "speed": f"{duration / elapsed:.2f}x" if elapsed > 0 else "N/A",
                "input_size": input_size,
                "output_size": output_size,
                "size_change": f"{(1 - output_size / input_size) * 100:.1f}%" if input_size > 0 else "N/A",
                "video_reencoded": needs_video_reencode,
                "audio_reencoded": needs_audio_reencode,
            }

            logger.success(f"Conversion complete in {format_duration(elapsed)} ({stats['speed']} realtime)")

            # Show summary panel
            if RICH_AVAILABLE:
                table = Table(show_header=False, box=None)
                table.add_row("Input:", format_size(input_size))
                table.add_row("Output:", format_size(output_size))
                table.add_row("Reduction:", stats['size_change'])
                table.add_row("Speed:", stats['speed'])
                console.print(Panel(table, title="[green]Conversion Complete", border_style="green"))

            return True, stats
        else:
            stderr = process.stderr.read()
            logger.error(f"FFmpeg failed: {stderr[:500]}")
            return False, {"error": stderr[:500]}

    except Exception as e:
        logger.error(f"Conversion error: {e}")
        return False, {"error": str(e)}

# ============================================================================
# CLI Commands
# ============================================================================

@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--config", "-c", type=click.Path(), help="Config file path")
@click.pass_context
def cli(ctx, verbose, config):
    """Media Sync - Sync and convert videos for Samsung TV."""
    ctx.ensure_object(dict)

    # Load config
    cfg = DEFAULT_CONFIG.copy()
    if config and Path(config).exists():
        with open(config) as f:
            cfg.update(json.load(f))

    ctx.obj["config"] = cfg
    ctx.obj["verbose"] = verbose

    # Setup logging
    setup_logging(cfg["log_dir"], verbose)

@cli.command()
@click.option("--dry-run", is_flag=True, help="Show what would be done without doing it")
@click.option("--limit", type=int, default=0, help="Limit number of files to process")
@click.pass_context
def sync(ctx, dry_run, limit):
    """Sync and convert new video files from remote server."""
    cfg = ctx.obj["config"]

    logger.info("=" * 60)
    logger.info("Media Sync Started")
    logger.info("=" * 60)

    # Initialize database
    db = ProcessedFilesDB(cfg["database_file"])
    db.clear_in_progress()  # Clean up from any previous crashes

    # Get remote files
    remote_files = get_remote_files(
        cfg["remote_host"],
        cfg["remote_path"],
        cfg["video_extensions"],
        cfg["min_file_size_mb"],
    )

    if not remote_files:
        logger.info("No files found on remote server")
        return

    # Filter out already processed
    new_files = [f for f in remote_files if not db.is_processed(f)]
    logger.info(f"New files to process: {len(new_files)}")

    if limit > 0:
        new_files = new_files[:limit]
        logger.info(f"Limited to {limit} files")

    if dry_run:
        logger.info("DRY RUN - would process:")
        for f in new_files:
            logger.info(f"  {Path(f).name}")
        return

    # Process each file
    processed = 0
    failed = 0

    for idx, remote_path in enumerate(new_files, 1):
        filename = Path(remote_path).name

        # Show file header with rich panel
        if RICH_AVAILABLE:
            file_info = get_remote_file_info(cfg["remote_host"], remote_path)
            size_str = format_size(file_info.get("size", 0)) if file_info else "unknown"
            console.print()
            console.print(Panel(
                f"[bold]{filename}[/bold]\n[dim]Size: {size_str}[/dim]",
                title=f"[cyan]File {idx}/{len(new_files)}[/cyan]",
                border_style="cyan"
            ))
        else:
            logger.info("-" * 40)
            logger.info(f"Processing: {filename}")

        # Skip if previously failed too many times
        if db.is_failed(remote_path):
            fail_info = db.data["failed"].get(remote_path, {})
            if fail_info.get("attempts", 0) >= 3:
                logger.warning(f"Skipping (failed {fail_info['attempts']} times): {filename}")
                continue

        try:
            # Sync file
            local_path = sync_file(cfg["remote_host"], remote_path, cfg["local_sync_dir"])
            if not local_path:
                db.mark_failed(remote_path, "Sync failed")
                failed += 1
                continue

            db.mark_in_progress(remote_path, local_path)

            # Determine output path
            output_name = Path(local_path).stem
            # Clean up filename
            output_name = re.sub(r'[^\w\s\-\.]', '', output_name)
            output_name = f"tv-{output_name}.mp4"
            output_path = str(Path(cfg["output_dir"]) / output_name)

            # Convert
            success, stats = convert_video(
                local_path,
                output_path,
                preset=cfg["ffmpeg_preset"],
                crf=cfg["ffmpeg_crf"],
                audio_bitrate=cfg["audio_bitrate"],
                keep_surround=cfg["keep_surround"],
            )

            if success:
                db.mark_processed(remote_path, output_path, stats)
                processed += 1

                # Cleanup local synced file
                try:
                    os.remove(local_path)
                    logger.debug(f"Removed local copy: {local_path}")
                except Exception as e:
                    logger.warning(f"Could not remove local copy: {e}")

                # Update minidlna database
                try:
                    db_path = Path("/var/cache/minidlna/files.db")
                    if db_path.exists():
                        os.remove(db_path)
                        logger.debug("Cleared minidlna database")
                except Exception as e:
                    logger.debug(f"Could not clear minidlna db: {e}")
            else:
                db.mark_failed(remote_path, stats.get("error", "Unknown error"))
                failed += 1

                # Cleanup failed output
                if os.path.exists(output_path):
                    os.remove(output_path)

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            db.mark_failed(remote_path, str(e))
            failed += 1

    # Summary
    if RICH_AVAILABLE:
        console.print()
        summary_table = Table(show_header=False, box=None)
        summary_table.add_row("[green]Processed:[/green]", str(processed))
        summary_table.add_row("[red]Failed:[/red]", str(failed))
        summary_table.add_row("[blue]Total:[/blue]", str(len(new_files)))

        style = "green" if failed == 0 else "yellow"
        console.print(Panel(
            summary_table,
            title=f"[bold {style}]Sync Complete[/bold {style}]",
            border_style=style
        ))
    else:
        logger.info("=" * 60)
        logger.info(f"Sync Complete: {processed} processed, {failed} failed")
        logger.info("=" * 60)

@cli.command()
@click.pass_context
def status(ctx):
    """Show processing status and statistics."""
    cfg = ctx.obj["config"]
    db = ProcessedFilesDB(cfg["database_file"])

    stats = db.get_stats()

    if RICH_AVAILABLE:
        # Stats table
        stats_table = Table(show_header=False, box=None)
        stats_table.add_row("[green]Processed:[/green]", str(stats['processed']))
        stats_table.add_row("[red]Failed:[/red]", str(stats['failed']))
        stats_table.add_row("[yellow]In Progress:[/yellow]", str(stats['in_progress']))

        console.print()
        console.print(Panel(stats_table, title="[bold]Media Sync Status[/bold]", border_style="blue"))

        # Recent files
        if db.data["processed"]:
            files_table = Table(show_header=True, header_style="bold cyan")
            files_table.add_column("Source", style="dim")
            files_table.add_column("Output", style="green")
            files_table.add_column("Date", style="dim")

            recent = list(db.data["processed"].items())[-5:]
            for path, info in recent:
                date = info.get("processed_at", "")[:10]
                files_table.add_row(
                    Path(path).name[:40] + "..." if len(Path(path).name) > 40 else Path(path).name,
                    Path(info['output_path']).name,
                    date
                )

            console.print()
            console.print(Panel(files_table, title="[bold green]Recent Processed[/bold green]", border_style="green"))

        # Failed files
        if db.data["failed"]:
            failed_table = Table(show_header=True, header_style="bold red")
            failed_table.add_column("File", style="dim")
            failed_table.add_column("Error", style="red")
            failed_table.add_column("Attempts", style="yellow")

            for path, info in db.data["failed"].items():
                failed_table.add_row(
                    Path(path).name[:30] + "..." if len(Path(path).name) > 30 else Path(path).name,
                    info['error'][:40] + "..." if len(info.get('error', '')) > 40 else info.get('error', 'Unknown'),
                    str(info.get('attempts', 1))
                )

            console.print()
            console.print(Panel(failed_table, title="[bold red]Failed Files[/bold red]", border_style="red"))

    else:
        click.echo("\nMedia Sync Status")
        click.echo("=" * 40)
        click.echo(f"Processed:   {stats['processed']}")
        click.echo(f"Failed:      {stats['failed']}")
        click.echo(f"In Progress: {stats['in_progress']}")
        click.echo()

        if db.data["processed"]:
            click.echo("Recent processed files:")
            recent = list(db.data["processed"].items())[-5:]
            for path, info in recent:
                click.echo(f"  - {Path(path).name}")
                click.echo(f"    -> {Path(info['output_path']).name}")

        if db.data["failed"]:
            click.echo("\nFailed files:")
            for path, info in db.data["failed"].items():
                click.echo(f"  - {Path(path).name}")
                click.echo(f"    Error: {info['error'][:50]}...")

@cli.command("list-remote")
@click.pass_context
def list_remote(ctx):
    """List video files on remote server."""
    cfg = ctx.obj["config"]

    files = get_remote_files(
        cfg["remote_host"],
        cfg["remote_path"],
        cfg["video_extensions"],
        cfg["min_file_size_mb"],
    )

    db = ProcessedFilesDB(cfg["database_file"])

    if RICH_AVAILABLE:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Status", style="dim", width=6)
        table.add_column("File", style="cyan")
        table.add_column("Size", justify="right", style="green")

        total_size = 0
        for f in files:
            if db.is_processed(f):
                status_icon = "[green]✓[/green]"
            elif db.is_failed(f):
                status_icon = "[red]✗[/red]"
            else:
                status_icon = "[dim]○[/dim]"

            file_info = get_remote_file_info(cfg["remote_host"], f)
            size = file_info.get("size", 0)
            total_size += size

            table.add_row(
                status_icon,
                Path(f).name,
                format_size(size)
            )

        console.print()
        console.print(Panel(
            table,
            title=f"[bold]Remote Files ({cfg['remote_host']}:{cfg['remote_path']})[/bold]",
            border_style="magenta"
        ))
        console.print(f"\n[bold]Total:[/bold] {len(files)} files, {format_size(total_size)}")
        console.print("[dim]Legend: ✓ = processed, ✗ = failed, ○ = pending[/dim]")

    else:
        click.echo(f"\nFiles on {cfg['remote_host']}:{cfg['remote_path']}")
        click.echo("=" * 60)

        for f in files:
            status = "✓" if db.is_processed(f) else "✗" if db.is_failed(f) else " "
            click.echo(f"[{status}] {Path(f).name}")

        click.echo(f"\nTotal: {len(files)} files")

@cli.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.pass_context
def convert(ctx, filepath, output):
    """Convert a single local video file."""
    cfg = ctx.obj["config"]

    if not output:
        stem = Path(filepath).stem
        output = str(Path(cfg["output_dir"]) / f"tv-{stem}.mp4")

    success, stats = convert_video(
        filepath,
        output,
        preset=cfg["ffmpeg_preset"],
        crf=cfg["ffmpeg_crf"],
        audio_bitrate=cfg["audio_bitrate"],
        keep_surround=cfg["keep_surround"],
    )

    if success:
        click.echo(f"\nConversion successful!")
        click.echo(f"Output: {output}")
        click.echo(f"Time: {stats['duration']:.0f}s ({stats['speed']})")
        click.echo(f"Size: {stats['size_change']}")
    else:
        click.echo(f"\nConversion failed: {stats.get('error', 'Unknown error')}")
        sys.exit(1)

@cli.command()
@click.pass_context
def init_config(ctx):
    """Create default config file."""
    config_path = Path("/home/sbulaev/p/homevideo/config.json")

    with open(config_path, 'w') as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)

    click.echo(f"Config created: {config_path}")
    click.echo("Edit this file to customize settings.")

if __name__ == "__main__":
    cli()
