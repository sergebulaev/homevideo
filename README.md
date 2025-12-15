# HomeVideo

Video sync and conversion toolkit for Samsung TV playback via MiniDLNA.

`Python` `FFmpeg` `rsync` `DLNA/UPnP` `SSH`

## What it does

- Syncs video files from a remote server over SSH/rsync
- Converts to Samsung TV compatible format (H.264 + AAC)
- Preserves all audio tracks and surround sound
- Tracks processed files to avoid re-processing
- Provides DLNA server scanner for debugging

## Requirements

- Python 3.8+
- FFmpeg
- rsync
- SSH access to remote server (optional, for sync feature)

## Installation

```bash
git clone https://github.com/sergebulaev/homevideo.git
cd homevideo
python3 -m venv venv
source venv/bin/activate
pip install click loguru rich ffmpeg-python requests
```

## Usage

### Convert a video

```bash
python scripts/convert.py movie.mkv -p surround -o output.mp4
```

Presets:
- `samsung-safe` - H.264 + AAC stereo (default, maximum compatibility)
- `surround` - H.264 + AAC 5.1 (keeps surround sound)
- `samsung-4k` - HEVC + AAC (for newer 4K TVs)
- `remux` - copy streams, just change container

### Sync from remote server

```bash
# Edit config in scripts/media_sync.py or create config.json
python scripts/media_sync.py sync
python scripts/media_sync.py sync --dry-run  # preview
python scripts/media_sync.py status          # check progress
```

### Scan DLNA servers

```bash
python scripts/dlna_scanner.py
python scripts/dlna_scanner.py --deep --search "movie"
```

## Cron setup

```bash
0 */6 * * * /path/to/homevideo/scripts/media_sync.sh
```

## License

Apache 2.0

## Author

[Serge Bulaev](https://github.com/sergebulaev)

- Telegram: [@sergeb](https://t.me/sergeb)
- Blog: [bulaev.net](https://www.bulaev.net)
