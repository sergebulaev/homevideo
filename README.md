# HomeVideo

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FFmpeg](https://img.shields.io/badge/FFmpeg-007808?style=for-the-badge&logo=ffmpeg&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![License](https://img.shields.io/badge/License-Apache_2.0-green?style=for-the-badge)
![Claude](https://img.shields.io/badge/Built_with-Claude_AI-CC785C?style=for-the-badge&logo=anthropic&logoColor=white)

Video sync and conversion toolkit for Samsung TV playback via MiniDLNA.

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

[![Telegram](https://img.shields.io/badge/Follow-@sergiobulaev-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/sergiobulaev)
[![Blog](https://img.shields.io/badge/Blog-bulaev.net-FF5722?style=for-the-badge&logo=substack&logoColor=white)](https://www.bulaev.net)
