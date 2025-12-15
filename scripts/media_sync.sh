#!/bin/bash
# Media Sync Crontab Wrapper
#
# This script is designed to be run from crontab to sync and convert
# video files from the remote server.
#
# Usage:
#   ./media_sync.sh              # Run sync
#   ./media_sync.sh --dry-run    # Preview only
#
# Crontab example (run every 6 hours):
#   0 */6 * * * /home/sbulaev/p/homevideo/scripts/media_sync.sh >> /home/sbulaev/p/homevideo/logs/cron.log 2>&1
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/venv"
LOCK_FILE="$PROJECT_DIR/.media_sync.lock"
LOG_FILE="$PROJECT_DIR/logs/cron.log"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Timestamp function
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# Log function
log() {
    echo "[$(timestamp)] $1"
}

# Check for lock file (prevent concurrent runs)
if [ -f "$LOCK_FILE" ]; then
    PID=$(cat "$LOCK_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        log "ERROR: Another instance is running (PID: $PID)"
        exit 1
    else
        log "WARNING: Stale lock file found, removing"
        rm -f "$LOCK_FILE"
    fi
fi

# Create lock file
echo $$ > "$LOCK_FILE"
trap "rm -f $LOCK_FILE" EXIT

log "=========================================="
log "Media Sync Started"
log "=========================================="

# Activate virtual environment
if [ ! -d "$VENV_DIR" ]; then
    log "ERROR: Virtual environment not found at $VENV_DIR"
    exit 1
fi

source "$VENV_DIR/bin/activate"

# Check if SSH key is available (for non-interactive use)
if ! ssh-add -l &>/dev/null; then
    log "WARNING: No SSH keys loaded, may need manual authentication"
fi

# Run the sync command
cd "$PROJECT_DIR"
python scripts/media_sync.py sync "$@"

EXIT_CODE=$?

log "=========================================="
log "Media Sync Completed (exit code: $EXIT_CODE)"
log "=========================================="

# Restart minidlna if we have permission
if command -v systemctl &>/dev/null; then
    if systemctl restart minidlna 2>/dev/null; then
        log "Restarted minidlna"
    else
        log "Could not restart minidlna (may need sudo)"
    fi
fi

exit $EXIT_CODE
