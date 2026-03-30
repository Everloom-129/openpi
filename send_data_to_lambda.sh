#!/usr/bin/env bash
set -euo pipefail

# Script to sync data directory to Lambda Labs instance
# Usage: ./send_data_to_lambda.sh [--dry-run]

LAMBDA_HOST="ubuntu@64.181.234.106"
SSH_KEY="$HOME/.ssh/id_lambda_transfer"
LOCAL_DATA="$HOME/projects/openpi_vis/data/"
REMOTE_DATA="~/projects/openpi/data/"

# Check if SSH key exists
if [[ ! -f "$SSH_KEY" ]]; then
    echo "Error: SSH key not found at $SSH_KEY"
    exit 1
fi

# Parse arguments
DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "Running in dry-run mode (no files will be transferred)"
fi

# Run rsync
echo "Syncing data to Lambda Labs instance..."
echo "Local:  $LOCAL_DATA"
echo "Remote: $LAMBDA_HOST:$REMOTE_DATA"
echo ""

rsync -avzP $DRY_RUN \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    -e "ssh -i $SSH_KEY" \
    "$LOCAL_DATA" \
    "$LAMBDA_HOST:$REMOTE_DATA"

echo ""
echo "Sync complete!"