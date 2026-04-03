    #!/bin/bash

# Check if input directory is provided
if [ -z "$2" ]; then
    echo "Usage: $0 <input_directory> <name>" 
    exit 1
fi

INPUT_DIR="$1"
NAME="$2"
REMOTE_USER="edward"
REMOTE_HOST="158.130.50.39"
REMOTE_PATH="/mnt/sda/edward/project_archive/$NAME"

# Ensure the input is a directory
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: '$INPUT_DIR' is not a directory."
    exit 1
fi

# Extract just the directory name (e.g., 2025-03-23_20-52-19)
DIR_NAME=$(basename "$INPUT_DIR")

# Optionally create the destination directory on the remote server
ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p \"$REMOTE_PATH\""

# Transfer the entire folder (not just its contents)
rsync -av --progress "$INPUT_DIR" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"

# Final status message
echo "Directory '$INPUT_DIR' has been transferred to '${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}'"

