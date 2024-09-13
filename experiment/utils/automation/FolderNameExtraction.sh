#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 target_directory output_file"
    exit 1
fi

# Assign arguments to variables
TARGET_DIR="$1"
OUTPUT_FILE="$2"

# Check if the target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "The specified target directory does not exist: $TARGET_DIR"
    exit 1
fi

# Write the folder names to the output file
find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; > "$OUTPUT_FILE"

# Confirm completion
echo "Folder names from '$TARGET_DIR' have been written to '$OUTPUT_FILE'."
