#!/bin/bash

# Check if source and destination directories are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <source_directory> <destination_directory>"
    exit 1
fi

source_dir="$1"
dest_dir="$2"

# Check if source directory exists
if [ ! -d "$source_dir" ]; then
    echo "Error: Source directory does not exist."
    exit 1
fi

# Create destination directory if it doesn't exist
mkdir -p "$dest_dir"

# Find all files in source directory and its subdirectories, then copy them to destination
find "$source_dir" -type f -exec cp {} "$dest_dir" \;

echo "All files have been copied to $dest_dir"