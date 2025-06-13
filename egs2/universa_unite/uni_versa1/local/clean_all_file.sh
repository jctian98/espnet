#!/bin/bash

# Define the root directory containing all the raw data
ROOT_DIR="dump/raw"

# Path to the Python script
PYTHON_SCRIPT="local/remove_nan.py"

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script $PYTHON_SCRIPT not found."
    exit 1
fi

# Find all metric.scp files in the directory structure
find "$ROOT_DIR" -name "metric.scp" | while read -r file_path; do
    echo "Processing: $file_path"
    
    # Create a temporary file for output
    temp_file="${file_path}.cleaned"
    
    # Run the Python script to clean the file
    python "$PYTHON_SCRIPT" -i "$file_path" -o "$temp_file"
    
    # Check if the Python script executed successfully
    if [ $? -eq 0 ]; then
        # Make a backup of the original file (optional)
        cp "$file_path" "${file_path}.bak"
        
        # Replace the original file with the cleaned one
        mv "$temp_file" "$file_path"
        echo "Successfully cleaned and replaced: $file_path"
    else
        echo "Error processing: $file_path"
        # Remove the temp file if it exists
        [ -f "$temp_file" ] && rm "$temp_file"
    fi
    
    echo "----------------------------------------"
done

echo "All files processed!"
