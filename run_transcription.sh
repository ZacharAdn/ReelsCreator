#!/bin/bash

# Print header
echo "🎬 Starting Hebrew Transcription Pipeline"
echo "========================================"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the project root directory
cd "$SCRIPT_DIR"

# Activate the virtual environment
echo "🔄 Activating virtual environment..."
source reels_extractor_env/bin/activate

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo "✅ Virtual environment activated"

# Change to the scripts directory
cd "src/quick scripts"

# Check if directory change was successful
if [ $? -ne 0 ]; then
    echo "❌ Failed to change to scripts directory"
    deactivate
    exit 1
fi

echo "🚀 Running transcription script..."
echo "----------------------------------------"

# Run the transcription script
python transcribe_advanced.py

# Store the exit code
EXIT_CODE=$?

# Deactivate the virtual environment
deactivate

# Check if the script ran successfully
if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Transcription failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "----------------------------------------"
echo "✨ All done! Check the results directory for output files"
