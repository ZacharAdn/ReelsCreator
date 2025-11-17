#!/bin/bash

# Print header
echo "🎬 Starting Hebrew Transcription Pipeline"
echo "========================================"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the project root directory
cd "$SCRIPT_DIR"

# Check if Ollama is already running
OLLAMA_WAS_RUNNING=false
if pgrep -x "ollama" > /dev/null; then
    echo "✅ Ollama is already running"
    OLLAMA_WAS_RUNNING=true
else
    echo "🤖 Starting Ollama for AI analysis..."

    # Start Ollama in the background
    brew services start ollama > /dev/null 2>&1

    # Wait for Ollama to be ready (max 10 seconds)
    echo "⏳ Waiting for Ollama to initialize..."
    for i in {1..10}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "✅ Ollama is ready"
            break
        fi
        sleep 1
    done

    # Check if Ollama started successfully
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "⚠️  Warning: Ollama failed to start (transcription will continue without AI analysis)"
    fi
fi

# Check if the Hebrew model is available
if curl -s http://localhost:11434/api/tags 2>/dev/null | grep -q "aya-expanse"; then
    echo "✅ Hebrew model (aya-expanse:8b) is available"
else
    echo "📥 Downloading Hebrew model (aya-expanse:8b)..."
    echo "   (This is a one-time download of ~5GB, may take 5-10 minutes)"

    ollama pull aya-expanse:8b

    if [ $? -eq 0 ]; then
        echo "✅ Model downloaded successfully"
    else
        echo "⚠️  Warning: Model download failed (transcription will continue without AI analysis)"
    fi
fi

# Activate the virtual environment
echo "🔄 Activating virtual environment..."
source reels_extractor_env/bin/activate

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate virtual environment"

    # Stop Ollama if we started it
    if [ "$OLLAMA_WAS_RUNNING" = false ]; then
        echo "🛑 Stopping Ollama..."
        brew services stop ollama > /dev/null 2>&1
    fi

    exit 1
fi

echo "✅ Virtual environment activated"

echo "🚀 Running transcription script..."
echo "----------------------------------------"

# Run the transcription script (using project root path)
python src/scripts/transcribe_advanced.py

# Store the exit code
EXIT_CODE=$?

# Deactivate the virtual environment
deactivate

# Stop Ollama if we started it (save RAM)
if [ "$OLLAMA_WAS_RUNNING" = false ]; then
    # Check if there are other transcription processes still running
    CURRENT_PID=$$
    OTHER_TRANSCRIPTIONS=$(ps aux | grep "transcribe_advanced.py" | grep -v grep | grep -v $CURRENT_PID | wc -l)

    if [ "$OTHER_TRANSCRIPTIONS" -gt 0 ]; then
        echo "ℹ️  Keeping Ollama running (detected $OTHER_TRANSCRIPTIONS other transcription(s) in progress)"
    else
        echo "🛑 Stopping Ollama to free RAM..."
        brew services stop ollama > /dev/null 2>&1

        # Wait a moment for shutdown
        sleep 1

        if pgrep -x "ollama" > /dev/null; then
            echo "⚠️  Warning: Ollama may still be running"
        else
            echo "✅ Ollama stopped successfully"
        fi
    fi
fi

# Check if the script ran successfully
if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Transcription failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "----------------------------------------"
echo "✨ All done! Check the results directory for output files"
