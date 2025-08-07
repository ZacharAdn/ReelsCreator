# TikTok Content Extractor

A Python-based system for automatically extracting valuable segments from long educational recordings and preparing them for short-form content (TikTok, Reels, Shorts).

## Overview

This system helps content creators efficiently process long educational recordings by:
1. Transcribing audio with timestamps using Whisper
2. Segmenting into overlapping chunks (45s with 10s overlap)
3. Generating embeddings for semantic analysis
4. Evaluating content value using GPT-4
5. Exporting high-value segments for editing

## Features

- **Automatic Transcription**: Uses OpenAI Whisper for accurate transcription
- **Smart Segmentation**: Creates overlapping segments to capture complete thoughts
- **Semantic Analysis**: Uses sentence-transformers for content similarity
- **AI Evaluation**: GPT-4 powered content quality assessment
- **Multiple Output Formats**: JSON, CSV, and detailed summaries
- **Batch Processing**: Process multiple files efficiently

## Project Status
✅ Initial Implementation Complete 🚧

See the [SPECS](SPECS/content_extractor_spec.md) folder for detailed technical specifications.

## Requirements
- Python 3.8+
- FFmpeg (system dependency for video processing)
- GPU recommended for better performance
- No paid API dependencies - uses free open-source models

## Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install FFmpeg (if not already installed)
```bash
# On Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# On macOS with Homebrew
brew install ffmpeg

# On Windows
# Download from https://ffmpeg.org/download.html
```

### 3. Run the Extractor

#### Using CLI:
```bash
python -m src path/to/your/video.mp4 -o results.json --export-csv segments.csv
```

#### Using Python:
```python
from src.content_extractor import ContentExtractor
from src.models import ProcessingConfig

# Configure
config = ProcessingConfig(
    segment_duration=45,
    overlap_duration=10,
    min_score_threshold=0.7
)

# Initialize and process
extractor = ContentExtractor(config)
result = extractor.process_video_file("path/to/video.mp4", "results.json")

# View results
print(f"Found {len(result.high_value_segments)} high-value segments")
```

### 4. View Results
- **JSON Output**: Complete results with metadata
- **CSV Export**: Spreadsheet-friendly format
- **Console Summary**: Key statistics and top segments

## Configuration

### Processing Parameters
- `segment_duration`: Length of each segment (default: 45s)
- `overlap_duration`: Overlap between segments (default: 10s)
- `min_score_threshold`: Minimum quality score (default: 0.7)
- `whisper_model`: Whisper model size (tiny/base/small/medium/large)

### Example Config File
```json
{
    "segment_duration": 45,
    "overlap_duration": 10,
    "min_score_threshold": 0.7,
    "whisper_model": "base",
    "embedding_model": "all-MiniLM-L6-v2"
}
```

## Output Format

### High-Value Segments Include:
- **Timestamps**: Precise start/end times
- **Transcribed Text**: Clean, readable text
- **Quality Score**: 0-1 rating with reasoning
- **Confidence**: Transcription confidence
- **Embeddings**: For similarity analysis

### Example Output:
```json
{
    "segments": [...],
    "high_value_segments": [
        {
            "start_time": 120.5,
            "end_time": 165.2,
            "text": "Here's a practical example of using pandas...",
            "value_score": 0.85,
            "reasoning": "Clear explanation with practical demonstration"
        }
    ],
    "summary": {
        "total_segments": 24,
        "high_value_count": 8,
        "average_score": 0.72
    }
}
```

## Testing

Run the test suite:
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test categories
pytest tests/test_core.py
pytest tests/test_data_driven.py
```

## License
MIT 