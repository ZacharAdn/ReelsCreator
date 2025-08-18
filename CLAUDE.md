# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Setup and Installation
```bash
# Create and activate virtual environment
python -m venv reels_extractor_env
source reels_extractor_env/bin/activate  # macOS/Linux
# or reels_extractor_env\Scripts\activate on Windows

# Install dependencies (standard full installation)
pip install -r requirements.txt

# Install basic dependencies only (if full install fails)
pip install -r requirements_basic.txt

# Setup environment (M1 Mac optimization, Hebrew models)
python setup_environment.py
```

### Running the System
```bash
# Basic video processing
python -m src path/to/video.mp4

# Fast processing (70% faster, good for testing)
python -m src path/to/video.mp4 --profile draft

# High quality processing (20% slower)
python -m src path/to/video.mp4 --profile quality

# For balanced speed/accuracy:
python -m src video.mp4 --profile balanced

# With CSV export and custom output
python -m src path/to/video.mp4 -o results.json --export-csv segments.csv

# With advanced features
python -m src path/to/video.mp4 --enable-speaker-detection --enable-similarity --primary-speaker-only
```

### Testing and Development
```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest --cov=src tests/ --cov-report=html

# Run specific test categories
pytest tests/test_core.py
pytest tests/test_data_driven.py

# Run performance benchmarks
pytest tests/test_benchmark.py
```

### Troubleshooting Commands
```bash
# Check FFmpeg installation
ffmpeg -version

# Test basic video processing without AI models
python -c "from src.video_processing import extract_audio; extract_audio('test.mp4', 'test.wav')"

# Test Whisper installation
python -c "import whisper; print(whisper.available_models())"
```

## Architecture Overview

This is a **multilingual educational content extraction pipeline** specifically optimized for Hebrew/English data science tutorials. The system processes long-form educational recordings to extract high-value segments suitable for short-form content creation.

### Core Components

**Main Orchestrator**: `src/content_extractor.py` - ContentExtractor class manages the entire pipeline

**Processing Pipeline**:
1. **Video Processing** (`src/video_processing.py`) - FFmpeg-based audio extraction
2. **Transcription** (`src/transcription.py`) - OpenAI Whisper with Hebrew/English support
3. **Segmentation** (`src/segmentation.py`) - Direct Whisper segment boundaries
4. **Speaker Analysis** (`src/speaker_analysis.py`) - Teacher vs student identification
5. **Evaluation** (`src/evaluation.py`) - Local LLM quality assessment (Qwen2.5, Phi-3)
6. **Embeddings** (`src/embeddings.py`) - Sentence-transformers semantic analysis

**Advanced Features**:
- **Speaker Diarization**: `src/stages/01_speaker_segmentation/` - Advanced speaker detection module
- **Language Processing**: `src/language_processor.py` - Hebrew/English multilingual handling
- **Data Models**: `src/models.py` - Configuration and result structures

### Key Technologies

- **AI Models**: OpenAI Whisper (transcription), Sentence-Transformers (embeddings), Qwen2.5/Phi-3 (evaluation)
- **Media Processing**: FFmpeg, MoviePy, librosa
- **Language Support**: spaCy (Hebrew), LangDetect, PyAnnote.Audio
- **Performance**: PyTorch with M1 Mac MPS optimization, batch processing
- **No External APIs**: Uses only local open-source models

### Configuration System

The system uses `ProcessingConfig` class with three optimized profiles:
- **draft**: 70% faster, minimal quality (good for testing)
- **balanced**: Default settings
- **quality**: 20% slower, maximum quality

Key parameters:
- `segment_duration`: 45s (typical TikTok/Reel length)
- `overlap_duration`: 10s (prevents content splitting)
- `min_score_threshold`: 0.7 (quality threshold)
- `whisper_model`: "base" (balance of speed/accuracy)
- `evaluation_model`: Qwen2.5 or Phi-3 for local evaluation

### Data Flow

1. **Input**: MP4/MOV/AVI video files
2. **Audio Extraction**: FFmpeg converts to WAV
3. **Transcription**: Whisper generates timestamped text
4. **Segmentation**: Natural speech boundaries preserved
5. **Speaker Analysis**: Teacher/student identification (optional)
6. **Content Evaluation**: Local LLM scores segments
7. **Output**: JSON/CSV with high-value segments, embeddings, metadata

### Output Structure

Results include:
- **High-value segments**: Start/end times, transcribed text, quality scores
- **Metadata**: Processing configuration, performance metrics
- **Embeddings**: For similarity analysis and clustering
- **Speaker information**: Teacher vs student identification
- **Quality reasoning**: AI-generated evaluation explanations

### Hebrew/English Specifics

- **Primary language**: Hebrew with English technical terms
- **Technical preservation**: Maintains data science terminology in English
- **spaCy Hebrew model**: Required for proper language processing
- **Character encoding**: UTF-8 throughout pipeline

### Performance Considerations

- **M1 Mac optimization**: MPS (Metal Performance Shaders) support
- **GPU acceleration**: CUDA/MPS for faster processing
- **Batch processing**: Configurable batch sizes for memory management
- **Minimal mode**: Skip non-essential processing for maximum speed
- **Memory efficient**: Streaming processing for large videos

### Common Development Patterns

When modifying this codebase:
1. **Follow modular architecture**: Each stage is independent
2. **Use configuration objects**: ProcessingConfig for all parameters
3. **Maintain Hebrew/English support**: Test with multilingual content
4. **Preserve error handling**: Comprehensive logging throughout
5. **Test thoroughly**: Core, integration, and performance tests
6. **M1 Mac compatibility**: Always test MPS acceleration paths