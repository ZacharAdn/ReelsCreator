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

# Note: requirements_basic.txt and setup_environment.py have been removed
# Use standard installation only
```

### Running the System
```bash
# Basic video processing
python -m src path/to/video.mp4

# Fast processing (70% faster, good for testing)
python -m src path/to/video.mp4 --profile draft

# ⚠️ High quality processing (CURRENTLY HANGS - USE BALANCED INSTEAD)
# python -m src path/to/video.mp4 --profile quality

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

# Test basic video processing without AI models (updated path)
python -c "from src.stages._01_audio_extraction.code.video_processing import VideoProcessor; print('Audio extraction available')"

# Test Whisper installation
python -c "import whisper; print(whisper.available_models())"

# Test stage-based architecture
python -c "from src.orchestrator.pipeline_orchestrator import PipelineOrchestrator; print('Pipeline orchestrator available')"
```

## Architecture Overview

This is a **multilingual educational content extraction pipeline** specifically optimized for Hebrew/English data science tutorials. The system processes long-form educational recordings to extract high-value segments suitable for short-form content creation.

## ⚠️ **CRITICAL KNOWN ISSUES**

Before working with this codebase, be aware of these production-blocking issues:

1. **Quality Profile Hangs**: The `--profile quality` option hangs indefinitely during LLM model loading. **Always use `--profile balanced` for production.**

2. **Uniform Quality Scores**: Current evaluation system gives all segments identical 0.75 scores, making it impossible to distinguish high-value content from low-value content.

3. **Limited Speaker Features**: Advanced speaker diarization requires Python 3.9+ (currently running 3.8).

**For current status and planned fixes, see `SPECS/PROJECT_STATUS.md`**

---

### Core Components (v2.0 Stage-based Architecture)

**Main Orchestrator**: `src/orchestrator/pipeline_orchestrator.py` - PipelineOrchestrator manages the 6-stage pipeline

**Processing Stages (6-Stage Architecture)**:
1. **Audio Extraction** (`src/stages/_01_audio_extraction/code/video_processing.py`) - FFmpeg-based video to audio
2. **Transcription** (`src/stages/_02_transcription/code/transcription.py`) - Whisper with Hebrew/English + technical terms
3. **Content Segmentation** (`src/stages/_03_content_segmentation/code/segmentation.py`) - Reels-optimized segments
4. **Speaker Segmentation** (`src/stages/_04_speaker_segmentation/code/`) - Advanced speaker analysis & classification
5. **Content Evaluation** (`src/stages/_05_content_evaluation/code/evaluation.py`) - LLM quality assessment
6. **Output Generation** (`src/stages/_06_output_generation/code/`) - Multi-format exports (JSON/CSV)

**Infrastructure**:
- **Stage Management**: `src/orchestrator/` - Pipeline coordination, configuration, performance monitoring
- **Shared Utilities**: `src/shared/` - Base classes, exceptions, models, common utilities  
- **Legacy Interface**: `src/content_extractor.py` - Backwards compatibility wrapper

### Key Technologies

- **AI Models**: OpenAI Whisper (transcription), Sentence-Transformers (embeddings), Qwen2.5/Phi-3 (evaluation)
- **Media Processing**: FFmpeg, MoviePy, librosa
- **Language Support**: spaCy (Hebrew), LangDetect, PyAnnote.Audio
- **Performance**: PyTorch with M1 Mac MPS optimization, batch processing
- **No External APIs**: Uses only local open-source models

### Configuration System

The system uses `ProcessingConfig` class with three optimized profiles:
- **draft**: 70% faster, minimal quality (good for testing) ✅ Working
- **balanced**: Default settings ✅ Working (Recommended)
- **quality**: 20% slower, maximum quality ⚠️ **CURRENTLY HANGS** - Do not use

**⚠️ CRITICAL ISSUE**: Quality profile hangs indefinitely during LLM model loading. Use balanced profile for production.

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
- **⚠️ KNOWN ISSUE**: Current evaluation gives all segments identical 0.75 scores (cannot distinguish quality)

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
1. **Follow stage-based architecture**: Each of the 6 stages in `src/stages/` is independent
2. **Use configuration objects**: ProcessingConfig for all parameters  
3. **Maintain Hebrew/English support**: Test with multilingual content
4. **Preserve error handling**: Comprehensive logging throughout
5. **Test thoroughly**: Core, integration, and performance tests
6. **M1 Mac compatibility**: Always test MPS acceleration paths
7. **Avoid quality profile**: Use balanced profile due to hanging issues
8. **Be aware of scoring limitations**: Current evaluation provides uniform scores