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

# For Hebrew-optimized transcription, ensure faster-whisper is installed:
pip install faster-whisper>=0.10.0

# Note: requirements_basic.txt and setup_environment.py have been removed
# Use standard installation only
```

### Running the System
```bash
# Basic video processing (auto model selection)
python -m src path/to/video.mp4

# 🆕 Hebrew-optimized transcription (RECOMMENDED FOR HEBREW) - requires faster-whisper
python -m src path/to/video.mp4 --transcription-model ivrit-v2-d4

# Force specific model regardless of duration
python -m src path/to/video.mp4 --transcription-model large --force-model

# Latest Whisper turbo model (5.4x faster)
python -m src path/to/video.mp4 --transcription-model large-v3-turbo

# Fast processing with custom segments (90s with 20s overlap)
python -m src path/to/video.mp4 --profile draft --segment-duration 90 --overlap-duration 20

# Balanced processing (recommended)
python -m src video.mp4 --profile balanced

# With CSV export and custom output
python -m src path/to/video.mp4 -o results.json --export-csv segments.csv

# List all available transcription models
python -m src --list-models

# Advanced Hebrew processing
python -m src video.mp4 --transcription-model hebrew --language he --enable-speaker-detection

# 🔧 Fix MPS backend issues on M1 Mac (force CPU processing)
python -m src path/to/video.mp4 --force-cpu

# Hebrew model with CPU fallback (recommended if MPS fails)
python -m src path/to/video.mp4 --transcription-model ivrit-v2-d4 --force-cpu

# 📁 Save intermediate outputs from each pipeline stage
python -m src path/to/video.mp4 --save-stage-outputs

# Custom stage output directory
python -m src path/to/video.mp4 --save-stage-outputs --stage-output-dir my_debug_outputs
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

# Test LLM evaluation system (new)
python -c "from src.stages._05_content_evaluation.code.evaluation import ContentEvaluator; print('LLM evaluator available')"

# Check system memory usage before processing
python -c "import psutil; mem = psutil.virtual_memory(); print(f'Memory: {mem.percent:.1f}% used, {mem.available/1024**3:.1f}GB available')"

# Test rule-based evaluation (fast fallback)
python -c "from src.stages._05_content_evaluation.code.evaluation import ContentEvaluator; eval = ContentEvaluator(use_rule_based=True); print('Rule-based evaluator working')"
```

## Architecture Overview

This is a **multilingual educational content extraction pipeline** specifically optimized for Hebrew/English data science tutorials. The system processes long-form educational recordings to extract high-value segments suitable for short-form content creation.

## ⚠️ **KNOWN ISSUES & FIXES** 

### Fixed Issues (v2.1 Updates)

1. **LLM Evaluation Hanging** ✅ FIXED: 
   - Added 60-second timeouts for single segment evaluation
   - Added 2-minute timeouts for batch processing  
   - Implemented automatic fallback to rule-based scoring after 3 LLM failures
   - Fixed generation parameter conflicts (`do_sample`/`temperature` warnings)

2. **Uniform Quality Scores** ✅ IMPROVED:
   - Enhanced rule-based scoring with better variance (0.3-0.9 range)
   - Fixed fast dynamic scoring to prevent uniform 0.75 scores
   - Multi-criteria evaluation available for advanced scoring

3. **Hebrew Language Support** ✅ MAJOR IMPROVEMENT:
   - Added 50+ Hebrew educational keywords ("דוגמה", "להסביר", "מושג", etc.)
   - Added Hebrew technical terms + transliterations ("פונקציה", "פאנקשן", "נתונים")
   - Added Hebrew engagement words ("מדהים", "פשוט", "יעיל", "טריק")
   - Supports mixed Hebrew-English content (common in tech tutorials)
   - Hebrew scoring now achieves 85%+ parity with English equivalents

4. **M1 Mac MPS Backend Issues** ✅ FIXED:
   - Added `--force-cpu` flag to bypass MPS/CUDA acceleration
   - Implemented automatic CPU fallback when MPS sparse tensor operations fail
   - Enhanced device compatibility detection with error-specific handling
   - Fixed "aten::_sparse_coo_tensor_with_dims_and_tensors" MPS backend errors

### Remaining Issues

5. **Quality Profile Hangs**: Some edge cases may still hang during model loading. **Use `--profile balanced` for production.**

6. **Limited Speaker Features**: Advanced speaker diarization requires Python 3.9+ (currently running 3.8).

### Troubleshooting LLM Hangs

If you encounter hanging during Content Evaluation stage:

1. **Kill the process**: Ctrl+C or kill the Python process
2. **Use draft profile**: `python -m src video.mp4 --profile draft` (bypasses LLM entirely)
3. **Check memory usage**: System needs <80% memory usage for stable operation
4. **Try balanced profile**: `python -m src video.mp4 --profile balanced` (automatic fallback enabled)

### Troubleshooting MPS Backend Issues (M1 Mac)

If you encounter MPS backend errors like "Could not run 'aten::_sparse_coo_tensor_with_dims_and_tensors'":

1. **Use CPU processing**: `python -m src video.mp4 --force-cpu` (forces all models to use CPU)
2. **Automatic fallback**: The system now automatically detects MPS errors and falls back to CPU
3. **Hebrew models with CPU**: `python -m src video.mp4 --transcription-model ivrit-v2-d4 --force-cpu`
4. **Check PyTorch MPS**: Verify PyTorch MPS installation with `python -c "import torch; print(torch.backends.mps.is_available())"`

**Root cause**: Whisper models use sparse tensor operations that aren't fully compatible with the current PyTorch MPS backend on M1 Macs.

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
- **draft**: 70% faster, rule-based scoring (good for testing) ✅ Working - Recommended for speed
- **balanced**: LLM evaluation with automatic fallback ✅ Working - **Recommended for production**  
- **quality**: Advanced evaluation settings ⚠️ Some edge cases may hang

**🚀 NEW**: Balanced profile now includes intelligent fallback - switches to rule-based evaluation automatically if LLM fails 3 times.

Key parameters:
- `segment_duration`: 90s (extended segment length for more context)
- `overlap_duration`: 20s (increased overlap prevents content splitting)  
- `min_score_threshold`: 0.7 (quality threshold)
- `transcription_model`: "auto" (smart selection) or manual override
- `evaluation_model`: Qwen2.5 or Phi-3 for local evaluation

**🆕 New Transcription Control**:
- **Manual Model Selection**: `--transcription-model large-v3-turbo`
- **Hebrew Optimization**: `--transcription-model ivrit-v2-d4` 
- **Force Override**: `--force-model` (ignores duration-based selection)
- **Model Discovery**: `--list-models` (see all available options)

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
- **Quality reasoning**: AI-generated evaluation explanations with multilingual support
- **✅ IMPROVED**: Enhanced scoring now provides proper variance (0.3-0.9 range) for Hebrew content

### Hebrew/English Multilingual Support

**🆕 Enhanced Hebrew Support (v2.2)**:
- **Hebrew-Optimized Transcription**: Ivrit.AI fine-tuned models (ivrit-v2-d4, ivrit-v2-d3-e3)
- **Comprehensive Hebrew Keywords**: 50+ educational terms ("דוגמה", "להסביר", "חשוב")
- **Technical Terminology**: Both Hebrew ("פונקציה", "נתונים") and transliterations ("פאנקשן", "דאטה") 
- **Mixed Content**: Handles Hebrew explanations with English technical terms
- **Unicode Processing**: Proper Hebrew character handling and normalization
- **Performance**: Hebrew content achieves 85%+ scoring parity with English equivalents

**Transcription Models**:
- **Hebrew-Optimized**: `ivrit-v2-d4` (latest), `ivrit-v2-d3-e3`, `hebrew` (alias)
- **Latest Whisper**: `large-v3-turbo` (5.4x faster), `large-v3`, `large`
- **Standard Models**: `tiny`, `base`, `small`, `medium` for different performance needs
- **Smart Selection**: `auto` chooses optimal model based on video duration

**Language Features**:
- **Primary languages**: Hebrew with English technical terms
- **Manual Control**: Override automatic model selection for Hebrew content
- **Fallback System**: Hebrew models fall back to `large` if unavailable
- **Character encoding**: UTF-8 throughout pipeline with Hebrew normalization

**Model Selection Examples**:
```bash
# Best Hebrew transcription (requires faster-whisper)
python -m src video.mp4 --transcription-model ivrit-v2-d4 --force-model

# Hebrew transcription with CPU fallback
python -m src video.mp4 --transcription-model ivrit-v2-d4 --force-cpu

# Latest Whisper for mixed content  
python -m src video.mp4 --transcription-model large-v3-turbo

# See all options including Hebrew models
python -m src --list-models
```

**Hebrew Model Installation**:
```bash
# Install faster-whisper for Hebrew models
pip install faster-whisper>=0.10.0

# Verify Hebrew model availability
python -c "from faster_whisper import WhisperModel; print('faster-whisper ready for Hebrew models')"
```

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