# System Specification - Content Extractor
Date: 2024

## System Purpose
An automated system for extracting high-quality content from long data science tutorial recordings and preparing it for short-form social media content.

## General Architecture
The system is built as a Python package containing several core modules that perform the processing pipeline.

### Core Dependencies
- `whisper`: Audio transcription with timestamps
- `sentence-transformers`: Embedding generation for semantic analysis
- `moviepy` or `ffmpeg-python`: Video processing and audio extraction
- `open-source-llm`: Free LLM for content evaluation (GPT-OSS/QWEN)
- `torch`: ML processing infrastructure
- `pandas`: Data management and export

## Current Workflow

### 1. Audio Extraction and Processing
- Extract audio from video files using FFmpeg
  - Supports: MP4, MOV, AVI, MKV
  - Optional: Keep extracted audio file (`--keep-audio`)
- Transcribe using OpenAI Whisper
  - Model options: tiny/base/small/medium/large
  - Normalized confidence scores (0-1 range)
  - Timestamped output with word-level alignment

### 2. Language Processing and Analysis
- Direct segment usage from Whisper
  - Configurable model size (tiny/base/small/medium/large)
  - Language-aware processing (Hebrew + English)
  - Technical term preservation
- Generate semantic embeddings
  - Using `all-MiniLM-L6-v2` by default
  - Batched processing (32 segments/batch)
  - Optional embedding export (`--include-embeddings`)

### 3. Content Evaluation
- Quality assessment using Qwen2.5-0.5B-Instruct
  - Runs locally, no API costs
  - GPU acceleration when available
  - Deterministic scoring with `.eval()` mode
  - Robust JSON parsing with fallbacks
- Evaluation criteria:
  - Educational value
  - Clarity and understandability
  - Practical demonstration
  - Short-form content potential

### 4. Export Options
- JSON output (configurable):
  - Full segment metadata
  - Timestamps and transcriptions
  - Quality scores and reasoning
  - Optional embeddings inclusion
- CSV export for analysis:
  - Start/end times
  - Text content
  - Confidence scores
  - Quality metrics
- Console summary:
  - Processing statistics
  - Top segments preview
  - Quality distribution

## Data Structure

### Segment (dataclass)
```python
@dataclass
class Segment:
    start_time: float
    end_time: float
    text: str
    confidence: float
    embedding: Optional[List[float]]
    value_score: Optional[float]
    reasoning: Optional[str]
```

## Configuration Parameters

### Core Processing
- `min_score_threshold`: Quality cutoff (default: 0.7)
- `whisper_model`: Model size (tiny/base/small/medium/large)
- `primary_language`: Primary content language (default: "he")
- `preserve_technical_terms`: Keep English technical terms (default: true)

### Performance Profiles
- **Draft Mode** (70% faster):
  ```python
  config = ProcessingConfig.create_profile("draft")
  # Uses: tiny whisper model, minimal processing
  ```
- **Balanced Mode** (default):
  ```python
  config = ProcessingConfig.create_profile("balanced")
  # Uses: base whisper model, standard features
  ```
- **Quality Mode** (20% slower):
  ```python
  config = ProcessingConfig.create_profile("quality")
  # Uses: medium whisper model, all features
  ```

### Performance Options
- `evaluation_batch_size`: LLM batch size (default: 5)
- `embedding_batch_size`: Embedding batch size (default: 32)
- `enable_similarity_analysis`: Enable similarity detection (default: false)
- `minimal_mode`: Skip non-essential processing (default: false)
- `enable_technical_terms`: Process technical terms (default: true)
- `keep_audio`: Retain extracted audio (default: false)
- `include_embeddings_in_json`: Export embeddings (default: false)

### Model Selection
- `embedding_model`: Sentence transformer model (default: all-MiniLM-L6-v2)
- `whisper_model`: Transcription model (tiny/base/small/medium/large)
- `evaluation_model`: LLM model for content evaluation:
  - Default: "Qwen/Qwen2.5-0.5B-Instruct" (fast)
  - Quality: "microsoft/Phi-3-mini-4k-instruct" (better but slower)
  - Alternative: "microsoft/DialoGPT-small" (balanced)

## Current Limitations & Future Work

### Current Focus (v1)
- Optimized processing profiles (draft/balanced/quality)
- Batch LLM evaluation (5-8 segments at once)
- Memory-efficient processing options
- Performance monitoring and metrics

### Future Enhancements (v2+)
- Advanced segmentation with overlapping
- Speaker diarization integration
- Complex language mixing analysis
- Full parallel processing capabilities

### Performance Roadmap
- **Phase 1** ✅ (Current):
  - LLM batch processing
  - Processing profiles
  - Optional features
  - Memory optimization
- **Phase 2** (Planned):
  - Full parallel processing
  - GPU memory optimization
  - Caching layer
  - Real-time feedback

### Future Features
- Web interface for easier usage
- Real-time processing feedback
- Integration with video editors
- Automated segment selection
- Multi-language support optimization

## Technical Notes
1. Requires Python 3.8+
2. GPU recommended for better performance
3. FFmpeg required for video processing (system dependency)
4. No paid API dependencies - uses free open-source models 