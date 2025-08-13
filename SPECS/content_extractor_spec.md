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

### 2. Segmentation and Analysis
- Create overlapping segments
  - Configurable duration (default: 45s, can be shorter)
  - Smart overlap (default: 10s)
  - Deduplication of repeated content
  - Progress tracking for long operations
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
- `segment_duration`: Length of each segment (default: 45s)
- `overlap_duration`: Overlap between segments (default: 10s)
- `min_score_threshold`: Quality cutoff (default: 0.7)
- `whisper_model`: Model size (tiny/base/small/medium/large)

### Performance Options
- `embedding_batch_size`: Batch size for embeddings (default: 32)
- `keep_audio`: Retain extracted audio (default: false)
- `include_embeddings_in_json`: Export embeddings (default: false)

### Model Selection
- `embedding_model`: Sentence transformer model (default: all-MiniLM-L6-v2)
- Evaluator: Fixed to Qwen2.5-0.5B-Instruct (local inference)

## Current Limitations & Future Work

### Performance Issues
- Segmentation process is slow for longer videos
- Memory usage can spike with large batch sizes
- No parallel processing for segment creation

### Planned Improvements
- Optimize segmentation algorithm
- Add batch processing for multiple videos
- Implement progress tracking for all steps
- Add parallel processing where possible

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