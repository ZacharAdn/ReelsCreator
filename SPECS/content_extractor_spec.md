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

## Workflow

### 1. Audio Extraction and Transcription
- Extract audio from video files (MP4, MOV, AVI, etc.)
- Use Whisper for transcription with timestamps
- Split into 45 second segments
- Create 10 second overlaps between segments
- Store metadata for each segment (start time, end time, text, confidence)

### 2. Semantic Analysis
- Generate embeddings for each segment using Sentence-Transformers
  - Recommended model: `all-MiniLM-L6-v2`
  - Potential upgrade to OpenAI embeddings later
- Calculate similarity between segments to identify related topics and sections

### 3. Content Evaluation
- Use free open-source LLMs (GPT-OSS/QWEN) for content quality assessment
- Evaluation parameters:
  - Is there an insight or practical demonstration?
  - Is the content clear and understandable?
  - Is there significant educational value?
- Score for each segment (0-1) with verbal explanation
- No dependency on paid OpenAI API

### 4. Results Export
- Export to JSON including:
  - Segment metadata
  - Quality scores and evaluations
  - Precise timestamps
  - Transcribed text
  - Embeddings (optional)

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

## Tuning Parameters
- Segment length (default: 45 seconds)
- Overlap length (default: 10 seconds)
- Minimum score threshold for segment retention (default: 0.7)
- Whisper model size (tiny/base/small/medium/large)

## Phase 2 (Future)
- Simple user interface
- Additional format support
- Parameter optimization
- Integration with editing tools

## Technical Notes
1. Requires Python 3.8+
2. GPU recommended for better performance
3. FFmpeg required for video processing (system dependency)
4. No paid API dependencies - uses free open-source models 