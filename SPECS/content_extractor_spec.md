# System Specification - Content Extractor
Date: August 2025 (Updated for v2.0 Stage-based Architecture)

## System Purpose
An automated system for extracting high-quality content from long data science tutorial recordings and preparing it for short-form social media content.

## General Architecture
The system is built as a **stage-based modular architecture** with 6 distinct processing stages, orchestrated through a central pipeline manager. Each stage is independent, testable, and can be configured separately.

### Stage-Based Pipeline Architecture
```
src/
├── stages/                    # 6 processing stages
│   ├── _01_audio_extraction/  # Video → Audio conversion
│   ├── _02_transcription/     # Audio → Text with timestamps  
│   ├── _03_content_segmentation/ # Text → Segments
│   ├── _04_speaker_segmentation/ # Advanced speaker analysis
│   ├── _05_content_evaluation/   # Quality scoring
│   └── _06_output_generation/    # Results export
├── orchestrator/             # Pipeline coordination
├── shared/                   # Common utilities
└── main.py                  # Entry point
```

### Core Dependencies (v2.0)
- **Audio Processing**: `openai-whisper>=20231117` (official OpenAI Whisper)
- **Language Models**: `transformers>=4.30.0` with Qwen2.5-0.5B-Instruct, Phi-3-mini
- **Embeddings**: `sentence-transformers>=2.2.2` (all-MiniLM-L6-v2)
- **Video Processing**: `ffmpeg-python` + system FFmpeg installation
- **ML Infrastructure**: `torch>=2.0.0` with M1 Mac MPS support
- **CLI & Progress**: `click`, `rich` for enhanced user experience
- **Data Export**: `pandas` for CSV/JSON output management

## Current Workflow (Stage-based v2.0)

### Stage 1: Audio Extraction (`_01_audio_extraction/`)
**Purpose**: Convert video files to audio format for processing
- **Input**: Video files (MP4, MOV, AVI, MKV)
- **Process**: FFmpeg-based extraction to WAV format
- **Output**: High-quality audio file + metadata
- **Key Features**: Batch processing, format validation, quality preservation

### Stage 2: Transcription (`_02_transcription/`)  
**Purpose**: Convert audio to timestamped text with multilingual support
- **Input**: Audio file from Stage 1
- **Process**: OpenAI Whisper transcription with Hebrew/English support
- **Output**: Timestamped segments with confidence scores
- **Key Features**: 
  - M1 Mac GPU acceleration (MPS)
  - Technical term preservation (74 data science terms)
  - Confidence score normalization (0-1 range)

### Stage 3: Content Segmentation (`_03_content_segmentation/`)
**Purpose**: Create content segments optimized for short-form content
- **Input**: Transcribed segments from Stage 2
- **Process**: Smart segmentation with overlap management
- **Output**: 15-45 second segments with natural boundaries
- **Key Features**: Deduplication, topic boundary detection, length optimization

### Stage 4: Speaker Segmentation (`_04_speaker_segmentation/`) 
**Purpose**: Advanced speaker analysis and teacher/student identification
- **Input**: Audio file + transcription segments
- **Process**: Hybrid frequency analysis + speaker classification
- **Output**: Speaker labels, teacher/student roles, confidence scores
- **Key Features**: Frequency analysis, temporal smoothing, role classification

### Stage 5: Content Evaluation (`_05_content_evaluation/`)
**Purpose**: Quality scoring for educational value assessment  
- **Input**: Segmented content with speaker information
- **Process**: Local LLM evaluation using Qwen2.5-0.5B-Instruct
- **Output**: Quality scores (0-1) with detailed reasoning
- **Key Features**: 
  - Batch processing (5-8 segments simultaneously)
  - Educational value criteria
  - Technical content weighting
  - **⚠️ Current Issue**: Uniform scoring problem

### Stage 6: Output Generation (`_06_output_generation/`)
**Purpose**: Export results in multiple formats
- **Input**: Evaluated segments with quality scores
- **Process**: Format conversion and summary generation
- **Output**: JSON/CSV files + console summary
- **Key Features**: Configurable exports, embeddings inclusion, filtering

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

## Configuration System (v2.0)

### Processing Profiles (Pre-configured Optimizations)
```bash
# Draft Profile (70% faster, basic quality)
python -m src video.mp4 --profile draft

# Balanced Profile (default, good quality/speed balance)  
python -m src video.mp4 --profile balanced

# Quality Profile (⚠️ CURRENTLY HANGS - being fixed)
python -m src video.mp4 --profile quality
```

### Core Processing Parameters
- `min_score_threshold`: Quality cutoff (0.7 default, 0.6 draft, 0.8 quality)
- `whisper_model`: Model size (tiny/base/small/medium/large)
- `primary_language`: "he" (Hebrew) with English technical terms
- `evaluation_model`: Qwen2.5-0.5B-Instruct (draft/balanced), Phi-3-mini (quality)
- `evaluation_batch_size`: 5-8 segments processed simultaneously

### Multilingual Support
- `preserve_technical_terms`: Maintains 74 data science terms in English
- `enable_technical_terms`: Toggle technical term processing (performance optimization)
- `primary_language`: Hebrew base with automatic English detection
- `technical_language`: English for data science terminology

### Advanced Options
- `enable_similarity_analysis`: Cross-segment similarity detection (default: false)
- `minimal_mode`: Skip non-essential features for maximum speed (default: false)
- `keep_audio`: Retain extracted audio files (default: false)  
- `include_embeddings_in_json`: Export embeddings in output (default: false)
- `enable_speaker_detection`: Advanced speaker analysis (default: false)
- `primary_speaker_only`: Filter to primary speaker segments (requires Python 3.9+)

### Model Configuration (v2.0)
- **Transcription**: OpenAI Whisper (tiny→large models available)
- **Embeddings**: `all-MiniLM-L6-v2` (384-dimensional vectors)
- **Evaluation**: Qwen2.5-0.5B-Instruct (fast), Phi-3-mini-4k-instruct (quality)
- **Language**: Hebrew spaCy model + 74 English technical terms
- **GPU**: M1 Mac MPS acceleration, CUDA support on compatible hardware

## Current Limitations & Known Issues (v2.0)

### 🚨 **Critical Issues Requiring Fix**
1. **Segment Quality Uniformity**: All segments receive identical 0.75 scores
   - **Impact**: Cannot distinguish high-value content
   - **Status**: Solution designed, needs implementation

2. **Quality Profile Performance Hang**: LLM model loading causes indefinite hangs
   - **Impact**: Quality profile unusable in production
   - **Status**: Timeout and fallback mechanisms needed

3. **Speaker Diarization Limitations**: Requires Python 3.9+ (currently 3.8)
   - **Impact**: Limited advanced speaker features
   - **Status**: Infrastructure ready, blocked by Python version

### ⚠️ **Minor Limitations**  
- Processing requires local GPU for optimal performance
- Hebrew language requires system-level language support
- Large videos (>30 minutes) may require memory optimization
- Quality profile currently unstable (use balanced mode)

### 🚀 **Development Roadmap**

**v2.1 (September 2025) - Stability Release:**
- Fix segment quality variance issue
- Resolve quality profile performance hangs  
- Complete project file cleanup
- Expand test coverage to >80%

**v2.2 (October 2025) - Performance Optimization:**
- Python 3.9+ migration for full speaker diarization
- GPU memory optimization improvements
- Advanced segmentation with intelligent overlaps
- Performance monitoring dashboard

**v3.0 (Q4 2025) - Advanced Features:**
- Web interface for easier usage
- Real-time processing feedback with progress bars
- Integration APIs for video editors
- Automated content curation workflows
- Multi-speaker educational content optimization

## Technical Notes (v2.0)
1. **Python Requirements**: 3.8+ (3.9+ recommended for full speaker features)
2. **GPU Support**: M1 Mac MPS acceleration, NVIDIA CUDA compatible
3. **System Dependencies**: FFmpeg required for video processing
4. **Memory Requirements**: 4-8GB RAM depending on processing profile  
5. **Storage**: ~2GB for models (Whisper, transformers, embeddings)
6. **No API Costs**: Uses only local open-source models (Qwen2.5, Phi-3, Whisper)

## Quick Start (v2.0)
```bash
# Setup
python -m venv reels_extractor_env && source reels_extractor_env/bin/activate
pip install -r requirements.txt

# Basic Usage (Recommended)
python -m src path/to/video.mp4 --profile balanced

# Fast Processing (Testing)  
python -m src path/to/video.mp4 --profile draft --export-csv results.csv

# ⚠️ Avoid Quality Profile (Currently Hangs)
# python -m src path/to/video.mp4 --profile quality
```