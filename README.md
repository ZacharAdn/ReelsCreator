# Reels Extractor 🎬

A powerful Python-based system for automatically extracting high-value segments from long educational recordings and preparing them for short-form content creation (TikTok, Reels, Shorts, YouTube Shorts).

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active Beta](https://img.shields.io/badge/Status-Active%20Beta-orange.svg)](#)

## 🎯 Overview

This intelligent content extraction system helps creators efficiently process long educational recordings by:

1. **🎤 Multilingual Transcription** - Uses OpenAI Whisper with Hebrew/English support and M1 Mac optimization
2. **✂️ Direct Segmentation** - Uses Whisper segments directly for optimal performance
3. **🧠 Semantic Analysis** - Uses sentence-transformers for content similarity and clustering
4. **⭐ Local AI Evaluation** - Qwen2.5 LLM powered quality assessment (no API costs)
5. **🌍 Technical Term Preservation** - Maintains Hebrew + English data science terminology
6. **📊 Multiple Output Formats** - JSON, CSV, and detailed summaries
7. **⚡ Batch Processing** - Process multiple files efficiently with GPU acceleration

Perfect for data science tutorials, educational content, podcasts, and any long-form video that needs to be converted into engaging short-form content.

## ✨ Features

### Core Capabilities
- **🎥 Multi-format Support**: MP4, MOV, AVI, MKV, and more
- **🎯 Direct Segmentation**: Uses Whisper segments for natural speech boundaries
- **🌍 Multilingual Processing**: Hebrew primary language with English technical terms
- **👥 Speaker Diarization**: Automatically identifies teacher vs student speech
- **🔍 Semantic Analysis**: Find related content and cluster similar segments
- **📈 Local AI Evaluation**: Qwen2.5 LLM for content quality assessment
- **🚀 M1 Mac Optimization**: GPU acceleration with MPS (Metal Performance Shaders)
- **📁 Batch Processing**: Handle multiple files efficiently
- **🔄 Configurable Parameters**: Customize models, thresholds, and processing options

### Output Formats
- **JSON**: Complete results with metadata and embeddings
- **CSV**: Spreadsheet-friendly format for analysis
- **Console Summary**: Key statistics and top segments
- **Detailed Reports**: Comprehensive analysis and recommendations

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher (required for speaker diarization)
- FFmpeg (for video processing)
- GPU recommended for better performance
- No paid API dependencies - uses free open-source models
- M1 Mac optimized with MPS support

### 1. Clone and Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/tiktok_agent.git
cd tiktok_agent

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install FFmpeg
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS with Homebrew
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
# Add to PATH environment variable
```

### 3. Run the Extractor

#### Command Line Interface
```bash
# Basic usage
python -m src path/to/your/video.mp4

# Fast processing for testing (70% faster)
python -m src path/to/your/video.mp4 --profile draft

# High quality processing (20% slower)
python -m src path/to/your/video.mp4 --profile quality

# With custom output and CSV export
python -m src path/to/your/video.mp4 -o results.json --export-csv segments.csv

# With performance optimizations
python -m src path/to/your/video.mp4 --evaluation-batch-size 8 --minimal-mode

# With custom evaluation model
python -m src path/to/your/video.mp4 --evaluation-model "microsoft/Phi-3-mini-4k-instruct" --min-score 0.7

# With similarity analysis enabled
python -m src path/to/your/video.mp4 --enable-similarity

# With speaker detection (teacher vs student)
python -m src path/to/your/video.mp4 --enable-speaker-detection --primary-speaker-only
```

#### Python API
```python
from src.content_extractor import ContentExtractor
from src.models import ProcessingConfig

# Option 1: Use optimized profiles
config = ProcessingConfig.create_profile("draft")     # 70% faster
# config = ProcessingConfig.create_profile("balanced") # Default
# config = ProcessingConfig.create_profile("quality")  # 20% slower

# Option 2: Custom configuration with performance options
# config = ProcessingConfig(
#     segment_duration=45,         # Segment length in seconds
#     overlap_duration=10,         # Overlap between segments
#     min_score_threshold=0.7,     # Minimum quality score
#     whisper_model="base",        # Whisper model size
#     evaluation_batch_size=5,     # Batch LLM processing for speed
#     enable_similarity_analysis=True,  # Enable similarity detection
#     minimal_mode=False           # Skip non-essential processing
# )

# Initialize extractor
extractor = ContentExtractor(config)

# Process video file
result = extractor.process_video_file("path/to/video.mp4", "results.json")

# View results
print(f"Found {len(result.high_value_segments)} high-value segments")
print(f"Average quality score: {result.summary.average_score:.2f}")

# Access individual segments
for segment in result.high_value_segments:
    print(f"Segment {segment.start_time:.1f}s - {segment.end_time:.1f}s")
    print(f"Score: {segment.value_score:.2f}")
    print(f"Text: {segment.text[:100]}...")
    print("---")
```

## ⚙️ Configuration

### Processing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `segment_duration` | 45 | Length of each segment in seconds |
| `overlap_duration` | 10 | Overlap between segments in seconds |
| `min_score_threshold` | 0.7 | Minimum quality score (0-1) |
| `whisper_model` | "base" | Whisper model size (tiny/base/small/medium/large) |
| `embedding_model` | "all-MiniLM-L6-v2" | Sentence transformer model |
| `evaluation_model` | "Qwen/Qwen2.5-0.5B-Instruct" | LLM model for content evaluation |

### Performance Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `processing_profile` | "balanced" | Profile: draft (70% faster), balanced, quality (20% slower) |
| `evaluation_batch_size` | 5 | Number of segments to evaluate in parallel |
| `enable_similarity_analysis` | False | Enable similarity detection and clustering |
| `minimal_mode` | False | Skip non-essential processing for maximum speed |
| `enable_technical_terms` | True | Process technical terminology |
| `enable_speaker_detection` | False | Enable teacher vs student speech detection |
| `primary_speaker_only` | False | Keep only teacher's speech segments |

### Configuration File
Create a `config.json` file for persistent settings:

```json
{
    "segment_duration": 45,
    "overlap_duration": 10,
    "min_score_threshold": 0.7,
    "whisper_model": "base",
    "embedding_model": "all-MiniLM-L6-v2",
    "batch_size": 4,
    "gpu_acceleration": true
}
```

## 📊 Output Format

### High-Value Segments Include:
- **⏰ Timestamps**: Precise start/end times
- **📝 Transcribed Text**: Clean, readable text
- **⭐ Quality Score**: 0-1 rating with detailed reasoning
- **🎯 Confidence**: Transcription confidence level
- **🧠 Embeddings**: For similarity analysis and clustering

### Example Output Structure:
```json
{
    "metadata": {
        "input_file": "tutorial.mp4",
        "processing_time": "2024-01-15T10:30:00Z",
        "total_duration": 3600.5,
        "whisper_model": "base"
    },
    "segments": [
        {
            "start_time": 120.5,
            "end_time": 165.2,
            "text": "Here's a practical example of using pandas for data analysis...",
            "confidence": 0.95,
            "embedding": [0.1, 0.2, ...],
            "value_score": 0.85,
            "reasoning": "Clear explanation with practical demonstration and code examples"
        }
    ],
    "high_value_segments": [...],
    "summary": {
        "total_segments": 24,
        "high_value_count": 8,
        "average_score": 0.72,
        "processing_duration": 125.3
    }
}
```

## 🧪 Testing

Run the comprehensive test suite:

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

## 📁 Project Structure

```
tiktok_agent/
├── src/                    # Main source code
│   ├── __main__.py        # CLI entry point
│   ├── content_extractor.py  # Main orchestrator
│   ├── transcription.py   # Whisper transcription
│   ├── segmentation.py    # Video segmentation
│   ├── embeddings.py      # Semantic embeddings
│   ├── evaluation.py      # AI content evaluation
│   ├── video_processing.py # Video/audio processing
│   └── models.py          # Data models
├── tests/                 # Test suite
├── data/                  # Sample videos and data
├── SPECS/                 # Technical specifications
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Advanced Usage

### Batch Processing
```python
import os
from src.content_extractor import ContentExtractor

extractor = ContentExtractor()

# Process all videos in a directory
video_dir = "path/to/videos/"
for video_file in os.listdir(video_dir):
    if video_file.endswith(('.mp4', '.mov', '.avi')):
        result = extractor.process_video_file(
            os.path.join(video_dir, video_file),
            f"results_{video_file}.json"
        )
```

### Custom Evaluation Criteria
```python
from src.evaluation import ContentEvaluator

# Custom evaluation prompt
custom_prompt = """
Evaluate this educational content segment for:
1. Practical value (0-1)
2. Clarity of explanation (0-1)
3. Engagement potential (0-1)
4. Actionability (0-1)
"""

evaluator = ContentEvaluator(custom_prompt=custom_prompt)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone and setup development environment
git clone https://github.com/yourusername/tiktok_agent.git
cd tiktok_agent
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .  # Install in development mode

# Run tests
pytest tests/
```

## 📈 Performance Tips

1. **GPU Acceleration**: Use CUDA-enabled PyTorch for faster processing
2. **Model Selection**: Choose appropriate Whisper model size for your needs
3. **Batch Processing**: Process multiple files together for efficiency
4. **Memory Management**: Adjust batch sizes based on available RAM

## 🐛 Troubleshooting

### Common Issues

**FFmpeg not found:**
```bash
# Ensure FFmpeg is installed and in PATH
ffmpeg -version
```

**CUDA out of memory:**
```python
# Reduce batch size or use CPU
config = ProcessingConfig(batch_size=1, gpu_acceleration=False)
```

**Whisper model download issues:**
```python
# Use smaller model or check internet connection
config = ProcessingConfig(whisper_model="tiny")
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for transcription
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [MoviePy](https://zulko.github.io/moviepy/) for video processing
- [FFmpeg](https://ffmpeg.org/) for media handling

## 📞 Support

- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/tiktok_agent/issues)
- 📖 Documentation: [Wiki](https://github.com/yourusername/tiktok_agent/wiki)

---

**Made with ❤️ for content creators** 