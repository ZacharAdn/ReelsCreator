# Hebrew Video Transcription Tool 🎬

A simple, powerful Python script for transcribing Hebrew and English educational videos with automatic chunking and real-time output.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

This tool transcribes long videos (educational content, lectures, tutorials) into timestamped text with:

- **🇮🇱 Hebrew-Optimized Transcription** - Uses Ivrit.AI models + Whisper large-v3-turbo
- **⚡ Chunk Processing** - Processes videos in 2-minute chunks with progress tracking
- **💾 Real-time Output** - Saves transcripts as each chunk completes
- **📁 Organized Results** - Each run creates a timestamped directory with all outputs
- **🔄 Automatic Fallback** - Hebrew model → Whisper turbo → Whisper large

Perfect for transcribing data science tutorials, educational content, and mixed Hebrew-English recordings.

## ✨ Features

- **Multi-chunk processing** - Handle videos of any length (tested up to 20+ minutes)
- **Progress tracking** - See exactly which chunk is being processed
- **Real-time saves** - Access partial results even if processing is interrupted
- **Timestamped outputs** - Each run creates a unique directory: `results/YYYY-MM-DD_HHMMSS/`
- **Multiple output formats** - Individual chunk files, cumulative transcript, metadata
- **M1 Mac compatible** - Automatic CPU fallback for MPS backend issues

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for video/audio processing)
- ~5GB free disk space for models

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Reels_extractor

# Create virtual environment
python -m venv reels_extractor_env

# Activate virtual environment
source reels_extractor_env/bin/activate  # macOS/Linux
# or: reels_extractor_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (if not already installed)
# macOS:
brew install ffmpeg
# Ubuntu/Debian:
sudo apt install ffmpeg
```

### 2. Basic Usage

```bash
# Activate virtual environment
source reels_extractor_env/bin/activate

# Run the transcription script
python "src/quick scripts/transcribe_advanced.py"

# Or use the helper script
./run_transcription.sh
```

**Note:** The script looks for videos in the `data/` directory by default. Edit the video file list at the bottom of `transcribe_advanced.py` to specify which videos to process.

### 3. Output Structure

Each transcription run creates a timestamped directory in `results/`:

```
results/2025-10-05_145645/
├── chunk_01.txt              # First 2-minute chunk transcript
├── chunk_01_metadata.txt     # Processing stats for chunk 1
├── chunk_02.txt              # Second chunk
├── chunk_02_metadata.txt     # Processing stats for chunk 2
├── full_transcript.txt       # Cumulative transcript (updated after each chunk)
└── IMG_4225_final_summary.txt # Complete results with all segments
```

## ⚙️ Configuration

### Chunk Size

You can adjust the chunk size by editing `CHUNK_SIZE_MINUTES` at the top of the script:

```python
# In transcribe_advanced.py
CHUNK_SIZE_MINUTES = 2  # Change to 1, 3, 5, etc.
```

- **Smaller chunks (1-2 min)** = More frequent progress updates, more output files
- **Larger chunks (5+ min)** = Fewer updates, potentially faster overall processing

### Video Selection

Edit the `video_files` list at the bottom of the script:

```python
video_files = [
    "data/IMG_4225.MP4",  # Your video
    "data/lecture.MOV",   # Add more videos here
]
```

## 📊 Output Files Explained

### Individual Chunk Files (`chunk_01.txt`, etc.)
Contains the transcript for that specific 2-minute segment with:
- Time range
- Processing time
- Full text
- Timestamped segments

### Cumulative Transcript (`full_transcript.txt`)
Updated after each chunk completes. If processing is interrupted, you'll have all transcribed content up to that point.

### Metadata Files (`chunk_01_metadata.txt`, etc.)
Processing statistics:
- Chunk number
- Start time and duration
- Processing time
- Language detected
- Number of segments

### Final Summary (`{video_name}_final_summary.txt`)
Complete results with:
- Full transcript
- All timestamped segments
- Processing information
- Model used

## 🧪 Supported Models

The script automatically tries these models in order:

1. **Hebrew-Optimized (Hugging Face)**
   - `imvladikon/wav2vec2-large-xlsr-53-hebrew`
   - Best for pure Hebrew content

2. **Whisper large-v3-turbo**
   - 5.4x faster than large
   - Great for mixed Hebrew-English

3. **Whisper large (fallback)**
   - Most reliable, slower
   - Works for all content

## 🔧 Troubleshooting

### FFmpeg Not Found
```bash
# Check FFmpeg installation
ffmpeg -version

# Install if missing (macOS)
brew install ffmpeg
```

### MPS Backend Errors (M1 Mac)
The script automatically falls back to CPU if MPS fails. No action needed!

### Memory Issues
Reduce chunk size to 1 minute if your system runs out of memory:
```python
CHUNK_SIZE_MINUTES = 1
```

### Model Download Failures
Ensure you have internet connection and ~5GB free space. The models will download automatically on first run.

## 📁 Project Structure

```
Reels_extractor/
├── src/
│   └── quick scripts/
│       └── transcribe_advanced.py  # Main transcription script
├── data/                           # Place your videos here
├── results/                        # Timestamped output directories
│   ├── 2025-10-05_145645/
│   └── 2025-10-05_183042/
├── reels_extractor_env/           # Virtual environment
├── run_transcription.sh           # Helper script
├── requirements.txt               # Dependencies
├── README.md                      # This file
└── CLAUDE.md                      # Developer instructions
```

## 🎯 Use Cases

- **Educational Content**: Transcribe data science lectures and tutorials
- **Meeting Notes**: Convert recorded meetings to searchable text
- **Podcast Transcription**: Create text versions of audio content
- **Content Creation**: Extract quotes and segments for social media
- **Accessibility**: Generate captions and subtitles

## 📈 Performance

- **Short video (3-4 min)**: ~6 minutes processing time
- **Medium video (20 min)**: ~30-40 minutes processing time
- **Processing speed**: ~3x real-time (varies by model and hardware)

## 🤝 Contributing

This is a simple, focused tool. Contributions welcome for:
- Bug fixes
- Performance improvements
- Additional language support
- Better error handling

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Transcription models
- [Ivrit.AI](https://huggingface.co/imvladikon) - Hebrew-optimized models
- [MoviePy](https://zulko.github.io/moviepy/) - Video processing
- [FFmpeg](https://ffmpeg.org/) - Media handling

---

**Made for Hebrew content creators** ❤️
