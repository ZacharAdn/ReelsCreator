# Hebrew Video Transcription & Reel Creator 🎬

A simple, powerful toolkit for transcribing Hebrew and English educational videos with automatic chunking, real-time output, and video segment extraction for creating Reels/Shorts.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

This toolkit provides two powerful tools:

### 1. **Video Transcription** (`transcribe_advanced.py`)
Transcribes long videos (educational content, lectures, tutorials) into timestamped text with:

- **🇮🇱 Hebrew-Optimized Transcription** - Uses Ivrit.AI models + Whisper large-v3-turbo
- **⚡ Chunk Processing** - Processes videos in 2-minute chunks with progress tracking
- **💾 Real-time Output** - Saves transcripts as each chunk completes
- **📁 Organized Results** - Each run creates a timestamped directory with all outputs
- **🔄 Automatic Fallback** - Hebrew model → Whisper turbo → Whisper large

### 2. **Video Segment Cutter** (`cut_video_segments.py`)
Extracts and concatenates specific time ranges to create Reels/Shorts:

- **✂️ Precise Cutting** - Extract exact time ranges from videos
- **🔗 Auto-Concatenation** - Joins all segments into one video
- **⚡ Two Methods** - MoviePy (simple) or FFmpeg (faster)
- **🎯 Smart Naming** - Outputs to `generated_data/VideoName_REEL.MP4`
- **📝 Flexible Input** - Interactive mode or command-line arguments

Perfect for transcribing data science tutorials, educational content, and creating engaging short-form content from long videos.

## ✨ Features

### Transcription Features
- **Interactive video selection** - Browse all videos in your project with metadata
- **Multi-chunk processing** - Handle videos of any length (tested up to 20+ minutes)
- **Progress tracking** - See exactly which chunk is being processed
- **Real-time saves** - Access partial results even if processing is interrupted
- **Timestamped outputs** - Each run creates a unique directory: `results/VideoName_YYYY-MM-DD_HHMMSS/`
- **Multiple output formats** - Individual chunk files, cumulative transcript, metadata
- **M1 Mac compatible** - Automatic CPU fallback for MPS backend issues

### Video Cutting Features
- **Interactive range input** - Add time ranges one by one with validation
- **Smart file naming** - Auto-increments (`_REEL_2.MP4`, `_REEL_3.MP4`) to avoid overwrites
- **Dual processing modes** - MoviePy (easy) or FFmpeg (fast)
- **Format flexibility** - Supports MM:SS.MS, M:SS.MS, SS.MS, MM:SS formats
- **Visual metadata** - See duration, size, and date before selecting

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

#### Transcription (Interactive Mode)

```bash
# Activate virtual environment
source reels_extractor_env/bin/activate

# Run the transcription script
python "src/quick scripts/transcribe_advanced.py"
```

**Interactive Flow:**
1. 🔍 Script scans project for all directories with videos
2. 📁 Select directory (shows video count for each)
3. 📹 View all videos with duration, size, and date
4. ✅ Select video to transcribe
5. ⚡ Processing begins with real-time progress updates

**Example:**
```
📁 Directories with videos:
================================================================================

1. ./data
   (3 videos)

2. ./archive/lectures
   (5 videos)

Select directory number: 1

📹 Available Videos
================================================================================

1. IMG_4225.MP4
   Duration: 3:45.26 | Size: 125.3MB | Date: 2025-10-05

2. lecture.MOV
   Duration: 12:30.00 | Size: 450.2MB | Date: 2025-10-04

Select video number: 1
```

#### Video Segment Cutting (Interactive Mode)

```bash
# Activate virtual environment
source reels_extractor_env/bin/activate

# Run in interactive mode (recommended)
python "src/quick scripts/cut_video_segments.py"
```

**Interactive Flow:**
1. 📹 Script scans `data/` directory for videos
2. 📊 View all videos with duration, size, and date
3. ✅ Select video to cut
4. ✂️ Enter time ranges one by one (press Enter when done)
5. ⚡ Choose MoviePy (default) or FFmpeg (faster)
6. 🎬 Video segments are extracted and concatenated
7. 💾 Output saved with unique name (auto-increments if file exists)

**Example:**
```
📹 Available Videos
================================================================================

1. IMG_4225.MP4
   Duration: 3:45.26 | Size: 125.3MB | Date: 2025-10-05

Select video number: 1

Range #1 (or press Enter to finish):
> 1:00.26-1:07.16
   ✓ Added: 1:00.26 - 1:07.16 (6.9s)

Range #2 (or press Enter to finish):
> 1:27.64-1:31.72
   ✓ Added: 1:27.64 - 1:31.72 (4.1s)

Range #3 (or press Enter to finish):
> [Enter]

📊 Total: 2 ranges | Total duration: 0:11.00
```

#### Command-Line Mode (For Automation)

```bash
# Video cutting with command-line arguments
python "src/quick scripts/cut_video_segments.py" \
  --video data/IMG_4225.MP4 \
  --ranges "1:00.26-1:07.16, 1:27.64-1:31.72, 1:42.30-1:49.04, 2:00.08-2:06.68"

# Use FFmpeg for faster processing (no re-encoding)
python "src/quick scripts/cut_video_segments.py" \
  --video data/IMG_4225.MP4 \
  --ranges "1:00-2:00, 3:00-4:00" \
  --use-ffmpeg

# Custom output location
python "src/quick scripts/cut_video_segments.py" \
  --video data/IMG_4225.MP4 \
  --ranges "1:00-2:00" \
  --output my_custom_reel.mp4
```

### 3. Output Structure

Each transcription run creates a timestamped directory in `results/`:

```
results/IMG_4225_2025-10-05_145645/
├── chunk_01.txt              # First 2-minute chunk transcript
├── chunk_01_metadata.txt     # Processing stats for chunk 1
├── chunk_02.txt              # Second chunk
├── chunk_02_metadata.txt     # Processing stats for chunk 2
├── full_transcript.txt       # Cumulative transcript (updated after each chunk)
└── IMG_4225_final_summary.txt # Complete results with all segments
```

## ⚙️ Configuration

### Transcription - Chunk Size

You can adjust the chunk size by editing `CHUNK_SIZE_MINUTES` at the top of `transcribe_advanced.py`:

```python
# In transcribe_advanced.py
CHUNK_SIZE_MINUTES = 2  # Change to 1, 3, 5, etc.
```

- **Smaller chunks (1-2 min)** = More frequent progress updates, more output files
- **Larger chunks (5+ min)** = Fewer updates, potentially faster overall processing

### Video Cutting - Time Range Format

Time ranges support multiple formats:
- **MM:SS.MS** - `1:23.45` (1 minute, 23.45 seconds)
- **M:SS.MS** - `1:00.26` (1 minute, 0.26 seconds)
- **SS.MS** - `45.50` (45.5 seconds)
- **MM:SS** - `1:23` (1 minute, 23 seconds)

Multiple ranges separated by commas:
```
"1:00.26-1:07.16, 1:27.64-1:31.72, 1:42.30-1:49.04, 2:00.08-2:06.68"
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
│       ├── transcribe_advanced.py  # Transcription script
│       └── cut_video_segments.py   # Video segment cutter
├── data/                           # Place your videos here
├── results/                        # Transcription output directories
│   ├── IMG_4225_2025-10-05_145645/
│   └── lecture_2025-10-05_183042/
├── generated_data/                 # Cut video output directory
│   ├── IMG_4225_REEL.MP4
│   ├── IMG_4225_REEL_2.MP4         # Auto-incremented versions
│   └── lecture_REEL.MP4
├── reels_extractor_env/           # Virtual environment
├── run_transcription.sh           # Helper script
├── requirements.txt               # Dependencies
├── README.md                      # This file
└── CLAUDE.md                      # Developer instructions
```

## 🎯 Use Cases

### Transcription
- **Educational Content**: Transcribe data science lectures and tutorials
- **Meeting Notes**: Convert recorded meetings to searchable text
- **Podcast Transcription**: Create text versions of audio content
- **Accessibility**: Generate captions and subtitles

### Video Cutting
- **Reels/Shorts Creation**: Extract best moments from long videos
- **Highlight Compilation**: Combine key segments into one video
- **Content Repurposing**: Create short-form content from lectures
- **Social Media**: Generate TikTok, Instagram Reels, YouTube Shorts
- **Tutorial Snippets**: Extract specific examples or demonstrations

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
