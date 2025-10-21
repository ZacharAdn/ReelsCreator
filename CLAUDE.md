# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **simple Hebrew-optimized video processing toolkit** with two main tools:

1. **`transcribe_advanced.py`** - Transcribes videos with automatic chunking and real-time output
2. **`cut_video_segments.py`** - Extracts and concatenates specific time ranges to create Reels/Shorts

## Essential Commands

### Setup and Installation

```bash
# Create and activate virtual environment
python -m venv reels_extractor_env
source reels_extractor_env/bin/activate  # macOS/Linux
# or: reels_extractor_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (required for video processing)
# macOS:
brew install ffmpeg
# Ubuntu/Debian:
sudo apt install ffmpeg
```

### Running the Scripts

**Transcription (Interactive Mode):**
```bash
# Activate virtual environment first
source reels_extractor_env/bin/activate

# Run the script in interactive mode
python src/scripts/transcribe_advanced.py

# The script will:
# 1. Scan project for directories with videos
# 2. Show all directories and video count
# 3. Let you select a directory
# 4. Display all videos with duration, size, and date
# 5. Let you select a video to transcribe
```

**Video Segment Cutting (Interactive Mode):**
```bash
# Activate virtual environment
source reels_extractor_env/bin/activate

# Interactive mode (recommended)
python src/scripts/cut_video_segments.py

# The script will:
# 1. Scan data/ directory for videos
# 2. Display all videos with duration, size, and date
# 3. Let you select a video
# 4. Prompt for time ranges one by one (press Enter to finish)
# 5. Ask if you want to use FFmpeg (faster)
# 6. Generate unique output name (adds _2, _3, etc. if file exists)

# Command-line mode (for automation)
python src/scripts/cut_video_segments.py \
  --video data/IMG_4225.MP4 \
  --ranges "1:00.26-1:07.16, 1:27.64-1:31.72, 1:42.30-1:49.04, 2:00.08-2:06.68"

# Use FFmpeg for faster processing
python src/scripts/cut_video_segments.py \
  --video data/IMG_4225.MP4 \
  --ranges "1:00-2:00" \
  --use-ffmpeg
```

### Configuration

**Adjust transcription chunk size:**
Edit `CHUNK_SIZE_MINUTES` at the top of `transcribe_advanced.py`:
```python
CHUNK_SIZE_MINUTES = 2  # Change to 1, 3, 5, etc.
```

**Note:** Both scripts now run in interactive mode by default. No need to manually edit video file lists!

## Architecture

### Two-Script Design

The codebase is intentionally minimal:
- **`transcribe_advanced.py`** - Video transcription with chunk processing
- **`cut_video_segments.py`** - Video segment extraction and concatenation
- **No complex pipeline**: Direct, focused functionality
- **No external dependencies** between scripts

### Core Functionality

```
transcribe_advanced.py does everything:
1. Audio Extraction - Uses MoviePy to extract audio from video
2. Model Loading - Tries Hebrew model → Whisper turbo → Whisper large
3. Chunk Processing - Splits long videos into 2-minute chunks
4. Real-time Output - Saves results after each chunk
5. Timestamp Organization - Creates YYYY-MM-DD_HHMMSS directories
```

### Key Functions

**Transcription:**
- `load_optimal_model()` - Loads best available transcription model with fallback
- `process_chunk()` - Processes a single 2-minute audio chunk
- `ensure_results_dir()` - Creates timestamped output directory
- `write_chunk_output()` - Saves chunk results in real-time
- `format_timestamp()` - Converts seconds to MM:SS format
- `transcribe_video()` - Main orchestration function
- `interactive_mode()` - Handles directory and video selection
- `find_directories_with_videos()` - Scans project for video directories
- `scan_directory_for_videos()` - Gets video metadata in a directory
- `get_video_info()` - Extracts duration, size, and date from video

**Video Cutting:**
- `parse_timestamp()` - Converts time strings (MM:SS.MS) to seconds
- `parse_time_range()` - Parses single time range (start-end)
- `parse_ranges()` - Parses comma-separated ranges
- `cut_segments_moviepy()` - Cuts video using MoviePy
- `cut_segments_ffmpeg()` - Cuts video using FFmpeg (faster)
- `get_unique_output_path()` - Generates unique filename with _2, _3, etc.
- `interactive_mode()` - Handles video selection and range input

### Technologies Used

- **OpenAI Whisper**: Standard transcription models
- **Hugging Face Transformers**: Hebrew-optimized wav2vec2 model
- **MoviePy**: Video/audio extraction
- **FFmpeg**: Media processing backend

## Output Structure

Each run creates a timestamped directory in `/results/` with format `YYYY-MM-DD_HHMMSS_VideoName`:

```
results/2025-10-05_145645_IMG_4225/
├── chunk_01.txt              # Individual chunk transcripts
├── chunk_01_metadata.txt     # Processing stats
├── chunk_02.txt
├── chunk_02_metadata.txt
├── full_transcript.txt       # Cumulative (updated in real-time)
└── IMG_4225_final_summary.txt  # Complete results
```

## Supported Models

The script tries these models in order (automatic fallback):

1. **Hebrew-Optimized (Hugging Face)**
   - Model: `imvladikon/wav2vec2-large-xlsr-53-hebrew`
   - Best for pure Hebrew content
   - Fastest, but may fail on some systems

2. **Whisper large-v3-turbo**
   - 5.4x faster than large
   - Great for mixed Hebrew-English
   - Good balance of speed and accuracy

3. **Whisper large (fallback)**
   - Most reliable
   - Works on all systems
   - Slower but highest quality

## Common Development Patterns

When modifying this codebase:

1. **Keep it simple**: This is intentionally a single-file script
2. **Preserve fallback logic**: Hebrew model → Turbo → Large is important
3. **Maintain real-time output**: The `write_chunk_output()` calls are critical
4. **Test with Hebrew content**: Always test with mixed Hebrew-English videos
5. **Don't break timestamping**: The YYYY-MM-DD_HHMMSS directory structure is essential

## Troubleshooting Commands

```bash
# Check FFmpeg installation
ffmpeg -version

# Test Whisper installation
python -c "import whisper; print(whisper.available_models())"

# Test Hugging Face transformers
python -c "from transformers import pipeline; print('Transformers working')"

# Test MoviePy
python -c "from moviepy.editor import VideoFileClip; print('MoviePy working')"

# Check if results directory exists
ls -la results/
```

## Known Issues & Fixes

### MPS Backend Errors (M1 Mac)

**Issue**: Whisper models may fail with MPS sparse tensor errors on M1 Macs

**Solution**: The script automatically detects MPS errors and falls back to CPU. No action needed.

**Manual workaround** (if needed): The device is set to `"auto"` in the Hugging Face pipeline. You can modify this to `"cpu"` in the `load_optimal_model()` function.

### Memory Issues

**Issue**: Large videos may cause memory problems

**Solution**: Reduce `CHUNK_SIZE_MINUTES` to 1 minute for lower memory usage

### Model Download Failures

**Issue**: First run may fail to download models

**Solution**: Ensure internet connection and ~5GB free disk space. Models download automatically on first run.

## File Paths

All file paths in the codebase:

- **Scripts**:
  - `src/scripts/transcribe_advanced.py` (transcription)
  - `src/scripts/cut_video_segments.py` (video cutting)
- **Helper**: `run_transcription.sh`
- **Videos**: `data/` (default input location)
- **Transcription Output**: `results/YYYY-MM-DD_HHMMSS_VideoName/`
- **Cut Video Output**: `generated_data/VideoName_REEL.MP4` (auto-increments: `_REEL_2.MP4`, `_REEL_3.MP4`, etc.)
- **Dependencies**: `requirements.txt`

## Testing

To test the transcription script:

```bash
# 1. Place a short test video in data/
cp /path/to/test.mp4 data/

# 2. Run the script in interactive mode
source reels_extractor_env/bin/activate
python src/scripts/transcribe_advanced.py

# 3. Select data/ directory from the list
# 4. Select your test video
# 5. Check results
ls -la results/
cat results/*/full_transcript.txt
```

To test the video cutting script:

```bash
# 1. Place a test video in data/
cp /path/to/test.mp4 data/

# 2. Run the script in interactive mode
python src/scripts/cut_video_segments.py

# 3. Select your video from the list
# 4. Enter time ranges (e.g., 0:10-0:20)
# 5. Check output
ls -la generated_data/
```

## Important Reminders

1. **Results directory is sacred**: Never delete `results/` - it contains all historical transcriptions
2. **Always activate venv**: The script requires the virtual environment
3. **FFmpeg is required**: The script will fail without FFmpeg installed
4. **First run is slow**: Models download on first run (~5GB total)
5. **Chunk processing is interruptible**: Press Ctrl+C anytime - partial results are saved

## Performance Expectations

- **Short video (3-4 min)**: ~6 minutes processing time
- **Medium video (20 min)**: ~30-40 minutes processing time
- **Long video (60 min)**: ~2 hours processing time
- **Processing speed**: ~3x real-time (varies by model and hardware)

## Future Enhancements (Optional)

Potential improvements to consider:
- Command-line arguments for video path and chunk size
- Support for batch processing multiple videos
- Better error recovery and retry logic
- GPU acceleration support
- Subtitle file generation (SRT, VTT)
