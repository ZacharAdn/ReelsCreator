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
# 1. Scan project for directories with videos
# 2. Show all directories and video count
# 3. Let you select a directory
# 4. Display all videos with duration, size, and date
# 5. Let you select a video
# 6. Prompt for time ranges one by one (press Enter to finish)
# 7. Ask if you want to use FFmpeg (faster)
# 8. Generate unique output name (adds _2, _3, etc. if file exists)

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

### Optional: AI Analysis with Ollama

**NEW FEATURE**: If Ollama is installed and running, transcription automatically includes AI-powered content analysis:

```bash
# 1. Install Ollama (one-time)
brew install ollama  # macOS
# or download from https://ollama.ai

# 2. Start Ollama server
ollama serve  # Keep running in background

# 3. Download Hebrew-optimized model (one-time, ~5GB)
ollama pull aya-expanse:8b

# 4. Run transcription as normal - AI analysis is automatic!
python src/scripts/transcribe_advanced.py
```

**What you get**:
- Auto-generated summary (3-5 sentences)
- Topic extraction (3-5 main topics)
- Hashtag suggestions (5-7 hashtags)
- Reel segment recommendations (2-3 clips with timestamps)

**Output files** (added to results directory):
- `ai_summary.txt` - Summary, topics, hashtags
- `suggested_reels.txt` - Timestamped reel suggestions with copy-paste commands

**Completely optional** - Works perfectly without Ollama installed.

### NEW: Automated Reel Generation

**LATEST FEATURE**: Automatically generate optimal 45-70 second reels with AI-powered multi-part selection!

**How it works**:
1. After transcription completes, you'll be prompted: "Auto-generate best reel?"
2. AI analyzes the full transcript + all reel suggestions
3. Intelligently selects THE BEST segments (can combine 2-4 non-contiguous parts!)
4. Automatically cuts and concatenates to create perfect reel

**Example**: AI might select:
- 4:15-4:45 (hook)
- 6:20-6:55 (main explanation)
- 8:10-8:25 (conclusion)
= 65-second engaging reel!

**Standalone mode**:
```bash
python src/scripts/generate_auto_reel.py \
  --results-dir results/2025-01-17_221415_IMG_4314 \
  --video data/IMG_4314.MP4

# Or interactive (auto-detect latest):
python src/scripts/generate_auto_reel.py
```

**Output**:
- `generated_data/VideoName_AUTO_REEL.MP4` - Final reel
- `generated_data/VideoName_AUTO_REEL_metadata.txt` - Selection details

**Requirements**: Ollama with aya-expanse:8b model (same as transcription AI)

## Architecture

### Three-Script Design

The codebase is intentionally minimal:
- **`transcribe_advanced.py`** - Video transcription with chunk processing + AI analysis
- **`cut_video_segments.py`** - Video segment extraction and concatenation
- **`generate_auto_reel.py`** - **NEW!** Automated optimal reel generation (45-70s)
- **Clean separation**: Each script has focused functionality
- **Composable**: Scripts can be used together or independently

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

**AI Analysis (Optional - Ollama):**
- `OllamaAnalyzer` class - Manages Ollama LLM integration
- `OllamaAnalyzer.is_available()` - Checks if Ollama is running
- `OllamaAnalyzer.analyze()` - Generates AI analysis (summary, topics, hashtags, reels)
- `save_ollama_analysis()` - Saves AI results to files

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
├── IMG_4225_final_summary.txt  # Complete results
│
# Optional: If Ollama is available
├── ai_summary.txt            # AI-generated summary, topics, hashtags
└── suggested_reels.txt       # Reel suggestions with timestamps
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

### ✅ FIXED: Device Detection for Hugging Face Pipeline

**Previous Issue**: Hebrew model failed with "device type auto not recognized"

**Solution**: Implemented proper device detection:
```python
import torch
if torch.cuda.is_available():
    device = 0  # CUDA GPU
elif torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon
else:
    device = -1  # CPU
```

Now works correctly on M1/M2/M3 Macs with MPS acceleration.

### ✅ FIXED: Hugging Face Pipeline KeyError

**Previous Issue**: `KeyError: 'chunks'` when using Hebrew model

**Solution**: Added `return_timestamps='word'` parameter and updated chunk processing to handle timestamp tuples properly. Also removes `[PAD]` tokens from output.

### Memory Issues

**Issue**: Large videos may cause memory problems

**Solution**: Reduce `CHUNK_SIZE_MINUTES` to 1 minute for lower memory usage

### Model Download Failures

**Issue**: First run may fail to download models

**Solution**: Ensure internet connection and ~5GB free disk space. Models download automatically on first run.

## File Paths

All file paths in the codebase:

- **Scripts**:
  - `src/scripts/transcribe_advanced.py` (transcription + AI analysis)
  - `src/scripts/cut_video_segments.py` (video cutting)
  - `src/scripts/generate_auto_reel.py` (automated reel generation) **NEW!**
- **Tests**: `src/tests/test_cut_video_segments.py` (unit tests)
- **Helper**: `run_transcription.sh` (automated Ollama management)
- **Videos**: `data/` (default input location)
- **Transcription Output**: `results/YYYY-MM-DD_HHMMSS_VideoName/`
  - `ai_summary.txt` - AI-generated summary and reel suggestions
  - `suggested_reels.txt` - **NEW!** Copy-paste commands for suggested reels
  - `full_transcript.txt` - Cumulative transcript
  - `chunk_*.txt` - Individual chunk transcripts
- **Generated Reels**: `generated_data/`
  - `VideoName_REEL.MP4` - Manual cuts (auto-increments: `_REEL_2.MP4`, etc.)
  - `VideoName_AUTO_REEL.MP4` - **NEW!** AI-generated optimal reels
- **Dependencies**: `requirements.txt`
- **Progress Context**: `progress_context/YYYY-MM-DD/` (development history and bug fixes)

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

## Progress Context

**🚨 CRITICAL - DOCUMENTATION IS MANDATORY 🚨**

This project maintains detailed development history in `progress_context/`. **EVERY code change, feature addition, or bug fix MUST be documented**.

```
progress_context/
└── YYYY-MM-DD/
    ├── README.md           # Daily summary
    ├── 01_feature.md       # Individual feature/fix documentation
    ├── 02_bugfix.md        # Detailed bug analysis and solution
    └── ...
```

### Why Documentation is Non-Negotiable

1. **Knowledge Preservation**: Prevents losing context about why changes were made
2. **Regression Prevention**: Future developers understand what was fixed and why
3. **Debugging Speed**: Detailed problem descriptions help identify similar issues faster
4. **Collaboration**: Team members can understand changes without asking
5. **Learning**: Documents solutions to complex problems for future reference

### Documentation Requirements

**EVERY change must include:**
- ✅ **Problem description** with error messages (if applicable)
- ✅ **Root cause analysis** explaining what caused the issue
- ✅ **Solution** with before/after code examples
- ✅ **Testing results** showing the fix works
- ✅ **Impact assessment** (files changed, backward compatibility, etc.)

### Mandatory Workflow for Claude Code

**When working on this codebase, you MUST:**

1. ✅ **BEFORE coding**: Check `progress_context/` for recent fixes and context
2. ✅ **DURING development**: Take notes about the problem, solution approach, and decisions
3. ✅ **AFTER completion**: Create a numbered markdown file in `progress_context/YYYY-MM-DD/`
4. ✅ **Include**:
   - Clear problem statement (user's request or bug description)
   - User's original message (if in Hebrew, include translation)
   - Detailed solution with code snippets
   - Testing verification (command output showing it works)
   - Files modified with line numbers
   - Impact on other parts of the codebase

### Documentation Template

Use this structure for all documentation:

```markdown
# [Feature/Bug/Fix]: Brief Title

**Date**: YYYY-MM-DD
**Type**: Feature Enhancement | Bug Fix | Refactoring | Documentation
**Status**: Completed | In Progress | Blocked

## Problem Statement
[Clear description of what needed to be done or what was broken]

## User Request
[Original user message, with translation if Hebrew]

## Solution
[Detailed explanation of the fix/feature]

### Changes Made
[Before/after code snippets with explanations]

## Testing Results
[Command output showing the fix works]

## Files Modified
- `path/to/file.py` (lines X-Y): Description of changes

## Impact
- **User Experience**: ⬆️ Improved | ➡️ Neutral | ⬇️ Degraded
- **Code Quality**: ⬆️ Improved | ➡️ Neutral | ⬇️ Degraded
- **Breaking Changes**: ✅ Yes | ❌ No
- **Performance**: ⬆️ Faster | ➡️ Neutral | ⬇️ Slower
```

### Examples

See existing documentation:
- `progress_context/2025-01-17/06_directory_selection_feature.md` - Feature addition
- Other entries in dated directories

**❌ NEVER skip documentation - it's as important as the code itself!**

This helps maintain project knowledge and prevents regression.

## Future Enhancements (Optional)

Potential improvements to consider:
- Command-line arguments for video path and chunk size
- Support for batch processing multiple videos
- Better error recovery and retry logic
- GPU acceleration support
- Subtitle file generation (SRT, VTT)
