## Plan: Create Video Segment Cutter Script

### Goal
Create a new script `cut_video_segments.py` that takes time ranges and produces a single concatenated video with all segments.

### Input Format
```
Video: data/IMG_4225.MP4
Time ranges: 1:00.26-1:07.16, 1:27.64-1:31.72, 1:42.30-1:49.04, 2:00.08-2:06.68
Output: generated_data/IMG_4225_REEL.MP4
```

### Implementation Approach

**Option 1: MoviePy (Recommended for simplicity)**
- Pro: Simple Python API, already in requirements
- Pro: Works with existing MoviePy dependency
- Con: Re-encodes video (slower, quality loss)
- Best for: Quick implementation, small videos

**Option 2: FFmpeg filter_complex (Fastest)**
- Pro: Single command, no intermediate files
- Pro: Can use stream copy (no re-encoding)
- Con: Complex command syntax
- Best for: Production use, large videos

I'll implement **both methods** with MoviePy as default (simpler) and FFmpeg as optional (faster).

### Files to Create

1. **`src/quick scripts/cut_video_segments.py`** - Main script
   - Parse time ranges in format `MM:SS.MS-MM:SS.MS` or `M:SS.MS-M:SS.MS`
   - Support both comma-separated input and interactive mode
   - Use MoviePy `VideoFileClip.subclip()` + `concatenate_videoclips()`
   - Add optional `--use-ffmpeg` flag for advanced users
   - Create `generated_data/` directory if it doesn't exist
   - Show progress bar for processing

2. **Example usage modes:**
   ```bash
   # Interactive mode
   python "src/quick scripts/cut_video_segments.py"
   
   # Command-line mode
   python "src/quick scripts/cut_video_segments.py" \
     --video data/IMG_4225.MP4 \
     --ranges "1:00.26-1:07.16,1:27.64-1:31.72,1:42.30-1:49.04,2:00.08-2:06.68"
   
   # Using FFmpeg (faster)
   python "src/quick scripts/cut_video_segments.py" \
     --video data/IMG_4225.MP4 \
     --ranges "1:00.26-1:07.16,1:27.64-1:31.72" \
     --use-ffmpeg
   ```

3. **Update `requirements.txt`** - No changes needed (MoviePy already included)

4. **Update `README.md`** - Add section about video cutting tool

5. **Update `CLAUDE.md`** - Add instructions for the new script

### Script Features

- ✅ Parse time ranges: `MM:SS.MS` or `M:SS.MS` format
- ✅ Support both file paths: relative and absolute
- ✅ Auto-create `generated_data/` directory
- ✅ Output naming: `{VideoName}_REEL.MP4`
- ✅ Progress indication during processing
- ✅ Error handling for invalid time ranges
- ✅ Validate time ranges are within video duration
- ✅ Audio included automatically
- ✅ Two methods: MoviePy (simple) and FFmpeg (fast)

### Output Structure
```
generated_data/
├── IMG_4225_REEL.MP4
├── IMG_4216_REEL.MP4
└── lecture_REEL.MP4
```

### Implementation Details

**Time parsing function:**
```python
def parse_timestamp(ts: str) -> float:
    """Convert MM:SS.MS or M:SS.MS to seconds"""
    # Examples: "1:00.26" → 60.26, "1:27.64" → 87.64
```

**MoviePy approach:**
```python
from moviepy.editor import VideoFileClip, concatenate_videoclips

clips = []
for start, end in time_ranges:
    clips.append(video.subclip(start, end))
    
final = concatenate_videoclips(clips, method="compose")
final.write_videofile(output_path)
```

**FFmpeg approach (optional):**
```python
import subprocess

# Build filter_complex command for each segment
# Execute single FFmpeg command with all segments
```

This will give you a simple, powerful tool to create Reels from timestamped segments!