# Auto-Reel Content Selection and Rotation Fix

**Date**: 2025-11-19
**Type**: Critical Bug Fix
**Status**: Completed

## Problem Statement

Two major issues were discovered with the auto-reel generation:

1. **Video Rotation Bug**: Generated reels were displayed rotated/flipped (minor issue)
2. **LLM Selection Failure**: LLM selected the first minute (0:00-1:00) which was just casual small talk instead of the actual educational content (critical issue)

### User's Report (Hebrew Translation)
> "The generated video is also rotated (minor issue - visible in screenshot) and in general it just took the first minute, which is just small talk (the bigger issue that we probably need to break down into sub-tasks/scripts/parts to understand and solve)"

## Root Cause Analysis

### Issue 1: Video Rotation Bug

**Location**: `src/scripts/cut_video_segments.py` line 317

**Root Cause**:
- Source video (IMG_4314.MOV): Has rotation metadata (`rotation=-90`)
- FFmpeg re-encodes and applies rotation to pixels during filter processing
- But `-map_metadata 0` copies the original rotation metadata to output
- Result: "Double rotation" - pixels are rotated + metadata says rotate again
- Player applies rotation twice, causing incorrect display

**Before**:
```python
cmd = [
    'ffmpeg',
    # ... other args ...
    '-map_metadata', '0',  # Copies ALL metadata including rotation
]
```

### Issue 2: LLM Selection Failure

**Location**: `src/scripts/generate_auto_reel.py` function `analyze_with_llm()`

**Root Causes** (3 compounding issues):

#### A. Transcript Truncation (lines 247-250)
```python
limited_transcript = transcript[:4000]  # Only 1% of 400KB transcript!
```

The LLM only saw the first 4,000 characters (~2-3 minutes) of a 29-minute video.

#### B. No Timestamps in Transcript
The truncated raw text had no timestamp information. The LLM had no way to know WHERE in the video each piece of content appeared.

#### C. No Reel Suggestions
The `ai_summary.txt` file had no "REEL SUGGESTIONS" section (older format), so `ai_data['suggestions']` was empty.

**Result**: LLM defaulted to selecting 0:00-1:00 because that's all it could see.

## Solution

### Fix 1: Video Rotation

**File**: `src/scripts/cut_video_segments.py`
**Line**: 319

Clear rotation metadata since FFmpeg already applies rotation during re-encoding:

```python
# Before
'-map_metadata', '0',  # Copy metadata from input

# After
'-metadata:s:v', 'rotate=0',  # Clear rotation metadata (already applied by FFmpeg)
```

### Fix 2: Use Chunk Summaries Instead of Truncated Transcript

**New Function**: `get_chunk_summaries_from_ai_analysis()`

Instead of sending truncated raw transcript, send ALL 15 chunk summaries with timestamps:

```python
def get_chunk_summaries_from_ai_analysis(results_dir: str) -> str:
    """
    Extract per-chunk summaries with timestamps from ai_summary.txt

    This provides the LLM with a structured view of ALL video content,
    not just truncated raw transcript.
    """
    # Parse CHUNK X ANALYSIS sections
    # Format output as:
    # [0:00.00 - 2:00.00] Chunk 1:
    # Summary of this chunk...
    #
    # [2:00.00 - 4:00.00] Chunk 2:
    # Summary of this chunk...
```

**Benefits**:
- LLM sees ALL 15 chunks (entire video)
- Each chunk has timestamp range
- Summaries are already LLM-generated (quality content)

### Fix 3: Updated Function Signature

**Before**:
```python
def analyze_with_llm(transcript: str, suggestions: List[Dict],
                     min_duration: int = 45, max_duration: int = 70) -> Dict:
```

**After**:
```python
def analyze_with_llm(chunk_summaries: str, suggestions: List[Dict],
                     cumulative_summary: str = "",
                     min_duration: int = 45, max_duration: int = 70) -> Dict:
```

### Fix 4: Improved LLM Prompt

The original prompt was vague about duration. The new prompt:

1. Emphasizes duration constraint clearly
2. Explains chunks are 2 minutes each
3. Asks for SHORT PORTIONS (15-30s) not entire chunks
4. Provides example output with duration

```python
prompt = f"""You are an expert content strategist for short-form video.

GOAL: Create a SHORT reel of exactly {min_duration}-{max_duration} seconds total.

Each chunk below is ~2 minutes long. You must select SHORT PORTIONS (15-30 seconds each) from within these chunks, NOT entire chunks.

CRITICAL DURATION CONSTRAINT:
- Target: {min_duration}-{max_duration} seconds TOTAL (NOT minutes!)
- Each part should be 15-30 seconds
- Example: 3 parts of 20 seconds each = 60 seconds total
- DO NOT select entire 2-minute chunks!

EXAMPLE OUTPUT:
PARTS:
1. [8:15 - 8:35] Explains the key concept clearly (20s)
2. [12:40 - 13:00] Gives practical example (20s)
3. [18:30 - 18:50] Summarizes the insight (20s)

TOTAL_DURATION: 60s
...
"""
```

### Fix 5: Validation with Retry

**New Function**: `validate_llm_selection()`

Validates LLM output before cutting video:
- Checks all parts have parseable timestamps
- Validates duration is within range
- Validates each part is at least 5 seconds

**Retry Mechanism**:
```python
MAX_ATTEMPTS = 3
for attempt in range(1, MAX_ATTEMPTS + 1):
    llm_result = analyze_with_llm(...)
    is_valid, error_msg = validate_llm_selection(llm_result, ...)
    if is_valid:
        break
```

## Testing Results

### Before Fix
```
Selected: 0:00 - 1:00 (first minute of small talk)
Duration: 60s
Issue: Wrong content (introductions/small talk)
```

### After Fix
```
================================================================================
🎯 SELECTED REEL
================================================================================

📹 Title: Machine Learning Fundamentals
📝 Narrative: This selection focuses on three distinct but interconnected
aspects of building a machine learning model: understanding the foundation
in data (1), preparing that data for algorithms (2), and the vital process
of dividing it into training and testing sets (3).

🎬 Parts (3):
  1. 0:25 - 0:40 - "The Data Foundation" (15s)
  2. 3:15 - 3:35 - "Structuring Your Data" (20s)
  3. 6:00 - 6:25 - "Training and Testing Sets" (25s)

⏱️  Total Duration: 60s
✅ Within target range (45-70s)
```

**Key Improvements**:
- Content from multiple parts of video (0:25, 3:15, 6:00) NOT just first minute
- Each segment is ~15-25 seconds, not entire 2-minute chunks
- Coherent narrative about ML model building
- Total duration correctly within 45-70s range

## Files Modified

### 1. src/scripts/cut_video_segments.py

**Line 319**: Rotation metadata fix
```python
# Before
'-map_metadata', '0',

# After
'-metadata:s:v', 'rotate=0',
```

### 2. src/scripts/generate_auto_reel.py

**Lines 133-190**: New function `get_chunk_summaries_from_ai_analysis()`

**Lines 221-294**: Updated `analyze_with_llm()`:
- Changed function signature
- New prompt with duration emphasis and examples
- Uses chunk summaries instead of truncated transcript

**Lines 423-473**: New function `validate_llm_selection()`

**Lines 693-742**: Updated main function:
- Calls `get_chunk_summaries_from_ai_analysis()`
- Uses retry mechanism with validation
- Better error handling

## Impact Assessment

### Before
- ❌ Video displayed rotated/flipped
- ❌ LLM selected first minute (small talk)
- ❌ No validation of LLM output
- ❌ LLM only saw 1% of video content

### After
- ✅ Video displays correctly
- ✅ LLM selects actual educational content
- ✅ Validation ensures correct duration
- ✅ LLM sees ALL 15 chunks with timestamps
- ✅ Retry mechanism for robustness

## Key Learnings

1. **Provide Complete Context**: LLM needs to see ALL content (via summaries) not truncated raw text
2. **Include Timestamps**: Without timestamps, LLM can't make informed time-based decisions
3. **Be Explicit About Constraints**: "45-70 seconds" is not enough - need examples and emphasis
4. **Validate Before Acting**: Always validate LLM output before expensive operations (video cutting)
5. **Explain the Data**: LLM needed to know chunks are ~2 min to understand to select portions

## Statistics

- **Lines added**: ~200 lines
- **Functions added**: 2 (get_chunk_summaries_from_ai_analysis, validate_llm_selection)
- **Functions modified**: 2 (analyze_with_llm, main)
- **Bug fixes**: 2 (rotation, content selection)
- **Test runs**: 3
- **Final result**: Successful reel generation with correct content

## Output Files

- **Generated Reel**: `generated_data/IMG_4314_AUTO_REEL_2.MP4` (60s, correct rotation)
- **Metadata**: `generated_data/IMG_4314_AUTO_REEL_2_metadata.txt`

## Future Improvements

1. **Add debug logging**: Print LLM prompt and response for debugging
2. **Increase context window**: Use larger Ollama context for longer videos
3. **Add content scoring**: Pre-score chunks to help LLM selection
4. **Human confirmation**: Optional step before cutting
