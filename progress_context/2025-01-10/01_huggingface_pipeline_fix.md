# Bug Fix: Hugging Face Pipeline KeyError

**Date**: 2025-01-17
**Type**: Bug Fix
**Severity**: Critical
**Status**: ✅ Fixed & Tested

## Problem

When using the Hebrew-optimized model from Hugging Face, transcription failed immediately with:
```
❌ Error during transcription: 'chunks'
```

The error occurred at line 90 in `process_chunk()` when trying to iterate over `result["chunks"]`.

## Root Cause

The Hugging Face `pipeline` for automatic speech recognition returns different output formats depending on parameters:

**Without timestamps:**
```python
{'text': 'transcription here'}
```

**With timestamps:**
```python
{
    'text': 'full transcription',
    'chunks': [
        {'text': ' word', 'timestamp': (0.0, 1.1)},
        ...
    ]
}
```

The code was calling the pipeline without `return_timestamps`, so the `chunks` key didn't exist.

## Solution

### 1. Added `return_timestamps='word'` parameter
**File**: `src/scripts/transcribe_advanced.py`
**Line**: 83

```python
# Before:
result = model(audio_path, chunk_length_s=30)

# After:
result = model(audio_path, chunk_length_s=30, return_timestamps='word')
```

### 2. Updated chunk processing logic
**File**: `src/scripts/transcribe_advanced.py`
**Lines**: 85-116

**Before:**
```python
for segment in result["chunks"]:
    cleaned_text = clean_rtl_markers(segment)  # ❌ treating chunk as string
    segment_dict = {
        'start': current_time,
        'end': current_time + 30,  # ❌ approximating timestamps
        'text': cleaned_text
    }
```

**After:**
```python
# Use full text directly from result
full_text = clean_rtl_markers(result.get('text', ''))

# Process chunks with actual timestamps
for chunk in result.get("chunks", []):
    chunk_text = clean_rtl_markers(chunk['text'])
    timestamp = chunk.get('timestamp', (0, 0))

    # Handle timestamp tuple (start, end)
    if isinstance(timestamp, tuple) and len(timestamp) == 2:
        chunk_start_time, chunk_end_time = timestamp
    else:
        # Fallback if timestamp format is unexpected
        chunk_start_time = start_time
        chunk_end_time = start_time + 30

    segment_dict = {
        'start': start_time + chunk_start_time,
        'end': start_time + chunk_end_time,
        'text': chunk_text
    }
```

### 3. Added [PAD] token cleaning
**File**: `src/scripts/transcribe_advanced.py`
**Line**: 213

Hugging Face models were outputting padding tokens in the text:
```
[PAD]ז[PAD]ה[PAD] מ[PAD]ע[PAD]ו[PAD]ל[PAD]ה[PAD]
```

Extended `clean_rtl_markers()` function:
```python
def clean_rtl_markers(text: str) -> str:
    # Remove RTL markers
    text = RTL_PATTERN.sub('', text)

    # Remove [PAD] tokens from Hugging Face models
    text = text.replace('[PAD]', '')

    return text
```

## Testing

Created and ran test script (`test_hf_fix.py`) on first 30 seconds of `IMG_3738.mov`:

### Test Results ✅

```
✅ Model loaded: huggingface
✅ Processing successful!
⏱️  Processing time: 2.5s
📝 Text length: 329 chars (was 1584 with [PAD] tokens)
📊 Segments: 67 segments
🕐 Duration: 30.0s

First segment:
   Time: 0.00s - 0.38s
   Text: זה...

Full text (first 200 chars):
   זה מעולה זה טוב אני חושב בשביל ה יודע משהו...
```

**Before fix**: Text had 1584 chars with `[PAD]` tokens everywhere
**After fix**: Clean 329 chars of Hebrew text

## Impact

- Hebrew-optimized model now works correctly
- Accurate word-level timestamps
- Clean Hebrew text without padding artifacts
- Fast processing (2.5s for 30s of audio)

## Related Fixes

This fix revealed and required fixing:
- Device detection bug (`device="auto"` → proper CUDA/MPS/CPU detection)
- See: `02_device_detection_fix.md`
